"""ModelPolicy: model-assisted decisions with a local-policy fallback.

Contract (implements `resonance.policy_types.PolicyProtocol`, kind="model"):

* The LOCAL policy is always run first, exactly once per decision. This keeps
  the local RNG stream (`ctx.rng`) advancing identically whether or not the
  model succeeds, so a model failure never shifts later local decisions, and it
  provides the temperature fields of the trace. If the model returns a valid
  answer, the trace is the local trace with `chosen`, `scores`,
  `probabilities` overridden, `policy_kind="model"` and
  `exploration_draw=None`. `temperature`/`temperature_base` remain the local
  policy's (computed, not used for the model's choice). Otherwise the local
  trace is returned unchanged except for a note explaining the fallback.
* The receiver prompt is built from `ctx.observation` ONLY (plus the public
  pattern set, the local learner's scores, the agent's own state and its own
  retrieved memories). The receiver context has no target by construction;
  any dict key containing "target" is additionally stripped from dumps.
* Every model attempt emits a `model_call` event via `ctx.emit` with the
  (redacted) request and response. Sender-side payloads contain the target and
  carry `visibility_hint: "sender_private"`.
* Retries: up to `cfg.max_retries` extra attempts on exception / provider error /
  invalid JSON / out-of-range values. Timeouts are the provider's own (HTTP
  timeout for API providers; llama.cpp has none and is bounded by max_tokens).
* Budget: every attempt counts. Once `calls_made >= cfg.call_budget` the model
  is skipped and the local decision is used; ONE `model_call` event with error
  "call budget exhausted" is emitted the first time this happens, and every
  later trace carries a note.
* Roles: the model is consulted only for roles listed in `cfg.use_for`
  ("receiver" -> choose_pattern, "sender" -> choose_motif / propose_phrase,
  "narrative" -> generate_narrative). Otherwise the local decision is returned.
"""

from __future__ import annotations

import json
import math
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from resonance import schemas as S
from resonance.models.base import REPAIRED_FLAG, ModelProvider, schema_for_prompt
from resonance.models.prompts import PromptLibrary
from resonance.models.redact import redact, redact_obj
from resonance.policy_types import EmitFn, PolicyProtocol, ReceiverContext, SenderContext

RATIONALE_MAX = 200
PHRASE_SCHEMA_MAX_NOTES = 8
DECISION_MAX_TOKENS = 80  # pattern_id + confidence + a <=200-char rationale
PHRASE_MAX_TOKENS = 448
NARRATIVE_MAX_TOKENS = 256
NARRATIVE_MAX_CHARS = 600
NARRATIVE_LABEL = "generated interpretation — not a measurement"
BUDGET_EXHAUSTED = "call budget exhausted"

STATE_DEFINITIONS: dict[str, str] = {
    "activation": "0..1, raised by dense/loud input and surprise, decays to baseline",
    "expected_value": "0..1, running expected score",
    "uncertainty": "0..1, recent prediction error",
    "affiliation": "-1..1, credit toward partner from shared outcomes",
}

_PURPOSES = set(typing.get_args(S.ModelRequest.model_fields["purpose"].annotation))


# ---------------------------------------------------------------------------
# Prompt building helpers (pure functions; unit-tested)
# ---------------------------------------------------------------------------


def strip_target_keys(obj: Any) -> Any:
    """Recursively drop dict entries whose key contains 'target' (defensive)."""
    if isinstance(obj, dict):
        return {k: strip_target_keys(v) for k, v in obj.items() if "target" not in str(k).lower()}
    if isinstance(obj, list):
        return [strip_target_keys(v) for v in obj]
    return obj


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        # 2 decimals, trailing zeros stripped (fewer prompt tokens: 0.5 not 0.50)
        out = f"{v:.2f}".rstrip("0").rstrip(".")
        return "0" if out in ("", "-0") else out
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def feature_summary(obs: S.Observation) -> str:
    """Summary of what the receiver observed, from the Observation ONLY.

    Music channel: the measured SymbolicFeatures fields (2 decimals). The
    normalised vector phi is omitted because it re-encodes the same fields and
    doubles prompt length (prompt evaluation dominates local-model latency)."""
    if obs.channel == "symbol":
        return (
            f"symbol #{obs.symbol_id} (an abstract message identity chosen by the partner; "
            f"no musical content)"
        )
    if obs.features is None:
        return "no features available for this observation"
    fields = strip_target_keys(obs.features.model_dump(exclude={"vector", "kind"}))
    return "\n".join(f"- {k}: {_fmt(v)}" for k, v in fields.items())


def state_summary(state: S.StateVector) -> str:
    d = state.model_dump()
    return "\n".join(
        f"- {k} = {_fmt(float(d[k]))} ({STATE_DEFINITIONS.get(k, '')})" for k in S.STATE_DIMS if k in d
    )


def memories_summary(retrieved: list[S.RetrievedMemory], k: int) -> str:
    if not retrieved or k <= 0:
        return "(none)"
    lines = []
    for m in retrieved[:k]:
        it = m.item
        lines.append(
            f"- step {it.step}: role={it.role} action={it.action} score={_fmt(float(it.score))} "
            f"similarity={_fmt(float(m.similarity))}"
        )
    return "\n".join(lines)


def patterns_summary(patterns: list[S.TimingPattern]) -> str:
    return "\n".join(f"- {p.id}: {p.name}, onsets {_fmt([float(o) for o in p.onsets])}" for p in patterns)


def indexed_values(values: list[float], label: str) -> str:
    if not values:
        return "(none)"
    return "\n".join(f"- {label} {i}: {_fmt(float(v))}" for i, v in enumerate(values))


def choice_schema(field: str, n: int) -> dict[str, Any]:
    # `enum` (not minimum/maximum) because llama.cpp's grammar converter enforces enum
    # but ignores numeric bounds; every field is re-validated in `validate_choice`.
    return {
        "type": "object",
        "properties": {
            field: {"type": "integer", "enum": list(range(n))},
            "confidence": {"type": "number"},
            "rationale": {"type": "string", "maxLength": RATIONALE_MAX},
        },
        "required": [field, "confidence", "rationale"],
        "additionalProperties": False,
    }


def phrase_schema(length_beats: float, tempo_min: float, tempo_max: float) -> dict[str, Any]:
    """Schema sent to the model. Ranges are expressed as enums on a grid (quarter
    beats, 5-bpm tempo steps) so llama.cpp's grammar can enforce them, and the
    schema asks for at most PHRASE_SCHEMA_MAX_NOTES notes to bound generation time.
    `validate_phrase` applies the full contract (1-16 notes, any value in range)."""
    grid = [round(i * 0.25, 2) for i in range(int(length_beats / 0.25))]
    tempos = [float(t) for t in range(int(math.ceil(tempo_min / 5) * 5), int(tempo_max) + 1, 5)]
    return {
        "type": "object",
        "properties": {
            "tempo_bpm": {"type": "number", "enum": tempos},
            "notes": {
                "type": "array",
                "minItems": 1,
                "maxItems": PHRASE_SCHEMA_MAX_NOTES,
                "items": {
                    "type": "object",
                    "properties": {
                        "pitch": {"type": "integer", "enum": list(range(48, 85))},
                        "onset": {"type": "number", "enum": grid},
                        "duration": {"type": "number", "enum": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]},
                        "velocity": {"type": "integer", "enum": list(range(30, 121, 10))},
                    },
                    "required": ["pitch", "onset", "duration", "velocity"],
                    "additionalProperties": False,
                },
            },
            "rationale": {"type": "string", "maxLength": RATIONALE_MAX},
        },
        "required": ["tempo_bpm", "notes", "rationale"],
        "additionalProperties": False,
    }


NARRATIVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"narrative": {"type": "string", "maxLength": NARRATIVE_MAX_CHARS}},
    "required": ["narrative"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Validation (strict; returns (value, None) or (None, reason))
# ---------------------------------------------------------------------------


def _as_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return None


def _as_num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def validate_choice(
    parsed: dict[str, Any] | None, field: str, n: int
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(parsed, dict):
        return None, "response is not a JSON object"
    idx = _as_int(parsed.get(field))
    if idx is None:
        return None, f"{field} missing or not an integer"
    if not 0 <= idx < n:
        return None, f"{field}={idx} out of range [0, {n - 1}]"
    conf = _as_num(parsed.get("confidence"))
    if conf is None or not 0.0 <= conf <= 1.0:
        return None, "confidence missing or outside [0, 1]"
    rationale = parsed.get("rationale", "")
    if not isinstance(rationale, str):
        return None, "rationale is not a string"
    return {
        field: idx,
        "confidence": conf,
        "rationale": rationale[:RATIONALE_MAX],
        "rationale_truncated": len(rationale) > RATIONALE_MAX or bool(parsed.get(REPAIRED_FLAG)),
    }, None


def validate_phrase(
    parsed: dict[str, Any] | None, length_beats: float, tempo_min: float, tempo_max: float
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(parsed, dict):
        return None, "response is not a JSON object"
    tempo = _as_num(parsed.get("tempo_bpm"))
    if tempo is None or not tempo_min <= tempo <= tempo_max:
        return None, f"tempo_bpm missing or outside [{tempo_min}, {tempo_max}]"
    notes = parsed.get("notes")
    if not isinstance(notes, list) or not 1 <= len(notes) <= 16:
        return None, "notes must be a list of 1-16 notes"
    out_notes = []
    for i, n in enumerate(notes):
        if not isinstance(n, dict):
            return None, f"note {i} is not an object"
        pitch, vel = _as_int(n.get("pitch")), _as_int(n.get("velocity"))
        onset, dur = _as_num(n.get("onset")), _as_num(n.get("duration"))
        if pitch is None or not 48 <= pitch <= 84:
            return None, f"note {i}: pitch missing or outside [48, 84]"
        if vel is None or not 30 <= vel <= 120:
            return None, f"note {i}: velocity missing or outside [30, 120]"
        if onset is None or onset < 0:
            return None, f"note {i}: onset missing or negative"
        if dur is None or not 0.25 <= dur <= 4.0:
            return None, f"note {i}: duration missing or outside [0.25, 4]"
        if onset + dur > length_beats + 1e-9:
            return None, f"note {i}: onset+duration={onset + dur:g} exceeds length {length_beats:g} beats"
        out_notes.append({"pitch": pitch, "onset": onset, "duration": dur, "velocity": vel})
    rationale = parsed.get("rationale", "")
    rationale = rationale[:RATIONALE_MAX] if isinstance(rationale, str) else ""
    return {"tempo_bpm": tempo, "notes": out_notes, "rationale": rationale}, None


def validate_narrative(parsed: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("narrative"), str):
        return None, "narrative missing or not a string"
    text = parsed["narrative"].strip()
    if not text:
        return None, "narrative is empty"
    return {"narrative": text[:NARRATIVE_MAX_CHARS]}, None


def narrative_event_payload(text: str, provider: str, model_id: str) -> dict[str, Any]:
    """Payload for a `generated_narrative` event. Never feed it into state or learning."""
    return redact_obj({"text": text, "provider": provider, "model_id": model_id, "label": NARRATIVE_LABEL})


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


@dataclass
class CallOutcome:
    value: dict[str, Any] | None
    reason: str | None
    attempts: int
    response: S.ModelResponse | None = None


class ModelPolicy:
    kind = "model"

    def __init__(
        self, provider: ModelProvider, fallback: PolicyProtocol, cfg: S.ModelConfig, prompts: PromptLibrary
    ):
        self.provider = provider
        self.fallback = fallback
        self.cfg = cfg
        self.prompts = prompts
        self.calls_made = 0
        self._budget_event_emitted = False

    # -- bookkeeping ---------------------------------------------------------

    @property
    def budget_remaining(self) -> int:
        return max(0, int(self.cfg.call_budget) - self.calls_made)

    def uses(self, role: str) -> bool:
        return role in self.cfg.use_for

    def status(self) -> S.ModelStatus:
        st = self.provider.status()
        return st.model_copy(update={"calls_made": self.calls_made, "call_budget": int(self.cfg.call_budget)})

    def _request(
        self, purpose: str, system: str, user: str, schema: dict[str, Any] | None, max_tokens: int
    ) -> tuple[S.ModelRequest, str]:
        # ModelRequest.purpose is a closed Literal in schemas.py (read at import time);
        # any purpose outside it is sent as "choose_pattern" and the true purpose is
        # recorded in the model_call event payload.
        req_purpose = purpose if purpose in _PURPOSES else "choose_pattern"
        req = S.ModelRequest(
            purpose=req_purpose,
            system_prompt=system,
            user_prompt=user,
            json_schema=schema,
            input_modality="symbolic_features",
            max_tokens=max_tokens,
            temperature=float(self.cfg.temperature),
        )
        return req, purpose

    def _emit(self, emit: EmitFn, payload: dict[str, Any]) -> None:
        try:
            emit("model_call", redact_obj(payload))
        except Exception:  # an emit failure must never break a decision
            pass

    def _call(
        self,
        *,
        purpose: str,
        role: str,
        agent_id: str | None,
        step: int | None,
        system: str,
        user: str,
        schema: dict[str, Any] | None,
        max_tokens: int,
        validator: Callable[[dict[str, Any] | None], tuple[dict[str, Any] | None, str | None]],
        emit: EmitFn,
    ) -> CallOutcome:
        req, true_purpose = self._request(purpose, system, user, schema, max_tokens)
        max_attempts = 1 + max(0, int(self.cfg.max_retries))
        base = {
            "purpose": true_purpose,
            "role": role,
            "agent_id": agent_id,
            "step": step,
            "provider": self.provider.name,
            "model_id": self.provider.model_id,
            "input_modality": req.input_modality,
            "accepts_audio": bool(self.provider.accepts_audio),
            "visibility_hint": {"sender": "sender_private", "narrative": "experimenter"}.get(role, "public"),
            "max_attempts": max_attempts,
            "call_budget": int(self.cfg.call_budget),
        }
        reason: str | None = None
        last: S.ModelResponse | None = None
        attempts = 0
        for attempt in range(1, max_attempts + 1):
            if self.calls_made >= self.cfg.call_budget:
                reason = BUDGET_EXHAUSTED
                if not self._budget_event_emitted:
                    self._budget_event_emitted = True
                    self._emit(
                        emit,
                        {
                            **base,
                            "attempt": attempt,
                            "request": None,
                            "response": {
                                "content": "",
                                "parsed": None,
                                "latency_ms": 0.0,
                                "usage": {},
                                "error": BUDGET_EXHAUSTED,
                            },
                            "ok": False,
                            "fallback_used": True,
                            "will_retry": False,
                            "validation_error": None,
                            "calls_made": self.calls_made,
                            "budget_remaining": 0,
                            "tested_in_this_environment": False,
                        },
                    )
                break
            self.calls_made += 1
            attempts = attempt
            try:
                resp = self.provider.complete(req)
            except Exception as exc:  # providers should not raise, but never trust that
                resp = S.ModelResponse(
                    ok=False,
                    provider=self.provider.name,
                    model_id=self.provider.model_id,
                    error=f"exception: {type(exc).__name__}: {exc}",
                )
            last = resp
            if resp.ok:
                value, why = validator(resp.parsed)
            else:
                value, why = None, resp.error or "provider returned ok=False"
            final = value is not None or attempt == max_attempts or self.calls_made >= self.cfg.call_budget
            self._emit(
                emit,
                {
                    **base,
                    "attempt": attempt,
                    "request": {"system": req.system_prompt, "user": req.user_prompt},
                    "response": {
                        "content": resp.content,
                        "parsed": resp.parsed,
                        "latency_ms": round(resp.latency_ms, 1),
                        "usage": resp.usage,
                        "error": resp.error,
                    },
                    "ok": value is not None,
                    "validation_error": None if resp.ok is False else why,
                    "fallback_used": value is None and final,
                    "will_retry": value is None and not final,
                    "calls_made": self.calls_made,
                    "budget_remaining": self.budget_remaining,
                    "tested_in_this_environment": bool(resp.tested_in_this_environment),
                },
            )
            if value is not None:
                return CallOutcome(value, None, attempt, resp)
            reason = why
        return CallOutcome(None, reason, attempts, last)

    def _model_trace(
        self, local: S.PolicyTrace, chosen: int, confidence: float, n: int, notes: list[str]
    ) -> S.PolicyTrace:
        # One-hot-ish representation of the model's stated confidence (floored at 1/n so
        # the chosen option is always the argmax). Not a sampled distribution.
        p_chosen = max(confidence, 1.0 / n) if n > 0 else 1.0
        rest = (1.0 - p_chosen) / (n - 1) if n > 1 else 0.0
        probs = [p_chosen if i == chosen else rest for i in range(n)]
        return local.model_copy(
            update={
                "policy_kind": "model",
                "chosen": chosen,
                "scores": list(probs),
                "probabilities": probs,
                "exploration_draw": None,
                "notes": [*local.notes, *notes],
            }
        )

    def _fallback_trace(self, local: S.PolicyTrace, note: str) -> S.PolicyTrace:
        return local.model_copy(update={"notes": [*local.notes, note]})

    def _decision_notes(self, kind: str, value: dict[str, Any], field: str, local_choice: int) -> list[str]:
        notes = [
            f"model decision via {self.provider.name}/{self.provider.model_id} "
            f"(input: symbolic features, no audio): {kind} {value[field]}, stated confidence "
            f"{value['confidence']:.2f}; local policy had sampled {local_choice}",
            f"model rationale (generated text, not a measurement): {redact(value['rationale'])}",
            "temperature fields are the local policy's (computed, not used for this choice)",
        ]
        if value.get("rationale_truncated"):
            notes.append(
                f"rationale truncated (output cut at max_tokens or longer than {RATIONALE_MAX} chars)"
            )
        return notes

    # -- PolicyProtocol ------------------------------------------------------

    def choose_pattern(self, ctx: ReceiverContext) -> S.PolicyTrace:
        local = self.fallback.choose_pattern(ctx)  # exactly once; advances ctx.rng as in a local run
        if not self.uses("receiver"):
            return self._fallback_trace(local, "model not consulted: 'receiver' not in model.use_for")
        system, user, schema = self.receiver_prompts(ctx)
        out = self._call(
            purpose="choose_pattern",
            role="receiver",
            agent_id=ctx.agent_id,
            step=ctx.step,
            system=system,
            user=user,
            schema=schema,
            max_tokens=DECISION_MAX_TOKENS,
            validator=lambda p: validate_choice(p, "pattern_id", ctx.n_patterns),
            emit=ctx.emit,
        )
        if out.value is None:
            return self._fallback_trace(
                local, f"model fallback -> local policy ({out.reason}; attempts={out.attempts})"
            )
        return self._model_trace(
            local,
            out.value["pattern_id"],
            out.value["confidence"],
            ctx.n_patterns,
            self._decision_notes("pattern", out.value, "pattern_id", local.chosen),
        )

    def choose_motif(self, ctx: SenderContext) -> S.PolicyTrace:
        local = self.fallback.choose_motif(ctx)
        if not self.uses("sender"):
            return self._fallback_trace(local, "model not consulted: 'sender' not in model.use_for")
        system, user, schema = self.sender_prompts(ctx)
        out = self._call(
            purpose="choose_motif",
            role="sender",
            agent_id=ctx.agent_id,
            step=ctx.step,
            system=system,
            user=user,
            schema=schema,
            max_tokens=DECISION_MAX_TOKENS,
            validator=lambda p: validate_choice(p, "motif_index", ctx.n_motifs),
            emit=ctx.emit,
        )
        if out.value is None:
            return self._fallback_trace(
                local, f"model fallback -> local policy ({out.reason}; attempts={out.attempts})"
            )
        return self._model_trace(
            local,
            out.value["motif_index"],
            out.value["confidence"],
            ctx.n_motifs,
            self._decision_notes("motif", out.value, "motif_index", local.chosen),
        )

    def propose_phrase(self, ctx: SenderContext) -> S.Phrase | None:
        """Ask the model to compose a phrase. Returns None (caller uses the local
        phrase) on any failure, when 'sender' is not in use_for, or on budget."""
        if not self.uses("sender"):
            return None
        music = ctx.config.music
        length, tmin, tmax = float(music.phrase_length_beats), float(music.tempo_min), float(music.tempo_max)
        schema = phrase_schema(length, tmin, tmax)
        system = self.prompts.render(
            "propose_phrase.system",
            length_beats=_fmt(length),
            tempo_min=_fmt(tmin),
            tempo_max=_fmt(tmax),
            schema=schema_for_prompt(schema),
        )
        user = self.prompts.render(
            "propose_phrase.user",
            step=ctx.step,
            partner_id=ctx.partner_id,
            target_id=ctx.target_id,
            target_name=ctx.target_pattern.name,
            target_onsets=_fmt([float(o) for o in ctx.target_pattern.onsets]),
            length_beats=_fmt(length),
            motifs=self._motif_list(ctx),
            state=state_summary(ctx.state),
        )
        out = self._call(
            purpose="propose_phrase",
            role="sender",
            agent_id=ctx.agent_id,
            step=ctx.step,
            system=system,
            user=user,
            schema=schema,
            max_tokens=PHRASE_MAX_TOKENS,
            validator=lambda p: validate_phrase(p, length, tmin, tmax),
            emit=ctx.emit,
        )
        if out.value is None:
            return None
        instrument = next((a.instrument for a in ctx.config.agents if a.id == ctx.agent_id), "pluck")
        try:
            return S.Phrase(
                id=f"model-{ctx.agent_id}-{ctx.step}",
                notes=[S.Note(**n) for n in out.value["notes"]],
                tempo_bpm=out.value["tempo_bpm"],
                length_beats=length,
                instrument=instrument,
                origin=S.PhraseOrigin(kind="model", agent_id=ctx.agent_id),
                tags=["model_proposed"],
            )
        except Exception:
            return None

    def generate_narrative(
        self,
        episode_summary: dict[str, Any],
        state_before: dict[str, Any],
        state_after: dict[str, Any],
        emit: EmitFn = lambda _t, _p: None,
        step: int | None = None,
    ) -> dict[str, Any] | None:
        """Budgeted narrative. Returns a `generated_narrative` payload or None.
        Only runs when cfg.narrative_enabled and 'narrative' in cfg.use_for."""
        if not narrative_enabled(self.cfg):
            return None
        system, user = narrative_prompts(self.prompts, episode_summary, state_before, state_after)
        out = self._call(
            purpose="narrative",
            role="narrative",
            agent_id=None,
            step=step,
            system=system,
            user=user,
            schema=NARRATIVE_SCHEMA,
            max_tokens=NARRATIVE_MAX_TOKENS,
            validator=validate_narrative,
            emit=emit,
        )
        if out.value is None:
            return None
        return narrative_event_payload(out.value["narrative"], self.provider.name, self.provider.model_id)

    # -- prompts (public so tests and docs can inspect exactly what is sent) --

    def receiver_prompts(self, ctx: ReceiverContext) -> tuple[str, str, dict[str, Any]]:
        n = ctx.n_patterns
        schema = choice_schema("pattern_id", n)
        k = int(ctx.config.memory.retrieval_k)
        system = self.prompts.render(
            "choose_pattern.system", n_patterns=n, max_pattern_id=n - 1, schema=schema_for_prompt(schema)
        )
        user = self.prompts.render(
            "choose_pattern.user",
            step=ctx.step,
            channel=ctx.observation.channel,
            feature_summary=feature_summary(ctx.observation),
            patterns=patterns_summary(ctx.patterns[:n]),
            learner_scores=indexed_values(list(ctx.learner_scores), "pattern"),
            state=state_summary(ctx.state),
            memories=memories_summary(list(ctx.retrieved), k),
        )
        return system, user, schema

    def sender_prompts(self, ctx: SenderContext) -> tuple[str, str, dict[str, Any]]:
        n = ctx.n_motifs
        schema = choice_schema("motif_index", n)
        k = int(ctx.config.memory.retrieval_k)
        system = self.prompts.render(
            "choose_motif.system", max_motif_index=n - 1, schema=schema_for_prompt(schema)
        )
        user = self.prompts.render(
            "choose_motif.user",
            step=ctx.step,
            partner_id=ctx.partner_id,
            target_id=ctx.target_id,
            target_name=ctx.target_pattern.name,
            target_onsets=_fmt([float(o) for o in ctx.target_pattern.onsets]),
            motifs=self._motif_list(ctx),
            learner_values=indexed_values(list(ctx.learner_values), "motif"),
            state=state_summary(ctx.state),
            memories=memories_summary(list(ctx.retrieved), k),
        )
        return system, user, schema

    @staticmethod
    def _motif_list(ctx: SenderContext) -> str:
        names = list(ctx.motif_names) or [f"motif {i}" for i in range(ctx.n_motifs)]
        return "\n".join(f"- {i}: {name}" for i, name in enumerate(names[: ctx.n_motifs]))


def narrative_enabled(cfg: S.ModelConfig) -> bool:
    return bool(cfg.narrative_enabled) and "narrative" in cfg.use_for


def narrative_prompts(
    prompts: PromptLibrary,
    episode_summary: dict[str, Any],
    state_before: dict[str, Any],
    state_after: dict[str, Any],
) -> tuple[str, str]:
    def dump(x: Any) -> str:
        return json.dumps(x, default=str, indent=None, separators=(", ", ": "))

    system = prompts.render("narrative.system", schema=schema_for_prompt(NARRATIVE_SCHEMA))
    user = prompts.render(
        "narrative.user",
        episode_summary=dump(episode_summary),
        state_before=dump(state_before),
        state_after=dump(state_after),
    )
    return system, user
