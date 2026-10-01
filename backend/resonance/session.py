"""A session: two agents, one task, one append-only event log.

`step()` runs exactly the episode protocol of docs/CONTRACTS.md:

  1. target drawn (rng_env)                         -> target_assigned (sender_private)
  2. sender policy picks a motif, phrase generated  -> phrase_sent
  3. channel: perturbation / symbol; Observation    -> phrase_received (payload.observation only)
  4. receiver policy scores patterns from x         -> action_chosen
  5. task scores chosen vs target                   -> outcome
  6. per agent: learning update, state update, memory append -> learning_update, state_update
  7.                                                -> episode_complete; roles alternate (A sends on even steps)

RNG streams: SeedSequence(seed).spawn(5) -> rng_env, rng_policy_A, rng_policy_B, rng_gen, rng_perturb.
Same seed + same config => identical event payloads (timestamps `t` aside).
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from resonance.agents.agent import Agent
from resonance.agents.learner import encode_observation, reference_vector
from resonance.agents.policy import expressive_modulation, temperature
from resonance.agents.state import StateDrive, StateEngine
from resonance.channel import (
    music_observation,
    nearest_motif,
    partner_similarity,
    perturb,
    symbol_observation,
)
from resonance.conditions import effective_settings, with_description
from resonance.env.timing_task import TimingTask
from resonance.music.features import symbolic_features
from resonance.music.motifs import generate, get_bank, motif_index_from_id, motif_names
from resonance.music.motifs import motif_id as make_motif_id
from resonance.policy_types import ReceiverContext, SenderContext
from resonance.schemas import (
    AgentInspection,
    ConditionSpec,
    EpisodeSummary,
    Event,
    ExperimentConfig,
    HumanPhraseRequest,
    Intervention,
    MemoryItem,
    Observation,
    Phrase,
    PhraseOrigin,
    PolicyTrace,
    RunExport,
    SessionMetrics,
    SessionSnapshot,
    SessionSummary,
    StateUpdateInputs,
    StateVector,
)

# Config paths that change array shapes or identities; not changeable on a live session.
STRUCTURAL_PATHS = ("task", "agents", "seed", "music.n_motifs", "music.motif_bank")
ROLLING_WINDOW = 20


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _dump(model: Any) -> Any:
    return model.model_dump(mode="json")


@dataclass
class EpisodeScript:
    """Preset scripting: force the sender's motif and/or the target (None = normal behaviour)."""

    motif_index: int | None = None
    target_id: int | None = None
    note: str = ""


class Session:
    def __init__(
        self,
        config: ExperimentConfig,
        condition: ConditionSpec,
        seed: int,
        preset: str | None = None,
        mode: str = "live",
        *,
        session_id: str | None = None,
        script: EpisodeScript | None = None,
        allow_model: bool = True,
        created_at: str | None = None,
        reset: bool = False,
    ) -> None:
        if len(config.agents) != 2:
            raise ValueError("a session needs exactly two agents")
        self.id = session_id or uuid.uuid4().hex[:12]
        self.created_at = created_at or now_iso()
        self.mode = mode
        self.preset = preset
        self.seed = int(seed)
        self.condition = with_description(condition)
        self.config = config.model_copy(update={"seed": self.seed}, deep=True)
        self.initial_config = self.config.model_copy(deep=True)
        self.script = script
        self.task = TimingTask(self.config.task.n_patterns, self.config.task.partial_credit)
        self.settings = effective_settings(self.condition, self.config)
        streams = [np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(5)]
        self.rng_env, rng_a, rng_b, self.rng_gen, self.rng_perturb = streams
        self.agents = [self._make_agent(i, p, allow_model and mode == "live") for i, p in enumerate(self.config.agents)]
        self.agent_by_id = {a.id: a for a in self.agents}
        self.rng_policy = {self.agents[0].id: rng_a, self.agents[1].id: rng_b}
        self.events: list[Event] = []
        self.step_index = 0
        self.scores: list[float] = []
        self.current_episode: EpisodeSummary | None = None
        self.pending_phrase: Phrase | None = None
        self.channel_transform: str = "none"
        self.sent_phrases: dict[str, Phrase] = {}
        self._recorded: dict[int, list[Event]] = {}
        self._replay_steps: list[int] = []
        self._replay_cursor = 0
        if mode != "replay":
            self._log_creation(reset)

    # ------------------------------------------------------------------ setup
    def _make_agent(self, index: int, params: Any, allow_model: bool) -> Agent:
        return Agent(
            params,
            self.config,
            channel=self.settings["channel"],
            memory_capacity=self.settings["memory_capacity"],
            coupling_enabled=self.settings["coupling_enabled"],
            frozen_state=self.settings["state_frozen"],
            init_seed=self.seed * 1000 + index,
            allow_model=allow_model,
        )

    def _log_creation(self, reset: bool) -> None:
        payload = {
            "seed": self.seed,
            "condition": _dump(self.condition),
            "settings": dict(self.settings),
            "preset": self.preset,
            "mode": self.mode,
            "agents": [a.id for a in self.agents],
            "script": None if self.script is None else self.script.__dict__,
        }
        self._emit("session_reset" if reset else "session_created", payload, visibility="experimenter")
        for agent in self.agents:
            for warning in agent.init_warnings:
                self._emit("model_call", warning, agent_id=agent.id, visibility="experimenter")

    @property
    def reference(self) -> np.ndarray:
        m = self.config.music
        return reference_vector(m.motif_bank, m.n_motifs, m.feature_weights)

    def _emit(
        self, type_: str, payload: dict[str, Any], agent_id: str | None = None, visibility: str = "public"
    ) -> Event:
        ev = Event(
            seq=len(self.events),
            step=self.step_index,
            t=now_iso(),
            type=type_,  # type: ignore[arg-type]
            agent_id=agent_id,
            visibility=visibility,  # type: ignore[arg-type]
            payload=payload,
        )
        self.events.append(ev)
        return ev

    def _emitter(self, agent_id: str) -> Callable[[str, dict[str, Any]], Event]:
        """emit() handed to policies (model_call events). Honours payload["visibility_hint"]
        (e.g. "sender_private" for sender-side model calls that contain the target)."""

        def emit(type_: str, payload: dict[str, Any]) -> Event:
            hint = payload.get("visibility_hint")
            visibility = hint if hint in ("public", "sender_private", "experimenter") else "experimenter"
            return self._emit(type_, payload, agent_id=agent_id, visibility=visibility)

        return emit

    def roles(self, step: int) -> tuple[Agent, Agent]:
        """(sender, receiver): agent 0 sends on even steps, agent 1 on odd steps."""
        return (self.agents[0], self.agents[1]) if step % 2 == 0 else (self.agents[1], self.agents[0])

    # ------------------------------------------------------------------ episodes
    def step(self, n: int = 1) -> list[Event]:
        if self.mode == "replay":
            raise RuntimeError("replay sessions advance with replay_step()")
        start = len(self.events)
        for _ in range(n):
            self._episode()
        return self.events[start:]

    def human_phrase(self, req: HumanPhraseRequest) -> list[Event]:
        """One episode with a human-supplied phrase as the message (sender agent skipped)."""
        if self.mode == "replay":
            raise RuntimeError("cannot add human phrases to a replay session")
        start = len(self.events)
        self._episode(human=req)
        return self.events[start:]

    def _episode(self, human: HumanPhraseRequest | None = None) -> None:
        k = self.step_index
        sender, receiver = self.roles(k)
        if human is not None and human.receiver_id is not None:
            receiver = self.agent_by_id[human.receiver_id]
            sender = self.agents[1] if receiver is self.agents[0] else self.agents[0]
        target = self._draw_target(human)
        pattern = self.task.patterns()[target]
        self._emit(
            "target_assigned",
            {"target_id": target, "pattern": _dump(pattern), "sender": "human" if human else sender.id},
            agent_id=None if human else sender.id,
            visibility="sender_private",
        )
        sender_info: dict[str, Any] = {"step": k, "role": "sender", "target_id": target}
        if human is None:
            sender.information_received.append(sender_info)

        # 2. sender
        phrase, s_trace, motif_idx, sender_chose = self._sender_turn(sender, receiver, target, human)
        self.sent_phrases[phrase.id] = phrase
        if human is None:
            sender.last_trace = s_trace
        self._emit(
            "phrase_sent",
            {"phrase": _dump(phrase), "motif_index": motif_idx, "trace": _dump(s_trace)},
            agent_id=None if human else sender.id,
        )

        # 3. channel
        _heard, obs = self._channel(phrase, motif_idx, sender)
        obs_dump = _dump(obs)
        self._emit("phrase_received", {"observation": obs_dump}, agent_id=receiver.id)
        receiver_info: dict[str, Any] = {"step": k, "role": "receiver", "observation": obs_dump}
        receiver.information_received.append(receiver_info)

        # 4. receiver
        x = encode_observation(obs, self.config.music.n_motifs, self.reference)
        r_trace = self._receiver_turn(receiver, sender, obs, x)
        self._emit("action_chosen", {"trace": _dump(r_trace)}, agent_id=receiver.id)

        # 5. outcome
        score = round(self.task.score(r_trace.chosen, target), 6)
        self._emit(
            "outcome",
            {
                "target_id": target,
                "chosen_pattern_id": r_trace.chosen,
                "score": score,
                "exact": score == 1.0,
                "note": "target_id is shown to the experimenter; the receiver learns only from its own choice and the score",
            },
            agent_id=receiver.id,
            visibility="experimenter",
        )
        receiver_info["feedback"] = {"own_choice": r_trace.chosen, "score": score}
        sender_info["feedback"] = {"score": score}

        # 6. learning + state + memory
        receiver_state_before = receiver.state
        sent_vec, heard_vec = self._episode_vectors(phrase, obs, motif_idx)
        if human is None:
            sim = self._similarity(sent_vec, receiver.last_sent_vector)
            self._update_sender(sender, receiver, phrase, motif_idx, target, score, sent_vec, sim, learn=sender_chose)
            sender.last_sent_vector = sent_vec if obs.channel == "music" else None
        sim_r = self._similarity(heard_vec, receiver.last_sent_vector)
        self._update_receiver(receiver, sender, phrase, obs, x, r_trace.chosen, score, heard_vec, sim_r)

        # 7. complete
        summary = EpisodeSummary(
            step=k,
            sender_id=sender.id,
            receiver_id=receiver.id,
            target_id=target,
            chosen_pattern_id=r_trace.chosen,
            score=score,
            phrase=phrase,
            observation=obs,
        )
        self.current_episode = summary
        self.scores.append(score)
        self._emit("episode_complete", _dump(summary))
        self._maybe_narrative(summary, receiver, receiver_state_before)
        self.step_index += 1

    def _maybe_narrative(self, summary: EpisodeSummary, receiver: Agent, before: StateVector) -> None:
        """GENERATED narrative (WS3 model layer), only when model.narrative_enabled. Logged as a
        `generated_narrative` event for the experimenter; never read back by the simulation."""
        if not self.config.model.narrative_enabled:
            return
        policy = next((a.policy for a in self.agents if hasattr(a.policy, "generate_narrative")), None)
        if policy is None:
            return
        try:
            payload = policy.generate_narrative(  # type: ignore[attr-defined]
                _dump(summary), _dump(before), _dump(receiver.state), emit=self._emitter(receiver.id), step=summary.step
            )
        except Exception as exc:  # narrative must never break a run
            self._emit("model_call", {"ok": False, "purpose": "narrative", "error": f"{type(exc).__name__}: {exc}"},
                       agent_id=receiver.id, visibility="experimenter")
            return
        if payload:
            self._emit("generated_narrative", {**payload, "kind": "generated_narrative"}, agent_id=receiver.id,
                       visibility="experimenter")

    def _draw_target(self, human: HumanPhraseRequest | None) -> int:
        if human is not None and human.target_id is not None:
            if not 0 <= human.target_id < self.task.n_patterns:
                raise ValueError(f"target_id must be in [0, {self.task.n_patterns - 1}]")
            return human.target_id
        if human is None and self.script is not None and self.script.target_id is not None:
            return self.script.target_id
        return self.task.draw_target(self.rng_env)

    # ------------------------------------------------------------------ sender
    def _sender_turn(
        self, sender: Agent, receiver: Agent, target: int, human: HumanPhraseRequest | None
    ) -> tuple[Phrase, PolicyTrace, int | None, bool]:
        """Returns (phrase, trace, motif_index or None, whether the sender policy chose)."""
        k = self.step_index
        values = [float(v) for v in sender.sender_learner.values(target)]
        bank = self.config.music.motif_bank
        if human is not None:
            phrase = human.phrase.model_copy(update={"origin": PhraseOrigin(kind="human")})
            idx = motif_index_from_id(phrase.motif_id, bank)
            return phrase, self._fixed_trace(sender, values, idx, "human", "phrase supplied by a human; sender agent not consulted"), idx, False
        if self.pending_phrase is not None:
            src, self.pending_phrase = self.pending_phrase, None
            phrase = src.model_copy(
                update={
                    "id": f"replay-{k}-{src.id}",
                    "origin": PhraseOrigin(kind="replay", agent_id=sender.id, source_phrase_id=src.id),
                }
            )
            idx = motif_index_from_id(phrase.motif_id, bank)
            return phrase, self._fixed_trace(sender, values, idx, "replay", "replay_motif intervention: sender policy not consulted"), idx, False
        if self.script is not None and self.script.motif_index is not None:
            m = self.script.motif_index
            mod = expressive_modulation(sender.state, self.config, sender.coupling_enabled)
            phrase = self._generate(m, sender, mod, origin_kind="preset")
            return phrase, self._fixed_trace(sender, values, m, "replay", f"scripted by preset: {self.script.note}"), m, False
        ctx = SenderContext(
            agent_id=sender.id,
            partner_id=receiver.id,
            step=k,
            target_id=target,
            target_pattern=self.task.patterns()[target],
            n_motifs=self.config.music.n_motifs,
            motif_names=motif_names(bank, self.config.music.n_motifs),
            learner_values=values,
            state=sender.state,
            coupling_enabled=sender.coupling_enabled,
            config=self.config,
            rng=self.rng_policy[sender.id],
            emit=self._emitter(sender.id),
        )
        trace = sender.policy.choose_motif(ctx)
        m = int(np.clip(trace.chosen, 0, self.config.music.n_motifs - 1))
        mod = trace.expressive_modulation or expressive_modulation(sender.state, self.config, sender.coupling_enabled)
        return self._generate(m, sender, mod), trace, m, True

    def _generate(self, m: int, sender: Agent, mod: dict[str, float], origin_kind: str = "agent") -> Phrase:
        mc = self.config.music
        return generate(
            m,
            None,
            sender.params.instrument,
            self.rng_gen,
            mc.generation_noise,
            tempo_multiplier=float(mod.get("tempo_multiplier", 1.0)),
            velocity_offset=float(mod.get("velocity_offset", 0.0)),
            agent_id=sender.id,
            step=self.step_index,
            bank=mc.motif_bank,
            length_beats=mc.phrase_length_beats,
            tempo_bounds=(mc.tempo_min, mc.tempo_max),
            origin_kind=origin_kind,
        )

    def _fixed_trace(self, agent: Agent, values: list[float], chosen: int | None, kind: str, note: str) -> PolicyTrace:
        n = len(values)
        probs = [1.0 if i == chosen else 0.0 for i in range(n)]
        return PolicyTrace(
            role="sender",
            policy_kind=kind,  # type: ignore[arg-type]
            scores=[round(v, 6) for v in values],
            probabilities=probs,
            temperature=round(temperature(agent.state, self.config, agent.coupling_enabled), 6),
            temperature_base=self.config.learning.temperature_base,
            coupling_enabled=agent.coupling_enabled,
            chosen=-1 if chosen is None else chosen,
            expressive_modulation=expressive_modulation(agent.state, self.config, agent.coupling_enabled),
            notes=[note],
        )

    # ------------------------------------------------------------------ channel / receiver
    def _channel(self, phrase: Phrase, motif_idx: int | None, sender: Agent) -> tuple[Phrase | None, Observation]:
        k = self.step_index
        mc = self.config.music
        if self.settings["channel"] == "symbol":
            sid = motif_idx if motif_idx is not None else nearest_motif(phrase, mc.motif_bank, mc.n_motifs, mc.feature_weights)
            return None, symbol_observation(sid, sender.id, k)
        kinds = [self.settings["perturbation"], self.channel_transform]
        heard, _ = perturb(phrase, kinds, self.rng_perturb)  # type: ignore[arg-type]
        return heard, music_observation(heard, sender.id, k, mc.feature_weights)

    def _receiver_turn(self, receiver: Agent, sender: Agent, obs: Observation, x: np.ndarray) -> PolicyTrace:
        retrieved = receiver.memory.retrieve(self._memory_vector(obs), self.config.memory.retrieval_k)
        receiver.last_retrieved = retrieved
        ctx = ReceiverContext(
            agent_id=receiver.id,
            partner_id=sender.id,
            step=self.step_index,
            observation=obs,
            n_patterns=self.task.n_patterns,
            patterns=self.task.patterns(),
            learner_scores=[float(v) for v in receiver.receiver_learner.scores(x)],
            state=receiver.state,
            coupling_enabled=receiver.coupling_enabled,
            config=self.config,
            rng=self.rng_policy[receiver.id],
            retrieved=retrieved,
            emit=self._emitter(receiver.id),
        )
        trace = receiver.policy.choose_pattern(ctx)
        if not 0 <= trace.chosen < self.task.n_patterns:
            trace = trace.model_copy(update={"chosen": int(np.clip(trace.chosen, 0, self.task.n_patterns - 1))})
        receiver.last_trace = trace
        return trace

    def _memory_vector(self, obs: Observation) -> list[float]:
        if obs.channel == "symbol":
            return [1.0 if i == obs.symbol_id else 0.0 for i in range(self.config.music.n_motifs)]
        assert obs.features is not None
        return list(obs.features.vector)

    def _episode_vectors(self, phrase: Phrase, obs: Observation, motif_idx: int | None) -> tuple[list[float], list[float]]:
        """(sender's own phrase vector, receiver's heard vector) for memory + similarity."""
        if obs.channel == "symbol":
            one_hot = self._memory_vector(obs)
            return one_hot, one_hot
        sent = symbolic_features(phrase, self.config.music.feature_weights).vector
        return list(sent), self._memory_vector(obs)

    def _similarity(self, a: list[float] | None, b: list[float] | None) -> float | None:
        if self.settings["channel"] == "symbol":
            return None  # no graded similarity in the symbol channel
        return partner_similarity(a, b, self.reference)

    # ------------------------------------------------------------------ updates
    def _lr(self, agent: Agent, base: float) -> float:
        return agent.effective_learning_rate(base, self.config.coupling.affiliation_to_learning_rate)

    def _update_sender(
        self, sender: Agent, receiver: Agent, phrase: Phrase, motif_idx: int | None, target: int, score: float,
        sent_vec: list[float], sim: float | None, learn: bool,
    ) -> None:
        payload: dict[str, Any] = {"role": "sender", "action": motif_idx, "score": score}
        if not self.settings["learning_enabled"]:
            payload.update(skipped=True, reason="learning disabled")
        elif not learn or motif_idx is None:
            payload.update(skipped=True, reason="sender did not choose this phrase (replay/scripted)")
        else:
            lr = self._lr(sender, self.config.learning.sender_learning_rate)
            payload.update(skipped=False, lr_base=self.config.learning.sender_learning_rate, lr_eff=round(lr, 6))
            payload.update(sender.sender_learner.update(target, motif_idx, score, lr))
        self._emit("learning_update", payload, agent_id=sender.id, visibility="sender_private")
        drive = StateDrive(score=score, heard=None, partner_similarity=sim, notes=["sender: own phrase is not fed back as acoustic input"])
        self._apply_state(sender, receiver, "sender", drive, phrase, motif_idx, sent_vec, motif_idx if motif_idx is not None else -1, score)

    def _update_receiver(
        self, receiver: Agent, sender: Agent, phrase: Phrase, obs: Observation, x: np.ndarray, chosen: int, score: float,
        heard_vec: list[float], sim: float | None,
    ) -> None:
        payload: dict[str, Any] = {"role": "receiver", "action": chosen, "score": score}
        if not self.settings["learning_enabled"]:
            payload.update(skipped=True, reason="learning disabled")
        else:
            lr = self._lr(receiver, self.config.learning.receiver_learning_rate)
            payload.update(skipped=False, lr_base=self.config.learning.receiver_learning_rate, lr_eff=round(lr, 6))
            payload.update(receiver.receiver_learner.update(x, chosen, score, lr))
        self._emit("learning_update", payload, agent_id=receiver.id)
        heard = obs.features if obs.channel == "music" else None
        notes = [] if heard is not None else ["symbol channel: no acoustic input"]
        drive = StateDrive(score=score, heard=heard, partner_similarity=sim, notes=notes)
        motif_idx = obs.symbol_id if obs.channel == "symbol" else motif_index_from_id(phrase.motif_id, self.config.music.motif_bank)
        self._apply_state(receiver, sender, "receiver", drive, obs.phrase or phrase, motif_idx, heard_vec, chosen, score)

    def _motif_id_or_none(self, motif_idx: int | None) -> str | None:
        bank = self.config.music.motif_bank
        if motif_idx is None or not 0 <= motif_idx < len(get_bank(bank)):
            return None
        return make_motif_id(motif_idx, bank)

    def _apply_state(
        self, agent: Agent, partner: Agent, role: str, drive: StateDrive, phrase: Phrase, motif_idx: int | None,
        vector: list[float], action: int, score: float,
    ) -> None:
        before = agent.state
        after, inputs, detail = StateEngine.update_detailed(
            before, drive, self.config.state, agent.params.sensitivity, agent.frozen_state, agent.baseline
        )
        agent.set_state(after)
        agent.last_state_inputs = inputs
        item = MemoryItem(
            step=self.step_index,
            role=role,  # type: ignore[arg-type]
            partner_id=partner.id,
            phrase_id=phrase.id,
            motif_id=self._motif_id_or_none(motif_idx),
            feature_vector=vector,
            action=action,
            score=score,
            state_before=before,
            state_after=after,
        )
        agent.memory.append(item)
        self._emit(
            "state_update",
            {
                "role": role,
                "before": _dump(before),
                "after": _dump(after),
                "inputs": _dump(inputs),
                "drive": detail.drive,
                "delta": detail.delta,
                "memory_item": _dump(item) if agent.memory.capacity > 0 else None,
            },
            agent_id=agent.id,
        )

    # ------------------------------------------------------------------ interventions
    def intervene(self, iv: Intervention) -> Event:
        if self.mode == "replay":
            raise RuntimeError("interventions are not allowed on a replay session")
        agents = self._target_agents(iv.agent_id)
        details: dict[str, Any] = {}
        value: Any = iv.value
        if iv.kind == "replay_motif":
            phrase = iv.phrase or self.lookup_phrase(iv.phrase_id)
            self.pending_phrase = phrase
            details["source_phrase_id"] = phrase.id
        elif iv.kind == "reset_memory":
            for a in agents:
                a.reset_memory()
        elif iv.kind == "reset_state":
            for a in agents:
                a.reset_state()
            details["state_after"] = {a.id: _dump(a.state) for a in agents}
        elif iv.kind == "freeze_state":
            value = True if value is None else bool(value)
            for a in agents:
                a.frozen_state = value
        elif iv.kind == "set_coupling":
            value = True if value is None else bool(value)
            for a in agents:
                a.coupling_enabled = value
        elif iv.kind == "swap_feature":
            value = iv.transform or (str(value) if value else "none")
            self.channel_transform = value
        elif iv.kind == "set_param":
            if not iv.path:
                raise ValueError("set_param needs a dotted `path`")
            self._set_param(iv.path, value)
            details["path"] = iv.path
        payload = {"kind": iv.kind, "agent_id": iv.agent_id, "value": value, "effective_from_step": self.step_index, **details}
        return self._emit("intervention", payload, agent_id=iv.agent_id, visibility="experimenter")

    def _target_agents(self, agent_id: str | None) -> list[Agent]:
        if agent_id is None:
            return list(self.agents)
        if agent_id not in self.agent_by_id:
            raise KeyError(f"unknown agent {agent_id!r}")
        return [self.agent_by_id[agent_id]]

    def lookup_phrase(self, phrase_id: str | None) -> Phrase:
        """A previously sent phrase, or a bank motif id ('m3-hop' / 'motif-m3-hop')."""
        if not phrase_id:
            raise KeyError("replay_motif needs phrase_id or phrase")
        if phrase_id in self.sent_phrases:
            return self.sent_phrases[phrase_id]
        mid = phrase_id.removeprefix("motif-")
        idx = motif_index_from_id(mid, self.config.music.motif_bank)
        if idx is None or idx >= self.config.music.n_motifs:
            raise KeyError(f"unknown phrase id {phrase_id!r}")
        sender, _ = self.roles(self.step_index)
        mod = {"tempo_multiplier": 1.0, "velocity_offset": 0.0}
        phrase = self._generate(idx, sender, mod, origin_kind="replay")
        return phrase.model_copy(update={"id": f"motif-{make_motif_id(idx, self.config.music.motif_bank)}"})

    def _set_param(self, path: str, value: Any) -> None:
        if any(path == p or path.startswith(p + ".") for p in STRUCTURAL_PATHS):
            raise ValueError(f"'{path}' is structural; reset the session with a new config instead")
        data = self.config.model_dump(mode="json")
        node = data
        parts = path.split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise ValueError(f"invalid config path {path!r}")
            node = node[part]
        node[parts[-1]] = value
        self.config = ExperimentConfig.model_validate(data)  # re-validate the whole config
        self.settings = effective_settings(self.condition, self.config)
        for a in self.agents:
            a.memory.set_capacity(self.settings["memory_capacity"])
            if path.startswith("coupling.enabled"):
                a.coupling_enabled = self.settings["coupling_enabled"]

    # ------------------------------------------------------------------ views
    @property
    def status(self) -> str:
        if self.mode == "replay":
            if self._replay_cursor == 0:
                return "ready"
            return "finished" if self._replay_cursor >= len(self._replay_steps) else "running"
        if self.step_index == 0:
            return "ready"
        return "finished" if self.step_index >= self.config.episodes else "running"

    def metrics(self) -> SessionMetrics:
        s = self.scores
        return SessionMetrics(
            episodes=len(s),
            mean_score=round(float(np.mean(s)), 6) if s else 0.0,
            rolling_score=round(float(np.mean(s[-ROLLING_WINDOW:])), 6) if s else 0.0,
            success_rate=round(float(np.mean([x == 1.0 for x in s])), 6) if s else 0.0,
            score_history=list(s),
        )

    def summary(self) -> SessionSummary:
        return SessionSummary(
            id=self.id,
            created_at=self.created_at,
            mode=self.mode,  # type: ignore[arg-type]
            preset=self.preset,
            seed=self.seed,
            condition=self.condition,
            step=self.step_index,
            status=self.status,  # type: ignore[arg-type]
        )

    def snapshot(self) -> SessionSnapshot:
        return SessionSnapshot(
            id=self.id,
            created_at=self.created_at,
            mode=self.mode,  # type: ignore[arg-type]
            preset=self.preset,
            seed=self.seed,
            condition=self.condition,
            config=self.config,
            step=self.step_index,
            status=self.status,  # type: ignore[arg-type]
            agents=[a.snapshot() for a in self.agents],
            current_episode=self.current_episode,
            metrics=self.metrics(),
            event_count=len(self.events),
            patterns=self.task.patterns(),
        )

    def inspect(self, agent_id: str) -> AgentInspection:
        a = self.agent_by_id[agent_id]
        return AgentInspection(
            agent=a.snapshot(),
            memory=a.memory.items(),
            retrieved=a.last_retrieved,
            last_state_inputs=a.last_state_inputs,
            state_history=list(a.state_history),
            information_received=list(a.information_received),
        )

    # ------------------------------------------------------------------ export / replay / reset
    def export(self) -> RunExport:
        events = self.events
        final = self.snapshot()
        if self.mode == "replay" and self._replay_cursor < len(self._replay_steps):
            events = self.recorded_events()
            final = Session.replayed_to_end(self).snapshot()
        return RunExport(
            exported_at=now_iso(),
            session=self.summary(),
            config=self.initial_config,
            condition=self.condition,
            events=list(events),
            final_snapshot=final,
        )

    @classmethod
    def from_export(cls, export: RunExport, session_id: str | None = None) -> Session:
        """A replay session: replay_step() re-emits recorded events and applies their recorded
        values (states, learner rows, memory items, scores) without recomputing anything."""
        return cls.from_events(
            export.config, export.condition, export.session.seed, export.events,
            session_id=session_id, preset=export.session.preset, created_at=export.session.created_at,
        )

    @classmethod
    def from_events(
        cls, config: ExperimentConfig, condition: ConditionSpec, seed: int, events: list[Event], *,
        session_id: str | None = None, preset: str | None = None, created_at: str | None = None,
    ) -> Session:
        s = cls(config, condition, seed, preset=preset, mode="replay", session_id=session_id,
                allow_model=False, created_at=created_at)
        grouped: dict[int, list[Event]] = defaultdict(list)
        for ev in sorted(events, key=lambda e: e.seq):
            grouped[ev.step].append(ev)
        s._recorded = dict(grouped)
        s._replay_steps = sorted(grouped)
        return s

    def recorded_events(self) -> list[Event]:
        """All recorded events of a replay session (in seq order)."""
        return [ev for step in self._replay_steps for ev in self._recorded[step]]

    @staticmethod
    def replayed_to_end(session: Session) -> Session:
        """A copy of a replay session advanced through all recorded events."""
        copy = Session.from_events(
            session.initial_config, session.condition, session.seed, session.recorded_events(),
            session_id=session.id, preset=session.preset, created_at=session.created_at,
        )
        copy.replay_step(len(copy._replay_steps))
        return copy

    def replay_step(self, n: int = 1) -> list[Event]:
        if self.mode != "replay":
            raise RuntimeError("only replay sessions can replay_step()")
        out: list[Event] = []
        for _ in range(n):
            if self._replay_cursor >= len(self._replay_steps):
                break
            for ev in self._recorded[self._replay_steps[self._replay_cursor]]:
                self._apply_recorded(ev)
                self.events.append(ev)
                out.append(ev)
            self._replay_cursor += 1
        return out

    @property
    def replay_total_steps(self) -> int:
        return len(self._replay_steps)

    def _apply_recorded(self, ev: Event) -> None:
        p = ev.payload
        agent = self.agent_by_id.get(ev.agent_id or "")
        if ev.type == "intervention":
            self._apply_recorded_intervention(p)
        elif ev.type == "target_assigned" and agent is not None:
            agent.information_received.append({"step": ev.step, "role": "sender", "target_id": p["target_id"]})
        elif ev.type == "phrase_sent" and agent is not None:
            agent.last_trace = PolicyTrace.model_validate(p["trace"])
        elif ev.type == "phrase_received" and agent is not None:
            agent.information_received.append({"step": ev.step, "role": "receiver", "observation": p["observation"]})
        elif ev.type == "action_chosen" and agent is not None:
            agent.last_trace = PolicyTrace.model_validate(p["trace"])
        elif ev.type == "outcome":
            self._apply_recorded_feedback(ev)
        elif ev.type == "learning_update" and agent is not None and not p.get("skipped", True):
            if p["role"] == "receiver":
                agent.receiver_learner.set_row(p["row"], p["weights_after"], p["bias_after"])
            else:
                agent.sender_learner.set_value(p["target"], p["motif"], p["q_after"], p["count_after"])
        elif ev.type == "state_update" and agent is not None:
            agent.set_state(StateVector.model_validate(p["after"]))
            agent.last_state_inputs = StateUpdateInputs.model_validate(p["inputs"])
            if p.get("memory_item"):
                agent.memory.append(MemoryItem.model_validate(p["memory_item"]))
        elif ev.type == "episode_complete":
            summary = EpisodeSummary.model_validate(p)
            self.current_episode = summary
            if summary.score is not None:
                self.scores.append(summary.score)
            self.step_index = summary.step + 1

    def _apply_recorded_feedback(self, ev: Event) -> None:
        for agent in self.agents:
            if agent.information_received and agent.information_received[-1]["step"] == ev.step:
                entry = agent.information_received[-1]
                if entry["role"] == "receiver":
                    entry["feedback"] = {"own_choice": ev.payload["chosen_pattern_id"], "score": ev.payload["score"]}
                else:
                    entry["feedback"] = {"score": ev.payload["score"]}

    def _apply_recorded_intervention(self, p: dict[str, Any]) -> None:
        agents = self._target_agents(p.get("agent_id"))
        kind = p["kind"]
        if kind == "reset_memory":
            for a in agents:
                a.reset_memory()
        elif kind == "reset_state":
            for a in agents:
                a.set_state(StateVector.model_validate(p["state_after"][a.id]))
        elif kind == "freeze_state":
            for a in agents:
                a.frozen_state = bool(p["value"])
        elif kind == "set_coupling":
            for a in agents:
                a.coupling_enabled = bool(p["value"])
        elif kind == "swap_feature":
            self.channel_transform = p["value"]
        elif kind == "set_param":
            self._set_param(p["path"], p["value"])

    @classmethod
    def rebuild(cls, config: ExperimentConfig, condition: ConditionSpec, seed: int, **kwargs: Any) -> Session:
        """Fresh agents, same seed and config; the event log restarts with `session_reset`."""
        return cls(config, condition, seed, reset=True, **kwargs)

    def reset_copy(self) -> Session:
        return Session.rebuild(
            self.initial_config, self.condition, self.seed, preset=self.preset, session_id=self.id,
            script=self.script, created_at=self.created_at,
        )
