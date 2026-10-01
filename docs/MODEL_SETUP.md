# Model setup — optional model-assisted layer

Resonance Lab runs without any model. With `model.provider: none` (the default), every decision
comes from the local policy and the simulation is fully deterministic for a given seed and config.
The model layer (`backend/resonance/models/`) is optional. It can make receiver decisions
(`choose_pattern`) and sender decisions (`choose_motif`, `propose_phrase`) from a **symbolic feature
summary computed from the note list**, and it can write a labelled narrative. On any failure it falls
back to the local policy.

## What is tested here, and what is not

| `model.provider` | Status in this build environment | Accepts audio | Input to the model |
|---|---|---|---|
| `none` (default) | Local policy only | — | — |
| `scripted` | **Tested.** Deterministic, not a language model: fixed script or a hash of the prompt | no | symbolic features |
| `llamacpp` | **Tested against a real local model**: Qwen2.5-0.5B-Instruct, Q4_K_M GGUF, CPU (`llama-cpp-python` 0.3.35) | no | symbolic features |
| `openai_compatible` | **Untested against any live service.** Request/response plumbing is tested against a local HTTP stub implementing `/v1/chat/completions` | no | symbolic features |
| `anthropic` | **Untested against the live API** (no credentials here). Request building and response parsing are tested offline with a stand-in client | no | symbolic features |

**No provider in this build accepts audio.** The "analyse the actual audio" enhancement is **not
available**. Every model sees only the symbolic feature summary (or `symbol #k` on the symbol channel),
and every prompt says so: "You are given a SYMBOLIC FEATURE SUMMARY computed from the note list. You
have not heard audio." The Anthropic Messages API does not take audio input. The OpenAI-compatible
provider sends text only. `GET /api/model/status` reports `accepts_audio: false` and the true
`input_modality`.

The local model is small. Its decisions are an **honest-but-weak baseline**. In the recorded example
below it ignored both the learner scores and the most similar memory. Report model-assisted runs
separately from local-policy results, and never pool them.

## Enabling a provider

Edit the `model:` block of `config/default.yaml`, or send the same fields in the `config` of
`POST /api/sessions`. The model is consulted only when **both** of these hold:

1. the agent has `policy_kind: model` (in `agents:`); and
2. the role is listed in `model.use_for`: `receiver` → `choose_pattern`, `sender` → `choose_motif`
   (and `propose_phrase`, if the session calls it), `narrative` → narrative (which also needs
   `narrative_enabled: true`).

With `use_for: []` (the default), nothing is delegated even when a provider is configured.

```yaml
# Local model (tested here). Download once into models/ (see "Local model" below).
model:
  provider: llamacpp
  model_id: qwen2.5-0.5b-instruct-q4_k_m     # label only; the file is local_model_path
  local_model_path: models/qwen2.5-0.5b-instruct-q4_k_m.gguf   # relative to the project root
  n_threads: 4
  temperature: 0.2
  call_budget: 200
  max_retries: 1
  use_for: [receiver]          # any of: sender, receiver, narrative
  narrative_enabled: false
agents:
  - {id: A, name: Aria, color: "#7c9cff", instrument: pluck, sensitivity: 1.0, policy_kind: local}
  - {id: B, name: Bram, color: "#ffb86b", instrument: marimba, sensitivity: 1.0, policy_kind: model}
```

```yaml
# Anthropic (untested here). The key lives in an env var; config stores only its NAME.
model:
  provider: anthropic
  model_id: claude-opus-5-5      # default if empty
  api_key_env: ANTHROPIC_API_KEY
  base_url: null                 # set only for a proxy/gateway
  timeout_seconds: 20
  use_for: [receiver, sender]
```

```yaml
# Any OpenAI-compatible server (untested here): OpenAI, Ollama, vLLM, LM Studio, llama.cpp server.
model:
  provider: openai_compatible
  model_id: qwen2.5:0.5b                  # whatever the server calls the model
  base_url: http://localhost:11434/v1     # required
  api_key_env: ""                         # optional; e.g. OPENAI_API_KEY for api.openai.com
  timeout_seconds: 20
  use_for: [receiver]
```

```yaml
# Deterministic, no model at all: exercises the whole model path reproducibly.
model:
  provider: scripted
  use_for: [receiver, sender]
```

Relative paths (`local_model_path`, `prompts_dir`) resolve against the project root (the parent of
`backend/`). Prompt text lives in `prompts/*.txt` (see `prompts/README.md`), not in code.

### Local model

The tested model file is `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` (Qwen2.5-0.5B-Instruct, Q4_K_M,
491 MB). Install the optional dependency with `pip install -e "backend[localmodel]"`
(`llama-cpp-python`). If the file or the package is missing, `status()` reports
`available: false` and the agent silently uses the local policy.

### Keys

`api_key_env` holds the **name** of an environment variable. Its value is read at call time, never
stored on the provider or in config, and registered for redaction. Every `model_call` and narrative
payload passes through `redact()`, which masks `sk-…` keys, `Bearer …` tokens, long values after
`key`/`token`/`secret`/`password` labels, and the exact values of registered key variables.

## Call budget, timeout, retries

- **Budget.** `model.call_budget` is the maximum number of model calls *per `ModelPolicy` instance*.
  The simulation builds one per model-assisted agent, so in practice the budget is per agent per
  session. Every attempt, including retries, counts. Once the budget is spent, the model is skipped
  and the local decision is used. One `model_call` event with `error: "call budget exhausted"` is
  emitted the first time this happens, and every later trace carries a note. `GET /api/model/status`
  shows `calls_made` and `call_budget`.
- **Retries.** Up to `model.max_retries` extra attempts after an exception, provider error, invalid
  JSON or out-of-range value. SDK-level retries are disabled, so the budget counts every attempt.
- **Timeout.** `anthropic` and `openai_compatible` use `timeout_seconds` as the HTTP timeout.
  `llamacpp` runs in-process and has **no wall-clock timeout**. Its bound is `max_tokens`: 80 for
  decisions, 448 for phrases, 256 for narrative, never more than 512. A decision took about 10–11 s on
  this 4-core CPU when otherwise idle, and about 22 s while other CPU-heavy jobs ran: roughly 4.6 s of
  prompt evaluation for about 720 tokens, plus grammar-constrained generation.
- **Validation.** Every answer is validated in the policy, whatever the provider promised:
  - `pattern_id` / `motif_index`: an integer in range;
  - `confidence`: in [0, 1];
  - `rationale`: a string, truncated to 200 characters;
  - phrase proposals: 1–16 notes, pitch 48–84, velocity 30–120, duration 0.25–4,
    onset + duration ≤ the phrase length, tempo within `music.tempo_min..tempo_max`.

  Failure means fallback.

### llama.cpp specifics we measured

- `response_format={"type": "json_object", "schema": …}` compiles the schema into a grammar. That
  grammar enforces `enum`, `minItems` and `maxItems`, but **not** `minimum`/`maximum` (a TODO in
  llama-cpp-python 0.3.35). Ranges are therefore expressed as enums where possible, and are always
  re-validated.
- `maxLength` made constrained sampling 5–10x slower (353–662 ms/token versus 68 ms/token without
  it), so it is stripped before the grammar is built. Without it, the 0.5B model sometimes writes a
  long rationale until `max_tokens`. The provider then closes the truncated JSON string and object
  and flags the result (`_repaired_truncated_output`), and the trace notes "rationale truncated".
- `propose_phrase` is slow with this model: about 38 s per call (about 1,000 prompt tokens and
  370–448 completion tokens). An earlier schema version that used `minimum`/`maximum` produced
  out-of-range values (velocity 0, duration 5), and validation rejected them. With the enum-constrained
  schema, both later test calls produced valid phrases. For the "offbeats" pattern, one put its notes
  on beats 0.5/1.5/2.5/3.5. Phrase proposals are off by default and are never called unless the
  session asks for them.
- Sender decisions (`choose_motif`) took about 6.5 s (about 590 prompt tokens). In two test calls the
  model picked motif 0 with "confidence 0.88" while the learner's best motif was 2 (0.71).

## Graceful degradation

| Situation | Behaviour |
|---|---|
| `provider: none` | `build_model_policy` returns the local policy itself |
| Provider unusable at build time (missing GGUF/SDK/key/base_url, missing prompt templates) | Local policy; reason logged and visible in `status().detail` |
| Exception, timeout, HTTP error, refusal, invalid JSON, out-of-range answer | Retry up to `max_retries`, then the local decision for that step; `trace.notes` says `model fallback -> local policy (<reason>)` |
| Budget exhausted | Local decision; one `model_call` event with the error, then notes |
| Role not in `use_for` | Local decision; note "model not consulted" |

The local policy is **always run first, exactly once**, even when the model is used. This keeps the
agent's RNG stream advancing exactly as in a local run, so a model failure never shifts later local
decisions. It also supplies `temperature`/`temperature_base`, which in a model trace are the local
policy's values: computed, but not used for the model's choice. A model trace has
`policy_kind: "model"`, `exploration_draw: null`, and `probabilities` representing the model's
**stated** confidence on the chosen option (floored at 1/n, the rest spread evenly). That is not a
sampled distribution.

## Recording and replay

- Every model attempt is logged as a `model_call` event. The payload holds `purpose`, `role`,
  `provider`, `model_id`, `input_modality`, `accepts_audio`, `request.system`, `request.user`,
  `response.content`, `response.parsed`, `latency_ms`, `usage`, `error`, `ok`, `fallback_used`,
  `will_retry`, `attempt`, `calls_made`, `budget_remaining` and `visibility_hint`. It is redacted
  before emission. Sender payloads contain the target and are marked `visibility_hint:
  "sender_private"`. The session currently logs all `model_call` events with visibility
  `experimenter`.
- **Live model calls are not deterministic.** Hardware, build, sampling and server-side changes all
  matter, so re-running a seed with a model provider can give different decisions. Replay of a
  recorded run (`POST /api/sessions/import` → `/replay/step`) uses the recorded events, which include
  every model output, and calls no model. That is how model-assisted runs are reproduced.
- `provider: scripted` is deterministic. Two runs with the same seed and config produced identical
  event payloads in an integration check (144 events, 0 differences).
- Generated narrative is emitted as a `generated_narrative` event with payload
  `{text, provider, model_id, label: "generated interpretation — not a measurement"}`. It is display
  only: nothing in the simulation reads it, so it never reaches state, learning, memory or metrics.

## Recorded example (real call, this environment)

- **Model:** `qwen2.5-0.5b-instruct-q4_k_m`, i.e. Qwen2.5-0.5B-Instruct-Q4_K_M (local llama.cpp).
- **Input modality:** `symbolic_features`.
- **Command:** `cd backend && . .venv/bin/activate && python -m resonance.models.demo`
- **Date:** 2026-10-01.
- **Wall time:** 11.17 s, including the first-call model load.
- **Context:** the fixed demo receiver context from `resonance/models/testing.py`. The learner favours
  pattern 1 (0.62). The most similar memory (similarity 0.93) is action 1 with score 1.0.

The run went through WS1's simulation `Session` as well (one step, agent B model-assisted, real
sim-computed features). It gave latency 11.4 s and 666 prompt / 80 completion tokens, and the model
chose pattern 0 when the target was 1 (score 0.0). Both runs show a weak decision-maker.

Request, system prompt (verbatim, redacted):

```text
You are the RECEIVER agent in Resonance Lab, an engineered two-agent research simulation. State values are engineered numbers with operational definitions, not feelings.

Your partner played a short phrase meant to indicate one of 4 timing patterns. Pick the pattern_id your partner most likely meant.

Input modality: You are given a SYMBOLIC FEATURE SUMMARY computed from the note list. You have not heard audio. On the "symbol" channel you get only an abstract message number.

Best evidence: your local learner's expected scores and your retrieved memories (what scored well before).

Output JSON only, no prose, no code fences, matching this schema:
{"type":"object","properties":{"pattern_id":{"type":"integer","enum":[0,1,2,3]},"confidence":{"type":"number"},"rationale":{"type":"string","maxLength":200}},"required":["pattern_id","confidence","rationale"],"additionalProperties":false}
pattern_id: integer 0 to 3. confidence: number 0 to 1. rationale: one short sentence of at most 15 words.
```

Request, user prompt:

```text
Step 12. Channel: music.

Observed phrase features (from the note list; no audio):
- note_count: 5
- duration_seconds: 4.36
- note_density: 1.15
- rhythmic_regularity: 0.71
- syncopation: 0
- mean_pitch: 63.4
- pitch_range: 7
- contour: 0.42
- contour_class: arch
- repetition: 0.25
- interval_histogram: [0, 0.5, 0.5, 0, 0]
- mean_velocity: 87.6
- velocity_variation: 0.06
- dissonance_proxy: 0
- tempo_bpm: 110

Timing patterns (public set; onsets in beats within a 4-beat bar):
- 0: four_on_floor, onsets [0, 1, 2, 3]
- 1: offbeats, onsets [0.5, 1.5, 2.5, 3.5]
- 2: syncopated, onsets [0, 0.75, 1.5, 2.5, 3]
- 3: sparse, onsets [0, 2]

Local learner expected score per pattern:
- pattern 0: 0.1
- pattern 1: 0.62
- pattern 2: 0.2
- pattern 3: 0.08

Your state:
- activation = 0.61 (0..1, raised by dense/loud input and surprise, decays to baseline)
- expected_value = 0.48 (0..1, running expected score)
- uncertainty = 0.37 (0..1, recent prediction error)
- affiliation = 0.12 (-1..1, credit toward partner from shared outcomes)

Retrieved memories (your action then and its score):
- step 7: role=receiver action=1 score=1 similarity=0.93
- step 3: role=receiver action=2 score=0.25 similarity=0.71

Reply with the JSON object only.
```

Response and bookkeeping (from the `model_call` event):

```json
{
  "purpose": "choose_pattern",
  "role": "receiver",
  "agent_id": "B",
  "step": 12,
  "provider": "llamacpp",
  "model_id": "qwen2.5-0.5b-instruct-q4_k_m",
  "input_modality": "symbolic_features",
  "accepts_audio": false,
  "visibility_hint": "public",
  "max_attempts": 1,
  "call_budget": 200,
  "attempt": 1,
  "response": {
    "content": "{\"pattern_id\": 3, \"confidence\": 0.08, \"rationale\": \"The phrase contains a rhythmic pattern (syncopated) and a tempo of 110, which is consistent with the timing pattern (3) in the public set. The score is 0.08, indicating a low confidence in the timing pattern.\"}",
    "parsed": {
      "pattern_id": 3,
      "confidence": 0.08,
      "rationale": "The phrase contains a rhythmic pattern (syncopated) and a tempo of 110, which is consistent with the timing pattern (3) in the public set. The score is 0.08, indicating a low confidence in the timing pattern."
    },
    "latency_ms": 11168.7,
    "usage": {
      "prompt_tokens": 718,
      "completion_tokens": 76,
      "total_tokens": 794
    },
    "error": null
  },
  "ok": true,
  "validation_error": null,
  "fallback_used": false,
  "will_retry": false,
  "calls_made": 1,
  "budget_remaining": 199,
  "tested_in_this_environment": true
}
```

Resulting `PolicyTrace` (`action_chosen` payload). The scores and probabilities are uniform because
the stated confidence (0.08) is below 1/4:

```json
{
  "role": "receiver",
  "policy_kind": "model",
  "scores": [
    0.25,
    0.25,
    0.25,
    0.25
  ],
  "probabilities": [
    0.25,
    0.25,
    0.25,
    0.25
  ],
  "temperature": 0.12,
  "temperature_base": 0.12,
  "coupling_enabled": true,
  "chosen": 3,
  "exploration_draw": null,
  "expressive_modulation": {},
  "notes": [
    "fake local policy (tests)",
    "model decision via llamacpp/qwen2.5-0.5b-instruct-q4_k_m (input: symbolic features, no audio): pattern 3, stated confidence 0.08; local policy had sampled 1",
    "model rationale (generated text, not a measurement): The phrase contains a rhythmic pattern (syncopated) and a tempo of 110, which is consistent with the timing pattern (3) in the public set. The score is 0.08, indicating a low confidence in the timing ",
    "temperature fields are the local policy's (computed, not used for this choice)",
    "rationale truncated (output cut at max_tokens or longer than 200 chars)"
  ]
}
```

This model copied the learner's score for pattern 3 (0.08) as its "confidence" and picked pattern 3
against both the learner and the memory. That is what "honest-but-weak baseline" means here.

## Tests

```bash
cd backend && . .venv/bin/activate
python -m pytest tests/test_models_*.py -q                 # includes ~11 s real llama.cpp call
python -m pytest tests/test_models_*.py -q -k "not slow"   # skip the real model call
RESONANCE_SKIP_SLOW=1 python -m pytest tests/test_models_*.py -q   # skip all llama.cpp tests
```

The llama.cpp tests skip cleanly if the GGUF file or `llama_cpp` is missing. No test touches the
network. The OpenAI-compatible tests bind a stub server on 127.0.0.1, and the Anthropic tests never
construct a real client.
