# Shared contracts

Source of truth for all shapes: `backend/resonance/schemas.py` (Pydantic v2).
TypeScript types are generated from the FastAPI OpenAPI document into
`frontend/src/api/schema.d.ts` (`make types`). Do not hand-edit the generated file.

## Vocabulary (keep these separate in code, UI and docs)

| Kind | Examples | Where |
|---|---|---|
| Measured | `SymbolicFeatures` (from notes), `AudioFeatures` (from rendered waveform) | `resonance/music/features.py`, `resonance/music/synth.py` |
| Engineered | `StateVector`, update equations, `LearnerSnapshot`, `MemoryItem` | `resonance/agents/*` |
| Generated | `generated_narrative` events, model rationales | `resonance/models/*`, never read by the sim |

## Episode protocol (one `step` = one episode)

1. Environment draws `target_id` with `rng_env` → event `target_assigned` (visibility `sender_private`).
2. Sender policy chooses motif index given target (contextual bandit, `rng_policy[sender]`),
   generates `Phrase` from motif with expressive modulation from state (if coupling enabled),
   `rng_gen` jitter → event `phrase_sent` (public, full phrase).
3. Channel: condition may perturb the phrase or replace it with a `symbol_id` →
   `Observation` built for the receiver → event `phrase_received` with
   `payload.observation` = exactly what the receiver saw. **Must never contain target_id.**
4. Receiver policy scores patterns from `Observation.features.vector` (or one-hot symbol),
   softmax with state-coupled temperature, samples with `rng_policy[receiver]` →
   event `action_chosen` with `PolicyTrace`.
5. Environment scores `chosen_pattern_id` vs target → event `outcome` (`score` in [0,1];
   exact = 1.0, partial credit by onset overlap when enabled).
6. Both agents: memory append, learner update (if learning enabled), state update
   (if not frozen) → events `learning_update`, `state_update` (payload includes
   `before`, `after`, `inputs: StateUpdateInputs`).
7. Event `episode_complete`. Roles alternate next step.

## HTTP API (prefix `/api`)

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/health` | | `HealthResponse` |
| GET | `/config/default` | | `ExperimentConfig` |
| GET | `/patterns` | | `TimingPattern[]` |
| GET | `/motifs` | | `Phrase[]` (base motif bank rendered at default tempo) |
| GET | `/presets` | | `PresetInfo[]` |
| POST | `/presets/{name}/run` | `{seed?: int}` | `PresetResult` |
| POST | `/sessions` | `CreateSessionRequest` | `SessionSnapshot` |
| GET | `/sessions` | | `SessionSummary[]` |
| GET | `/sessions/{id}` | | `SessionSnapshot` |
| POST | `/sessions/{id}/step` | `StepRequest` | `StepResult` |
| POST | `/sessions/{id}/reset` | | `SessionSnapshot` (same seed/config, fresh agents) |
| POST | `/sessions/{id}/intervene` | `Intervention` | `StepResult` (events = [intervention event]) |
| POST | `/sessions/{id}/human_phrase` | `HumanPhraseRequest` | `StepResult` |
| GET | `/sessions/{id}/events?from_seq=&limit=` | | `Event[]` |
| GET | `/sessions/{id}/agents/{agent_id}/inspect` | | `AgentInspection` |
| GET | `/sessions/{id}/export` | | `RunExport` |
| POST | `/sessions/import` | `RunExport` | `SessionSnapshot` (mode `replay`) |
| POST | `/sessions/{id}/replay/step` | `StepRequest` | `StepResult` (replays recorded events) |
| POST | `/phrases/features` | `Phrase` | `PhraseFeatures` |
| POST | `/phrases/render` | `Phrase` | `audio/wav` bytes |
| POST | `/phrases/transform` | `{phrase, transform}` | `Phrase` |
| GET/POST | `/motifs/saved` | `SavedMotif` | `SavedMotif[]` |
| POST | `/experiments` | `ExperimentRequest` | `ExperimentResult` |
| GET | `/experiments` | | `ExperimentResult[]` (summaries) |
| GET | `/experiments/{id}` | | `ExperimentResult` |
| GET | `/experiments/{id}/csv` | | `text/csv` |
| GET | `/model/status` | | `ModelStatus` |

Sessions live in memory while the server runs and are persisted to SQLite
(`data/resonance.db`) as event logs so they survive restarts and can be exported.

## Conditions

| name | learning | memory | state updates | coupling | channel |
|---|---|---|---|---|---|
| full | on | on | on | on | music |
| no_history | off | capacity 0 | on | on | music |
| state_fixed | on | on | off (held at baseline) | n/a | music |
| state_decoupled | on | on | on | off | music |
| symbol | on | on | on | on | one-hot of sender's motif index (identity preserved, no acoustic drive, no graded similarity) |
| perturbed | on | on | on | on | music, transform applied in channel |

## Reset semantics

- `reset_memory(agent)`: episodic memory cleared, learner weights re-initialised. State vector untouched.
- `reset_state(agent)`: state := baseline. Memory and learner untouched.
- `session reset`: new agents, same seed and config, event log restarted.

## RNG streams

`numpy.random.SeedSequence(seed).spawn(5)` → `rng_env`, `rng_policy_A`, `rng_policy_B`, `rng_gen`, `rng_perturb`. Same seed + same config ⇒ identical event log in local mode.

## Frontend ownership of time

The backend never auto-advances. The frontend drives `step` calls; at 1× speed it waits for the
sent phrase's duration (audio) before requesting the next step. Pause = stop calling. This keeps
the backend deterministic and makes replay trivial.
