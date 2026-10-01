# Resonance Lab — progress & checkpoint

Orchestrator: Fable (planning, contracts, integration, verification).
Implementation: Opus workstreams (see ownership). This file is the durable
resume point; update it at every checkpoint.

## Environment facts (verified 2026-10-01)

- Python 3.11.15 + uv; Node 22.22; npm 10.9; Playwright 1.56 (global) with Chromium at /opt/pw-browsers.
- No runtime model API keys in the environment (only harness/infra credentials, NOT used by the app).
- Local model route verified: `llama-cpp-python 0.3.35` (prebuilt CPU wheel) + `models/qwen2.5-0.5b-instruct-q4_k_m.gguf`
  (491 MB, downloaded). JSON-schema constrained decision: load 0.7 s, ~2.9 s per call, valid JSON. This is the
  real, tested model-assisted provider. Anthropic / OpenAI-compatible providers get plumbing + unit tests with a
  stub only → must be labelled "untested against live service".
- All six research URLs reachable (HTTP 200).
- Agent model aliases available to the orchestrator: `opus`, `sonnet`, `haiku`, `fable`. Environment documents
  Opus 5.5 = `claude-opus-5-5`; the alias `opus` is used for implementation workstreams.

## Decisions (practical defaults; change only with reason)

- Stack: FastAPI + Pydantic v2 + numpy + SQLite (stdlib) backend; React 19 + TS + Vite 8 frontend; Web Audio synth
  in browser; numpy synth on server for WAV export + audio-measured features. Types generated from OpenAPI.
- Backend never auto-advances; the frontend drives `step`. Pause/step/speed are frontend concerns.
- Task: cooperative timing task (4 patterns default). Sender = contextual bandit over 8 motifs given target.
  Receiver = linear softmax over 16 normalised symbolic features (+bias) → generalises across related phrases.
- State: 4 bounded dims (activation, expected_value, uncertainty, affiliation) with inertia, decay, bounded
  steps; drives from (a) hand-authored acoustic influence (switchable), (b) prediction error, (c) outcome.
- Coupling (switchable): state → softmax temperature, expressive tempo/velocity, learning-rate scaling.
- Conditions: full, no_history, state_fixed, state_decoupled, symbol (one-hot motif identity), perturbed(kind).
- Presets: first_encounter, shared_history (vs memory-reset control), same_phrase_different_history (constructed).

## Ownership

| Workstream | Owner | Files |
|---|---|---|
| WS1 simulation core + API + experiments + persistence | Opus agent "sim" | `backend/resonance/**` except `models/`, `backend/tests/**` |
| WS2 frontend + audio | Opus agent "web" | `frontend/**` (except generated `src/api/schema.d.ts`) |
| WS3 research note + model providers + prompts + docs/report skeleton | Opus agent "research" | `backend/resonance/models/**`, `backend/tests/test_models_*.py`, `prompts/**`, `docs/RESEARCH_NOTES.md`, `docs/MODEL_SETUP.md` |
| Contracts, config, integration, smoke test, final verification | Fable | `backend/resonance/schemas.py`, `backend/resonance/api.py` signatures, `config/`, `docs/CONTRACTS.md`, `PROGRESS.md`, `README.md` |

## Status

- [x] Workspace inspected, toolchain verified, local model verified
- [x] Contracts: `schemas.py`, `docs/CONTRACTS.md`, `config/default.yaml`, API signatures, generated TS types
- [x] WS1 sim core — 143 tests pass (+3 slow llama); ruff clean; server verified end to end by orchestrator (health, step, isolation, reset_memory, WAV, export/import)
- [ ] WS2 frontend (lab screen, audio, piano roll, plots, controls, interventions, inspector, experiments view)
- [x] WS3 research note, providers (llamacpp tested with real local call; anthropic/openai-compatible untested against live services, plumbing unit-tested), prompts, model setup docs — 56 tests pass (53 + 3 slow)
- [ ] Integration: real API ↔ frontend, `make types` regenerated, `./start.sh` works
- [ ] Browser smoke test (Playwright) at desktop size
- [x] Batch experiments run (6 conditions × 10 seeds, ablations, perturbations, 300-episode horizon, presets) → `data/exports/`, `docs/EXPERIMENT_REPORT.md`
- [ ] README + architecture + definitions docs
- [ ] Final verification against acceptance criteria

## Blockers

None.

## Exact next action

When WS2 reports: build frontend, run Playwright smoke against the REAL backend at 1440×900, fix integration issues, verify `./start.sh`, finalize README, push.

## Commands

- `./start.sh` → http://localhost:8000 (builds frontend, serves API + UI)
- `make dev` → API :8000 + Vite :5173 (dev)
- `make test` / `make smoke` / `make types` / `make experiments`

## Validation log

- 2026-10-01 06:55 — Orchestrator verification of WS1: `RESONANCE_SKIP_SLOW=1 pytest` → 143 passed, 3 skipped (12 s); ruff clean after StrEnum fix. Live uvicorn on :8011: health ok; create+step 3 → 30 events, receiver observation has no target; reset_memory → memory 0, state unchanged; render → 200 audio/wav 237,744 B; export 32 events → import → mode replay.
- 2026-10-01 06:58 — Real local-model session (llamacpp, agent B receiver, budget 3): 2 model_call events ok, modality symbolic_features, ~10 s each, no "target" in receiver prompt, no env values in export. Script: backend/scripts/model_e2e.py.
- 2026-10-01 06:57 — Ablations (backend/scripts/ablations.py → data/exports/experiment_ablation_*.json): acoustic off 0.638±0.127; no affiliation→lr 0.654±0.090; no expressive 0.640±0.114; no temperature coupling 0.638±0.120; pitch_shuffle 0.566±0.108 lowest; 300 episodes: full 0.771, symbol 0.833, no_history 0.326.

- 2026-10-01 06:35 — WS3: `RESONANCE_SKIP_SLOW=1 pytest tests/test_models_*.py` → 53 passed, 3 skipped; ruff clean. Real local-model receiver decision: ~11 s idle, 22 s under load; the 0.5B model is a weak decision-maker (recorded honestly).
- 2026-10-01 05:58 — schemas import; `config/default.yaml` validates; OpenAPI: 25 routes, 51 schemas; TS types generated (2544 lines).
