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
- [ ] WS1 sim core (music, features, synth, agents, env, session, conditions, experiments, persistence, API)
- [ ] WS2 frontend (lab screen, audio, piano roll, plots, controls, interventions, inspector, experiments view)
- [ ] WS3 research note, providers (llamacpp tested; anthropic/openai untested), prompts, model setup docs
- [ ] Integration: real API ↔ frontend, `make types` regenerated, `./start.sh` works
- [ ] Browser smoke test (Playwright) at desktop size
- [ ] Batch experiments run; sample exports + report in `data/exports/` and `docs/EXPERIMENT_REPORT.md`
- [ ] README + architecture + definitions docs
- [ ] Final verification against acceptance criteria

## Blockers

None.

## Exact next action

Launch WS1/WS2/WS3 Opus agents with the briefs below; while they run, write README skeleton.

## Commands

- `./start.sh` → http://localhost:8000 (builds frontend, serves API + UI)
- `make dev` → API :8000 + Vite :5173 (dev)
- `make test` / `make smoke` / `make types` / `make experiments`

## Validation log

- 2026-10-01 05:58 — schemas import; `config/default.yaml` validates; OpenAPI: 25 routes, 51 schemas; TS types generated (2544 lines).
