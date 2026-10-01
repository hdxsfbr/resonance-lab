# Resonance Lab

An experimental laboratory where two **engineered** agents exchange short musical phrases
to solve a cooperative timing task, while a bounded, persistent internal state and a set of
learned associations evolve with their interaction history. You can hear the phrases,
watch the state and the learned associations change, intervene (replay a motif, erase one
agent's memory, freeze state, decouple state from behaviour, perturb the music), and run
matched control conditions over many seeds.

**What this is:** a transparent agent simulation with hand-written update equations and a
simple associative learner, built to test one computational hypothesis:

> Agents with persistent internal state, memory and learned musical associations may
> develop behaviour that depends on their musical interaction history, beyond immediate
> stimulus–response rules or prompted emotional role-play.

**What this is not:** a claim that anything here experiences feelings. State variables such
as `activation` or `affiliation` are operational definitions (see
[docs/DEFINITIONS.md](docs/DEFINITIONS.md)), hand-authored dynamics are labelled as such,
and generated text is never treated as a measurement. The novelty of the project has not
been established; see [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md) for what we borrow
from prior work.

## Quick start

Requirements: Python 3.11+, Node 22+, `uv` (optional; falls back to `python -m venv`). Tested on Linux.

```bash
./start.sh
# → http://localhost:8000   (API docs at http://localhost:8000/docs)
```

The script creates the Python environment, installs dependencies, builds the frontend and
serves everything on one port. No API keys are needed. For development with hot reload:

```bash
make dev          # API on :8000, Vite dev server on :5173 (proxies /api)
make test         # backend pytest + frontend vitest
make smoke        # Playwright browser smoke test
make experiments  # headless multi-seed batch for all six conditions → data/exports/
```

## Five-minute exploration

1. Open http://localhost:8000 and click **Enable audio** (browsers require a gesture before sound).
2. On the start screen choose **First encounter → Start live session**. The badge in the header reads
   **LIVE SIMULATION**. Press **▶ Start**. Two agents alternate as sender and receiver: each episode the
   sender sees a private target rhythm, plays a phrase (you hear it, and see it on the piano roll), the
   receiver picks a rhythm, and both get the same score. Watch the score plot and the four state bars.
3. Press **Pause**, then **Step** a few times. Click **Inspect Aria** (or Bram): state before → after with
   the inputs that caused it (the acoustic term is labelled *hand-authored*), the policy's scores →
   probabilities with base and effective temperature, the learned weights, and the *information actually
   received* panel, which confirms the receiver saw no target field.
4. In the **Interventions** tab: **Clear memory** on one agent (its learned associations vanish, its state
   bars do not move), **Reset state** (state returns to baseline, associations stay), tick **freeze
   state**, or tick **disable coupling** and step once: the inspector's effective temperature now equals
   the base temperature. Every intervention appears in the timeline with "effective from step N".
5. Click **↻ Replay** on an earlier timeline row: the phrase plays again and is queued as the next sent
   phrase, so the receiver really receives it.
6. Open **Experiments**, keep all six conditions, enter five seeds and 120 episodes, and **Run experiment**
   (no audio). Read the per-run dots, not just the means. Download the CSV.
7. Open **Compose**, place a few notes on the keyboard or grid, choose a target, and **Send as human**.

Screenshots from the live browser smoke test are in `docs/screenshots/`.

## Modes

| Badge | Meaning |
|---|---|
| **LIVE SIMULATION** | Episodes are being computed now from seed and config |
| **REPLAY** | An exported run is being re-emitted; nothing is recomputed |
| **DEMO DATA** | The browser's mock client; no backend involved |

Local experimental mode (default) needs no model. Model-assisted mode is optional; see
[docs/MODEL_SETUP.md](docs/MODEL_SETUP.md).

## Verification

```bash
make test-backend                                   # 144 unit + integration tests (3 slow local-model tests skipped with RESONANCE_SKIP_SLOW=1)
make test-frontend                                  # 28 vitest tests
make smoke                                          # Playwright flow in mock mode
./start.sh &                                        # then, against the real server:
cd frontend && SMOKE_BASE_URL=http://127.0.0.1:8000 LIVE_SMOKE=1 npx playwright test tests/live.spec.ts
cd backend && .venv/bin/python scripts/model_e2e.py # real local-model session (needs models/*.gguf, see docs/MODEL_SETUP.md)
```

What was verified in the build session is recorded in [PROGRESS.md](PROGRESS.md). Audio was verified
by rendering the browser's Web Audio engine offline to a non-silent WAV and by the server render; no one
listened to it, so "sounds intentional" is a claim about the motif bank and synth spec, not a listening test.

## Documentation

- [docs/USER_MANUAL.md](docs/USER_MANUAL.md) — plain-language manual: what the lab is, every panel and button, how to read results
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the pieces fit
- [docs/DEFINITIONS.md](docs/DEFINITIONS.md) — state variables, update rules, learned components
- [docs/CONTRACTS.md](docs/CONTRACTS.md) — episode protocol, API, conditions, reset semantics
- [docs/CONTROLS_AND_CONFOUNDS.md](docs/CONTROLS_AND_CONFOUNDS.md) — the six conditions and what each can and cannot show
- [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md) — sources and adaptations
- [docs/MODEL_SETUP.md](docs/MODEL_SETUP.md) — optional runtime models
- [docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md) — initial results and next experiments
- [PROGRESS.md](PROGRESS.md) — build log and verification record

## Configuration

All experimental knobs are in [config/default.yaml](config/default.yaml). Every run records
its effective configuration. Same seed and config reproduce the same event log in local mode.
