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

Requirements: Python 3.11+, Node 22+, `uv` (optional; falls back to `python -m venv`).

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

1. Open the app, click **Enable audio** (browsers require a gesture before sound).
2. Pick the **First encounter** preset and press **Start**. Two agents alternate as sender
   and receiver. Each episode: the sender sees a private target rhythm, plays a phrase, the
   receiver picks a rhythm, both get the same score. Watch the score climb (or not) and the
   state bars move.
3. Press **Pause**, then **Step** a few times. Open an agent's **Inspector**: state before →
   after with the inputs that caused it, the retrieved memories, the learned weights, and the
   policy's scores → probabilities with the effective temperature.
4. Try an intervention: **Clear memory** on one agent (its learned associations vanish but its
   state does not); **Reset state** (state returns to baseline, associations stay); **Freeze
   state**; **Disable coupling** (state keeps evolving but can no longer influence actions —
   the inspector's temperature column shows the change).
5. Select an earlier exchange in the timeline and **Replay motif**: hear it again and, in a
   live session, have the receiver actually receive it next step.
6. Open **Experiments**, run all six conditions over five seeds (no audio), and read the
   per-run points, not just the means.
7. Compose a phrase in the **Phrase editor**, pick a target, and send it as a human.

## Modes

| Badge | Meaning |
|---|---|
| **LIVE SIMULATION** | Episodes are being computed now from seed and config |
| **REPLAY** | An exported run is being re-emitted; nothing is recomputed |
| **DEMO DATA** | The browser's mock client; no backend involved |

Local experimental mode (default) needs no model. Model-assisted mode is optional; see
[docs/MODEL_SETUP.md](docs/MODEL_SETUP.md).

## Documentation

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
