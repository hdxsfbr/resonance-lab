# Resonance Lab — frontend

Single-screen laboratory for watching two engineered agents exchange short musical phrases:
hear the phrases (Web Audio), see the engineered state and learned associations evolve,
intervene, replay motifs, compare conditions. Vite 8 + React 19 + TypeScript (strict), no UI
framework, hand-rolled SVG charts.

The UI keeps three kinds of quantity visibly separate, each with a tag:
**MEASURED** (features of the notes / waveform), **ENGINEERED** / **LEARNED** (state vector with
operational definitions, learner weights, memory), **GENERATED** (model output / narrative —
"not a measurement"). Hand-authored dynamics (the acoustic→activation drive) are tagged
**HAND-AUTHORED**.

## Run

```bash
npm install --legacy-peer-deps      # once
npm run dev                         # http://localhost:5173, proxies /api -> http://localhost:8000
npm run build                       # tsc -b (strict, zero errors) + vite build -> dist/
npm run preview                     # serve dist/ (also proxies /api)
VITE_API_TARGET=http://127.0.0.1:8765 npm run dev   # point the proxy at another backend
```

From the repo root, `make dev` starts API + Vite; `./start.sh` builds `dist/` and serves it from the API.

Append `?mock=1` to force the in-browser demo data source.

## Data-source badge

A persistent badge in the header says where every number on screen comes from:

| badge | meaning |
|---|---|
| **LIVE SIMULATION** | the backend answered `GET /api/health`; sessions are stepped live through `/api/sessions/{id}/step` |
| **REPLAY** | an imported run (`POST /api/sessions/import`); Step/Start call `/api/sessions/{id}/replay/step`, which re-emits recorded events; interventions are disabled |
| **DEMO DATA (mock)** | `GET /api/health` failed / returned non-2xx (e.g. 501 while the backend is a stub), or `?mock=1`: an in-browser mock implements the same client interface. Nothing shown is a backend result |

REPLAY and DEMO can appear together (a replay imported into the mock). The footer repeats the
source; the tooltip on the badge gives the selection reason.

## Architecture

```
src/
  api/schema.d.ts      GENERATED from the backend OpenAPI (make types) — never hand-edit
  api/types.ts         aliases over components['schemas']
  api/client.ts        LabApi interface + HttpLabApi (each call bound to `paths` at compile time) + chooseClient()
  api/events.ts        tolerant readers for Event payloads (payload is dict[str, Any] in the contract)
  api/mock.ts          MockLabApi: same interface, in-browser (lazy-loaded chunk)
  api/mock/*           mock simulation (episode protocol, update rules from docs/DEFINITIONS.md),
                       experiments, presets, replay, transforms, seeded RNG streams
  audio/               synth spec math, Web Audio graph, engine (autoplay-safe), playhead, WAV  -> SYNTH_SPEC.md
  state/store.ts       reducer (episodes, trajectories, interventions, log) — unit-tested
  state/LabContext.tsx React context + controller: run loop ("frontend owns time"), interventions, import/export
  components/          Header, ControlsBar, AgentPanel, TaskCard, PianoRoll, Timeline, TrajectoryPlot,
                       ScorePlot, InterventionsPanel/EventLog, Inspector, PhraseEditor, ExperimentsView, PresetsMenu
  lib/definitions.ts   operational definitions copied from schemas.py / CONTRACTS.md (keep in sync)
```

Time: the backend never advances on its own. At 1× the loop waits for the sent phrase's audio
(incl. release and 0.5 s tail) to finish before requesting the next step; "no audio / fast"
calls `step` with `n=10`. See `src/audio/SYNTH_SPEC.md` for the pacing table.

## Synth

`src/audio/SYNTH_SPEC.md` (browser realisation) and `../docs/SYNTH_SPEC.md` (shared spec with the
server numpy synth). Envelopes and both one-pole filters follow the server's difference equations;
the remaining known difference is band-limited vs naive oscillators. Audio is created/resumed only
from the **Enable audio** button (browser autoplay policy); the header shows the audio state.
**⤓ WAV** uses `POST /api/phrases/render` in live mode and an `OfflineAudioContext` render in mock
mode; **⤓ JSON** downloads the symbolic phrase.

## Tests

```bash
npx vitest run                                   # unit tests (src/**/*.test.ts)
npx playwright test                              # e2e in MOCK mode against `vite preview` (builds first)
LIVE_API=http://127.0.0.1:8000 npx vitest run src/api/live.test.ts                 # contract test vs a running backend
VITE_API_TARGET=http://127.0.0.1:8000 LIVE_SMOKE=1 npx playwright test tests/live.spec.ts  # browser flow vs live backend
```

- Unit: scheduling math (beats→seconds, timing, pacing), envelope shapes (ADSR, percussive,
  sampled curves), WAV encoding, mock determinism / protocol / information isolation / reset
  semantics / replay, reducer (episode folding, trajectories, interventions, badge semantics),
  event payload readers.
- `tests/smoke.spec.ts`: 1440×900, `?mock=1`, asserts no console errors, clicks Enable audio,
  starts a preset, steps 3×, opens the inspector, opens experiments and runs one; screenshots in
  `test-results/`.
- `tests/interactions.spec.ts`: run loop at 1× and fast, interventions, timeline replay, composer
  → save / measure / send as human, WAV export, preset comparison, export → import → REPLAY; plus
  a 1024-px layout check.
- Playwright uses the pre-installed Chromium at `/opt/pw-browsers/chromium-1194/chrome-linux/chrome`
  (override with `PW_CHROMIUM`); never run `playwright install` here.

## What the mock does (DEMO DATA)

Two agents (Aria/pluck, Bram/marimba), 8 motifs, 4 timing patterns, roles alternating. Sender =
softmax bandit `Q[target][motif]`; receiver = linear softmax over a 16-d feature vector
(standardised against the mock motif bank); state update and coupling follow
`docs/DEFINITIONS.md`. Deterministic per seed. Its learning defaults (η 0.3/0.5, τ 0.08) differ
from the backend's so that learning is visible in ~120 episodes; the config popover shows them.
Experiments, presets (with templated, labelled comparisons), export/import/replay, saved motifs
and feature measurement are all implemented in the browser. The mock never emits `model_call`
events (policies are local), so the inspector's model panel stays empty in demo mode.
