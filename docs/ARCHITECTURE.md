# Architecture

Resonance Lab is a small, local-first application: a Python simulation core behind a
FastAPI HTTP API, a React/TypeScript single-screen laboratory in the browser, and SQLite
for persistence. There are no distributed services, no vector database and no agent
framework. The whole thing runs with `./start.sh` and no API keys.

```
┌────────────────────────────── browser ───────────────────────────────┐
│ React lab UI  ── typed client (generated from OpenAPI) ──┐            │
│ Web Audio synth (instrument spec shared with server)     │            │
└──────────────────────────────────────────────────────────┼────────────┘
                                                           │ /api/*
┌──────────────────────────── backend (FastAPI) ───────────▼────────────┐
│ api.py          routes, session registry, static frontend             │
│ session.py      episode protocol, RNG streams, channel, interventions │
│ conditions.py   six matched conditions → effective settings           │
│ env/            Task protocol; TimingTask (replaceable)               │
│ agents/         StateEngine, EpisodicMemory, learners, LocalPolicy    │
│ music/          motifs, symbolic features, transforms, numpy synth    │
│ experiments.py  headless multi-seed runner, per-run metrics, CSV      │
│ presets.py      three reproducible presets with comparisons           │
│ models/         optional provider abstraction + ModelPolicy (WS3)     │
│ store.py        SQLite: sessions, events, experiments, saved motifs   │
└───────────────────────────────────────────────────────────────────────┘
```

## Where time lives

The backend never advances on its own. The frontend calls `POST /sessions/{id}/step`
and, at 1× speed, waits for the phrase's audio to finish before requesting the next
episode. Pause is simply "stop calling". This keeps the simulation deterministic,
makes single-stepping free, and makes replay a matter of re-emitting recorded events.

## The three kinds of quantity

| Kind | Produced by | Shown as |
|---|---|---|
| Measured | `music/features.py` (from notes), `music/synth.py` (from waveform) | feature tables, labelled *symbolic* or *audio-measured* |
| Engineered | `agents/state.py` equations, learner weights, memory | state bars with operational definitions; weight heatmaps; Q tables |
| Generated | `models/narrative.py`, model rationales | panels labelled *generated — not a measurement* |

The simulation's state transitions and the task scoring never read generated text.

## Information isolation

The hidden target is drawn by the environment and handed only to the sender's
`SenderContext`. The receiver's `ReceiverContext` is built from an `Observation`
(phrase + symbolic features, or a symbol id) that is constructed without the target.
The `phrase_received` event records the exact `Observation` so the UI can show what
the receiver actually saw. `tests/test_isolation.py` checks that no observation,
receiver information log entry, receiver context, or model prompt contains the target.

## Randomness

`SeedSequence(seed).spawn(5)` → `rng_env` (targets), `rng_policy_A`, `rng_policy_B`
(exploration), `rng_gen` (phrase jitter), `rng_perturb` (channel perturbations).
Changing exploration behaviour therefore does not change which targets are drawn.

## Model-assisted mode

`models/registry.py` builds a provider from the `model:` config block. `ModelPolicy`
implements the same `PolicyProtocol` as `LocalPolicy`; it formats a *symbolic* feature
summary, retrieved memories and the engineered state into a prompt, asks for a JSON
decision under a strict schema, validates it, and falls back to the local policy on
any failure, timeout or exhausted call budget. Every call is logged as a `model_call`
event with the request, the response and the input modality. No provider in this build
accepts audio; the UI says so.
