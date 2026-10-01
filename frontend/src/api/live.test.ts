/**
 * Opt-in contract test against a RUNNING backend:
 *   LIVE_API=http://127.0.0.1:8000 npx vitest run src/api/live.test.ts
 * Skipped by default (no backend in unit-test runs).
 */
import { describe, expect, it } from 'vitest'
import { HttpLabApi } from './client'
import { initialState, reducer, type LabState } from '../state/store'
import type { Phrase } from './types'

const base = (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env.LIVE_API
const d = base ? describe : describe.skip

d('live API contract (HttpLabApi)', () => {
  const api = new HttpLabApi(base)

  it('serves health, catalog and config', async () => {
    const h = await api.health()
    expect(h.status).toBe('ok')
    const [presets, patterns, motifs, cfg, model] = await Promise.all([api.presets(), api.patterns(), api.motifs(), api.defaultConfig(), api.modelStatus()])
    expect(presets.length).toBeGreaterThanOrEqual(3)
    expect(patterns.length).toBe(cfg.task?.n_patterns)
    expect(motifs.length).toBeGreaterThan(0)
    expect(typeof model.input_modality).toBe('string')
  })

  it('creates a preset session, steps, and the reducer folds the events into episodes', async () => {
    const snap = await api.createSession({ preset: 'first_encounter', condition: { name: 'full', perturbation: 'none', description: '' } })
    expect(snap.mode).toBe('live')
    let s: LabState = reducer(initialState, { type: 'source', kind: 'live', reason: 'test', health: null })
    s = reducer(s, { type: 'sessionLoaded', snapshot: snap })
    const r1 = await api.step(snap.id, { n: 1 })
    s = reducer(s, { type: 'stepResult', result: r1 })
    const r2 = await api.step(snap.id, { n: 10 })
    s = reducer(s, { type: 'stepResult', result: r2 })
    expect(s.episodes).toHaveLength(11)
    const e0 = s.episodes[0]
    expect(e0.complete).toBe(true)
    expect(e0.senderId).toBe(snap.agents[0].id)
    expect(e0.receiverId).toBe(snap.agents[1].id)
    expect(e0.targetId).not.toBeNull()
    expect(e0.chosenId).not.toBeNull()
    expect(e0.score).not.toBeNull()
    expect(e0.phrase?.notes.length).toBeGreaterThan(0)
    expect(e0.observation).not.toBeNull()
    expect(JSON.stringify(e0.observation)).not.toMatch(/target/i)
    expect(e0.senderTrace?.role).toBe('sender')
    expect(e0.receiverTrace?.role).toBe('receiver')
    expect(Object.keys(e0.stateChanges).length).toBe(2)
    expect(s.stepOffset).toBe(1)
    expect(s.statePoints[snap.agents[0].id].length).toBe(12)
    expect(s.session?.metrics.episodes).toBe(11)
  })

  it('applies every intervention kind and reports effective_from_step', async () => {
    const snap = await api.createSession({ condition: { name: 'full', perturbation: 'none', description: '' } })
    const r = await api.step(snap.id, { n: 3 })
    const sent = r.events.find((e) => e.type === 'phrase_sent')!.payload!.phrase as Phrase
    const A = snap.agents[0].id
    let s: LabState = reducer(reducer(initialState, { type: 'sessionLoaded', snapshot: snap }), { type: 'stepResult', result: r })
    for (const iv of [
      { kind: 'replay_motif' as const, phrase_id: sent.id, phrase: sent },
      { kind: 'reset_memory' as const, agent_id: A },
      { kind: 'reset_state' as const, agent_id: null },
      { kind: 'freeze_state' as const, agent_id: A, value: true },
      { kind: 'set_coupling' as const, agent_id: A, value: false },
      { kind: 'swap_feature' as const, transform: 'transpose' as const },
      { kind: 'set_param' as const, path: 'state.decay.activation', value: 0.2 },
    ]) {
      const res = await api.intervene(snap.id, iv)
      expect(res.events[0].type).toBe('intervention')
      expect(res.events[0].payload?.effective_from_step).toBe(3)
      s = reducer(s, { type: 'stepResult', result: res })
    }
    expect(s.interventions.map((i) => i.kind)).toEqual(['replay_motif', 'reset_memory', 'reset_state', 'freeze_state', 'set_coupling', 'swap_feature', 'set_param'])
    const a = s.session!.agents[0]
    expect(a.frozen_state).toBe(true)
    expect(a.coupling_enabled).toBe(false)
    expect(a.memory_size).toBe(0)
    expect(s.session!.config.state?.decay?.activation).toBe(0.2)
    const next = await api.step(snap.id, { n: 1 })
    const ph = next.events.find((e) => e.type === 'phrase_sent')!.payload!.phrase as Phrase
    expect(ph.origin?.kind).toMatch(/replay|transformed/)
  })

  // The WS2 brief lists `agents.0.sensitivity` as a set_param example (the UI has a per-agent
  // sensitivity slider). Kept as its own test so a backend divergence is visible.
  it('set_param agents.0.sensitivity (brief example path)', async () => {
    const snap = await api.createSession({ condition: { name: 'full', perturbation: 'none', description: '' } })
    const res = await api.intervene(snap.id, { kind: 'set_param', path: 'agents.0.sensitivity', value: 1.5 })
    expect(res.snapshot.config.agents?.[0].sensitivity).toBe(1.5)
  })

  it('human phrase, inspect, features, render, transform, saved motifs', async () => {
    const snap = await api.createSession({ condition: { name: 'full', perturbation: 'none', description: '' } })
    const phrase: Phrase = {
      id: 'human-test',
      notes: [
        { pitch: 60, onset: 0, duration: 0.5, velocity: 90 },
        { pitch: 64, onset: 1, duration: 0.5, velocity: 90 },
        { pitch: 67, onset: 2, duration: 0.5, velocity: 90 },
      ],
      tempo_bpm: 112,
      length_beats: 8,
      instrument: 'pluck',
    }
    const res = await api.humanPhrase(snap.id, { phrase, target_id: 1, receiver_id: snap.agents[1].id })
    let s: LabState = reducer(reducer(initialState, { type: 'sessionLoaded', snapshot: snap }), { type: 'stepResult', result: res })
    expect(s.episodes[0].human).toBe(true)
    expect(s.episodes[0].targetId).toBe(1)
    const insp = await api.inspect(snap.id, snap.agents[1].id)
    expect(insp.state_history.length).toBeGreaterThan(0)
    const f = await api.phraseFeatures(phrase)
    expect(f.symbolic.note_count).toBe(3)
    expect(f.symbolic.vector.length).toBe(16)
    const wav = await api.renderPhrase(phrase)
    expect(wav.size).toBeGreaterThan(1000)
    const head = new Uint8Array(await wav.slice(0, 4).arrayBuffer())
    expect(String.fromCharCode(...head)).toBe('RIFF')
    const t = await api.transformPhrase({ phrase, transform: 'transpose' })
    expect(t.notes[0].pitch).toBe(65)
    const list = await api.saveMotif({ name: 'live-test', phrase, created_at: new Date().toISOString(), features: null })
    expect(list.some((m) => m.name === 'live-test')).toBe(true)
    expect((await api.savedMotifs()).some((m) => m.name === 'live-test')).toBe(true)
    void s
  })

  it('export -> import -> replay/step re-emits the recorded run', async () => {
    const snap = await api.createSession({ condition: { name: 'full', perturbation: 'none', description: '' } })
    const r = await api.step(snap.id, { n: 4 })
    const exp = await api.exportRun(snap.id)
    expect(exp.events.length).toBeGreaterThan(0)
    const rep = await api.importRun(exp)
    expect(rep.mode).toBe('replay')
    const rs = await api.replayStep(rep.id, { n: 4 })
    const scores = (evs: typeof r.events) => evs.filter((e) => e.type === 'outcome').map((e) => e.payload?.score)
    expect(scores(rs.events)).toEqual(scores(r.events))
    let s: LabState = reducer(reducer(initialState, { type: 'sessionLoaded', snapshot: rep }), { type: 'stepResult', result: rs })
    expect(s.episodes).toHaveLength(4)
    void s
  })

  it('runs a tiny experiment, CSV, and a preset comparison', async () => {
    const cfg = await api.defaultConfig()
    const r = await api.runExperiment({
      name: 'contract',
      conditions: [
        { name: 'full', perturbation: 'none', description: '' },
        { name: 'symbol', perturbation: 'none', description: '' },
      ],
      seeds: [1, 2],
      config: { ...cfg, episodes: 20 },
    })
    expect(r.runs).toHaveLength(4)
    expect(r.aggregates).toHaveLength(2)
    const csv = await api.experimentCsv(r.id)
    expect(csv.split('\n').length).toBeGreaterThan(2)
    expect((await api.getExperiment(r.id)).id).toBe(r.id)
    const pr = await api.runPreset('first_encounter', { seed: 3 })
    expect(pr.preset).toBe('first_encounter')
    expect(Object.keys(pr.comparison).length).toBeGreaterThan(0)
  }, 60_000)
})
