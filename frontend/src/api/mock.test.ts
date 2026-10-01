import { describe, expect, it } from 'vitest'
import { MockLabApi } from './mock'
import type { Event, ExperimentConfig } from './types'

const clock = () => '2026-10-01T00:00:00.000Z'
const strip = (evs: Event[]) => evs.map(({ t: _t, ...rest }) => rest)

async function run(seed: number, n: number) {
  const api = new MockLabApi({ clock })
  const cfg = (await api.defaultConfig()) as ExperimentConfig
  const s = await api.createSession({ config: { ...cfg, seed }, condition: { name: 'full', perturbation: 'none', description: '' }, preset: null })
  const r = await api.step(s.id, { n })
  return { api, id: s.id, result: r }
}

describe('mock client', () => {
  it('is deterministic: same seed + config => identical event log and snapshot metrics', async () => {
    const a = await run(11, 25)
    const b = await run(11, 25)
    expect(strip(a.result.events)).toEqual(strip(b.result.events))
    expect(a.result.snapshot.metrics).toEqual(b.result.snapshot.metrics)
    const c = await run(12, 25)
    expect(strip(c.result.events)).not.toEqual(strip(a.result.events))
  })

  it('follows the episode protocol and alternates roles', async () => {
    const { result } = await run(3, 4)
    const types = result.events.filter((e) => e.step === 0).map((e) => e.type)
    expect(types.slice(0, 5)).toEqual(['target_assigned', 'phrase_sent', 'phrase_received', 'action_chosen', 'outcome'])
    expect(types[types.length - 1]).toBe('episode_complete')
    const senders = result.events.filter((e) => e.type === 'phrase_sent').map((e) => e.agent_id)
    expect(senders).toEqual(['A', 'B', 'A', 'B'])
    expect(result.snapshot.step).toBe(4)
    expect(result.snapshot.mode).toBe('demo')
  })

  it('never puts the target into what the receiver observes', async () => {
    const { result, api, id } = await run(5, 30)
    for (const e of result.events.filter((x) => x.type === 'phrase_received')) {
      expect(JSON.stringify(e.payload)).not.toMatch(/target/i)
    }
    const insp = await api.inspect(id, 'B')
    expect(JSON.stringify(insp.information_received)).not.toMatch(/target/i)
    const target = result.events.find((e) => e.type === 'target_assigned')
    expect(target?.visibility).toBe('sender_private')
  })

  it('reset_memory clears memory + associations but leaves state; reset_state does the converse', async () => {
    const { api, id } = await run(7, 20)
    const before = await api.getSession(id)
    const a = before.agents[0]
    expect(a.memory_size).toBeGreaterThan(0)
    const r1 = await api.intervene(id, { kind: 'reset_memory', agent_id: 'A' })
    const a1 = r1.snapshot.agents[0]
    expect(a1.memory_size).toBe(0)
    expect(a1.learner.updates).toBe(0)
    expect(a1.state).toEqual(a.state)
    expect(r1.events[0].payload?.effective_from_step).toBe(20)
    const r2 = await api.intervene(id, { kind: 'reset_state', agent_id: 'B' })
    const b2 = r2.snapshot.agents[1]
    expect(b2.state).toEqual(b2.baseline)
    expect(b2.memory_size).toBe(before.agents[1].memory_size)
  })

  it('replay_motif makes the queued phrase the next sent phrase', async () => {
    const { api, id, result } = await run(9, 3)
    const first = result.events.find((e) => e.type === 'phrase_sent')!.payload!.phrase as { id: string; notes: unknown[] }
    await api.intervene(id, { kind: 'replay_motif', phrase_id: first.id })
    const r = await api.step(id, { n: 1 })
    const sent = r.events.find((e) => e.type === 'phrase_sent')!.payload!.phrase as { notes: unknown[]; origin: { kind: string } }
    expect(sent.notes).toEqual(first.notes)
    expect(sent.origin.kind).toBe('replay')
  })

  it('export -> import -> replay re-emits the recorded episodes', async () => {
    const { api, id, result } = await run(4, 6)
    const exp = await api.exportRun(id)
    const snap = await api.importRun(exp)
    expect(snap.mode).toBe('replay')
    expect(snap.step).toBe(0)
    const r = await api.replayStep(snap.id, { n: 6 })
    expect(strip(r.events)).toEqual(strip(result.events))
    expect(r.snapshot.metrics.score_history).toEqual(result.snapshot.metrics.score_history)
    const end = await api.replayStep(snap.id, { n: 1 })
    expect(end.events).toHaveLength(0)
    expect(end.snapshot.status).toBe('finished')
  })

  it('runs a small experiment with per-condition aggregates', async () => {
    const api = new MockLabApi({ clock })
    const cfg = await api.defaultConfig()
    const r = await api.runExperiment({
      name: 't',
      conditions: [
        { name: 'full', perturbation: 'none', description: '' },
        { name: 'no_history', perturbation: 'none', description: '' },
      ],
      seeds: [1, 2],
      config: { ...cfg, episodes: 30 },
    })
    expect(r.runs).toHaveLength(4)
    expect(r.aggregates.map((a) => a.condition)).toEqual(['full', 'no_history'])
    expect(r.aggregates[0].per_run_final_block).toHaveLength(2)
    expect(r.aggregates[0].mean_learning_curve).toHaveLength(10)
    const csv = await api.experimentCsv(r.id)
    expect(csv.split('\n')[0]).toContain('final_block_score')
    expect(csv.trim().split('\n')).toHaveLength(5)
  })
})
