import { describe, expect, it } from 'vitest'
import { MockLabApi } from '../api/mock'
import { badgeFor, focusedEpisode, initialState, reducer, type LabState } from './store'

const clock = () => '2026-10-01T00:00:00.000Z'

async function loaded() {
  const api = new MockLabApi({ clock })
  const snap = await api.createSession({ condition: { name: 'full', perturbation: 'none', description: '' }, preset: 'first_encounter' })
  let s: LabState = reducer(initialState, { type: 'source', kind: 'mock', reason: 'test', health: null })
  s = reducer(s, { type: 'sessionLoaded', snapshot: snap })
  return { api, s, id: snap.id }
}

describe('lab reducer', () => {
  it('loads a session with initial state points and the preset episode target', async () => {
    const { s } = await loaded()
    expect(s.session?.preset).toBe('first_encounter')
    expect(s.episodesTarget).toBe(60)
    expect(s.episodes).toHaveLength(0)
    expect(Object.keys(s.statePoints)).toEqual(['A', 'B'])
    expect(s.statePoints.A[0].step).toBe(0)
  })

  it('folds step results into episodes, trajectories and metrics', async () => {
    const { api, s: s0, id } = await loaded()
    let s = s0
    s = reducer(s, { type: 'stepResult', result: await api.step(id, { n: 1 }) })
    s = reducer(s, { type: 'stepResult', result: await api.step(id, { n: 10 }) })
    expect(s.episodes).toHaveLength(11)
    expect(s.episodes.map((e) => e.step)).toEqual([...Array(11).keys()])
    const e0 = s.episodes[0]
    expect(e0.senderId).toBe('A')
    expect(e0.receiverId).toBe('B')
    expect(e0.targetId).not.toBeNull()
    expect(e0.chosenId).not.toBeNull()
    expect(e0.phrase).not.toBeNull()
    expect(e0.observation?.channel).toBe('music')
    expect(e0.senderTrace?.role).toBe('sender')
    expect(e0.receiverTrace?.role).toBe('receiver')
    expect(Object.keys(e0.stateChanges).sort()).toEqual(['A', 'B'])
    // 0-based steps -> x offset 1: one point per completed episode plus the initial point
    expect(s.stepOffset).toBe(1)
    expect(s.statePoints.A.map((p) => p.step)).toEqual([...Array(12).keys()])
    expect(s.session?.metrics.episodes).toBe(11)
  })

  it('records interventions with their effective step, and selection drives the focused episode', async () => {
    const { api, s: s0, id } = await loaded()
    let s = reducer(s0, { type: 'stepResult', result: await api.step(id, { n: 3 }) })
    s = reducer(s, { type: 'stepResult', result: await api.intervene(id, { kind: 'freeze_state', agent_id: 'A', value: true }) })
    expect(s.interventions).toHaveLength(1)
    expect(s.interventions[0]).toMatchObject({ kind: 'freeze_state', agentId: 'A', effectiveFromStep: 3 })
    expect(s.session?.agents[0].frozen_state).toBe(true)
    expect(focusedEpisode(s)?.step).toBe(2)
    s = reducer(s, { type: 'select', step: 0 })
    expect(focusedEpisode(s)?.step).toBe(0)
    s = reducer(s, { type: 'select', step: null })
    expect(focusedEpisode(s)?.step).toBe(2)
  })

  it('an immediate reset_state is drawn as a jump at the current x, keeping the pre-reset point', async () => {
    const { api, s: s0, id } = await loaded()
    let s = reducer(s0, { type: 'stepResult', result: await api.step(id, { n: 4 }) })
    const before = s.statePoints.B[s.statePoints.B.length - 1]
    s = reducer(s, { type: 'stepResult', result: await api.intervene(id, { kind: 'reset_state', agent_id: 'B' }) })
    const pts = s.statePoints.B
    expect(pts[pts.length - 2]).toEqual(before)
    expect(pts[pts.length - 1].step).toBe(before.step)
    expect(pts[pts.length - 1].state).toEqual(s.session!.agents[1].baseline)
    s = reducer(s, { type: 'stepResult', result: await api.step(id, { n: 1 }) })
    expect(s.statePoints.B[s.statePoints.B.length - 1].step).toBe(before.step + 1)
  })

  it('badge semantics: DEMO for mock, REPLAY for imported runs, LIVE otherwise', async () => {
    const { s } = await loaded()
    expect(badgeFor(s)).toEqual(['demo'])
    const live = { ...s, source: { ...s.source, kind: 'live' as const } }
    expect(badgeFor(live)).toEqual(['live'])
    const replay = { ...live, session: { ...s.session!, mode: 'replay' as const } }
    expect(badgeFor(replay)).toEqual(['replay'])
    expect(badgeFor({ ...replay, source: s.source })).toEqual(['replay', 'demo'])
  })

  it('keeps the log bounded', () => {
    let s = initialState
    for (let i = 0; i < 250; i++) s = reducer(s, { type: 'log', entry: { kind: 'info', text: `m${i}` } })
    expect(s.log).toHaveLength(200)
    expect(s.log[s.log.length - 1].text).toBe('m249')
  })
})
