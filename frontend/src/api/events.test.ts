import { describe, expect, it } from 'vitest'
import { emptyEpisode, foldEvent, readIntervention, statePointsFromEvents, type EpisodeRecord } from './events'
import type { Event } from './types'

const ev = (seq: number, step: number, type: Event['type'], payload: Record<string, unknown>, agent_id: string | null = null): Event => ({
  seq,
  step,
  t: '',
  type,
  agent_id,
  visibility: 'public',
  payload,
})

describe('event payload readers', () => {
  it('assemble an episode from backend-shaped payloads (flat episode_complete)', () => {
    const map = new Map<number, EpisodeRecord>()
    const sv = { activation: 0.5, expected_value: 0.5, uncertainty: 0.5, affiliation: 0 }
    const events: Event[] = [
      ev(0, 0, 'target_assigned', { target_id: 2, sender: 'A' }, 'A'),
      ev(1, 0, 'phrase_sent', { phrase: { id: 'p', notes: [], tempo_bpm: 100, length_beats: 8, instrument: 'pluck' }, motif_index: 3, trace: null }, 'A'),
      ev(2, 0, 'phrase_received', { observation: { channel: 'symbol', symbol_id: 3, sender_id: 'A', step: 0 } }, 'B'),
      ev(3, 0, 'action_chosen', { trace: { role: 'receiver', policy_kind: 'local', scores: [0, 1], probabilities: [0.3, 0.7], temperature: 0.1, temperature_base: 0.1, coupling_enabled: true, chosen: 1 } }, 'B'),
      ev(4, 0, 'outcome', { target_id: 2, chosen_pattern_id: 1, score: 0.2 }, 'B'),
      ev(5, 0, 'state_update', { before: sv, after: { ...sv, activation: 0.6 }, inputs: { decay_applied: true, frozen: false } }, 'B'),
      ev(6, 0, 'episode_complete', { step: 0, sender_id: 'A', receiver_id: 'B', target_id: 2, chosen_pattern_id: 1, score: 0.2 }),
    ]
    for (const e of events) foldEvent(map, e)
    const ep = map.get(0)!
    expect(ep).toMatchObject({ senderId: 'A', receiverId: 'B', targetId: 2, chosenId: 1, score: 0.2, motifIndex: 3, complete: true })
    expect(ep.observation?.symbol_id).toBe(3)
    expect(ep.receiverTrace?.chosen).toBe(1)
    expect(ep.stateChanges.B.after.activation).toBe(0.6)
    expect(statePointsFromEvents(events).B).toEqual([{ step: 0, state: { ...sv, activation: 0.6 } }])
    expect(emptyEpisode(4).complete).toBe(false)
  })

  it('read flat (backend) and nested intervention payloads', () => {
    const flat = readIntervention(ev(9, 5, 'intervention', { kind: 'reset_memory', agent_id: null, value: null, effective_from_step: 5 }))
    expect(flat).toMatchObject({ kind: 'reset_memory', agentId: null, effectiveFromStep: 5 })
    const nested = readIntervention(ev(10, 6, 'intervention', { intervention: { kind: 'set_coupling', agent_id: 'A', value: false }, effective_from_step: 7 }, 'A'))
    expect(nested).toMatchObject({ kind: 'set_coupling', agentId: 'A', effectiveFromStep: 7 })
    expect(readIntervention(ev(11, 6, 'outcome', {}))).toBeNull()
  })
})
