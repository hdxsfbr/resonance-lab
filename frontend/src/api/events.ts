/**
 * Tolerant reading of Event payloads.
 *
 * CONTRACT GAP: `Event.payload` is `dict[str, Any]` in schemas.py, so payload keys
 * are not covered by the generated types. The keys below are what this UI reads
 * (and what the mock emits). Each reader also accepts a few obvious variants so the
 * UI degrades gracefully if the backend chooses different names.
 *
 *   target_assigned   payload.target_id: int                       (visibility sender_private)
 *   phrase_sent       payload.phrase: Phrase, payload.motif_index?: int, payload.trace?: PolicyTrace (sender)
 *   phrase_received   payload.observation: Observation             (agent_id = receiver)
 *   action_chosen     payload.trace: PolicyTrace (or payload IS a PolicyTrace), agent_id = decider
 *   outcome           payload.score, payload.chosen_pattern_id, payload.target_id
 *   state_update      payload.before, payload.after: StateVector, payload.inputs: StateUpdateInputs (agent_id)
 *   learning_update   free-form (agent_id)
 *   intervention      payload.kind, payload.agent_id (null = both agents), payload.value, payload.effective_from_step: int
 *                     (+ details such as source_phrase_id, state_after, path). Mock adds payload.effect / payload.description.
 *                     A nested payload.intervention: Intervention is also accepted.
 *   model_call        payload.request?: ModelRequest, payload.response?: ModelResponse
 *   generated_narrative payload.text: string
 *   episode_complete  payload may carry EpisodeSummary fields (directly or under payload.episode)
 */
import type {
  EpisodeSummary,
  Event,
  Intervention,
  Observation,
  Phrase,
  PolicyTrace,
  StateUpdateInputs,
  StateVector,
} from './types'

type Rec = Record<string, unknown>

const isObj = (x: unknown): x is Rec => typeof x === 'object' && x !== null && !Array.isArray(x)
const num = (x: unknown): number | null => (typeof x === 'number' && Number.isFinite(x) ? x : null)
const str = (x: unknown): string | null => (typeof x === 'string' ? x : null)

export function isPhrase(x: unknown): x is Phrase {
  return isObj(x) && Array.isArray(x.notes) && typeof x.tempo_bpm === 'number' && typeof x.length_beats === 'number'
}
export function isPolicyTrace(x: unknown): x is PolicyTrace {
  return isObj(x) && Array.isArray(x.probabilities) && Array.isArray(x.scores) && typeof x.chosen === 'number'
}
export function isStateVector(x: unknown): x is StateVector {
  return (
    isObj(x) &&
    typeof x.activation === 'number' &&
    typeof x.expected_value === 'number' &&
    typeof x.uncertainty === 'number' &&
    typeof x.affiliation === 'number'
  )
}
export function isObservation(x: unknown): x is Observation {
  return isObj(x) && (x.channel === 'music' || x.channel === 'symbol')
}

function pick<T>(payload: Rec, keys: string[], guard: (x: unknown) => x is T): T | null {
  for (const k of keys) if (guard(payload[k])) return payload[k] as T
  return null
}

export interface StateChange {
  before: StateVector
  after: StateVector
  inputs: StateUpdateInputs | null
}

export interface InterventionRecord {
  seq: number
  step: number
  kind: string
  agentId: string | null
  effectiveFromStep: number
  effect: string
  description: string
  intervention: Partial<Intervention>
}

export interface EpisodeRecord {
  step: number
  senderId: string | null
  receiverId: string | null
  targetId: number | null
  chosenId: number | null
  score: number | null
  phrase: Phrase | null
  observation: Observation | null
  motifIndex: number | null
  motifId: string | null
  senderTrace: PolicyTrace | null
  receiverTrace: PolicyTrace | null
  stateChanges: Record<string, StateChange>
  modelCalls: Event[]
  narratives: Event[]
  complete: boolean
}

export function emptyEpisode(step: number): EpisodeRecord {
  return {
    step,
    senderId: null,
    receiverId: null,
    targetId: null,
    chosenId: null,
    score: null,
    phrase: null,
    observation: null,
    motifIndex: null,
    motifId: null,
    senderTrace: null,
    receiverTrace: null,
    stateChanges: {},
    modelCalls: [],
    narratives: [],
    complete: false,
  }
}

function payloadOf(e: Event): Rec {
  return isObj(e.payload) ? e.payload : {}
}

function applyTrace(ep: EpisodeRecord, trace: PolicyTrace, agentId: string | null | undefined) {
  if (trace.role === 'sender') {
    ep.senderTrace = trace
    if (agentId && !ep.senderId) ep.senderId = agentId
  } else {
    ep.receiverTrace = trace
    if (agentId && !ep.receiverId) ep.receiverId = agentId
    if (ep.chosenId === null) ep.chosenId = trace.chosen
  }
}

function applySummary(ep: EpisodeRecord, s: Partial<EpisodeSummary>) {
  if (typeof s.sender_id === 'string') ep.senderId = s.sender_id
  if (typeof s.receiver_id === 'string') ep.receiverId = s.receiver_id
  if (typeof s.target_id === 'number') ep.targetId = s.target_id
  if (typeof s.chosen_pattern_id === 'number') ep.chosenId = s.chosen_pattern_id
  if (typeof s.score === 'number') ep.score = s.score
  if (isPhrase(s.phrase) && !ep.phrase) ep.phrase = s.phrase
  if (isObservation(s.observation) && !ep.observation) ep.observation = s.observation
}

/** Fold one event into the per-episode record map (mutates `map`). */
export function foldEvent(map: Map<number, EpisodeRecord>, e: Event): void {
  const t = e.type
  if (t === 'session_created' || t === 'session_reset' || t === 'intervention') return
  let ep = map.get(e.step)
  if (!ep) {
    ep = emptyEpisode(e.step)
    map.set(e.step, ep)
  }
  const p = payloadOf(e)
  switch (t) {
    case 'target_assigned': {
      const tid = num(p.target_id) ?? num(p.target)
      if (tid !== null) ep.targetId = tid
      if (e.agent_id && !ep.senderId) ep.senderId = e.agent_id
      break
    }
    case 'phrase_sent': {
      const ph = pick(p, ['phrase'], isPhrase) ?? (isPhrase(p) ? (p as unknown as Phrase) : null)
      if (ph) {
        ep.phrase = ph
        ep.motifId = ph.motif_id ?? ep.motifId
      }
      ep.motifIndex = num(p.motif_index) ?? num(p.motif) ?? ep.motifIndex
      ep.motifId = str(p.motif_id) ?? ep.motifId
      if (e.agent_id) ep.senderId = e.agent_id
      const rid = str(p.receiver_id)
      if (rid) ep.receiverId = rid
      const tr = pick(p, ['trace', 'policy_trace'], isPolicyTrace)
      if (tr) applyTrace(ep, tr, e.agent_id)
      break
    }
    case 'phrase_received': {
      const ob = pick(p, ['observation'], isObservation) ?? (isObservation(p) ? (p as unknown as Observation) : null)
      if (ob) {
        ep.observation = ob
        if (!ep.senderId) ep.senderId = ob.sender_id
      }
      if (e.agent_id) ep.receiverId = e.agent_id
      break
    }
    case 'action_chosen': {
      const tr = pick(p, ['trace', 'policy_trace'], isPolicyTrace) ?? (isPolicyTrace(p) ? (p as unknown as PolicyTrace) : null)
      if (tr) applyTrace(ep, tr, e.agent_id)
      const c = num(p.chosen_pattern_id)
      if (c !== null) ep.chosenId = c
      break
    }
    case 'outcome': {
      const s = num(p.score)
      if (s !== null) ep.score = s
      const c = num(p.chosen_pattern_id) ?? num(p.chosen)
      if (c !== null) ep.chosenId = c
      const tid = num(p.target_id)
      if (tid !== null) ep.targetId = tid
      break
    }
    case 'state_update': {
      const before = pick(p, ['before', 'state_before'], isStateVector)
      const after = pick(p, ['after', 'state_after', 'state'], isStateVector)
      const inputs = isObj(p.inputs) ? (p.inputs as unknown as StateUpdateInputs) : null
      if (e.agent_id && after) ep.stateChanges[e.agent_id] = { before: before ?? after, after, inputs }
      break
    }
    case 'model_call':
      ep.modelCalls.push(e)
      break
    case 'generated_narrative':
      ep.narratives.push(e)
      break
    case 'episode_complete': {
      applySummary(ep, (isObj(p.episode) ? p.episode : isObj(p.summary) ? p.summary : p) as Partial<EpisodeSummary>)
      ep.complete = true
      break
    }
    default:
      break
  }
}

export function readIntervention(e: Event): InterventionRecord | null {
  if (e.type !== 'intervention') return null
  const p = payloadOf(e)
  const inner = (isObj(p.intervention) ? p.intervention : p) as Partial<Intervention>
  const kind = str(inner.kind) ?? str(p.kind) ?? 'intervention'
  const eff = num(p.effective_from_step) ?? e.step
  return {
    seq: e.seq,
    step: e.step,
    kind,
    agentId: (inner.agent_id as string | null | undefined) ?? e.agent_id ?? null,
    effectiveFromStep: eff,
    effect: str(p.effect) ?? '',
    description: str(p.description) ?? '',
    intervention: inner,
  }
}

export interface StatePoint {
  step: number
  state: StateVector
}

/** Per-agent state trajectory points from state_update events (value AFTER the episode's update). */
export function statePointsFromEvents(events: readonly Event[]): Record<string, StatePoint[]> {
  const out: Record<string, StatePoint[]> = {}
  for (const e of events) {
    if (e.type !== 'state_update' || !e.agent_id) continue
    const p = payloadOf(e)
    const after = pick(p, ['after', 'state_after', 'state'], isStateVector)
    if (!after) continue
    ;(out[e.agent_id] ??= []).push({ step: e.step, state: after })
  }
  return out
}

export function narrativeText(e: Event): string {
  const p = payloadOf(e)
  return str(p.text) ?? str(p.narrative) ?? JSON.stringify(p)
}
