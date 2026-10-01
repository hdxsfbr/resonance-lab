/**
 * In-browser MOCK simulation implementing the episode protocol in docs/CONTRACTS.md
 * and the update rules in docs/DEFINITIONS.md closely enough to produce plausible,
 * deterministic demo data. It is NOT the backend simulation; every number it
 * produces is shown under the "DEMO DATA (mock)" badge.
 */
import type {
  AgentInspection,
  AgentParams,
  AgentSnapshot,
  ConditionSpec,
  EpisodeSummary,
  Event,
  EventType,
  ExperimentConfig,
  HumanPhraseRequest,
  Intervention,
  MemoryItem,
  Observation,
  Perturbation,
  Phrase,
  PolicyTrace,
  RetrievedMemory,
  RunExport,
  SessionMetrics,
  SessionSnapshot,
  SessionSummary,
  StateDim,
  StateUpdateInputs,
  StateVector,
  TimingPattern,
} from '../types'
import { mockFeatures as computeSymbolicFeatures } from './normalise'
import { clamp, cosine, mean, softmax } from '../../lib/stats'
import { MOCK_MOTIFS, MOCK_PATTERNS, MOCK_TEMPO } from './data'
import { Rng, spawnStreams } from './rng'
import { transformPhrase } from './transforms'

export const N_FEATURES = 16
const DIMS: StateDim[] = ['activation', 'expected_value', 'uncertainty', 'affiliation']

export class MockHttpError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export interface Effective {
  learning: boolean
  memoryCap: number
  stateUpdates: boolean
  coupling: boolean
  channel: 'music' | 'symbol'
  perturbation: Perturbation
}

export function effectiveSettings(cond: ConditionSpec, cfg: ExperimentConfig): Effective {
  const learningOn = cfg.learning?.enabled ?? true
  const couplingOn = cfg.coupling?.enabled ?? true
  const cap = cfg.memory?.capacity ?? 200
  const base: Effective = {
    learning: learningOn,
    memoryCap: cap,
    stateUpdates: true,
    coupling: couplingOn,
    channel: 'music',
    perturbation: 'none',
  }
  switch (cond.name) {
    case 'no_history':
      return { ...base, learning: false, memoryCap: 0 }
    case 'state_fixed':
      return { ...base, stateUpdates: false }
    case 'state_decoupled':
      return { ...base, coupling: false }
    case 'symbol':
      return { ...base, channel: 'symbol' }
    case 'perturbed':
      return { ...base, perturbation: cond.perturbation }
    default:
      return base
  }
}

export function scorePatterns(a: TimingPattern, b: TimingPattern, partial: boolean): number {
  if (a.id === b.id) return 1
  if (!partial) return 0
  const sa = new Set(a.onsets.map((x) => x.toFixed(3)))
  const sb = new Set(b.onsets.map((x) => x.toFixed(3)))
  let inter = 0
  for (const x of sa) if (sb.has(x)) inter++
  const union = new Set([...sa, ...sb]).size
  return union ? (0.5 * inter) / union : 0
}

interface AgentInternal {
  index: number
  id: string
  state: StateVector
  baseline: StateVector
  memory: MemoryItem[]
  W: number[][]
  b: number[]
  Q: number[][]
  counts: number[][]
  updates: number
  lastTrace: PolicyTrace | null
  frozen: boolean
  coupling: boolean
  stateHistory: StateVector[]
  info: Record<string, unknown>[]
  lastInputs: StateUpdateInputs | null
  lastRetrieved: RetrievedMemory[]
  lastOwnVec: number[] | null
}

function deepClone<T>(x: T): T {
  return JSON.parse(JSON.stringify(x)) as T
}

function round(x: number, d = 4): number {
  const f = 10 ** d
  return Math.round(x * f) / f
}

export interface MockSimOptions {
  id: string
  seed: number
  config: ExperimentConfig
  condition: ConditionSpec
  preset?: string | null
  recordEvents?: boolean
  clock?: () => string
}

export class MockSim {
  readonly id: string
  readonly createdAt: string
  readonly preset: string | null
  seed: number
  config: ExperimentConfig
  condition: ConditionSpec
  eff: Effective
  step = 0
  events: Event[] = []
  scores: number[] = []
  current: EpisodeSummary | null = null
  agents: AgentInternal[] = []
  patterns: TimingPattern[]
  nMotifs: number
  pendingReplay: Phrase | null = null
  swap: Perturbation = 'none'
  phrasesById = new Map<string, Phrase>()
  private seq = 0
  private rng = spawnStreams(0)
  private readonly record: boolean
  private readonly clock: () => string

  constructor(o: MockSimOptions) {
    this.id = o.id
    this.clock = o.clock ?? (() => new Date().toISOString())
    this.createdAt = this.clock()
    this.preset = o.preset ?? null
    this.seed = o.seed
    this.config = deepClone(o.config)
    this.config.seed = o.seed
    this.condition = deepClone(o.condition)
    this.eff = effectiveSettings(this.condition, this.config)
    this.record = o.recordEvents ?? true
    const nP = clamp(this.config.task?.n_patterns ?? 4, 2, MOCK_PATTERNS.length)
    this.patterns = MOCK_PATTERNS.slice(0, nP)
    this.nMotifs = clamp(this.config.music?.n_motifs ?? 8, 2, MOCK_MOTIFS.length)
    this.init()
  }

  /** Fresh agents, same seed and config (also used by session reset). */
  init(): void {
    this.rng = spawnStreams(this.seed)
    this.step = 0
    this.scores = []
    this.current = null
    this.pendingReplay = null
    this.swap = 'none'
    this.phrasesById.clear()
    this.agents = (this.config.agents ?? []).slice(0, 2).map((p, i) => this.makeAgent(p, i))
    this.emit('session_created', 0, null, 'public', {
      seed: this.seed,
      condition: this.condition,
      note: 'DEMO DATA (mock): produced by the in-browser mock simulation, not the backend.',
    })
  }

  /** Session reset: new agents, same seed and config, event log restarted. */
  reset(): void {
    this.events = []
    this.seq = 0
    this.init()
    this.emit('session_reset', 0, null, 'public', { note: 'fresh agents, same seed and config; event log restarted' })
  }

  private baselineFor(p: AgentParams): StateVector {
    return deepClone(p.baseline_override ?? this.config.state?.baseline ?? { activation: 0.5, expected_value: 0.5, uncertainty: 0.5, affiliation: 0 })
  }

  private freshLearner(): Pick<AgentInternal, 'W' | 'b' | 'Q' | 'counts' | 'updates'> {
    const nP = this.patterns.length
    const init = this.config.learning?.optimistic_init ?? 0.5
    const scale = this.config.learning?.receiver_init_scale ?? 0
    return {
      W: Array.from({ length: nP }, () => Array.from({ length: N_FEATURES }, () => (scale ? (this.rng.gen.next() - 0.5) * 2 * scale : 0))),
      b: Array(nP).fill(init),
      Q: Array.from({ length: nP }, () => Array(this.nMotifs).fill(init)),
      counts: Array.from({ length: nP }, () => Array(this.nMotifs).fill(0)),
      updates: 0,
    }
  }

  private makeAgent(p: AgentParams, index: number): AgentInternal {
    const baseline = this.baselineFor(p)
    return {
      index,
      id: p.id,
      state: { ...baseline },
      baseline,
      memory: [],
      ...this.freshLearner(),
      lastTrace: null,
      frozen: false,
      coupling: true,
      stateHistory: [{ ...baseline }],
      info: [],
      lastInputs: null,
      lastRetrieved: [],
      lastOwnVec: null,
    }
  }

  params(a: AgentInternal): AgentParams {
    return (this.config.agents ?? [])[a.index]
  }

  agentById(id: string | null | undefined): AgentInternal {
    const a = this.agents.find((x) => x.id === id)
    if (!a) throw new MockHttpError(404, `unknown agent ${id}`)
    return a
  }

  private emit(type: EventType, step: number, agentId: string | null, visibility: Event['visibility'], payload: Record<string, unknown>): Event {
    const e: Event = { seq: this.seq++, step, t: this.clock(), type, agent_id: agentId, visibility, payload }
    if (this.record) this.events.push(e)
    return e
  }

  isCoupled(a: AgentInternal): boolean {
    return this.eff.coupling && a.coupling
  }

  temperature(a: AgentInternal): { tau: number; base: number } {
    const base = this.config.learning?.temperature_base ?? 0.35
    if (!this.isCoupled(a)) return { tau: base, base }
    const c = this.config.coupling!
    const tau = base * (1 + c.uncertainty_to_temperature * (a.state.uncertainty - 0.5)) * (1 + c.activation_to_temperature * (0.5 - a.state.activation))
    return { tau: clamp(tau, c.temperature_min, c.temperature_max), base }
  }

  private learningRate(a: AgentInternal, base: number): number {
    if (!this.isCoupled(a)) return base
    const k = this.config.coupling?.affiliation_to_learning_rate ?? 0.5
    return clamp(base * (1 + k * a.state.affiliation), 0.01, 1)
  }

  receiverScores(a: AgentInternal, x: number[]): number[] {
    return a.W.map((w, k) => w.reduce((s, wi, i) => s + wi * (x[i] ?? 0), 0) + a.b[k])
  }

  symbolVector(symbol: number | null): number[] {
    const v = Array(N_FEATURES).fill(0)
    if (symbol !== null && symbol >= 0 && symbol < N_FEATURES) v[symbol] = 1
    return v
  }

  private generatePhrase(sender: AgentInternal, motifIndex: number, step: number): { phrase: Phrase; modulation: Record<string, number> } {
    const motif = MOCK_MOTIFS[motifIndex]
    const c = this.config.coupling!
    const coupled = this.isCoupled(sender)
    const A = sender.state.activation
    const tempoMult = coupled ? 1 + c.activation_to_tempo * (A - 0.5) : 1
    const velOff = coupled ? c.activation_to_velocity * (A - 0.5) : 0
    const noise = this.config.music?.generation_noise ?? 0.05
    const tempo = clamp(MOCK_TEMPO * tempoMult, this.config.music?.tempo_min ?? 80, this.config.music?.tempo_max ?? 160)
    const notes = motif.notes.map((n) => ({
      ...n,
      velocity: Math.round(clamp(n.velocity + velOff + this.rng.gen.normal() * noise * 40, 1, 127)),
    }))
    const params = this.params(sender)
    const phrase: Phrase = {
      id: `ph-${this.id.slice(-4)}-${step}-${sender.id}`,
      notes,
      tempo_bpm: round(tempo, 2),
      length_beats: this.config.music?.phrase_length_beats ?? 8,
      instrument: params.instrument,
      origin: { kind: 'agent', agent_id: sender.id, source_phrase_id: null, transform: null },
      motif_id: motif.id,
      tags: [],
    }
    return { phrase, modulation: { tempo_multiplier: round(tempoMult), velocity_offset: round(velOff, 2) } }
  }

  private retrieve(a: AgentInternal, x: number[]): RetrievedMemory[] {
    const k = this.config.memory?.retrieval_k ?? 5
    return a.memory
      .map((item) => ({ item, similarity: round(cosine(x, item.feature_vector)) }))
      .sort((p, q) => q.similarity - p.similarity)
      .slice(0, k)
  }

  private updateState(
    a: AgentInternal,
    score: number,
    acousticVec: number[] | null,
    similarity: number | null,
  ): { before: StateVector; after: StateVector; inputs: StateUpdateInputs } {
    const before = { ...a.state }
    const sc = this.config.state!
    const pe = score - before.expected_value
    if (!this.eff.stateUpdates || a.frozen) {
      const inputs: StateUpdateInputs = {
        music_drive: { activation: 0 },
        prediction_error: round(pe),
        outcome: score,
        partner_similarity: similarity === null ? null : round(similarity),
        decay_applied: false,
        frozen: true,
        notes: [a.frozen ? 'frozen by freeze_state intervention: update skipped' : 'state_fixed condition: state held at baseline'],
      }
      return { before, after: { ...before }, inputs }
    }
    const sens = this.params(a).sensitivity ?? 1
    const music = sc.acoustic_activation_enabled && acousticVec ? sc.acoustic_activation_gain * (acousticVec[0] - 0.5 + (acousticVec[12] - 0.5)) : 0
    const peg = sc.prediction_error_gain ?? {}
    const og = sc.outcome_gain ?? {}
    const drive: Record<StateDim, number> = {
      activation: sens * (music + (peg.activation ?? 0) * pe),
      expected_value: sens * ((og.expected_value ?? 1) * pe),
      uncertainty: sens * ((peg.uncertainty ?? 0) * (Math.abs(pe) - before.uncertainty)),
      affiliation: sens * ((og.affiliation ?? 0) * (score - 0.5) + sc.partner_similarity_gain * (similarity ?? 0)),
    }
    const after = { ...before }
    for (const d of DIMS) {
      const inertia = sc.inertia?.[d] ?? 0.7
      const decay = sc.decay?.[d] ?? 0.05
      const maxStep = sc.max_step?.[d] ?? 0.3
      const delta = clamp((1 - inertia) * drive[d] + decay * (a.baseline[d] - before[d]), -maxStep, maxStep)
      const lo = d === 'affiliation' ? -1 : 0
      after[d] = round(clamp(before[d] + delta, lo, 1), 5)
    }
    a.state = after
    const inputs: StateUpdateInputs = {
      music_drive: { activation: round(sens * music) },
      prediction_error: round(pe),
      outcome: score,
      partner_similarity: similarity === null ? null : round(similarity),
      decay_applied: true,
      frozen: false,
      notes: sc.acoustic_activation_enabled ? ['music_drive is HAND-AUTHORED (density/velocity -> activation)'] : ['acoustic drive disabled'],
    }
    return { before, after, inputs }
  }

  private remember(a: AgentInternal, item: MemoryItem) {
    if (this.eff.memoryCap <= 0) return
    a.memory.push(item)
    if (a.memory.length > this.eff.memoryCap) a.memory.splice(0, a.memory.length - this.eff.memoryCap)
  }

  private buildObservation(phrase: Phrase, motifIndex: number | null, sender: string, step: number, rngPerturb: Rng): Observation {
    if (this.eff.channel === 'symbol') {
      return { channel: 'symbol', phrase: null, features: null, symbol_id: motifIndex, sender_id: sender, step }
    }
    const heard = this.eff.perturbation !== 'none' ? transformPhrase(phrase, this.eff.perturbation, rngPerturb) : phrase
    return { channel: 'music', phrase: heard, features: computeSymbolicFeatures(heard), symbol_id: null, sender_id: sender, step }
  }

  /** One episode. Returns the events it produced. */
  stepOnce(human?: HumanPhraseRequest): Event[] {
    const produced: Event[] = []
    const emit = (type: EventType, agentId: string | null, vis: Event['visibility'], payload: Record<string, unknown>) =>
      produced.push(this.emit(type, e, agentId, vis, payload))
    const e = this.step
    const nP = this.patterns.length
    const partial = this.config.task?.partial_credit ?? true
    let receiver: AgentInternal
    let sender: AgentInternal | null
    if (human) {
      receiver = human.receiver_id ? this.agentById(human.receiver_id) : this.agents[(e + 1) % 2]
      sender = null
    } else {
      sender = this.agents[e % 2]
      receiver = this.agents[(e + 1) % 2]
    }
    const senderId = sender ? sender.id : 'human'
    // 1. target
    const target = human && typeof human.target_id === 'number' ? clamp(human.target_id, 0, nP - 1) : this.rng.env.int(nP)
    emit('target_assigned', senderId, 'sender_private', { target_id: target, pattern_name: this.patterns[target].name })
    // 2. sender policy + phrase
    let phrase: Phrase
    let motifIndex: number | null = null
    let senderTrace: PolicyTrace | null = null
    if (human) {
      phrase = { ...human.phrase, origin: { kind: 'human', agent_id: null, source_phrase_id: null, transform: null } }
    } else if (this.pendingReplay) {
      const s = sender!
      phrase = { ...this.pendingReplay, id: `${this.pendingReplay.id}@r${e}`, origin: { kind: 'replay', agent_id: s.id, source_phrase_id: this.pendingReplay.id, transform: null } }
      const mi = MOCK_MOTIFS.findIndex((m) => m.id === phrase.motif_id)
      motifIndex = mi >= 0 ? mi : null
      const { tau, base } = this.temperature(s)
      const q = s.Q[target]
      senderTrace = {
        role: 'sender',
        policy_kind: 'replay',
        scores: q.map((v) => round(v)),
        probabilities: softmax(q, tau).map((v) => round(v)),
        temperature: round(tau),
        temperature_base: base,
        coupling_enabled: this.isCoupled(s),
        chosen: motifIndex ?? -1,
        exploration_draw: null,
        expressive_modulation: {},
        notes: ['replay_motif intervention: sender policy bypassed for this step'],
      }
      this.pendingReplay = null
    } else {
      const s = sender!
      const { tau, base } = this.temperature(s)
      const q = s.Q[target]
      const probs = softmax(q, tau)
      const rngP = s.index === 0 ? this.rng.policyA : this.rng.policyB
      const [m, draw] = rngP.categorical(probs)
      motifIndex = m
      const gen = this.generatePhrase(s, m, e)
      phrase = gen.phrase
      senderTrace = {
        role: 'sender',
        policy_kind: this.params(s).policy_kind === 'model' ? 'model' : 'local',
        scores: q.map((v) => round(v)),
        probabilities: probs.map((v) => round(v)),
        temperature: round(tau),
        temperature_base: base,
        coupling_enabled: this.isCoupled(s),
        chosen: m,
        exploration_draw: round(draw),
        expressive_modulation: gen.modulation,
        notes: [],
      }
    }
    if (this.swap !== 'none') phrase = { ...transformPhrase(phrase, this.swap, this.rng.perturb), motif_id: phrase.motif_id }
    this.phrasesById.set(phrase.id, phrase)
    if (sender && senderTrace) sender.lastTrace = senderTrace
    const sentFeatures = computeSymbolicFeatures(phrase)
    emit('phrase_sent', senderId, 'public', {
      phrase,
      motif_index: motifIndex,
      motif_id: phrase.motif_id ?? null,
      receiver_id: receiver.id,
      trace: senderTrace,
    })
    // 3. channel -> observation (never contains the target)
    const obs = this.buildObservation(phrase, motifIndex, senderId, e, this.rng.perturb)
    receiver.info.push({ step: e, observation: obs })
    emit('phrase_received', receiver.id, 'public', { observation: obs })
    // 4. receiver policy
    const x = obs.channel === 'symbol' ? this.symbolVector(obs.symbol_id ?? null) : obs.features!.vector
    receiver.lastRetrieved = this.retrieve(receiver, x)
    const rScores = this.receiverScores(receiver, x)
    const { tau: rTau, base: rBase } = this.temperature(receiver)
    const rProbs = softmax(rScores, rTau)
    const rRng = receiver.index === 0 ? this.rng.policyA : this.rng.policyB
    const [chosen, rDraw] = rRng.categorical(rProbs)
    const receiverTrace: PolicyTrace = {
      role: 'receiver',
      policy_kind: this.params(receiver).policy_kind === 'model' ? 'model' : 'local',
      scores: rScores.map((v) => round(v)),
      probabilities: rProbs.map((v) => round(v)),
      temperature: round(rTau),
      temperature_base: rBase,
      coupling_enabled: this.isCoupled(receiver),
      chosen,
      exploration_draw: round(rDraw),
      expressive_modulation: {},
      notes: [],
    }
    receiver.lastTrace = receiverTrace
    emit('action_chosen', receiver.id, 'public', { trace: receiverTrace, chosen_pattern_id: chosen })
    // 5. outcome
    const score = round(scorePatterns(this.patterns[chosen], this.patterns[target], partial))
    emit('outcome', null, 'public', { target_id: target, chosen_pattern_id: chosen, score, exact: score >= 0.999 })
    // 6. learning
    if (this.eff.learning) {
      const lrR = this.learningRate(receiver, this.config.learning?.receiver_learning_rate ?? 0.15)
      const l2 = this.config.learning?.receiver_l2 ?? 0.001
      const err = score - rScores[chosen]
      receiver.W[chosen] = receiver.W[chosen].map((w, i) => round(w + lrR * err * (x[i] ?? 0) - l2 * w, 5))
      receiver.b[chosen] = round(receiver.b[chosen] + lrR * err, 5)
      receiver.updates++
      emit('learning_update', receiver.id, 'experimenter', { role: 'receiver', pattern: chosen, error: round(err), learning_rate_effective: round(lrR) })
      if (sender && motifIndex !== null) {
        const lrS = this.learningRate(sender, this.config.learning?.sender_learning_rate ?? 0.2)
        const old = sender.Q[target][motifIndex]
        sender.Q[target][motifIndex] = round(old + lrS * (score - old), 5)
        sender.counts[target][motifIndex]++
        sender.updates++
        emit('learning_update', sender.id, 'experimenter', { role: 'sender', target_id: target, motif_index: motifIndex, error: round(score - old), learning_rate_effective: round(lrS) })
      }
    } else {
      emit('learning_update', null, 'experimenter', { skipped: true, reason: `learning disabled (${this.condition.name})` })
    }
    // 7. state updates (+ memory)
    const acoustic = obs.channel === 'music' ? obs.features!.vector : null
    const parts: { a: AgentInternal; role: 'sender' | 'receiver'; sim: number | null; action: number }[] = [
      { a: receiver, role: 'receiver', sim: receiver.lastOwnVec ? clamp(cosine(x, receiver.lastOwnVec), 0, 1) : null, action: chosen },
    ]
    if (sender) parts.unshift({ a: sender, role: 'sender', sim: null, action: motifIndex ?? -1 })
    for (const { a, role, sim, action } of parts) {
      const { before, after, inputs } = this.updateState(a, score, acoustic, obs.channel === 'music' ? sim : null)
      a.lastInputs = inputs
      a.stateHistory.push({ ...after })
      emit('state_update', a.id, 'public', { before, after, inputs })
      this.remember(a, {
        step: e,
        role,
        partner_id: role === 'sender' ? receiver.id : senderId,
        phrase_id: phrase.id,
        motif_id: phrase.motif_id ?? null,
        feature_vector: role === 'sender' ? sentFeatures.vector : x,
        action,
        score,
        state_before: before,
        state_after: after,
      })
    }
    if (sender) sender.lastOwnVec = sentFeatures.vector
    // 8. complete
    this.scores.push(score)
    this.current = {
      step: e,
      sender_id: senderId,
      receiver_id: receiver.id,
      target_id: target,
      chosen_pattern_id: chosen,
      score,
      phrase,
      observation: obs,
    }
    emit('episode_complete', null, 'public', { episode: this.current })
    this.step++
    return produced
  }

  stepN(n: number): Event[] {
    const out: Event[] = []
    for (let i = 0; i < n; i++) out.push(...this.stepOnce())
    return out
  }

  findPhrase(id: string): Phrase | null {
    return this.phrasesById.get(id) ?? null
  }

  /** agent_id null/undefined = both agents (as the backend). */
  private targets(agentId: string | null | undefined): AgentInternal[] {
    return agentId === null || agentId === undefined ? this.agents : [this.agentById(agentId)]
  }

  intervene(iv: Intervention): Event {
    const next = this.step
    let effect: 'immediately' | 'next step' = 'next step'
    let description = ''
    let value: unknown = iv.value ?? null
    const details: Record<string, unknown> = {}
    const ids = () => this.targets(iv.agent_id).map((a) => a.id).join('+')
    switch (iv.kind) {
      case 'replay_motif': {
        const ph = iv.phrase ?? (iv.phrase_id ? this.findPhrase(iv.phrase_id) : null)
        if (!ph) throw new MockHttpError(404, `phrase ${iv.phrase_id ?? '(none)'} not found`)
        this.pendingReplay = ph
        details.source_phrase_id = ph.id
        description = `phrase ${ph.id} will be sent at step ${next} regardless of sender policy`
        break
      }
      case 'reset_memory':
        for (const a of this.targets(iv.agent_id)) {
          a.memory = []
          Object.assign(a, this.freshLearner())
        }
        effect = 'immediately'
        description = `${ids()}: episodic memory cleared and learned associations re-initialised; state untouched`
        break
      case 'reset_state': {
        const after: Record<string, StateVector> = {}
        for (const a of this.targets(iv.agent_id)) {
          a.state = { ...a.baseline }
          a.stateHistory.push({ ...a.state })
          after[a.id] = { ...a.state }
        }
        details.state_after = after
        effect = 'immediately'
        description = `${ids()}: state := baseline; memory and associations untouched`
        break
      }
      case 'freeze_state':
        value = iv.value === null || iv.value === undefined ? true : Boolean(iv.value)
        for (const a of this.targets(iv.agent_id)) a.frozen = value as boolean
        description = `${ids()}: state updates ${value ? 'frozen' : 'unfrozen'}`
        break
      case 'set_coupling':
        value = iv.value === null || iv.value === undefined ? true : Boolean(iv.value)
        for (const a of this.targets(iv.agent_id)) a.coupling = value as boolean
        description = `${ids()}: state→behaviour coupling ${value ? 'on' : 'off'}`
        break
      case 'swap_feature':
        this.swap = iv.transform ?? 'none'
        value = this.swap
        description = `transform '${this.swap}' applied to every subsequent sent phrase`
        break
      case 'set_param':
        if (!iv.path) throw new MockHttpError(422, 'set_param needs a dotted `path`')
        setPath(this.config as unknown as Record<string, unknown>, iv.path, iv.value)
        this.eff = effectiveSettings(this.condition, this.config)
        details.path = iv.path
        description = `${iv.path} := ${String(iv.value)}`
        break
    }
    // Flat payload, same keys as the backend (+ mock-only `effect` and `description`).
    return this.emit('intervention', next, iv.agent_id ?? null, 'experimenter', {
      kind: iv.kind,
      agent_id: iv.agent_id ?? null,
      value,
      effective_from_step: next,
      ...details,
      effect,
      description,
    })
  }

  metrics(): SessionMetrics {
    const s = this.scores
    return {
      episodes: s.length,
      mean_score: round(mean(s)),
      rolling_score: round(mean(s.slice(-20))),
      success_rate: round(s.length ? s.filter((x) => x >= 0.999).length / s.length : 0),
      score_history: [...s],
    }
  }

  agentSnapshot(a: AgentInternal): AgentSnapshot {
    const p = this.params(a)
    return {
      id: a.id,
      name: p.name,
      color: p.color,
      instrument: p.instrument,
      params: deepClone(p),
      state: { ...a.state },
      baseline: { ...a.baseline },
      memory_size: a.memory.length,
      learner: {
        receiver_weights: a.W.map((r) => [...r]),
        receiver_bias: [...a.b],
        sender_values: a.Q.map((r) => [...r]),
        sender_counts: a.counts.map((r) => [...r]),
        updates: a.updates,
      },
      last_trace: a.lastTrace,
      frozen_state: a.frozen || !this.eff.stateUpdates,
      coupling_enabled: this.isCoupled(a),
      policy_kind: p.policy_kind,
    }
  }

  status(): SessionSnapshot['status'] {
    if (this.step === 0) return 'ready'
    return this.step >= this.config.episodes ? 'finished' : 'running'
  }

  snapshot(mode: SessionSnapshot['mode'] = 'demo'): SessionSnapshot {
    return {
      id: this.id,
      created_at: this.createdAt,
      mode,
      preset: this.preset,
      seed: this.seed,
      condition: deepClone(this.condition),
      config: deepClone(this.config),
      step: this.step,
      status: this.status(),
      agents: this.agents.map((a) => this.agentSnapshot(a)),
      current_episode: this.current,
      metrics: this.metrics(),
      event_count: this.events.length,
      patterns: this.patterns,
    }
  }

  summary(mode: SessionSnapshot['mode'] = 'demo'): SessionSummary {
    return {
      id: this.id,
      created_at: this.createdAt,
      mode,
      preset: this.preset,
      seed: this.seed,
      condition: deepClone(this.condition),
      step: this.step,
      status: this.status(),
    }
  }

  inspect(agentId: string): AgentInspection {
    const a = this.agentById(agentId)
    return {
      agent: this.agentSnapshot(a),
      memory: a.memory.slice(-50),
      retrieved: a.lastRetrieved,
      last_state_inputs: a.lastInputs,
      state_history: a.stateHistory,
      information_received: a.info.slice(-50),
    }
  }

  exportRun(): RunExport {
    return {
      format_version: '1',
      exported_at: this.clock(),
      session: this.summary(),
      config: deepClone(this.config),
      condition: deepClone(this.condition),
      events: deepClone(this.events),
      final_snapshot: this.snapshot(),
    }
  }
}

/** Set a dotted config path (e.g. `state.decay.activation`, `agents.0.sensitivity`). */
export function setPath(obj: Record<string, unknown>, path: string, value: unknown): void {
  const keys = path.split('.')
  let cur: Record<string, unknown> | unknown[] = obj
  for (let i = 0; i < keys.length - 1; i++) {
    const k = keys[i]
    const next: unknown = Array.isArray(cur) ? cur[Number(k)] : cur[k]
    if (typeof next !== 'object' || next === null) throw new MockHttpError(422, `invalid path ${path}`)
    cur = next as Record<string, unknown>
  }
  const last = keys[keys.length - 1]
  if (Array.isArray(cur)) cur[Number(last)] = value
  else cur[last] = value
}
