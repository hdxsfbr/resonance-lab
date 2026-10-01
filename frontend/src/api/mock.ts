/**
 * In-browser MOCK implementation of the LabApi interface. Selected automatically
 * when /api/health fails or returns 501, or forced with `?mock=1`.
 * Everything it returns is DEMO DATA and the UI labels it so.
 */
import { ApiError, type LabApi } from './client'
import type {
  AgentInspection,
  CreateSessionRequest,
  Event,
  ExperimentConfig,
  ExperimentRequest,
  ExperimentResult,
  HealthResponse,
  HumanPhraseRequest,
  Intervention,
  ModelStatus,
  Phrase,
  PhraseFeatures,
  PresetInfo,
  PresetResult,
  RunExport,
  RunPresetRequest,
  SavedMotif,
  SessionSnapshot,
  SessionSummary,
  StepRequest,
  StepResult,
  TimingPattern,
  TransformRequest,
} from './types'
import { mockFeatures as computeSymbolicFeatures } from './mock/normalise'
import { MOCK_MOTIFS, MOCK_PATTERNS, MOCK_PRESETS, mockDefaultConfig } from './mock/data'
import { MockHttpError, MockSim } from './mock/sim'
import { MockReplay, basePhrase, experimentCsv, runMockExperiment, runMockPreset } from './mock/analysis'
import { Rng } from './mock/rng'
import { transformPhrase } from './mock/transforms'

export const MOCK_VERSION = 'mock-0.1.0'

const MOCK_MODEL_STATUS: ModelStatus = {
  provider: 'none',
  model_id: '',
  available: false,
  tested_in_this_environment: false,
  accepts_audio: false,
  input_modality: 'none',
  detail: 'DEMO DATA (mock): no model provider; all policies are local.',
  calls_made: 0,
  call_budget: 0,
}

export interface MockOptions {
  /** Deterministic clock for tests. */
  clock?: () => string
  /** Simulated network latency in ms (default 0). */
  latencyMs?: number
}

function clone<T>(x: T): T {
  return JSON.parse(JSON.stringify(x)) as T
}

export class MockLabApi implements LabApi {
  readonly kind = 'mock' as const
  private sims = new Map<string, MockSim>()
  private replays = new Map<string, MockReplay>()
  private experiments = new Map<string, ExperimentResult>()
  private saved: SavedMotif[] = []
  private counter = 0
  private readonly clock: () => string
  private readonly latency: number

  constructor(opts: MockOptions = {}) {
    this.clock = opts.clock ?? (() => new Date().toISOString())
    this.latency = opts.latencyMs ?? 0
  }

  private nextId(prefix: string): string {
    this.counter += 1
    return `${prefix}-${String(this.counter).padStart(4, '0')}`
  }

  /** Run `fn` asynchronously; map mock errors to ApiError like the HTTP client. Results are deep-cloned. */
  private async run<T>(fn: () => T): Promise<T> {
    if (this.latency) await new Promise((r) => setTimeout(r, this.latency))
    try {
      return clone(fn())
    } catch (e) {
      if (e instanceof MockHttpError) throw new ApiError(e.status, `[mock] ${e.message}`)
      if (e instanceof ApiError) throw e
      throw new ApiError(500, `[mock] ${(e as Error).message}`)
    }
  }

  private sim(id: string): MockSim {
    const s = this.sims.get(id)
    if (!s) throw new MockHttpError(404, `session ${id} not found`)
    return s
  }

  health(): Promise<HealthResponse> {
    return this.run(() => ({
      status: 'ok' as const,
      version: MOCK_VERSION,
      mode_note: 'Engineered agent simulation. State labels are operational definitions, not claims about experience.',
      model: MOCK_MODEL_STATUS,
    }))
  }
  defaultConfig(): Promise<ExperimentConfig> {
    return this.run(() => mockDefaultConfig())
  }
  patterns(): Promise<TimingPattern[]> {
    return this.run(() => MOCK_PATTERNS)
  }
  motifs(): Promise<Phrase[]> {
    return this.run(() => MOCK_MOTIFS.map((_, i) => basePhrase(i)))
  }
  presets(): Promise<PresetInfo[]> {
    return this.run(() => MOCK_PRESETS)
  }
  runPreset(name: string, req: RunPresetRequest): Promise<PresetResult> {
    return this.run(() => {
      if (!MOCK_PRESETS.some((p) => p.name === name)) throw new MockHttpError(404, `preset ${name} not found`)
      return runMockPreset(name, req.seed ?? 7, mockDefaultConfig(), () => this.nextId('mock'), this.clock, (s) => this.sims.set(s.id, s))
    })
  }
  createSession(req: CreateSessionRequest): Promise<SessionSnapshot> {
    return this.run(() => {
      const preset = req.preset ? MOCK_PRESETS.find((p) => p.name === req.preset) : undefined
      if (req.preset && !preset) throw new MockHttpError(404, `preset ${req.preset} not found`)
      const config = req.config ?? mockDefaultConfig()
      if (preset && !req.config) config.episodes = preset.default_episodes
      const condition = req.condition ?? preset?.condition ?? { name: 'full', perturbation: 'none', description: '' }
      const sim = new MockSim({ id: this.nextId('mock'), seed: config.seed, config, condition, preset: req.preset ?? null, clock: this.clock })
      this.sims.set(sim.id, sim)
      return sim.snapshot()
    })
  }
  listSessions(): Promise<SessionSummary[]> {
    return this.run(() => [...[...this.sims.values()].map((s) => s.summary()), ...[...this.replays.values()].map((r) => ({ ...r.run.session, id: r.id, mode: 'replay' as const }))])
  }
  getSession(id: string): Promise<SessionSnapshot> {
    return this.run(() => {
      const r = this.replays.get(id)
      if (r) return r.snapshot()
      return this.sim(id).snapshot()
    })
  }
  step(id: string, req: StepRequest): Promise<StepResult> {
    return this.run(() => {
      if (this.replays.has(id)) throw new MockHttpError(409, 'session is a replay; use /replay/step')
      const s = this.sim(id)
      const events = s.stepN(Math.max(1, Math.min(500, req.n ?? 1)))
      return { events, snapshot: s.snapshot() }
    })
  }
  reset(id: string): Promise<SessionSnapshot> {
    return this.run(() => {
      const s = this.sim(id)
      s.reset()
      return s.snapshot()
    })
  }
  intervene(id: string, req: Intervention): Promise<StepResult> {
    return this.run(() => {
      if (this.replays.has(id)) throw new MockHttpError(409, 'interventions are not available in replay mode')
      const s = this.sim(id)
      const ev = s.intervene(req)
      return { events: [ev], snapshot: s.snapshot() }
    })
  }
  humanPhrase(id: string, req: HumanPhraseRequest): Promise<StepResult> {
    return this.run(() => {
      const s = this.sim(id)
      const events = s.stepOnce(req)
      return { events, snapshot: s.snapshot() }
    })
  }
  events(id: string, q: { from_seq?: number; limit?: number } = {}): Promise<Event[]> {
    return this.run(() => {
      const r = this.replays.get(id)
      const all = r ? r.emitted : this.sim(id).events
      const from = q.from_seq ?? 0
      return all.filter((e) => e.seq >= from).slice(0, q.limit ?? 500)
    })
  }
  inspect(id: string, agentId: string): Promise<AgentInspection> {
    return this.run(() => {
      const r = this.replays.get(id)
      if (r) {
        const snap = r.snapshot()
        const agent = snap.agents.find((a) => a.id === agentId)
        if (!agent) throw new MockHttpError(404, `agent ${agentId} not found`)
        const hist = r.emitted
          .filter((e) => e.type === 'state_update' && e.agent_id === agentId)
          .map((e) => (e.payload as { after: AgentInspection['state_history'][number] }).after)
        const info = r.emitted
          .filter((e) => e.type === 'phrase_received' && e.agent_id === agentId)
          .map((e) => ({ step: e.step, observation: (e.payload as { observation: unknown }).observation }))
        return { agent, memory: [], retrieved: [], last_state_inputs: null, state_history: [agent.baseline, ...hist], information_received: info.slice(-50) }
      }
      return this.sim(id).inspect(agentId)
    })
  }
  exportRun(id: string): Promise<RunExport> {
    return this.run(() => {
      const r = this.replays.get(id)
      if (r) return r.run
      return this.sim(id).exportRun()
    })
  }
  importRun(run: RunExport): Promise<SessionSnapshot> {
    return this.run(() => {
      if (!run || !Array.isArray(run.events) || !run.final_snapshot) throw new MockHttpError(422, 'not a RunExport document')
      const r = new MockReplay(this.nextId('replay'), clone(run))
      this.replays.set(r.id, r)
      return r.snapshot()
    })
  }
  replayStep(id: string, req: StepRequest): Promise<StepResult> {
    return this.run(() => {
      const r = this.replays.get(id)
      if (!r) throw new MockHttpError(404, `replay session ${id} not found`)
      const events = r.replayStep(Math.max(1, req.n ?? 1))
      return { events, snapshot: r.snapshot() }
    })
  }
  phraseFeatures(phrase: Phrase): Promise<PhraseFeatures> {
    return this.run(() => ({ phrase_id: phrase.id, symbolic: computeSymbolicFeatures(phrase), audio: null }))
  }
  async renderPhrase(phrase: Phrase): Promise<Blob> {
    const { renderPhraseWav } = await import('../audio/wav')
    return renderPhraseWav(phrase)
  }
  transformPhrase(req: TransformRequest): Promise<Phrase> {
    return this.run(() => transformPhrase(req.phrase, req.transform, new Rng(17)))
  }
  savedMotifs(): Promise<SavedMotif[]> {
    return this.run(() => this.saved)
  }
  saveMotif(m: SavedMotif): Promise<SavedMotif[]> {
    return this.run(() => {
      const withFeatures: SavedMotif = { ...m, features: m.features ?? computeSymbolicFeatures(m.phrase) }
      this.saved = [...this.saved.filter((x) => x.name !== m.name), withFeatures]
      return this.saved
    })
  }
  runExperiment(req: ExperimentRequest): Promise<ExperimentResult> {
    return this.run(() => {
      if (!req.conditions.length || !req.seeds.length) throw new MockHttpError(422, 'conditions and seeds must be non-empty')
      const r = runMockExperiment(req, mockDefaultConfig(), this.nextId('exp'), this.clock)
      this.experiments.set(r.id, r)
      return r
    })
  }
  listExperiments(): Promise<ExperimentResult[]> {
    return this.run(() => [...this.experiments.values()])
  }
  getExperiment(id: string): Promise<ExperimentResult> {
    return this.run(() => {
      const r = this.experiments.get(id)
      if (!r) throw new MockHttpError(404, `experiment ${id} not found`)
      return r
    })
  }
  experimentCsv(id: string): Promise<string> {
    return this.run(() => {
      const r = this.experiments.get(id)
      if (!r) throw new MockHttpError(404, `experiment ${id} not found`)
      return experimentCsv(r)
    })
  }
  modelStatus(): Promise<ModelStatus> {
    return this.run(() => MOCK_MODEL_STATUS)
  }
}

export function createMockClient(opts?: MockOptions): LabApi {
  return new MockLabApi(opts)
}
