/**
 * Typed HTTP client for the Resonance Lab API (contract: docs/CONTRACTS.md,
 * types: ./schema.d.ts generated from the backend OpenAPI document).
 *
 * `LabApi` is the interface the UI depends on. Two implementations exist:
 *   - HttpLabApi (this file): talks to the real backend under /api
 *   - MockLabApi (./mock.ts): in-browser demo implementation of the same interface
 */
import type { paths } from './schema'
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

export type DataSourceKind = 'live' | 'mock'

export interface LabApi {
  readonly kind: DataSourceKind
  health(): Promise<HealthResponse>
  defaultConfig(): Promise<ExperimentConfig>
  patterns(): Promise<TimingPattern[]>
  motifs(): Promise<Phrase[]>
  presets(): Promise<PresetInfo[]>
  runPreset(name: string, req: RunPresetRequest): Promise<PresetResult>
  createSession(req: CreateSessionRequest): Promise<SessionSnapshot>
  listSessions(): Promise<SessionSummary[]>
  getSession(id: string): Promise<SessionSnapshot>
  step(id: string, req: StepRequest): Promise<StepResult>
  reset(id: string): Promise<SessionSnapshot>
  intervene(id: string, req: Intervention): Promise<StepResult>
  humanPhrase(id: string, req: HumanPhraseRequest): Promise<StepResult>
  events(id: string, q?: { from_seq?: number; limit?: number }): Promise<Event[]>
  inspect(id: string, agentId: string): Promise<AgentInspection>
  exportRun(id: string): Promise<RunExport>
  importRun(run: RunExport): Promise<SessionSnapshot>
  replayStep(id: string, req: StepRequest): Promise<StepResult>
  phraseFeatures(phrase: Phrase): Promise<PhraseFeatures>
  /** WAV bytes. Live: server numpy synth. Mock: browser OfflineAudioContext render. */
  renderPhrase(phrase: Phrase): Promise<Blob>
  transformPhrase(req: TransformRequest): Promise<Phrase>
  savedMotifs(): Promise<SavedMotif[]>
  saveMotif(m: SavedMotif): Promise<SavedMotif[]>
  runExperiment(req: ExperimentRequest): Promise<ExperimentResult>
  listExperiments(): Promise<ExperimentResult[]>
  getExperiment(id: string): Promise<ExperimentResult>
  experimentCsv(id: string): Promise<string>
  modelStatus(): Promise<ModelStatus>
}

export class ApiError extends Error {
  readonly status: number
  readonly detail: unknown
  constructor(status: number, message: string, detail?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

// ---------------------------------------------------------------------------
// Compile-time binding of each call to the generated `paths` table.
// ---------------------------------------------------------------------------

type Method = 'get' | 'post'
type Op<P extends keyof paths, M extends Method> = NonNullable<paths[P][M]>
type JsonOk<O> = O extends { responses: { 200: { content: { 'application/json': infer R } } } } ? R : never
type JsonBody<O> = O extends { requestBody: { content: { 'application/json': infer B } } } ? B : undefined

function fillPath(template: string, params: Record<string, string>): string {
  return template.replace(/\{(\w+)\}/g, (_m, key: string) => {
    const v = params[key]
    if (v === undefined) throw new Error(`missing path param ${key}`)
    return encodeURIComponent(v)
  })
}

export class HttpLabApi implements LabApi {
  readonly kind = 'live' as const
  private readonly base: string

  constructor(base = '') {
    this.base = base
  }

  private async raw(method: Method, url: string, body?: unknown, accept = 'application/json'): Promise<Response> {
    const init: RequestInit = { method: method.toUpperCase(), headers: { Accept: accept } }
    if (body !== undefined) {
      init.body = JSON.stringify(body)
      ;(init.headers as Record<string, string>)['Content-Type'] = 'application/json'
    }
    const res = await fetch(this.base + url, init)
    if (!res.ok) {
      let detail: unknown = undefined
      try {
        detail = await res.json()
      } catch {
        /* non-JSON error body */
      }
      const msg =
        detail && typeof detail === 'object' && 'detail' in detail
          ? `${res.status}: ${JSON.stringify((detail as { detail: unknown }).detail)}`
          : `${res.status} ${res.statusText}`
      throw new ApiError(res.status, `${method.toUpperCase()} ${url} failed — ${msg}`, detail)
    }
    return res
  }

  private async call<P extends keyof paths, M extends Method>(
    path: P,
    method: M,
    opts: { params?: Record<string, string>; query?: Record<string, number | string | undefined>; body?: JsonBody<Op<P, M>> } = {},
  ): Promise<JsonOk<Op<P, M>>> {
    let url = fillPath(path as string, opts.params ?? {})
    if (opts.query) {
      const qs = Object.entries(opts.query)
        .filter(([, v]) => v !== undefined)
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
        .join('&')
      if (qs) url += `?${qs}`
    }
    const res = await this.raw(method, url, opts.body)
    return (await res.json()) as JsonOk<Op<P, M>>
  }

  health() {
    return this.call('/api/health', 'get')
  }
  defaultConfig() {
    return this.call('/api/config/default', 'get')
  }
  patterns() {
    return this.call('/api/patterns', 'get')
  }
  motifs() {
    return this.call('/api/motifs', 'get')
  }
  presets() {
    return this.call('/api/presets', 'get')
  }
  runPreset(name: string, req: RunPresetRequest) {
    return this.call('/api/presets/{name}/run', 'post', { params: { name }, body: req })
  }
  createSession(req: CreateSessionRequest) {
    return this.call('/api/sessions', 'post', { body: req })
  }
  listSessions() {
    return this.call('/api/sessions', 'get')
  }
  getSession(id: string) {
    return this.call('/api/sessions/{session_id}', 'get', { params: { session_id: id } })
  }
  step(id: string, req: StepRequest) {
    return this.call('/api/sessions/{session_id}/step', 'post', { params: { session_id: id }, body: req })
  }
  reset(id: string) {
    return this.call('/api/sessions/{session_id}/reset', 'post', { params: { session_id: id } })
  }
  intervene(id: string, req: Intervention) {
    return this.call('/api/sessions/{session_id}/intervene', 'post', { params: { session_id: id }, body: req })
  }
  humanPhrase(id: string, req: HumanPhraseRequest) {
    return this.call('/api/sessions/{session_id}/human_phrase', 'post', { params: { session_id: id }, body: req })
  }
  events(id: string, q: { from_seq?: number; limit?: number } = {}) {
    return this.call('/api/sessions/{session_id}/events', 'get', { params: { session_id: id }, query: q })
  }
  inspect(id: string, agentId: string) {
    return this.call('/api/sessions/{session_id}/agents/{agent_id}/inspect', 'get', {
      params: { session_id: id, agent_id: agentId },
    })
  }
  exportRun(id: string) {
    return this.call('/api/sessions/{session_id}/export', 'get', { params: { session_id: id } })
  }
  importRun(run: RunExport) {
    return this.call('/api/sessions/import', 'post', { body: run })
  }
  replayStep(id: string, req: StepRequest) {
    return this.call('/api/sessions/{session_id}/replay/step', 'post', { params: { session_id: id }, body: req })
  }
  phraseFeatures(phrase: Phrase) {
    return this.call('/api/phrases/features', 'post', { body: phrase })
  }
  async renderPhrase(phrase: Phrase): Promise<Blob> {
    const res = await this.raw('post', '/api/phrases/render', phrase, 'audio/wav')
    return await res.blob()
  }
  transformPhrase(req: TransformRequest) {
    return this.call('/api/phrases/transform', 'post', { body: req })
  }
  savedMotifs() {
    return this.call('/api/motifs/saved', 'get')
  }
  saveMotif(m: SavedMotif) {
    return this.call('/api/motifs/saved', 'post', { body: m })
  }
  runExperiment(req: ExperimentRequest) {
    return this.call('/api/experiments', 'post', { body: req })
  }
  listExperiments() {
    return this.call('/api/experiments', 'get')
  }
  getExperiment(id: string) {
    return this.call('/api/experiments/{experiment_id}', 'get', { params: { experiment_id: id } })
  }
  async experimentCsv(id: string): Promise<string> {
    const res = await this.raw('get', `/api/experiments/${encodeURIComponent(id)}/csv`, undefined, 'text/csv')
    return await res.text()
  }
  modelStatus() {
    return this.call('/api/model/status', 'get')
  }
}

export interface ClientChoice {
  api: LabApi
  reason: string
  health: HealthResponse | null
}

/**
 * Select the data source. Mock is used when `?mock=1` is present, or when
 * /api/health fails or returns non-2xx (e.g. 501 while the backend is a stub).
 */
export async function chooseClient(
  search: string,
  makeMock: () => LabApi | Promise<LabApi>,
  live: LabApi = new HttpLabApi(),
  timeoutMs = 2500,
): Promise<ClientChoice> {
  const params = new URLSearchParams(search)
  if (params.get('mock') === '1') {
    const api = await makeMock()
    return { api, reason: 'forced by ?mock=1', health: await api.health() }
  }
  try {
    const health = await Promise.race([
      live.health(),
      new Promise<never>((_, rej) => setTimeout(() => rej(new Error('health check timed out')), timeoutMs)),
    ])
    if (health && health.status === 'ok') return { api: live, reason: 'GET /api/health ok', health }
    throw new Error('unexpected health payload')
  } catch (e) {
    const api = await makeMock()
    const why = e instanceof ApiError ? `GET /api/health returned ${e.status}` : `GET /api/health failed (${(e as Error).message})`
    return { api, reason: `${why}; using in-browser mock`, health: await api.health() }
  }
}
