/**
 * Global lab store: a plain reducer (unit-tested in store.test.ts) used via
 * React context (LabContext.tsx). Fast-changing audio playhead state lives
 * outside the store (audio/playhead.ts).
 */
import type {
  ExperimentConfig,
  Event,
  HealthResponse,
  Phrase,
  PresetInfo,
  PresetResult,
  SessionSnapshot,
  StepResult,
  TimingPattern,
} from '../api/types'
import type { DataSourceKind } from '../api/client'
import {
  foldEvent,
  readIntervention,
  statePointsFromEvents,
  type EpisodeRecord,
  type InterventionRecord,
  type StatePoint,
} from '../api/events'
import type { Speed } from '../audio/synthSpec'

export type BadgeKind = 'live' | 'replay' | 'demo'

export interface LogEntry {
  id: number
  at: string
  kind: 'intervention' | 'session' | 'info' | 'error' | 'audio'
  text: string
  step?: number
  effect?: string
}

export interface LabState {
  source: { kind: DataSourceKind | null; reason: string; health: HealthResponse | null }
  presets: PresetInfo[]
  patterns: TimingPattern[]
  motifs: Phrase[]
  defaultConfig: ExperimentConfig | null
  session: SessionSnapshot | null
  episodes: EpisodeRecord[]
  interventions: InterventionRecord[]
  statePoints: Record<string, StatePoint[]>
  /** x offset added to an event's step to get "episodes completed" (1 for 0-based steps). */
  stepOffset: number
  selectedStep: number | null
  running: boolean
  speed: Speed
  episodesTarget: number
  log: LogEntry[]
  busy: boolean
  error: string | null
  inspector: { open: boolean; agentId: string | null }
  experimentsOpen: boolean
  startMenuOpen: boolean
  presetResult: PresetResult | null
  overlayPattern: boolean
}

export const initialState: LabState = {
  source: { kind: null, reason: '', health: null },
  presets: [],
  patterns: [],
  motifs: [],
  defaultConfig: null,
  session: null,
  episodes: [],
  interventions: [],
  statePoints: {},
  stepOffset: 1,
  selectedStep: null,
  running: false,
  speed: '1x',
  episodesTarget: 120,
  log: [],
  busy: false,
  error: null,
  inspector: { open: false, agentId: null },
  experimentsOpen: false,
  startMenuOpen: false,
  presetResult: null,
  overlayPattern: true,
}

export type Action =
  | { type: 'source'; kind: DataSourceKind; reason: string; health: HealthResponse | null }
  | { type: 'catalog'; presets?: PresetInfo[]; patterns?: TimingPattern[]; motifs?: Phrase[]; defaultConfig?: ExperimentConfig }
  | { type: 'sessionLoaded'; snapshot: SessionSnapshot; events?: Event[] }
  | { type: 'stepResult'; result: StepResult }
  | { type: 'select'; step: number | null }
  | { type: 'running'; running: boolean }
  | { type: 'speed'; speed: Speed }
  | { type: 'target'; episodes: number }
  | { type: 'log'; entry: Omit<LogEntry, 'id' | 'at'> & { at?: string } }
  | { type: 'busy'; busy: boolean }
  | { type: 'error'; message: string | null }
  | { type: 'inspector'; open: boolean; agentId?: string | null }
  | { type: 'experiments'; open: boolean }
  | { type: 'startMenu'; open: boolean }
  | { type: 'presetResult'; result: PresetResult | null }
  | { type: 'overlay'; on: boolean }

export function badgeFor(state: Pick<LabState, 'source' | 'session'>): BadgeKind[] {
  const out: BadgeKind[] = []
  if (state.session?.mode === 'replay') out.push('replay')
  if (state.source.kind === 'mock') out.push('demo')
  else if (state.session?.mode !== 'replay') out.push('live')
  return out
}

function foldAll(prev: EpisodeRecord[], events: readonly Event[]): EpisodeRecord[] {
  if (!events.length) return prev
  const map = new Map<number, EpisodeRecord>()
  const touched = new Set<number>()
  for (const e of events) if (e.type !== 'intervention' && e.type !== 'session_created' && e.type !== 'session_reset') touched.add(e.step)
  for (const ep of prev) map.set(ep.step, touched.has(ep.step) ? { ...ep, stateChanges: { ...ep.stateChanges }, modelCalls: [...ep.modelCalls], narratives: [...ep.narratives] } : ep)
  for (const e of events) foldEvent(map, e)
  return [...map.values()].filter((ep) => ep.complete || ep.phrase || ep.score !== null).sort((a, b) => a.step - b.step)
}

function initialPoints(s: SessionSnapshot): Record<string, StatePoint[]> {
  const out: Record<string, StatePoint[]> = {}
  for (const a of s.agents) out[a.id] = [{ step: s.step, state: { ...a.state } }]
  return out
}

function mergePoints(
  prev: Record<string, StatePoint[]>,
  events: readonly Event[],
  snapshot: SessionSnapshot,
  offset: number,
): Record<string, StatePoint[]> {
  const fromEvents = statePointsFromEvents(events)
  const out: Record<string, StatePoint[]> = { ...prev }
  for (const a of snapshot.agents) {
    const add = (fromEvents[a.id] ?? []).map((p) => ({ step: p.step + offset, state: p.state }))
    // immediate interventions (reset_state) change state without a state_update event: record the snapshot value
    const last = add.length ? add[add.length - 1] : null
    const pts = [...(out[a.id] ?? [])]
    for (const p of add) {
      if (pts.length && pts[pts.length - 1].step === p.step) pts[pts.length - 1] = p
      else pts.push(p)
    }
    if (!last) {
      const x = snapshot.step
      const tail = pts[pts.length - 1]
      const same = tail && JSON.stringify(tail.state) === JSON.stringify(a.state)
      if (!same) {
        if (tail && tail.step === x) pts[pts.length - 1] = { step: x, state: { ...a.state } }
        else pts.push({ step: x, state: { ...a.state } })
      }
    }
    out[a.id] = pts
  }
  return out
}

let logSeq = 0

export function reducer(state: LabState, action: Action): LabState {
  switch (action.type) {
    case 'source':
      return { ...state, source: { kind: action.kind, reason: action.reason, health: action.health } }
    case 'catalog':
      return {
        ...state,
        presets: action.presets ?? state.presets,
        patterns: action.patterns ?? state.patterns,
        motifs: action.motifs ?? state.motifs,
        defaultConfig: action.defaultConfig ?? state.defaultConfig,
      }
    case 'sessionLoaded': {
      const events = action.events ?? []
      const s = action.snapshot
      const episodes = foldAll([], events)
      const interventions = events.map(readIntervention).filter((x): x is InterventionRecord => x !== null)
      const firstStep = episodes.length ? episodes[0].step : null
      const offset = firstStep === null ? 1 : firstStep === 0 ? 1 : 0
      const base = initialPoints({ ...s, step: 0, agents: s.agents.map((a) => ({ ...a, state: a.baseline })) })
      const statePoints = events.length ? mergePoints(base, events, s, offset) : initialPoints(s)
      return {
        ...state,
        session: s,
        episodes,
        interventions,
        statePoints,
        stepOffset: offset,
        selectedStep: null,
        running: false,
        episodesTarget: Math.max(s.config.episodes ?? state.episodesTarget, s.step),
        patterns: s.patterns?.length ? s.patterns : state.patterns,
        error: null,
      }
    }
    case 'stepResult': {
      const { events, snapshot } = action.result
      const prevStep = state.session?.step ?? 0
      let offset = state.stepOffset
      if (state.episodes.length === 0) {
        const first = events.find((e) => e.type !== 'intervention' && e.type !== 'session_created' && e.type !== 'session_reset')
        if (first) offset = first.step === prevStep ? 1 : 0
      }
      const interventions = [
        ...state.interventions,
        ...events.map(readIntervention).filter((x): x is InterventionRecord => x !== null),
      ]
      return {
        ...state,
        session: snapshot,
        episodes: foldAll(state.episodes, events),
        interventions,
        statePoints: mergePoints(state.statePoints, events, snapshot, offset),
        stepOffset: offset,
        patterns: snapshot.patterns?.length ? snapshot.patterns : state.patterns,
      }
    }
    case 'select':
      return { ...state, selectedStep: action.step }
    case 'running':
      return { ...state, running: action.running }
    case 'speed':
      return { ...state, speed: action.speed }
    case 'target':
      return { ...state, episodesTarget: Math.max(1, Math.floor(action.episodes)) }
    case 'log': {
      const entry: LogEntry = { id: ++logSeq, at: action.entry.at ?? new Date().toISOString(), ...action.entry }
      return { ...state, log: [...state.log.slice(-199), entry] }
    }
    case 'busy':
      return { ...state, busy: action.busy }
    case 'error':
      return { ...state, error: action.message }
    case 'inspector':
      return { ...state, inspector: { open: action.open, agentId: action.agentId ?? state.inspector.agentId } }
    case 'experiments':
      return { ...state, experimentsOpen: action.open }
    case 'startMenu':
      return { ...state, startMenuOpen: action.open }
    case 'presetResult':
      return { ...state, presetResult: action.result }
    case 'overlay':
      return { ...state, overlayPattern: action.on }
  }
}

/** The episode the centre panels should show: the selected one, else the latest. */
export function focusedEpisode(state: Pick<LabState, 'episodes' | 'selectedStep'>): EpisodeRecord | null {
  if (!state.episodes.length) return null
  if (state.selectedStep !== null) return state.episodes.find((e) => e.step === state.selectedStep) ?? null
  return state.episodes[state.episodes.length - 1]
}
