import { createContext, useCallback, useContext, useEffect, useMemo, useReducer, useRef, useState, type ReactNode } from 'react'
import { chooseClient, type LabApi } from '../api/client'
import type { EpisodeRecord } from '../api/events'
import type { ConditionSpec, ExperimentConfig, Intervention, Phrase, RunExport, StepResult } from '../api/types'
import { audioEngine, type PlayHandle } from '../audio/engine'
import { pacingSeconds, stepsPerRequest } from '../audio/synthSpec'
import { downloadBlob, downloadJson, safeName } from '../lib/download'
import { errorMessage } from '../lib/format'
import { INTERVENTION_DEFS } from '../lib/definitions'
import { initialState, reducer, type Action, type LabState } from './store'

export interface LabActions {
  startSession(opts: { preset?: string | null; condition?: ConditionSpec; config?: ExperimentConfig | null }): Promise<void>
  runPresetComparison(name: string, seed?: number | null): Promise<void>
  stepOnce(): Promise<void>
  play(): void
  pause(): void
  resetSession(): Promise<void>
  intervene(iv: Intervention): Promise<boolean>
  replayMotif(ep: EpisodeRecord): Promise<void>
  playPhraseAudio(phrase: Phrase): PlayHandle
  sendHuman(phrase: Phrase, targetId: number | null, receiverId: string | null): Promise<void>
  exportRun(): Promise<void>
  importRun(file: File): Promise<void>
  downloadWav(phrase: Phrase): Promise<void>
  downloadPhraseJson(phrase: Phrase): void
}

interface LabContextValue {
  state: LabState
  dispatch: (a: Action) => void
  api: LabApi | null
  actions: LabActions
}

const Ctx = createContext<LabContextValue | null>(null)

// oxlint-disable-next-line react/only-export-components
export function useLab(): LabContextValue {
  const v = useContext(Ctx)
  if (!v) throw new Error('useLab outside LabProvider')
  return v
}

function lastSentPhrase(res: StepResult): Phrase | null {
  for (let i = res.events.length - 1; i >= 0; i--) {
    const e = res.events[i]
    if (e.type === 'phrase_sent') {
      const p = (e.payload as { phrase?: Phrase }).phrase
      if (p && Array.isArray(p.notes)) return p
    }
  }
  return res.snapshot.current_episode?.phrase ?? null
}

export function LabProvider({ children, apiOverride }: { children: ReactNode; apiOverride?: LabApi }) {
  const [state, dispatch] = useReducer(reducer, initialState)
  const [api, setApi] = useState<LabApi | null>(null)
  const stateRef = useRef(state)
  const apiRef = useRef<LabApi | null>(null)
  const runningRef = useRef(false)
  const wakeRef = useRef<(() => void) | null>(null)
  const handleRef = useRef<PlayHandle | null>(null)
  const inited = useRef(false)

  useEffect(() => {
    stateRef.current = state
  }, [state])

  const log = useCallback((kind: LabState['log'][number]['kind'], text: string, extra: { step?: number; effect?: string } = {}) => {
    dispatch({ type: 'log', entry: { kind, text, ...extra } })
  }, [])

  const fail = useCallback(
    (e: unknown, context: string) => {
      const msg = `${context}: ${errorMessage(e)}`
      dispatch({ type: 'error', message: msg })
      log('error', msg)
    },
    [log],
  )

  // ---- data source selection + catalog --------------------------------------------------
  useEffect(() => {
    if (inited.current) return
    inited.current = true
    void (async () => {
      const choice = apiOverride
        ? { api: apiOverride, reason: 'injected', health: await apiOverride.health().catch(() => null) }
        : await chooseClient(window.location.search, () => import('../api/mock').then((m) => m.createMockClient()))
      apiRef.current = choice.api
      setApi(choice.api)
      dispatch({ type: 'source', kind: choice.api.kind, reason: choice.reason, health: choice.health })
      log('info', `Data source: ${choice.api.kind === 'mock' ? 'DEMO DATA (mock)' : 'live API'} — ${choice.reason}`)
      const a = choice.api
      const [presets, patterns, motifs, cfg] = await Promise.allSettled([a.presets(), a.patterns(), a.motifs(), a.defaultConfig()])
      dispatch({
        type: 'catalog',
        presets: presets.status === 'fulfilled' ? presets.value : undefined,
        patterns: patterns.status === 'fulfilled' ? patterns.value : undefined,
        motifs: motifs.status === 'fulfilled' ? motifs.value : undefined,
        defaultConfig: cfg.status === 'fulfilled' ? cfg.value : undefined,
      })
      for (const [name, r] of [
        ['presets', presets],
        ['patterns', patterns],
        ['motifs', motifs],
        ['config', cfg],
      ] as const)
        if (r.status === 'rejected') log('error', `GET ${name} failed: ${errorMessage(r.reason)}`)
      dispatch({ type: 'startMenu', open: true })
    })()
  }, [apiOverride, log])

  const need = useCallback((): LabApi => {
    const a = apiRef.current
    if (!a) throw new Error('data source not ready')
    return a
  }, [])

  const stopLoop = useCallback(() => {
    runningRef.current = false
    wakeRef.current?.()
  }, [])

  const loadSession = useCallback(
    async (snapshot: LabState['session']) => {
      if (!snapshot) return
      let events: import('../api/types').Event[] = []
      if (snapshot.step > 0 && snapshot.mode !== 'replay') {
        try {
          let from = 0
          for (let page = 0; page < 40; page++) {
            const batch = await need().events(snapshot.id, { from_seq: from, limit: 500 })
            events = events.concat(batch)
            if (batch.length < 500) break
            from = batch[batch.length - 1].seq + 1
          }
        } catch (e) {
          log('error', `could not load event history: ${errorMessage(e)}`)
        }
      }
      dispatch({ type: 'sessionLoaded', snapshot, events })
    },
    [need, log],
  )

  const startSession: LabActions['startSession'] = useCallback(
    async ({ preset = null, condition, config = null }) => {
      stopLoop()
      dispatch({ type: 'busy', busy: true })
      try {
        const cond = condition ?? stateRef.current.presets.find((p) => p.name === preset)?.condition ?? { name: 'full', perturbation: 'none', description: '' }
        const snap = await need().createSession({ preset, condition: cond, config })
        await loadSession(snap)
        dispatch({ type: 'startMenu', open: false })
        log('session', `Session ${snap.id} created (preset ${preset ?? '—'}, condition ${snap.condition.name}${snap.condition.perturbation !== 'none' ? `:${snap.condition.perturbation}` : ''}, seed ${snap.seed})`)
      } catch (e) {
        fail(e, 'Create session failed')
      } finally {
        dispatch({ type: 'busy', busy: false })
      }
    },
    [need, loadSession, log, fail, stopLoop],
  )

  const runPresetComparison: LabActions['runPresetComparison'] = useCallback(
    async (name, seed) => {
      dispatch({ type: 'busy', busy: true })
      dispatch({ type: 'presetResult', result: null })
      try {
        const r = await need().runPreset(name, { seed: seed ?? null })
        dispatch({ type: 'presetResult', result: r })
        log('info', `Preset comparison '${name}' finished (seed ${r.seed})`)
      } catch (e) {
        fail(e, `Run preset ${name} failed`)
      } finally {
        dispatch({ type: 'busy', busy: false })
      }
    },
    [need, log, fail],
  )

  const doStep = useCallback(
    async (n: number): Promise<StepResult | null> => {
      const s = stateRef.current.session
      if (!s) return null
      const res = s.mode === 'replay' ? await need().replayStep(s.id, { n }) : await need().step(s.id, { n })
      dispatch({ type: 'stepResult', result: res })
      stateRef.current = { ...stateRef.current, session: res.snapshot }
      return res
    },
    [need],
  )

  const playPhraseAudio = useCallback((phrase: Phrase): PlayHandle => {
    handleRef.current?.stop()
    const h = audioEngine.playPhrase(phrase)
    handleRef.current = h
    return h
  }, [])

  const stepOnce = useCallback(async () => {
    if (runningRef.current) return
    try {
      const res = await doStep(1)
      if (!res) return
      if (res.snapshot.mode === 'replay' && res.events.length === 0) log('info', 'End of recorded run (replay).')
      const ph = lastSentPhrase(res)
      if (ph) playPhraseAudio(ph)
    } catch (e) {
      fail(e, 'Step failed')
    }
  }, [doStep, fail, log, playPhraseAudio])

  const sleep = (ms: number) =>
    new Promise<void>((resolve) => {
      const t = setTimeout(() => {
        wakeRef.current = null
        resolve()
      }, ms)
      wakeRef.current = () => {
        clearTimeout(t)
        wakeRef.current = null
        resolve()
      }
    })

  const play = useCallback(() => {
    if (runningRef.current || !stateRef.current.session) return
    runningRef.current = true
    dispatch({ type: 'running', running: true })
    void (async () => {
      try {
        while (runningRef.current) {
          const st = stateRef.current
          if (!st.session) break
          if (st.session.step >= st.episodesTarget) {
            log('info', `Episode target reached (${st.episodesTarget}).`)
            break
          }
          const speed = st.speed
          const n = Math.min(stepsPerRequest(speed), Math.max(1, st.episodesTarget - st.session.step))
          const res = await doStep(n)
          if (!res) break
          if (res.snapshot.mode === 'replay' && res.events.length === 0) {
            log('info', 'End of recorded run (replay).')
            break
          }
          if (!runningRef.current) break
          if (speed === 'fast') {
            await sleep(16)
            continue
          }
          const ph = lastSentPhrase(res)
          if (!ph) {
            await sleep(150)
            continue
          }
          playPhraseAudio(ph)
          // 1x: wait until the sent phrase's audio (incl. release + tail) has finished
          await sleep(pacingSeconds(ph, speed) * 1000)
        }
      } catch (e) {
        fail(e, 'Run loop stopped')
      } finally {
        runningRef.current = false
        dispatch({ type: 'running', running: false })
      }
    })()
  }, [doStep, fail, log, playPhraseAudio])

  const pause = useCallback(() => {
    stopLoop()
  }, [stopLoop])

  const resetSession = useCallback(async () => {
    const s = stateRef.current.session
    if (!s) return
    stopLoop()
    try {
      const snap = await need().reset(s.id)
      await loadSession(snap)
      log('session', `Session ${s.id} reset: fresh agents, same seed (${snap.seed}) and config; event log restarted.`)
    } catch (e) {
      fail(e, 'Reset failed')
    }
  }, [need, loadSession, log, fail, stopLoop])

  const intervene = useCallback(
    async (iv: Intervention): Promise<boolean> => {
      const s = stateRef.current.session
      if (!s) return false
      if (s.mode === 'replay') {
        log('error', 'Interventions are unavailable in replay mode (recorded events are re-emitted).')
        return false
      }
      try {
        const res = await need().intervene(s.id, iv)
        dispatch({ type: 'stepResult', result: res })
        const ev = res.events.find((e) => e.type === 'intervention')
        const p = (ev?.payload ?? {}) as { effective_from_step?: number; description?: string; effect?: string }
        const def = INTERVENTION_DEFS[iv.kind]
        log('intervention', `${def.label}${iv.agent_id ? ` [${iv.agent_id}]` : ''}: ${p.description || def.semantics}`, {
          step: p.effective_from_step ?? s.step,
          effect: p.effect || def.takesEffect,
        })
        return true
      } catch (e) {
        fail(e, `Intervention ${iv.kind} failed`)
        return false
      }
    },
    [need, log, fail],
  )

  const replayMotif = useCallback(
    async (ep: EpisodeRecord) => {
      if (!ep.phrase) return
      playPhraseAudio(ep.phrase)
      const s = stateRef.current.session
      if (!s || s.mode === 'replay') {
        log('audio', `Played phrase from step ${ep.step} locally (audio only — replay mode, no intervention sent).`)
        return
      }
      const ok = await intervene({ kind: 'replay_motif', phrase_id: ep.phrase.id, phrase: ep.phrase })
      if (ok) log('audio', `Played phrase from step ${ep.step} locally AND queued it as the next sent phrase (replay_motif).`)
    },
    [intervene, log, playPhraseAudio],
  )

  const sendHuman = useCallback(
    async (phrase: Phrase, targetId: number | null, receiverId: string | null) => {
      const s = stateRef.current.session
      if (!s) return
      try {
        const res = await need().humanPhrase(s.id, { phrase, target_id: targetId, receiver_id: receiverId })
        dispatch({ type: 'stepResult', result: res })
        playPhraseAudio(phrase)
        log('session', `Human phrase sent (target ${targetId ?? 'drawn by environment'}, receiver ${receiverId ?? 'auto'}).`)
      } catch (e) {
        fail(e, 'Send as human failed')
      }
    },
    [need, log, fail, playPhraseAudio],
  )

  const exportRun = useCallback(async () => {
    const s = stateRef.current.session
    if (!s) return
    try {
      const run = await need().exportRun(s.id)
      downloadJson(run, `resonance-run-${safeName(s.id)}.json`)
      log('info', `Exported run ${s.id} (${run.events.length} events).`)
    } catch (e) {
      fail(e, 'Export failed')
    }
  }, [need, log, fail])

  const importRun = useCallback(
    async (file: File) => {
      stopLoop()
      try {
        const run = JSON.parse(await file.text()) as RunExport
        const snap = await need().importRun(run)
        await loadSession(snap)
        dispatch({ type: 'startMenu', open: false })
        log('session', `Imported run as REPLAY session ${snap.id}: Step/Play re-emit recorded events.`)
      } catch (e) {
        fail(e, 'Import failed')
      }
    },
    [need, loadSession, log, fail, stopLoop],
  )

  const downloadWav = useCallback(
    async (phrase: Phrase) => {
      try {
        const blob = await need().renderPhrase(phrase)
        downloadBlob(blob, `${safeName(phrase.id)}.wav`)
      } catch (e) {
        fail(e, 'WAV render failed')
      }
    },
    [need, fail],
  )

  const downloadPhraseJson = useCallback((phrase: Phrase) => downloadJson(phrase, `${safeName(phrase.id)}.json`), [])

  const actions = useMemo<LabActions>(
    () => ({
      startSession,
      runPresetComparison,
      stepOnce,
      play,
      pause,
      resetSession,
      intervene,
      replayMotif,
      playPhraseAudio,
      sendHuman,
      exportRun,
      importRun,
      downloadWav,
      downloadPhraseJson,
    }),
    [startSession, runPresetComparison, stepOnce, play, pause, resetSession, intervene, replayMotif, playPhraseAudio, sendHuman, exportRun, importRun, downloadWav, downloadPhraseJson],
  )

  const value = useMemo(() => ({ state, dispatch, api, actions }), [state, api, actions])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}
