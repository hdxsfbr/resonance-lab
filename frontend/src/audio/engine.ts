/**
 * Audio engine: owns the single AudioContext. Respects browser autoplay policy:
 * the context is created / resumed ONLY from `enable()`, which the UI calls from
 * a user gesture (the "Enable audio" button).
 */
import type { Phrase } from '../api/types'
import { createMasterChain, ensureOnePole, hasOnePole, schedulePhrase } from './synth'
import { renderSeconds, secondsToBeats } from './synthSpec'
import { playhead } from './playhead'

export type AudioStatus = 'locked' | 'running' | 'suspended' | 'unsupported'

export interface AudioEngineState {
  status: AudioStatus
  muted: boolean
  volume: number
  sampleRate: number | null
  /** 'one-pole worklet' (matches server) or 'biquad fallback' for the pluck filter */
  pluckFilter: string | null
}

export interface PlayOptions {
  /** Seconds from now to start (default 0.05 s scheduling lead). */
  when?: number
  /** Called on animation frames with the beat position, and once with `null` when done. */
  onProgress?: (beat: number | null) => void
  /** Also drive the global playhead (default true). */
  drivePlayhead?: boolean
}

export interface PlayHandle {
  readonly phraseId: string
  readonly audible: boolean
  readonly totalSeconds: number
  readonly done: Promise<void>
  stop(): void
}

type Listener = (s: AudioEngineState) => void

const LEAD = 0.05

export class AudioEngine {
  private ctx: AudioContext | null = null
  private master: GainNode | null = null
  private volume: GainNode | null = null
  private state: AudioEngineState = { status: 'locked', muted: false, volume: 0.8, sampleRate: null, pluckFilter: null }
  private listeners = new Set<Listener>()
  private active = new Set<PlayHandle>()

  constructor() {
    if (typeof window === 'undefined' || typeof window.AudioContext === 'undefined') this.state.status = 'unsupported'
  }

  getState(): AudioEngineState {
    return this.state
  }

  subscribe(l: Listener): () => void {
    this.listeners.add(l)
    return () => this.listeners.delete(l)
  }

  private patch(p: Partial<AudioEngineState>) {
    this.state = { ...this.state, ...p }
    for (const l of this.listeners) l(this.state)
  }

  /** Must be called from a user gesture. */
  async enable(): Promise<void> {
    if (this.state.status === 'unsupported') return
    if (!this.ctx) {
      this.ctx = new AudioContext({ latencyHint: 'interactive' })
      const chain = createMasterChain(this.ctx, this.ctx.destination)
      this.master = chain.input
      this.volume = chain.volume
      this.applyGain()
      this.ctx.onstatechange = () => this.patch({ status: this.ctx?.state === 'running' ? 'running' : 'suspended' })
    }
    // resume first (still inside the user gesture), then load the filter worklet
    const resumed = this.ctx.state !== 'running' ? this.ctx.resume() : Promise.resolve()
    await Promise.all([resumed, ensureOnePole(this.ctx)])
    this.patch({
      status: this.ctx.state === 'running' ? 'running' : 'suspended',
      sampleRate: this.ctx.sampleRate,
      pluckFilter: hasOnePole(this.ctx) ? 'one-pole worklet' : 'biquad fallback',
    })
  }

  async suspend(): Promise<void> {
    if (this.ctx && this.ctx.state === 'running') await this.ctx.suspend()
    this.patch({ status: this.ctx ? 'suspended' : this.state.status })
  }

  private applyGain() {
    if (!this.volume || !this.ctx) return
    const g = this.state.muted ? 0 : this.state.volume
    this.volume.gain.setTargetAtTime(g, this.ctx.currentTime, 0.02)
  }

  setVolume(v: number): void {
    this.patch({ volume: Math.max(0, Math.min(1, v)) })
    this.applyGain()
  }

  setMuted(m: boolean): void {
    this.patch({ muted: m })
    this.applyGain()
  }

  get audible(): boolean {
    return this.state.status === 'running' && !!this.ctx
  }

  stopAll(): void {
    for (const h of [...this.active]) h.stop()
  }

  /**
   * Play a phrase. If audio is not enabled the handle is SILENT but still keeps
   * time (so visual playheads and 1x pacing behave identically).
   */
  playPhrase(phrase: Phrase, opts: PlayOptions = {}): PlayHandle {
    const when = Math.max(0, opts.when ?? LEAD)
    const total = renderSeconds(phrase)
    const ctx = this.audible ? this.ctx : null
    let sources: OscillatorNode[] = []
    let voiceBus: GainNode | null = null
    if (ctx && this.master) {
      voiceBus = ctx.createGain()
      voiceBus.connect(this.master)
      sources = schedulePhrase(ctx, voiceBus, phrase, ctx.currentTime + when)
    }
    const startedAtMs = performance.now() + when * 1000
    let stopped = false
    let raf = 0
    let timer: ReturnType<typeof setTimeout> | null = null
    let resolveDone: () => void = () => {}
    const done = new Promise<void>((res) => (resolveDone = res))
    const drive = opts.drivePlayhead ?? true
    if (drive)
      playhead.set({ phraseId: phrase.id, startedAtMs, tempoBpm: phrase.tempo_bpm, lengthBeats: phrase.length_beats, totalSeconds: total, audible: !!ctx })
    const finish = () => {
      if (stopped) return
      stopped = true
      if (raf) cancelAnimationFrame(raf)
      if (timer) clearTimeout(timer)
      opts.onProgress?.(null)
      if (drive && playhead.get()?.phraseId === phrase.id) playhead.set(null)
      this.active.delete(handle)
      resolveDone()
    }
    const tick = () => {
      if (stopped) return
      const sec = (performance.now() - startedAtMs) / 1000
      opts.onProgress?.(sec < 0 ? 0 : secondsToBeats(sec, phrase.tempo_bpm))
      raf = requestAnimationFrame(tick)
    }
    if (opts.onProgress && typeof requestAnimationFrame !== 'undefined') raf = requestAnimationFrame(tick)
    timer = setTimeout(finish, (when + total) * 1000)
    const handle: PlayHandle = {
      phraseId: phrase.id,
      audible: !!ctx,
      totalSeconds: total,
      done,
      stop: () => {
        if (ctx && voiceBus) {
          const t = ctx.currentTime
          voiceBus.gain.setTargetAtTime(0, t, 0.015)
          for (const s of sources) {
            try {
              s.stop(t + 0.1)
            } catch {
              /* already stopped */
            }
          }
        }
        finish()
      },
    }
    this.active.add(handle)
    return handle
  }
}

export const audioEngine = new AudioEngine()

/** Convenience wrapper over the singleton engine. */
export function playPhrase(phrase: Phrase, opts: PlayOptions = {}): PlayHandle {
  return audioEngine.playPhrase(phrase, opts)
}
