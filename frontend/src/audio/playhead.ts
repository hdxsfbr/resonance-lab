/** Tiny pub-sub for the currently sounding phrase, so the piano roll can animate a playhead without React re-renders. */
export interface PlayheadInfo {
  phraseId: string
  /** performance.now() timestamp (ms) at which beat 0 sounds */
  startedAtMs: number
  tempoBpm: number
  lengthBeats: number
  /** total seconds including release + tail */
  totalSeconds: number
  audible: boolean
}

type Listener = (p: PlayheadInfo | null) => void

let current: PlayheadInfo | null = null
const listeners = new Set<Listener>()

export const playhead = {
  get(): PlayheadInfo | null {
    return current
  },
  set(p: PlayheadInfo | null): void {
    current = p
    for (const l of listeners) l(p)
  },
  subscribe(l: Listener): () => void {
    listeners.add(l)
    return () => listeners.delete(l)
  },
  /** Beat position of the playhead now, or null if nothing is playing. */
  beatNow(nowMs = performance.now()): number | null {
    if (!current) return null
    const sec = (nowMs - current.startedAtMs) / 1000
    if (sec < 0 || sec > current.totalSeconds) return null
    return (sec * current.tempoBpm) / 60
  },
}
