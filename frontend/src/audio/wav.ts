/** WAV encoding + offline rendering with the same synth graph as live playback. */
import type { Phrase } from '../api/types'
import { createMasterChain, ensureOnePole, schedulePhrase } from './synth'
import { renderSeconds } from './synthSpec'

/** Encode mono float samples [-1,1] as 16-bit PCM WAV. */
export function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const bytes = 44 + samples.length * 2
  const buf = new ArrayBuffer(bytes)
  const v = new DataView(buf)
  const w = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i))
  }
  w(0, 'RIFF')
  v.setUint32(4, bytes - 8, true)
  w(8, 'WAVE')
  w(12, 'fmt ')
  v.setUint32(16, 16, true)
  v.setUint16(20, 1, true) // PCM
  v.setUint16(22, 1, true) // mono
  v.setUint32(24, sampleRate, true)
  v.setUint32(28, sampleRate * 2, true)
  v.setUint16(32, 2, true)
  v.setUint16(34, 16, true)
  w(36, 'data')
  v.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return buf
}

/** Render a phrase in the browser (OfflineAudioContext). Used by the MOCK data source's renderPhrase. */
export async function renderPhraseWav(phrase: Phrase, sampleRate = 44100): Promise<Blob> {
  if (typeof OfflineAudioContext === 'undefined') throw new Error('OfflineAudioContext unavailable')
  const length = Math.ceil(renderSeconds(phrase) * sampleRate)
  const ctx = new OfflineAudioContext(1, length, sampleRate)
  await ensureOnePole(ctx)
  const { input } = createMasterChain(ctx, ctx.destination)
  schedulePhrase(ctx, input, phrase, 0)
  const buf = await ctx.startRendering()
  return new Blob([encodeWav(buf.getChannelData(0), sampleRate)], { type: 'audio/wav' })
}
