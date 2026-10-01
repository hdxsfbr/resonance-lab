import { useEffect, useRef, useState } from 'react'
import type { Phrase, TimingPattern } from '../api/types'
import { useLab } from '../state/LabContext'
import { focusedEpisode } from '../state/store'
import { playhead } from '../audio/playhead'
import { phraseSeconds } from '../audio/synthSpec'
import { sequential } from '../lib/colors'
import { fmt, noteName } from '../lib/format'
import { useSize } from '../lib/useSize'
import { KindTag } from './KindTag'

interface RollProps {
  phrase: Phrase
  height: number
  width: number
  target?: TimingPattern | null
  chosen?: TimingPattern | null
  showOverlay?: boolean
}

function lanesFor(p: TimingPattern | null | undefined, lengthBeats: number): number[] {
  if (!p) return []
  const out: number[] = []
  for (let bar = 0; bar < lengthBeats; bar += p.length_beats) for (const o of p.onsets) if (bar + o < lengthBeats) out.push(bar + o)
  return out
}

/** SVG piano roll with a playhead driven by audio/playhead (no React re-render per frame). */
export function Roll({ phrase, height, width, target, chosen, showOverlay }: RollProps) {
  const lineRef = useRef<SVGLineElement | null>(null)
  const pitches = phrase.notes.map((n) => n.pitch)
  const lo = Math.min(...(pitches.length ? pitches : [60])) - 2
  const hi = Math.max(lo + 14, Math.max(...(pitches.length ? pitches : [72])) + 2)
  const padL = 46
  const laneH = showOverlay ? 22 : 0
  const padT = 4 + laneH
  const padB = 16
  const w = Math.max(100, width) - padL - 4
  const h = Math.max(30, height - padT - padB)
  const rows = hi - lo + 1
  const rowH = h / rows
  const x = (beat: number) => padL + (beat / phrase.length_beats) * w
  const y = (pitch: number) => padT + (hi - pitch) * rowH

  useEffect(() => {
    let raf = 0
    const loop = () => {
      const el = lineRef.current
      const ph = playhead.get()
      if (el) {
        const b = ph && ph.phraseId === phrase.id ? playhead.beatNow() : null
        if (b === null || b > phrase.length_beats + 0.01) el.style.display = 'none'
        else {
          el.style.display = ''
          const xx = padL + (b / phrase.length_beats) * w
          el.setAttribute('x1', String(xx))
          el.setAttribute('x2', String(xx))
        }
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [phrase.id, phrase.length_beats, w])

  const beats = Array.from({ length: Math.floor(phrase.length_beats) + 1 }, (_, i) => i)
  return (
    <svg width={width} height={Math.max(height, padT + padB + 30)} className="roll" role="img" aria-label={`Piano roll of phrase ${phrase.id}`} data-testid="piano-roll">
      {Array.from({ length: rows }, (_, i) => {
        const p = hi - i
        const black = [1, 3, 6, 8, 10].includes(((p % 12) + 12) % 12)
        return <rect key={p} x={padL} y={padT + i * rowH} width={w} height={rowH} className={black ? 'roll__row roll__row--black' : 'roll__row'} />
      })}
      {Array.from({ length: rows }, (_, i) => hi - i)
        .filter((p) => p % 12 === 0 || rows <= 16)
        .map((p) => (
          <text key={`l${p}`} x={padL - 4} y={y(p) + rowH / 2 + 3} className="axislabel" textAnchor="end">
            {noteName(p)}
          </text>
        ))}
      {beats.map((b) => (
        <g key={`b${b}`}>
          <line x1={x(b)} x2={x(b)} y1={padT} y2={padT + h} className={b % 4 === 0 ? 'gridline gridline--bar' : 'gridline'} />
          <text x={x(b)} y={padT + h + 12} className="axislabel" textAnchor="middle">
            {b}
          </text>
        </g>
      ))}
      {showOverlay && (
        <g aria-label="Onset overlay">
          <text x={padL - 4} y={11} className="axislabel" textAnchor="end">
            target
          </text>
          <text x={padL - 4} y={21} className="axislabel" textAnchor="end">
            chosen
          </text>
          {lanesFor(target, phrase.length_beats).map((o, i) => (
            <rect key={`t${i}`} x={x(o) - 1} y={4} width={4} height={8} rx={1} className="lane lane--target" />
          ))}
          {lanesFor(chosen, phrase.length_beats).map((o, i) => (
            <rect key={`c${i}`} x={x(o) - 1} y={14} width={4} height={8} rx={1} className="lane lane--chosen" />
          ))}
        </g>
      )}
      {phrase.notes.map((n, i) => (
        <rect
          key={i}
          x={x(n.onset)}
          y={y(n.pitch) + 0.5}
          width={Math.max(2, x(n.onset + n.duration) - x(n.onset) - 1)}
          height={Math.max(2, rowH - 1)}
          rx={2}
          fill={sequential(n.velocity / 127)}
        >
          <title>{`${noteName(n.pitch)} (${n.pitch}) · onset ${fmt(n.onset, 2)} · dur ${fmt(n.duration, 2)} beats · velocity ${n.velocity}`}</title>
        </rect>
      ))}
      <line ref={lineRef} x1={padL} x2={padL} y1={padT - laneH} y2={padT + h} className="playhead" style={{ display: 'none' }} />
    </svg>
  )
}

export function PianoRollPanel() {
  const { state, actions, dispatch } = useLab()
  const ep = focusedEpisode(state)
  const [view, setView] = useState<'sent' | 'received'>('sent')
  const [ref, size] = useSize<HTMLDivElement>({ width: 560, height: 190 })
  const received = ep?.observation?.phrase ?? null
  const differs = !!(received && ep?.phrase && received.id !== ep.phrase.id)
  const phrase = view === 'received' && differs ? received : (ep?.phrase ?? null)
  const target = state.patterns.find((p) => p.id === ep?.targetId) ?? null
  const chosen = state.patterns.find((p) => p.id === ep?.chosenId) ?? null
  const live = state.session && state.session.mode !== 'replay'
  const senderName = ep?.human ? 'human' : (state.session?.agents.find((a) => a.id === ep?.senderId)?.name ?? ep?.senderId ?? '—')
  return (
    <section className="panel roll-panel" aria-label="Current phrase">
      <div className="panel__bar">
        <h3 className="panel__title">
          Phrase {ep ? `· step ${ep.step}` : ''} <KindTag kind="measured" />
        </h3>
        {phrase && (
          <span className="small muted panel__meta" title={phrase.id}>
            {senderName} · {phrase.instrument} · {fmt(phrase.tempo_bpm, 0)} bpm · {fmt(phrase.length_beats, 0)} beats · {fmt(phraseSeconds(phrase), 2)} s
            {phrase.motif_id ? ` · ${phrase.motif_id}` : ''}
            {phrase.origin?.kind && phrase.origin.kind !== 'agent' ? ` · origin ${phrase.origin.kind}${phrase.origin.transform ? ` (${phrase.origin.transform})` : ''}` : ''}
          </span>
        )}
        {phrase && (
          <span className="row" style={{ margin: 0 }}>
            <button
              type="button"
              className="btn btn--tiny"
              onClick={() => void actions.downloadWav(phrase)}
              title={state.source.kind === 'mock' ? 'Download WAV — rendered in the browser with the same synth (mock data source)' : 'Download WAV — POST /api/phrases/render (server numpy synth)'}
            >
              ⤓ WAV
            </button>
            <button type="button" className="btn btn--tiny" onClick={() => actions.downloadPhraseJson(phrase)} title="Download the symbolic phrase as JSON">
              ⤓ JSON
            </button>
          </span>
        )}
        {differs && (
          <span className="seg" role="group" aria-label="Which phrase">
            <button type="button" className={`seg__btn${view === 'sent' ? ' seg__btn--on' : ''}`} onClick={() => setView('sent')}>
              sent
            </button>
            <button type="button" className={`seg__btn${view === 'received' ? ' seg__btn--on' : ''}`} onClick={() => setView('received')}>
              received (after channel)
            </button>
          </span>
        )}
      </div>
      <div className="roll-panel__body" ref={ref}>
        {phrase ? (
          <Roll phrase={phrase} width={size.width} height={size.height} target={target} chosen={chosen} showOverlay={state.overlayPattern} />
        ) : (
          <p className="muted empty">No phrase yet — press Step or Start.</p>
        )}
      </div>
      <div className="panel__bar panel__bar--bottom">
        <span className="legend">
          <span className="legend__ramp" aria-hidden /> velocity 1 → 127
        </span>
        <label className="check tiny" title="Draw the target pattern's and the receiver's chosen pattern's onsets above the roll (repeated every bar)">
          <input type="checkbox" checked={state.overlayPattern} onChange={(e) => dispatch({ type: 'overlay', on: e.target.checked })} /> onsets: target vs chosen
        </label>
        <span className="spacer" />
        {phrase && (
          <>
            <button type="button" className="btn btn--small" onClick={() => actions.playPhraseAudio(phrase)} title="Local audio playback only; no intervention">
              ▶ Play
            </button>
            {ep && (
              <button
                type="button"
                className="btn btn--small"
                onClick={() => void actions.replayMotif(ep)}
                title={live ? 'Plays locally AND posts replay_motif: the receiver gets this phrase at the next step' : 'Replay mode: local playback only'}
              >
                ↻ Replay{live ? ' → next' : ''}
              </button>
            )}
          </>
        )}
      </div>
    </section>
  )
}
