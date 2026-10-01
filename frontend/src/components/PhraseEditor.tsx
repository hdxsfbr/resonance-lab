import { useEffect, useRef, useState } from 'react'
import type { Instrument, Note, Phrase, PhraseFeatures, SavedMotif } from '../api/types'
import { useLab } from '../state/LabContext'
import { audioEngine } from '../audio/engine'
import { sequential } from '../lib/colors'
import { errorMessage, fmt, noteName } from '../lib/format'
import { KindTag } from './KindTag'

const SCALES = {
  major: [0, 2, 4, 5, 7, 9, 11],
  pentatonic: [0, 2, 4, 7, 9],
} as const
type ScaleName = keyof typeof SCALES

const INSTRUMENTS: Instrument[] = ['pluck', 'marimba', 'bell', 'sine', 'triangle', 'square']
const VELOCITIES = [
  { k: 'pp', v: 40 },
  { k: 'p', v: 64 },
  { k: 'mf', v: 88 },
  { k: 'f', v: 108 },
  { k: 'ff', v: 124 },
]
const LENGTH_BEATS = 8
const STEP = 0.5
const COLS = LENGTH_BEATS / STEP
const ROOT = 60

function scalePitches(scale: ScaleName): number[] {
  const out: number[] = []
  for (let oct = 0; oct < 2; oct++) for (const iv of SCALES[scale]) out.push(ROOT + oct * 12 + iv)
  out.push(ROOT + 24)
  return out
}

function previewNote(pitch: number, instrument: Instrument, velocity: number) {
  if (!audioEngine.audible) return
  audioEngine.playPhrase(
    { id: `kbd-${pitch}-${Date.now()}`, notes: [{ pitch, onset: 0, duration: 0.5, velocity }], tempo_bpm: 120, length_beats: 0.5, instrument },
    { drivePlayhead: false, when: 0.01 },
  )
}

export function PhraseEditor() {
  const { state, api, actions } = useLab()
  const [scale, setScale] = useState<ScaleName>('pentatonic')
  const [notes, setNotes] = useState<Note[]>([])
  const [vel, setVel] = useState(88)
  const [len, setLen] = useState(0.5)
  const [tempo, setTempo] = useState(112)
  const [instrument, setInstrument] = useState<Instrument>('pluck')
  const [cursor, setCursor] = useState(0)
  const [name, setName] = useState('my-motif')
  const [saved, setSaved] = useState<SavedMotif[]>([])
  const [loadSel, setLoadSel] = useState('')
  const [target, setTarget] = useState<string>('random')
  const [receiver, setReceiver] = useState<string>('auto')
  // measured features are only shown while they still describe the current notes/tempo
  const [measured, setMeasured] = useState<{ key: string; f: PhraseFeatures } | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const drag = useRef<{ idx: number; y: number; v0: number; moved: boolean } | null>(null)
  const pitches = scalePitches(scale)
  const phraseKey = JSON.stringify([notes, tempo, instrument])
  const rows = [...pitches].reverse()

  const phrase: Phrase = {
    id: `human-${name || 'phrase'}`,
    notes: [...notes].sort((a, b) => a.onset - b.onset || a.pitch - b.pitch),
    tempo_bpm: tempo,
    length_beats: LENGTH_BEATS,
    instrument,
    origin: { kind: 'human', agent_id: null, source_phrase_id: null, transform: null },
    motif_id: null,
    tags: ['human'],
  }

  useEffect(() => {
    if (!api) return
    let cancelled = false
    api
      .savedMotifs()
      .then((list) => !cancelled && setSaved(list))
      .catch((e) => !cancelled && setMsg(`load saved motifs failed: ${errorMessage(e)}`))
    return () => {
      cancelled = true
    }
  }, [api])

  const toggleAt = (col: number, pitch: number) => {
    const onset = col * STEP
    const i = notes.findIndex((n) => n.pitch === pitch && Math.abs(n.onset - onset) < 1e-6)
    if (i >= 0) setNotes(notes.filter((_, j) => j !== i))
    else {
      setNotes([...notes, { pitch, onset, duration: Math.min(len, LENGTH_BEATS - onset), velocity: vel }])
      previewNote(pitch, instrument, vel)
    }
  }

  const covering = (col: number, pitch: number) =>
    notes.findIndex((n) => n.pitch === pitch && col * STEP >= n.onset - 1e-6 && col * STEP < n.onset + n.duration - 1e-6)

  const onKey = (pitch: number) => {
    previewNote(pitch, instrument, vel)
    if (cursor >= COLS) return
    setNotes([...notes.filter((n) => !(n.pitch === pitch && Math.abs(n.onset - cursor * STEP) < 1e-6)), { pitch, onset: cursor * STEP, duration: Math.min(len, LENGTH_BEATS - cursor * STEP), velocity: vel }])
    setCursor(Math.min(COLS, cursor + Math.round(len / STEP)))
  }

  const save = async () => {
    if (!api || !phrase.notes.length) return
    try {
      const list = await api.saveMotif({ name: name || 'motif', phrase: { ...phrase, id: `saved-${name}` }, created_at: new Date().toISOString(), features: null })
      setSaved(list)
      setMsg(`Saved motif '${name}' (POST /api/motifs/saved).`)
    } catch (e) {
      setMsg(`save failed: ${errorMessage(e)}`)
    }
  }

  const load = (n: string) => {
    setLoadSel(n)
    const m = saved.find((s) => s.name === n)
    if (!m) return
    setNotes(m.phrase.notes.map((x) => ({ ...x })))
    setTempo(m.phrase.tempo_bpm)
    setInstrument(m.phrase.instrument)
    setName(m.name)
    setMsg(`Loaded '${m.name}'.`)
  }

  const measure = async () => {
    if (!api) return
    try {
      setMeasured({ key: phraseKey, f: await api.phraseFeatures(phrase) })
    } catch (e) {
      setMsg(`features failed: ${errorMessage(e)}`)
    }
  }

  const features = measured && measured.key === phraseKey ? measured.f : null
  const s = state.session
  const canSend = !!s && s.mode !== 'replay' && phrase.notes.length > 0
  const cell = 17
  return (
    <section className="panel composer" aria-label="Phrase editor">
      <h3 className="panel__title">Compose a phrase</h3>
      <div className="row wrap small">
        <span className="seg" role="group" aria-label="Scale">
          {(Object.keys(SCALES) as ScaleName[]).map((k) => (
            <button key={k} type="button" className={`seg__btn${scale === k ? ' seg__btn--on' : ''}`} onClick={() => setScale(k)}>
              {k === 'major' ? 'C major' : 'C pentatonic'}
            </button>
          ))}
        </span>
        <label>
          tempo <input className="input input--num" type="number" min={80} max={160} value={tempo} onChange={(e) => setTempo(Math.max(21, Math.min(400, Number(e.target.value) || 112)))} />
        </label>
        <select className="input" value={instrument} onChange={(e) => setInstrument(e.target.value as Instrument)} aria-label="Instrument">
          {INSTRUMENTS.map((i) => (
            <option key={i}>{i}</option>
          ))}
        </select>
      </div>
      <div className="row wrap small">
        <span className="muted">velocity</span>
        <span className="seg" role="group" aria-label="Velocity brush">
          {VELOCITIES.map((x) => (
            <button key={x.k} type="button" className={`seg__btn${vel === x.v ? ' seg__btn--on' : ''}`} onClick={() => setVel(x.v)} title={`velocity ${x.v}`}>
              {x.k}
            </button>
          ))}
        </span>
        <span className="muted">length</span>
        <span className="seg" role="group" aria-label="Note length">
          {[0.5, 1, 2].map((l) => (
            <button key={l} type="button" className={`seg__btn${len === l ? ' seg__btn--on' : ''}`} onClick={() => setLen(l)}>
              {l}
            </button>
          ))}
        </span>
      </div>
      <p className="tiny muted">Click a cell to add / remove a note. Drag a note up/down to change its velocity. Keys below insert at the cursor (▲).</p>
      <svg
        width={30 + COLS * cell}
        height={rows.length * cell + 12}
        className="grid-editor"
        role="grid"
        aria-label="Phrase grid editor"
        onPointerMove={(e) => {
          const d = drag.current
          if (!d) return
          const dy = d.y - e.clientY
          if (Math.abs(dy) > 3) d.moved = true
          if (d.moved) setNotes((ns) => ns.map((n, j) => (j === d.idx ? { ...n, velocity: Math.max(1, Math.min(127, Math.round(d.v0 + dy))) } : n)))
        }}
        onPointerUp={() => {
          const d = drag.current
          drag.current = null
          if (d && !d.moved) setNotes((ns) => ns.filter((_, j) => j !== d.idx))
        }}
        onPointerLeave={() => (drag.current = null)}
      >
        {rows.map((p, r) => (
          <g key={p}>
            <text x={26} y={r * cell + cell / 2 + 4} className="axislabel" textAnchor="end">
              {noteName(p)}
            </text>
            {Array.from({ length: COLS }, (_, c) => {
              const ci = covering(c, p)
              const startsHere = ci >= 0 && Math.abs(notes[ci].onset - c * STEP) < 1e-6
              return (
                <rect
                  key={c}
                  x={30 + c * cell}
                  y={r * cell}
                  width={cell - 2}
                  height={cell - 2}
                  rx={2}
                  className={ci >= 0 ? 'ge__note' : c % 2 === 0 ? 'ge__cell ge__cell--beat' : 'ge__cell'}
                  fill={ci >= 0 ? sequential(notes[ci].velocity / 127) : undefined}
                  opacity={ci >= 0 && !startsHere ? 0.6 : 1}
                  onPointerDown={(e) => {
                    if (ci >= 0) {
                      ;(e.currentTarget.ownerSVGElement as SVGSVGElement).setPointerCapture?.(e.pointerId)
                      drag.current = { idx: ci, y: e.clientY, v0: notes[ci].velocity, moved: false }
                    } else toggleAt(c, p)
                  }}
                >
                  <title>{ci >= 0 ? `${noteName(p)} onset ${notes[ci].onset} vel ${notes[ci].velocity}` : `${noteName(p)} @ beat ${c * STEP}`}</title>
                </rect>
              )
            })}
          </g>
        ))}
        <text x={30 + cursor * cell + 4} y={rows.length * cell + 10} className="axislabel">
          ▲
        </text>
      </svg>
      <div className="keyboard" role="group" aria-label="On-screen keyboard">
        {pitches.map((p) => (
          <button key={p} type="button" className={`key${p % 12 === 0 ? ' key--c' : ''}`} onClick={() => onKey(p)} title={`${noteName(p)} — insert at cursor`}>
            {noteName(p)}
          </button>
        ))}
      </div>
      <div className="row wrap">
        <button type="button" className="btn btn--small" onClick={() => setCursor(0)}>
          cursor ⟲
        </button>
        <button type="button" className="btn btn--small" onClick={() => actions.playPhraseAudio(phrase)} disabled={!phrase.notes.length}>
          ▶ Preview
        </button>
        <button type="button" className="btn btn--small" onClick={() => setNotes([])}>
          Clear
        </button>
        <button type="button" className="btn btn--small" onClick={() => actions.downloadPhraseJson(phrase)} disabled={!phrase.notes.length}>
          ⤓ JSON
        </button>
        <button type="button" className="btn btn--small" onClick={() => void measure()} disabled={!phrase.notes.length}>
          Measure features
        </button>
      </div>
      {features && (
        <div className="features small">
          <KindTag kind="measured" /> {features.symbolic.note_count} notes · density {fmt(features.symbolic.note_density, 2)}/s · regularity {fmt(features.symbolic.rhythmic_regularity, 2)} · syncopation{' '}
          {fmt(features.symbolic.syncopation, 2)} · contour {features.symbolic.contour_class} · mean vel {fmt(features.symbolic.mean_velocity, 0)}
          {state.source.kind === 'mock' && <span className="muted"> (browser mock features)</span>}
        </div>
      )}
      <div className="row wrap">
        <input className="input" value={name} onChange={(e) => setName(e.target.value)} aria-label="Motif name" />
        <button type="button" className="btn btn--small" onClick={() => void save()} disabled={!phrase.notes.length}>
          Save motif
        </button>
        <select className="input" value={loadSel} onChange={(e) => load(e.target.value)} aria-label="Load saved motif">
          <option value="">Load saved…</option>
          {saved.map((m) => (
            <option key={m.name} value={m.name}>
              {m.name}
            </option>
          ))}
        </select>
      </div>
      <div className="row wrap">
        <label className="small">
          target{' '}
          <select className="input" value={target} onChange={(e) => setTarget(e.target.value)}>
            <option value="random">drawn by environment</option>
            {state.patterns.map((p) => (
              <option key={p.id} value={String(p.id)}>
                #{p.id} {p.name}
              </option>
            ))}
          </select>
        </label>
        <label className="small">
          receiver{' '}
          <select className="input" value={receiver} onChange={(e) => setReceiver(e.target.value)}>
            <option value="auto">auto (alternation)</option>
            {(s?.agents ?? []).map((a) => (
              <option key={a.id} value={a.id}>
                {a.name}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="btn btn--small btn--accent"
          disabled={!canSend}
          onClick={() => void actions.sendHuman(phrase, target === 'random' ? null : Number(target), receiver === 'auto' ? null : receiver)}
          title="POST /api/sessions/{id}/human_phrase — you act as the sender for one episode"
        >
          Send as human
        </button>
      </div>
      {msg && <p className="tiny muted">{msg}</p>}
    </section>
  )
}
