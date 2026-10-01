import { useEffect, useRef, useState, useSyncExternalStore } from 'react'
import { useLab } from '../state/LabContext'
import { badgeFor, type BadgeKind } from '../state/store'
import { audioEngine } from '../audio/engine'
import { perturbationDef } from '../lib/definitions'
import { fmt } from '../lib/format'

const BADGE_TEXT: Record<BadgeKind, string> = {
  live: 'LIVE SIMULATION',
  replay: 'REPLAY',
  demo: 'DEMO DATA (mock)',
}

const BADGE_TITLE: Record<BadgeKind, string> = {
  live: 'Data comes from the backend simulation (/api), stepped live by this page.',
  replay: 'An imported run: Step/Play re-emit recorded events (POST /replay/step). Interventions are disabled.',
  demo: 'The backend is unavailable or ?mock=1 was set: an in-browser mock produces plausible demo data. Nothing here is a backend result.',
}

export function DataSourceBadges() {
  const { state } = useLab()
  if (!state.source.kind) return <span className="badge badge--pending">connecting…</span>
  return (
    <span className="badges" data-testid="data-source-badge">
      {badgeFor(state).map((b) => (
        <span key={b} className={`badge badge--${b}`} title={`${BADGE_TITLE[b]}\n${state.source.reason}`}>
          {BADGE_TEXT[b]}
        </span>
      ))}
    </span>
  )
}

function useAudioState() {
  return useSyncExternalStore(
    (cb) => audioEngine.subscribe(cb),
    () => audioEngine.getState(),
  )
}

export function AudioControls() {
  const a = useAudioState()
  const label = { locked: 'audio locked', running: 'audio on', suspended: 'audio paused', unsupported: 'no Web Audio' }[a.status]
  return (
    <div className="audioctl" aria-label="Audio controls">
      <span className={`audioctl__dot audioctl__dot--${a.status}`} aria-hidden />
      <span className="audioctl__status" data-testid="audio-status">
        {label}
      </span>
      {a.status === 'locked' && (
        <button type="button" className="btn btn--accent" onClick={() => void audioEngine.enable()} data-testid="enable-audio">
          Enable audio
        </button>
      )}
      {a.status === 'running' && (
        <button type="button" className="btn btn--small" onClick={() => void audioEngine.suspend()} title="Pause audio output (simulation unaffected)">
          ❚❚ Output
        </button>
      )}
      {a.status === 'suspended' && (
        <button type="button" className="btn btn--small" onClick={() => void audioEngine.enable()} title="Resume audio output">
          ▶ Output
        </button>
      )}
      <button
        type="button"
        className={`btn btn--small${a.muted ? ' btn--on' : ''}`}
        onClick={() => audioEngine.setMuted(!a.muted)}
        aria-pressed={a.muted}
        disabled={a.status === 'unsupported'}
      >
        {a.muted ? 'Muted' : 'Mute'}
      </button>
      <label className="audioctl__vol">
        <span className="sr-only">Volume</span>
        <input type="range" min={0} max={1} step={0.01} value={a.volume} onChange={(e) => audioEngine.setVolume(Number(e.target.value))} aria-label="Volume" />
      </label>
    </div>
  )
}

function ConfigPopover() {
  const { state } = useLab()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [open])
  const cfg = state.session?.config ?? state.defaultConfig
  const model = state.source.health?.model
  return (
    <div className="popover" ref={ref}>
      <button type="button" className="btn btn--small" onClick={() => setOpen((o) => !o)} aria-expanded={open} data-testid="config-button">
        Config ▾
      </button>
      {open && cfg && (
        <div className="popover__body" role="dialog" aria-label="Configuration summary">
          <h4>Configuration {state.session ? `(session ${state.session.id})` : '(default)'}</h4>
          <dl className="kv">
            <dt>episodes (config)</dt>
            <dd>{cfg.episodes}</dd>
            <dt>learning</dt>
            <dd>
              {cfg.learning?.enabled ? 'on' : 'off'} · receiver η {fmt(cfg.learning?.receiver_learning_rate, 3)} · sender η {fmt(cfg.learning?.sender_learning_rate, 3)}
            </dd>
            <dt>temperature τ_base</dt>
            <dd>{fmt(cfg.learning?.temperature_base, 3)}</dd>
            <dt>state decay / step</dt>
            <dd>
              {Object.entries(cfg.state?.decay ?? {})
                .map(([k, v]) => `${k} ${fmt(v, 3)}`)
                .join(' · ')}
            </dd>
            <dt>coupling (state→behaviour)</dt>
            <dd>{cfg.coupling?.enabled ? 'on' : 'off'}</dd>
            <dt>acoustic→activation drive</dt>
            <dd>{cfg.state?.acoustic_activation_enabled ? `on (gain ${fmt(cfg.state?.acoustic_activation_gain)}) — hand-authored` : 'off'}</dd>
            <dt>memory capacity</dt>
            <dd>{cfg.memory?.capacity}</dd>
            <dt>task</dt>
            <dd>
              {cfg.task?.n_patterns} timing patterns · partial credit {cfg.task?.partial_credit ? 'on' : 'off'}
            </dd>
            <dt>model provider</dt>
            <dd>
              {cfg.model?.provider ?? 'none'}
              {cfg.model?.model_id ? ` (${cfg.model.model_id})` : ''}
              {model ? ` · status: ${model.available ? 'available' : 'unavailable'}${model.tested_in_this_environment ? ', tested here' : ', untested here'}` : ''}
            </dd>
            <dt>model input modality</dt>
            <dd>
              {model?.input_modality ?? 'none'} {model && !model.accepts_audio ? '(no audio input)' : ''}
            </dd>
          </dl>
          {state.source.kind === 'mock' && <p className="note">DEMO DATA (mock): the mock uses its own learning defaults so learning is visible within ~120 episodes.</p>}
        </div>
      )}
    </div>
  )
}

export function Header() {
  const { state } = useLab()
  const s = state.session
  const cond = s?.condition
  return (
    <header className="header">
      <div className="header__brand">
        <span className="header__name">Resonance Lab</span>
        <DataSourceBadges />
      </div>
      <div className="header__meta" aria-label="Run identity">
        <span className="meta">
          <span className="meta__k">seed</span> <span className="meta__v">{s?.seed ?? '—'}</span>
        </span>
        <span className="meta">
          <span className="meta__k">condition</span>{' '}
          <span className="meta__v" title={cond ? perturbationDef(cond.perturbation).description : ''}>
            {cond ? cond.name : '—'}
            {cond && cond.perturbation !== 'none' ? ` · ${cond.perturbation}` : ''}
          </span>
        </span>
        <span className="meta">
          <span className="meta__k">preset</span> <span className="meta__v">{s?.preset ?? '—'}</span>
        </span>
        <span className="meta">
          <span className="meta__k">session</span> <span className="meta__v mono">{s?.id ?? '—'}</span>
        </span>
        <ConfigPopover />
      </div>
      <AudioControls />
    </header>
  )
}
