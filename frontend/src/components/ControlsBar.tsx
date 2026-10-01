import { useRef, useState } from 'react'
import { useLab } from '../state/LabContext'
import { SPEEDS, type Speed } from '../audio/synthSpec'

const SPEED_LABEL: Record<Speed, string> = {
  '0.5x': '0.5×',
  '1x': '1×',
  '2x': '2×',
  '4x': '4×',
  fast: 'no audio / fast',
}

const SPEED_HELP: Record<Speed, string> = {
  '0.5x': 'Plays each sent phrase, then waits as long again before the next step.',
  '1x': 'Waits for the sent phrase (incl. release + 0.5 s tail) to finish before requesting the next step.',
  '2x': 'Requests the next step halfway through; the previous phrase is cut off.',
  '4x': 'Requests the next step a quarter of the way through; the previous phrase is cut off.',
  fast: 'Calls step with n=10 and skips audio.',
}

export function ControlsBar() {
  const { state, dispatch, actions } = useLab()
  const [confirming, setConfirming] = useState(false)
  const fileRef = useRef<HTMLInputElement | null>(null)
  const s = state.session
  const replay = s?.mode === 'replay'
  return (
    <div className="controls" role="toolbar" aria-label="Simulation controls">
      {state.running ? (
        <button type="button" className="btn btn--primary" onClick={actions.pause} data-testid="pause">
          ❚❚ Pause
        </button>
      ) : (
        <button type="button" className="btn btn--primary" onClick={actions.play} disabled={!s} data-testid="start">
          ▶ Start
        </button>
      )}
      <button type="button" className="btn" onClick={() => void actions.stepOnce()} disabled={!s || state.running} data-testid="step" title={replay ? 'POST /replay/step (re-emit recorded episode)' : 'POST /step with n=1'}>
        Step
      </button>
      {confirming ? (
        <span className="confirm" role="alertdialog" aria-label="Confirm reset">
          Reset session? New agents, same seed & config; event log restarts.
          <button
            type="button"
            className="btn btn--danger btn--small"
            onClick={() => {
              setConfirming(false)
              void actions.resetSession()
            }}
          >
            Yes, reset
          </button>
          <button type="button" className="btn btn--small" onClick={() => setConfirming(false)}>
            Cancel
          </button>
        </span>
      ) : (
        <button type="button" className="btn" onClick={() => setConfirming(true)} disabled={!s || replay} data-testid="reset">
          Reset session
        </button>
      )}
      <span className="controls__group" role="group" aria-label="Speed">
        <span className="small muted">speed</span>
        {SPEEDS.map((sp) => (
          <button
            key={sp}
            type="button"
            className={`seg__btn${state.speed === sp ? ' seg__btn--on' : ''}`}
            onClick={() => dispatch({ type: 'speed', speed: sp })}
            title={SPEED_HELP[sp]}
            aria-pressed={state.speed === sp}
          >
            {SPEED_LABEL[sp]}
          </button>
        ))}
      </span>
      <label className="controls__group small">
        <span className="muted">episodes target</span>
        <input
          type="number"
          min={1}
          max={5000}
          value={state.episodesTarget}
          onChange={(e) => dispatch({ type: 'target', episodes: Number(e.target.value) || 1 })}
          className="input input--num"
        />
      </label>
      <span className="spacer" />
      <button type="button" className="btn btn--small" onClick={() => void actions.exportRun()} disabled={!s} data-testid="export">
        ⤓ Export run JSON
      </button>
      <button type="button" className="btn btn--small" onClick={() => fileRef.current?.click()} data-testid="import">
        ⤒ Import run JSON
      </button>
      <input
        ref={fileRef}
        type="file"
        accept="application/json,.json"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) void actions.importRun(f)
          e.target.value = ''
        }}
      />
      <button type="button" className="btn btn--small" onClick={() => dispatch({ type: 'startMenu', open: true })} data-testid="open-presets">
        Presets
      </button>
      <button type="button" className="btn btn--small btn--accent" onClick={() => dispatch({ type: 'experiments', open: true })} data-testid="open-experiments">
        Experiments
      </button>
    </div>
  )
}
