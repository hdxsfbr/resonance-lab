import { useLab } from '../state/LabContext'

export function ErrorBanner() {
  const { state, dispatch } = useLab()
  if (!state.error) return null
  return (
    <div className="errorbanner" role="alert">
      <span>⚠ {state.error}</span>
      <button type="button" className="btn btn--tiny" onClick={() => dispatch({ type: 'error', message: null })} aria-label="Dismiss error">
        ✕
      </button>
    </div>
  )
}
