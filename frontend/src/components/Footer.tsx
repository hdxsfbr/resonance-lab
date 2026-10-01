import { useLab } from '../state/LabContext'
import { FOOTER_NOTE } from '../lib/definitions'

export function Footer() {
  const { state } = useLab()
  return (
    <footer className="footer" data-testid="footer-note">
      <span>{FOOTER_NOTE}</span>
      <span className="muted">
        {state.source.kind === 'mock' ? 'Source: in-browser mock' : state.source.kind === 'live' ? `Source: /api (${state.source.health?.version ?? ''})` : ''}
      </span>
    </footer>
  )
}
