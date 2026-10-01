import { useId, useState, type ReactNode } from 'react'

/** ⓘ button with an accessible tooltip (hover or focus). */
export function InfoTip({ children, label = 'definition', wide = false }: { children: ReactNode; label?: string; wide?: boolean }) {
  const id = useId()
  const [open, setOpen] = useState(false)
  return (
    <span className="infotip" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button
        type="button"
        className="infotip__btn"
        aria-label={label}
        aria-describedby={open ? id : undefined}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((o) => !o)}
      >
        ⓘ
      </button>
      {open && (
        <span role="tooltip" id={id} className={`infotip__pop${wide ? ' infotip__pop--wide' : ''}`}>
          {children}
        </span>
      )}
    </span>
  )
}
