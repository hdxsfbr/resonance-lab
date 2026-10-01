import { useEffect, useRef, useState } from 'react'

/** Observe an element's content-box size. */
export function useSize<T extends HTMLElement>(fallback = { width: 600, height: 200 }) {
  const ref = useRef<T | null>(null)
  const [size, setSize] = useState(fallback)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver((entries) => {
      const r = entries[0].contentRect
      setSize((s) => (Math.abs(s.width - r.width) < 1 && Math.abs(s.height - r.height) < 1 ? s : { width: r.width, height: r.height }))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, size] as const
}
