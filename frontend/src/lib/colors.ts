/**
 * Chart colours (dark surface). Categorical slots follow the validated dark
 * categorical order from the dataviz reference palette; text never wears a
 * series colour.
 */
import type { StateDim } from '../api/types'

export const DIM_COLORS: Record<StateDim, string> = {
  activation: '#3987e5',
  expected_value: '#d95926',
  uncertainty: '#199e70',
  affiliation: '#c98500',
}

export const STATUS = {
  good: '#0ca30c',
  warning: '#fab219',
  serious: '#ec835a',
  critical: '#d03b3b',
}

/** Sequential blue ramp for magnitude (dark surface: low = dark, high = light). */
const SEQ = ['#184f95', '#1c5cab', '#256abf', '#2a78d6', '#3987e5', '#5598e7', '#6da7ec', '#86b6ef', '#9ec5f4', '#b7d3f6']

export function sequential(t: number): string {
  const i = Math.max(0, Math.min(SEQ.length - 1, Math.round(t * (SEQ.length - 1))))
  return SEQ[i]
}

function hexToRgb(h: string): [number, number, number] {
  const n = parseInt(h.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

function mix(a: string, b: string, t: number): string {
  const x = hexToRgb(a)
  const y = hexToRgb(b)
  const c = x.map((v, i) => Math.round(v + (y[i] - v) * t))
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`
}

const DIV_NEG = '#e66767'
const DIV_MID = '#383835'
const DIV_POS = '#3987e5'

/** Diverging red <-> grey <-> blue for signed values; t in [-1, 1]. */
export function diverging(t: number): string {
  const c = Math.max(-1, Math.min(1, t))
  return c < 0 ? mix(DIV_MID, DIV_NEG, -c) : mix(DIV_MID, DIV_POS, c)
}
