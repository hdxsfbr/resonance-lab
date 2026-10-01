import type { StateDim, StateVector } from '../api/types'
import { STATE_DEFS, STATE_DIMS, STATE_FOOTNOTE } from '../lib/definitions'
import { DIM_COLORS } from '../lib/colors'
import { fmt, signed } from '../lib/format'
import { InfoTip } from './InfoTip'

/** Four labelled bars with numeric values, baseline tick and per-dim operational definition. */
export function StateBars({ state, baseline, delta }: { state: StateVector; baseline: StateVector; delta?: Partial<Record<StateDim, number>> }) {
  return (
    <div className="statebars">
      {STATE_DIMS.map((d) => {
        const def = STATE_DEFS[d]
        const [lo, hi] = def.range
        const pos = (v: number) => ((v - lo) / (hi - lo)) * 100
        const v = state[d]
        const zero = pos(Math.max(lo, 0))
        const left = Math.min(zero, pos(v))
        const width = Math.abs(pos(v) - zero)
        const dd = delta?.[d]
        return (
          <div className="statebar" key={d} data-testid={`statebar-${d}`}>
            <div className="statebar__head">
              <span className="statebar__name">
                <span className="swatch" style={{ background: DIM_COLORS[d] }} aria-hidden />
                {def.label}
              </span>
              <InfoTip label={`${def.label} definition`}>
                <strong>
                  {def.label} [{lo}, {hi}]
                </strong>
                <br />
                {def.definition}
                <br />
                <em>{STATE_FOOTNOTE}</em>
              </InfoTip>
              <span className="statebar__val mono">{fmt(v, 3)}</span>
            </div>
            <div className="statebar__line">
              <div className="statebar__track" aria-hidden>
                {lo < 0 && <span className="statebar__zero" style={{ left: `${zero}%` }} />}
                <span className="statebar__fill" style={{ left: `${left}%`, width: `${width}%`, background: DIM_COLORS[d] }} />
                <span className="statebar__base" style={{ left: `${pos(baseline[d])}%` }} title={`baseline ${fmt(baseline[d], 2)}`} />
              </div>
              <span className={`statebar__delta mono${dd === undefined || Math.abs(dd) < 5e-4 ? ' statebar__delta--zero' : ''}`} title="change in the shown step">
                {dd === undefined ? '' : signed(dd, 3)}
              </span>
            </div>
          </div>
        )
      })}
    </div>
  )
}
