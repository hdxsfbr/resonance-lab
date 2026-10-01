/** Visible provenance label: MEASURED / ENGINEERED / GENERATED / HAND-AUTHORED / MOCK. */
export type Kind = 'measured' | 'engineered' | 'generated' | 'hand' | 'learned' | 'mock'

const TEXT: Record<Kind, string> = {
  measured: 'MEASURED',
  engineered: 'ENGINEERED',
  generated: 'GENERATED',
  hand: 'HAND-AUTHORED',
  learned: 'LEARNED',
  mock: 'MOCK',
}

const TITLE: Record<Kind, string> = {
  measured: 'Computed deterministically from the notes or the rendered waveform.',
  engineered: 'Engineered internal quantity with an operational definition; not a measurement of experience.',
  generated: 'Model output or narrative text. Not a measurement; never read by the simulation.',
  hand: 'Hand-authored dynamics chosen by the designers; switchable; not learned.',
  learned: 'Parameters fitted by the agent from its history (learned associations).',
  mock: 'Produced by the in-browser mock data source.',
}

export function KindTag({ kind }: { kind: Kind }) {
  return (
    <span className={`kindtag kindtag--${kind}`} title={TITLE[kind]}>
      {TEXT[kind]}
    </span>
  )
}
