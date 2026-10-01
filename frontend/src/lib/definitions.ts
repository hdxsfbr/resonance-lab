/**
 * Operational definitions shown in the UI. Text is copied from
 * backend/resonance/schemas.py (StateVector docstring, Field descriptions,
 * Perturbation comments) and docs/CONTRACTS.md. Keep in sync when they change.
 *
 * Wording rule: these are ENGINEERED quantities. Never relabel them with
 * experiential words.
 */
import type { ConditionName, Perturbation, StateDim, InterventionKind } from '../api/types'

export const STATE_DIMS: StateDim[] = ['activation', 'expected_value', 'uncertainty', 'affiliation']

export interface StateDimDef {
  dim: StateDim
  label: string
  range: [number, number]
  /** Verbatim from schemas.py StateVector docstring. */
  definition: string
}

export const STATE_DEFS: Record<StateDim, StateDimDef> = {
  activation: {
    dim: 'activation',
    label: 'activation',
    range: [0, 1],
    definition:
      'drive/arousal-like scalar. Raised by dense/loud input (if the hand-authored acoustic influence is enabled) and by surprising outcomes; decays toward baseline. Modulates expressive tempo/velocity and exploration temperature when coupling is enabled.',
  },
  expected_value: {
    dim: 'expected_value',
    label: 'expected_value',
    range: [0, 1],
    definition: 'running expectation of the coordination score.',
  },
  uncertainty: {
    dim: 'uncertainty',
    label: 'uncertainty',
    range: [0, 1],
    definition: 'running magnitude of recent prediction error.',
  },
  affiliation: {
    dim: 'affiliation',
    label: 'affiliation',
    range: [-1, 1],
    definition:
      'running credit toward the partner: rises with shared success, falls with shared failure. Scales effective learning rate when coupled.',
  },
}

export const STATE_FOOTNOTE =
  'Engineered quantity with an operational definition (backend/resonance/schemas.py · StateVector). Not a measurement of experience.'

/** Fixed order of the learner feature vector phi(phrase) (schemas.FEATURE_NAMES). */
export const FEATURE_NAMES = [
  'note_density',
  'rhythmic_regularity',
  'syncopation',
  'mean_pitch',
  'pitch_range',
  'contour',
  'repetition',
  'iv_unison',
  'iv_step',
  'iv_third',
  'iv_fourth_fifth',
  'iv_large',
  'mean_velocity',
  'velocity_variation',
  'dissonance_proxy',
  'tempo',
] as const

export interface ConditionDef {
  name: ConditionName
  summary: string
}

/** One-line descriptions derived from the Conditions table in docs/CONTRACTS.md. */
export const CONDITION_DEFS: ConditionDef[] = [
  { name: 'full', summary: 'Learning on, memory on, state updates on, coupling on; music channel. Reference condition.' },
  { name: 'no_history', summary: 'Learning off, memory capacity 0; state updates and coupling on; music channel.' },
  { name: 'state_fixed', summary: 'Learning and memory on; state held at baseline (no updates), so coupling has nothing to act on; music channel.' },
  { name: 'state_decoupled', summary: 'Learning, memory and state updates on; state cannot influence the policy (coupling off); music channel.' },
  {
    name: 'symbol',
    summary:
      "Receiver gets a one-hot of the sender's motif index instead of a phrase (identity preserved, no acoustic drive, no graded similarity).",
  },
  { name: 'perturbed', summary: 'Everything on; a transform is applied to every phrase in the channel (choose the kind).' },
]

export interface PerturbationDef {
  kind: Perturbation
  description: string
  /** Whether the rhythmic task message survives the transform (schemas.py comments + CONTROLS_AND_CONFOUNDS.md). */
  preservesMessage: 'yes' | 'partly' | 'no' | 'n/a'
}

export const PERTURBATION_DEFS: PerturbationDef[] = [
  { kind: 'none', description: 'No transform.', preservesMessage: 'n/a' },
  { kind: 'transpose', description: '+5 semitones; preserves intervals, contour, rhythm (expressive-structure change).', preservesMessage: 'yes' },
  { kind: 'velocity_flatten', description: 'All velocities -> 80; removes dynamics.', preservesMessage: 'yes' },
  { kind: 'tempo_shift', description: "Tempo x 1.25; changes notes/sec but not beat-relative structure.", preservesMessage: 'yes' },
  { kind: 'contour_invert', description: 'Mirror pitches around mean; destroys contour, keeps rhythm.', preservesMessage: 'partly' },
  { kind: 'rhythm_shuffle', description: 'Permute inter-onset intervals; destroys rhythmic message content.', preservesMessage: 'no' },
  { kind: 'pitch_shuffle', description: 'Permute pitch order; destroys contour and interval sequence.', preservesMessage: 'partly' },
]

export function perturbationDef(kind: Perturbation): PerturbationDef {
  return PERTURBATION_DEFS.find((p) => p.kind === kind) ?? PERTURBATION_DEFS[0]
}

export const SCORING_RULE =
  "score in [0,1]: 1.0 if the receiver's chosen pattern equals the target; otherwise, when task.partial_credit is on, partial credit by onset overlap between chosen and target patterns (backend formula); 0 when partial credit is off."

export const MOCK_SCORING_RULE =
  'Mock formula: 0.5 x |onsets(chosen) ∩ onsets(target)| / |onsets(chosen) ∪ onsets(target)| for a wrong choice.'

export interface InterventionDef {
  kind: InterventionKind
  label: string
  semantics: string
  takesEffect: 'next step' | 'immediately'
}

/** From the Intervention docstring in schemas.py + Reset semantics in CONTRACTS.md. */
export const INTERVENTION_DEFS: Record<InterventionKind, InterventionDef> = {
  replay_motif: {
    kind: 'replay_motif',
    label: 'Replay motif',
    semantics: 'The chosen phrase is used as the NEXT sent phrase regardless of sender policy.',
    takesEffect: 'next step',
  },
  reset_memory: {
    kind: 'reset_memory',
    label: 'Clear memory',
    semantics: 'Clears episodic memory + learned associations; state untouched.',
    takesEffect: 'immediately',
  },
  reset_state: {
    kind: 'reset_state',
    label: 'Reset state',
    semantics: 'State := baseline; memory untouched.',
    takesEffect: 'immediately',
  },
  freeze_state: {
    kind: 'freeze_state',
    label: 'Freeze state',
    semantics: 'State updates are skipped while frozen.',
    takesEffect: 'next step',
  },
  set_coupling: {
    kind: 'set_coupling',
    label: 'State→behaviour coupling',
    semantics: 'When off, state may evolve but cannot influence the policy (temperature, expression, learning rate).',
    takesEffect: 'next step',
  },
  swap_feature: {
    kind: 'swap_feature',
    label: 'Swap musical characteristic',
    semantics: "Transform applied to every subsequent sent phrase until set to 'none'.",
    takesEffect: 'next step',
  },
  set_param: {
    kind: 'set_param',
    label: 'Set parameter',
    semantics: 'Live config change at a dotted path (e.g. state.decay.activation).',
    takesEffect: 'next step',
  },
}

export const FOOTER_NOTE =
  'Engineered agent simulation. State labels are operational definitions, not claims about experience. Narrative text, where shown, is generated and is not a measurement.'

export const EXPERIMENT_CAVEAT =
  'Runs (seeds) are the unit of analysis; turns within a run are correlated. Controls may match or beat the full condition; this is reported as-is.'

export const HAND_AUTHORED_ACOUSTIC =
  'HAND-AUTHORED: dense/loud phrases push activation up. Switchable; not a learned association.'
