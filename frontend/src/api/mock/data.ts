/**
 * MOCK data source content: motif bank, timing patterns, default config,
 * presets. These are demo stand-ins for the backend's resonance/music/motifs.py,
 * env/timing task and presets.py. Everything served from here is labelled
 * "DEMO DATA (mock)" in the UI.
 */
import type { ExperimentConfig, Note, PresetInfo, TimingPattern } from '../types'

export const MOCK_TEMPO = 112

export const MOCK_PATTERNS: TimingPattern[] = [
  { id: 0, name: 'four on the floor', onsets: [0, 1, 2, 3], length_beats: 4 },
  { id: 1, name: 'offbeats', onsets: [0.5, 1.5, 2.5, 3.5], length_beats: 4 },
  { id: 2, name: 'tresillo', onsets: [0, 1.5, 3], length_beats: 4 },
  { id: 3, name: 'gallop', onsets: [0, 0.75, 1, 2, 2.75, 3], length_beats: 4 },
]

function seq(pitches: number[], onsets: number[], dur: number | number[], vel: number | number[]): Note[] {
  return pitches.map((p, i) => ({
    pitch: p,
    onset: onsets[i],
    duration: Array.isArray(dur) ? dur[i] : dur,
    velocity: Array.isArray(vel) ? vel[i] : vel,
  }))
}

const range = (n: number, step = 1, start = 0) => Array.from({ length: n }, (_, i) => start + i * step)

export interface MockMotif {
  id: string
  notes: Note[]
}

/** 8 base motifs over 8 beats with deliberately different measured features. */
export const MOCK_MOTIFS: MockMotif[] = [
  { id: 'm0_rising_steps', notes: seq([60, 62, 64, 65, 67, 69, 71, 72], range(8), 0.9, 84) },
  {
    id: 'm1_falling_eighths',
    notes: seq([79, 77, 76, 74, 72, 71, 69, 67, 65, 64, 62, 60], range(12, 0.5), 0.45, 92),
  },
  { id: 'm2_arch', notes: seq([60, 64, 67, 72, 67, 64, 60], range(7), 0.9, [70, 80, 90, 100, 90, 80, 70]) },
  {
    id: 'm3_syncopated_valley',
    notes: seq([72, 67, 64, 60, 64, 67, 72], [0, 0.75, 1.5, 2.25, 3.75, 4.5, 5.25], 0.7, 88),
  },
  {
    id: 'm4_pulse_accents',
    notes: seq([67, 67, 67, 67, 67, 67, 67, 67], range(8), 0.5, [112, 58, 112, 58, 112, 58, 112, 58]),
  },
  { id: 'm5_sparse_low', notes: seq([48, 55, 48], [0, 3, 6], [2.5, 2.5, 1.8], 52) },
  {
    id: 'm6_zigzag_leaps',
    notes: seq([60, 72, 62, 74, 64, 76, 65, 77, 67, 79], range(10, 0.75), 0.6, 110),
  },
  {
    id: 'm7_dotted_gallop',
    notes: seq([64, 66, 64, 66, 64, 66, 64, 66, 70, 71, 70, 66], [0, 0.75, 1, 2, 2.75, 3, 4, 4.75, 5, 6, 6.75, 7], [0.6, 0.2, 0.9, 0.6, 0.2, 0.9, 0.6, 0.2, 0.9, 0.6, 0.2, 0.9], 96),
  },
]

export function mockDefaultConfig(): ExperimentConfig {
  return {
    seed: 7,
    episodes: 120,
    state: {
      baseline: { activation: 0.5, expected_value: 0.5, uncertainty: 0.5, affiliation: 0.0 },
      inertia: { activation: 0.6, expected_value: 0.8, uncertainty: 0.7, affiliation: 0.85 },
      decay: { activation: 0.15, expected_value: 0.02, uncertainty: 0.05, affiliation: 0.03 },
      max_step: { activation: 0.3, expected_value: 0.3, uncertainty: 0.3, affiliation: 0.3 },
      acoustic_activation_enabled: true,
      acoustic_activation_gain: 0.5,
      prediction_error_gain: { activation: 0.4, uncertainty: 1.0 },
      outcome_gain: { expected_value: 1.0, affiliation: 0.8 },
      partner_similarity_gain: 0.2,
    },
    coupling: {
      enabled: true,
      uncertainty_to_temperature: 1.0,
      activation_to_temperature: 0.6,
      activation_to_tempo: 0.3,
      activation_to_velocity: 30.0,
      affiliation_to_learning_rate: 0.5,
      temperature_min: 0.05,
      temperature_max: 5.0,
    },
    // MOCK defaults differ from config/default.yaml (0.15 / 0.2 / 0.35) so that the
    // in-browser demo shows learning within ~120 episodes. Shown truthfully in the config popover.
    learning: {
      enabled: true,
      receiver_learning_rate: 0.3,
      sender_learning_rate: 0.5,
      temperature_base: 0.08,
      receiver_l2: 0.001,
      optimistic_init: 0.5,
      receiver_init_scale: 0.0,
    },
    memory: { capacity: 200, retrieval_k: 5 },
    music: {
      phrase_length_beats: 8,
      tempo_min: 80,
      tempo_max: 160,
      motif_bank: 'default',
      n_motifs: 8,
      feature_weights: {},
      generation_noise: 0.05,
    },
    task: { kind: 'timing', n_patterns: 4, partial_credit: true },
    model: {
      provider: 'none',
      model_id: '',
      base_url: null,
      api_key_env: '',
      timeout_seconds: 20,
      max_retries: 1,
      call_budget: 200,
      temperature: 0.2,
      use_for: [],
      narrative_enabled: false,
      prompts_dir: 'prompts',
      local_model_path: 'models/qwen2.5-0.5b-instruct-q4_k_m.gguf',
      n_threads: 4,
    },
    agents: [
      { id: 'A', name: 'Aria', color: '#7c9cff', instrument: 'pluck', baseline_override: null, sensitivity: 1.0, policy_kind: 'local' },
      { id: 'B', name: 'Bram', color: '#ffb86b', instrument: 'marimba', baseline_override: null, sensitivity: 1.0, policy_kind: 'local' },
    ],
  }
}

export const MOCK_PRESETS: PresetInfo[] = [
  {
    name: 'first_encounter',
    title: 'First encounter (mock)',
    summary: 'Two fresh agents with no shared history start exchanging phrases.',
    manipulation: 'None: fresh agents, full condition, 60 episodes.',
    evidence_criterion: 'Final-block mean score exceeds the first-block mean and the chance level for uniform guessing.',
    constructed_note: '',
    default_episodes: 60,
    condition: { name: 'full', perturbation: 'none', description: 'reference condition' },
  },
  {
    name: 'shared_history',
    title: 'Shared history vs memory-reset control (mock)',
    summary: 'After 80 shared episodes, one branch keeps its history and a matched branch has both memories cleared.',
    manipulation: 'reset_memory on both agents at episode 80 in the control branch; same seed in both branches.',
    evidence_criterion: 'Mean score over the next 40 episodes is higher with history than after the memory reset.',
    constructed_note: '',
    default_episodes: 120,
    condition: { name: 'full', perturbation: 'none', description: 'reference condition' },
  },
  {
    name: 'same_phrase_different_history',
    title: 'Same phrase, different history (constructed, mock)',
    summary: 'Two receivers are trained on different phrase→pattern pairings, then hear the identical probe phrase.',
    manipulation: 'Training schedule: the same motif is paired with target 0 for receiver 1 and with target 2 for receiver 2 (40 episodes each).',
    evidence_criterion: 'Receivers assign different choice probabilities to the identical probe phrase.',
    constructed_note: 'Constructed: the difference is produced by the training schedule. It demonstrates history-dependence of the learned policy; it is not a discovery.',
    default_episodes: 40,
    condition: { name: 'full', perturbation: 'none', description: 'reference condition' },
  },
]
