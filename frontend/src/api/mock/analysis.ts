/** Mock experiment runner, preset runner and replay session (DEMO DATA only). */
import type {
  ConditionAggregate,
  ConditionSpec,
  Event,
  ExperimentConfig,
  ExperimentRequest,
  ExperimentResult,
  Phrase,
  PresetResult,
  RunExport,
  RunMetrics,
  SessionSnapshot,
  StateVector,
} from '../types'
import { mockFeatures as computeSymbolicFeatures } from './normalise'
import { blockMeans, kl, mean, pearson, sd, softmax } from '../../lib/stats'
import { EXPERIMENT_CAVEAT } from '../../lib/definitions'
import { MOCK_MOTIFS, MOCK_TEMPO } from './data'
import { Rng } from './rng'
import { MockSim, scorePatterns } from './sim'
import { transformPhrase } from './transforms'
import { isStateVector } from '../events'

export function basePhrase(i: number, instrument: Phrase['instrument'] = 'pluck', lengthBeats = 8): Phrase {
  const m = MOCK_MOTIFS[i]
  return {
    id: `motif-${m.id}`,
    notes: m.notes.map((n) => ({ ...n })),
    tempo_bpm: MOCK_TEMPO,
    length_beats: lengthBeats,
    instrument,
    origin: { kind: 'preset', agent_id: null, source_phrase_id: null, transform: null },
    motif_id: m.id,
    tags: ['base_motif', 'mock'],
  }
}

function argmax(xs: number[]): number {
  let b = 0
  for (let i = 1; i < xs.length; i++) if (xs[i] > xs[b]) b = i
  return b
}

function chanceLevel(sim: MockSim): number {
  const P = sim.patterns
  const partial = sim.config.task?.partial_credit ?? true
  return mean(P.map((t) => mean(P.map((c) => scorePatterns(c, t, partial)))))
}

function receiverInput(sim: MockSim, phrase: Phrase, motifIndex: number): number[] {
  return sim.eff.channel === 'symbol' ? sim.symbolVector(motifIndex) : computeSymbolicFeatures(phrase).vector
}

export function runMetrics(sim: MockSim): RunMetrics {
  const s = sim.scores
  const n = s.length
  const block = Math.max(1, Math.round(n * 0.2))
  const tauBase = sim.config.learning?.temperature_base ?? 0.35
  const nM = sim.nMotifs
  const policy_shift_kl: Record<string, number> = {}
  const persistence: Record<string, number> = {}
  const famUnfam: Record<string, number> = {}
  const rng = new Rng(sim.seed + 991)
  for (const a of sim.agents) {
    const uniformS = Array(nM).fill(1 / nM)
    const senderKl = mean(a.Q.map((row) => kl(softmax(row, tauBase), uniformS)))
    const nP = sim.patterns.length
    const uniformR = Array(nP).fill(1 / nP)
    const fam: number[] = []
    const unfam: number[] = []
    const recvKl: number[] = []
    for (let i = 0; i < nM; i++) {
      const ph = basePhrase(i)
      const p = softmax(sim.receiverScores(a, receiverInput(sim, ph, i)), tauBase)
      recvKl.push(kl(p, uniformR))
      fam.push(Math.max(...p))
      const tp = transformPhrase(ph, 'contour_invert', rng)
      unfam.push(Math.max(...softmax(sim.receiverScores(a, receiverInput(sim, tp, i)), tauBase)))
    }
    policy_shift_kl[a.id] = Number(((senderKl + mean(recvKl)) / 2).toFixed(4))
    famUnfam[`${a.id}_familiar`] = Number(mean(fam).toFixed(4))
    famUnfam[`${a.id}_unfamiliar`] = Number(mean(unfam).toFixed(4))
    // probe pulse: activation := 1, then decay only, count steps to within 0.05 of baseline
    if (!sim.eff.stateUpdates) persistence[a.id] = 0
    else {
      const b = a.baseline.activation
      const d = sim.config.state?.decay?.activation ?? 0.15
      let x = 1
      let k = 0
      while (Math.abs(x - b) >= 0.05 && k < 500) {
        x += d * (b - x)
        k++
      }
      persistence[a.id] = k
    }
  }
  // generalisation: greedy sender motif per target, transposed, greedy receiver choice
  const gen: number[] = []
  const partial = sim.config.task?.partial_credit ?? true
  for (const [si, ri] of [
    [0, 1],
    [1, 0],
  ]) {
    const S = sim.agents[si]
    const R = sim.agents[ri]
    if (!S || !R) continue
    for (const t of sim.patterns) {
      const m = argmax(S.Q[t.id])
      const tp = transformPhrase(basePhrase(m), 'transpose', rng)
      const c = argmax(sim.receiverScores(R, receiverInput(sim, tp, m)))
      gen.push(scorePatterns(sim.patterns[c], t, partial))
    }
  }
  const actA = sim.agents[0]?.stateHistory.map((v) => v.activation) ?? []
  const actB = sim.agents[1]?.stateHistory.map((v) => v.activation) ?? []
  return {
    seed: sim.seed,
    condition: sim.condition.name,
    perturbation: sim.condition.perturbation,
    episodes: n,
    mean_score: Number(mean(s).toFixed(4)),
    final_block_score: Number(mean(s.slice(-block)).toFixed(4)),
    first_block_score: Number(mean(s.slice(0, block)).toFixed(4)),
    success_rate: Number((n ? s.filter((x) => x >= 0.999).length / n : 0).toFixed(4)),
    learning_curve: blockMeans(s, 10).map((x) => Number(x.toFixed(4))),
    policy_shift_kl,
    state_effect_persistence: persistence,
    familiar_vs_unfamiliar: famUnfam,
    generalization_score: Number(mean(gen).toFixed(4)),
    state_similarity: Number(pearson(actA, actB).toFixed(4)),
    extra: { chance_level: Number(chanceLevel(sim).toFixed(4)) },
  }
}

export function aggregate(runs: RunMetrics[]): ConditionAggregate[] {
  const groups = new Map<string, RunMetrics[]>()
  for (const r of runs) {
    const key = `${r.condition}|${r.perturbation}`
    groups.set(key, [...(groups.get(key) ?? []), r])
  }
  return [...groups.values()].map((rs) => {
    const curveLen = Math.max(...rs.map((r) => r.learning_curve.length))
    const curve = Array.from({ length: curveLen }, (_, i) => rs.map((r) => r.learning_curve[i] ?? 0))
    const fb = rs.map((r) => r.final_block_score)
    const sr = rs.map((r) => r.success_rate)
    return {
      condition: rs[0].condition,
      perturbation: rs[0].perturbation,
      n_runs: rs.length,
      mean_final_block_score: Number(mean(fb).toFixed(4)),
      sd_final_block_score: Number(sd(fb).toFixed(4)),
      mean_success_rate: Number(mean(sr).toFixed(4)),
      sd_success_rate: Number(sd(sr).toFixed(4)),
      mean_learning_curve: curve.map((c) => Number(mean(c).toFixed(4))),
      sd_learning_curve: curve.map((c) => Number(sd(c).toFixed(4))),
      per_run_final_block: fb,
    }
  })
}

export function runMockExperiment(req: ExperimentRequest, base: ExperimentConfig, id: string, clock: () => string): ExperimentResult {
  const cfg = req.config ?? base
  const runs: RunMetrics[] = []
  for (const cond of req.conditions) {
    for (const seed of req.seeds) {
      const sim = new MockSim({ id: `${id}-${cond.name}-${seed}`, seed, config: cfg, condition: cond, recordEvents: false, clock })
      sim.stepN(cfg.episodes)
      runs.push(runMetrics(sim))
    }
  }
  return {
    id,
    name: req.name,
    created_at: clock(),
    config: cfg,
    runs,
    aggregates: aggregate(runs),
    notes: [
      'DEMO DATA (mock): produced by the in-browser mock simulation, not the backend.',
      EXPERIMENT_CAVEAT,
      'final_block_score = mean score over the last 20% of episodes; learning_curve = 10 block means.',
    ],
  }
}

export function experimentCsv(r: ExperimentResult): string {
  const head = ['experiment_id', 'condition', 'perturbation', 'seed', 'episodes', 'mean_score', 'first_block_score', 'final_block_score', 'success_rate', 'generalization_score', 'state_similarity']
  const rows = r.runs.map((m) =>
    [r.id, m.condition, m.perturbation, m.seed, m.episodes, m.mean_score, m.first_block_score, m.final_block_score, m.success_rate, m.generalization_score ?? '', m.state_similarity ?? ''].join(','),
  )
  return [head.join(','), ...rows].join('\n') + '\n'
}

export function runMockPreset(
  name: string,
  seed: number,
  base: ExperimentConfig,
  newId: () => string,
  clock: () => string,
  register: (sim: MockSim) => void,
): PresetResult {
  const full: ConditionSpec = { name: 'full', perturbation: 'none', description: '' }
  const mk = (preset: string) => {
    const sim = new MockSim({ id: newId(), seed, config: base, condition: full, preset, clock })
    register(sim)
    return sim
  }
  if (name === 'first_encounter') {
    const sim = mk(name)
    sim.stepN(60)
    const m = runMetrics(sim)
    const chance = chanceLevel(sim)
    return {
      preset: name,
      seed,
      sessions: [sim.summary()],
      comparison: {
        labels: {
          first_block_score: 'Mean score, first 20% of episodes',
          final_block_score: 'Mean score, last 20% of episodes',
          chance_level: 'Expected score for uniform guessing',
          success_rate: 'Fraction of exact matches',
          criterion_met: 'Evidence criterion met',
        },
        first_block_score: m.first_block_score,
        final_block_score: m.final_block_score,
        chance_level: Number(chance.toFixed(4)),
        success_rate: m.success_rate,
        criterion_met: m.final_block_score > m.first_block_score && m.final_block_score > chance,
      },
      narrative: '[templated, not model-generated] Compares early and late coordination scores of one mock run against chance.',
    }
  }
  if (name === 'shared_history') {
    const kept = mk(name)
    const reset = mk(name)
    kept.stepN(80)
    reset.stepN(80)
    for (const a of reset.agents) reset.intervene({ kind: 'reset_memory', agent_id: a.id })
    kept.stepN(40)
    reset.stepN(40)
    const withH = mean(kept.scores.slice(80))
    const without = mean(reset.scores.slice(80))
    return {
      preset: name,
      seed,
      sessions: [kept.summary(), reset.summary()],
      comparison: {
        labels: {
          with_history_next40: 'Mean score, episodes 81-120, history kept',
          memory_reset_next40: 'Mean score, episodes 81-120, after reset_memory on both agents',
          difference: 'Difference (kept − reset)',
          criterion_met: 'Evidence criterion met',
        },
        with_history_next40: Number(withH.toFixed(4)),
        memory_reset_next40: Number(without.toFixed(4)),
        difference: Number((withH - without).toFixed(4)),
        criterion_met: withH > without,
      },
      narrative: '[templated, not model-generated] Same seed in both branches; only the memory reset differs.',
    }
  }
  if (name === 'same_phrase_different_history') {
    const probe = basePhrase(3, 'pluck')
    const s1 = mk(name)
    const s2 = mk(name)
    for (let i = 0; i < 40; i++) {
      s1.stepOnce({ phrase: { ...probe, id: `${probe.id}-t${i}` }, target_id: 0, receiver_id: 'B' })
      s2.stepOnce({ phrase: { ...probe, id: `${probe.id}-t${i}` }, target_id: 2, receiver_id: 'B' })
    }
    const x = computeSymbolicFeatures(probe).vector
    const tau = base.learning?.temperature_base ?? 0.35
    const p1 = softmax(s1.receiverScores(s1.agentById('B'), x), tau).map((v) => Number(v.toFixed(4)))
    const p2 = softmax(s2.receiverScores(s2.agentById('B'), x), tau).map((v) => Number(v.toFixed(4)))
    const tv = 0.5 * p1.reduce((acc, v, i) => acc + Math.abs(v - p2[i]), 0)
    return {
      preset: name,
      seed,
      sessions: [s1.summary(), s2.summary()],
      comparison: {
        labels: {
          probe_motif: 'Probe phrase (identical for both receivers)',
          history_1_probabilities: 'Receiver B, history 1 (paired with pattern 0): P(pattern)',
          history_2_probabilities: 'Receiver B, history 2 (paired with pattern 2): P(pattern)',
          total_variation_distance: 'Total variation distance between the two distributions',
          constructed: 'Result constructed by the training schedule',
        },
        probe_motif: probe.motif_id,
        history_1_probabilities: p1,
        history_2_probabilities: p2,
        total_variation_distance: Number(tv.toFixed(4)),
        constructed: true,
      },
      narrative: '[templated, not model-generated] The difference is built by the training schedule.',
    }
  }
  throw new Error(`unknown preset ${name}`)
}

/** Replay of an imported RunExport: re-emits recorded events episode by episode. */
export class MockReplay {
  readonly id: string
  readonly run: RunExport
  readonly steps: number[]
  cursor = 0
  emitted: Event[] = []

  constructor(id: string, run: RunExport) {
    this.id = id
    this.run = run
    const s = new Set<number>()
    for (const e of run.events) if (e.type === 'episode_complete' || e.type === 'outcome') s.add(e.step)
    this.steps = [...s].sort((a, b) => a - b)
    this.emitted = run.events.filter((e) => e.type === 'session_created')
  }

  replayStep(n: number): Event[] {
    const take = this.steps.slice(this.cursor, this.cursor + n)
    this.cursor += take.length
    const set = new Set(take)
    const evs = this.run.events.filter((e) => set.has(e.step) && e.type !== 'session_created')
    this.emitted.push(...evs)
    return evs
  }

  snapshot(): SessionSnapshot {
    const f = this.run.final_snapshot
    const scores: number[] = []
    const lastState: Record<string, StateVector> = {}
    let current: SessionSnapshot['current_episode'] = null
    for (const e of this.emitted) {
      const p = e.payload ?? {}
      if (e.type === 'outcome' && typeof p.score === 'number') scores.push(p.score)
      if (e.type === 'state_update' && e.agent_id && isStateVector(p.after)) lastState[e.agent_id] = p.after
      if (e.type === 'episode_complete' && p.episode && typeof p.episode === 'object') current = p.episode as SessionSnapshot['current_episode']
    }
    const done = this.cursor >= this.steps.length
    return {
      ...f,
      id: this.id,
      mode: 'replay',
      step: this.cursor === 0 ? 0 : this.steps[this.cursor - 1] + 1,
      status: done ? 'finished' : this.cursor === 0 ? 'ready' : 'running',
      agents: f.agents.map((a) => ({ ...a, state: lastState[a.id] ?? a.baseline })),
      current_episode: current,
      metrics: {
        episodes: scores.length,
        mean_score: mean(scores),
        rolling_score: mean(scores.slice(-20)),
        success_rate: scores.length ? scores.filter((x) => x >= 0.999).length / scores.length : 0,
        score_history: scores,
      },
      event_count: this.emitted.length,
    }
  }
}
