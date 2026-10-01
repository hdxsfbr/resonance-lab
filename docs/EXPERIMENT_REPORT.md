# Initial experiment report

Date: 2026-10-01. All numbers below were produced by the headless runner
(`python -m resonance.cli experiments`, plus `backend/scripts/ablations.py`) on
the default configuration in `config/default.yaml`, seeds 1–10, 120 episodes unless stated.
Raw per-run data: `data/exports/experiment_*.json` and `.csv`.

**Reading guide.** Runs (seeds) are the unit of analysis; turns within a run are correlated.
"Final block" is the mean coordination score over the last 20 % of episodes. Chance with
partial credit is 0.326 (exact match scores 1.0; otherwise Jaccard overlap of onsets × 0.5).
State variables are engineered quantities with the operational definitions in
`docs/DEFINITIONS.md`. Nothing here is evidence of feelings; everything here is a statement
about implemented mechanisms and measured behaviour.

## 1. Six matched conditions

| condition | first block | final block | success rate | per-run final block |
|---|---|---|---|---|
| full | 0.365 ± 0.145 | 0.608 ± 0.145 | 0.448 ± 0.100 | 0.61 0.44 0.45 0.59 0.69 0.81 0.64 0.68 0.70 0.47 |
| no_history | 0.300 ± 0.107 | 0.308 ± 0.045 | 0.228 ± 0.039 | flat at chance |
| state_fixed | 0.354 ± 0.117 | 0.636 ± 0.120 | 0.464 ± 0.064 | |
| state_decoupled | 0.354 ± 0.117 | 0.636 ± 0.120 | 0.464 ± 0.064 | identical to state_fixed |
| symbol | 0.353 ± 0.112 | 0.601 ± 0.098 | 0.429 ± 0.099 | |
| perturbed: transpose | 0.357 ± 0.141 | 0.629 ± 0.107 | 0.435 ± 0.095 | |
| perturbed: rhythm_shuffle | 0.365 ± 0.143 | 0.618 ± 0.141 | 0.439 ± 0.103 | |

Mean ± sd over 10 runs. Learning curve for `full` (10 blocks of 12 episodes):
0.33 0.40 0.53 0.47 0.55 0.47 0.59 0.57 0.55 0.66.

**What this shows.**

- Learning works and is variable. `full` rises from chance to ~0.61 with a seed range of
  0.44–0.81. `no_history` stays at chance. The receiver learns a mapping from measured phrase
  features to expected score; the sender learns which motif to send for which target. With 4
  targets and 8 motifs the two learners can settle into partial-pooling equilibria, which is
  where the seed-to-seed spread comes from. This is a signalling game, and that spread is the
  expected behaviour of simple reinforcement in one.
- **The moving state does not help coordination.** `state_fixed` and `state_decoupled` score
  slightly higher than `full` (0.636 vs 0.608; the difference is within one sd). In this
  design state reaches behaviour only through coupling, so the two controls coincide exactly.
  The coupling's main cost is the affiliation → learning-rate term (section 2).
- **The arbitrary-symbol control matches music.** `symbol` reaches 0.601, statistically
  indistinguishable from `full` at 120 episodes, and *beats* it at 300 episodes (0.833 ± 0.091
  vs 0.771 ± 0.116; `no_history` 0.326 ± 0.054). A one-hot identity is at least as easy to
  decode as a graded feature vector, and it is immune to expressive modulation. Music, in this
  task, does not carry *more* task information than a symbol; what it adds is graded
  similarity (section 3) and the acoustic path into state, which did not translate into
  better coordination.
- **Perturbations.** Expressive-structure changes (transpose, velocity_flatten, tempo_shift)
  leave coordination unchanged (0.63–0.67). `pitch_shuffle` is the only clearly damaging one
  (0.566 ± 0.108), consistent with the motif bank coding targets partly in contour and
  interval sequence. `rhythm_shuffle` has little effect because half of the bank's motifs have
  evenly spaced onsets, so shuffling their inter-onset intervals changes nothing. These are
  communication effects, not affective effects.

Full perturbation table (final block, 10 seeds):

| transpose | velocity_flatten | tempo_shift | contour_invert | rhythm_shuffle | pitch_shuffle |
|---|---|---|---|---|---|
| 0.629 ± 0.112 | 0.669 ± 0.094 | 0.638 ± 0.094 | 0.657 ± 0.117 | 0.618 ± 0.149 | 0.566 ± 0.108 |

## 2. Coupling ablations (`full` with one path removed, 10 seeds)

| ablation | final block | success rate |
|---|---|---|
| full (reference) | 0.608 ± 0.145 | 0.448 ± 0.100 |
| acoustic → activation drive off | 0.638 ± 0.127 | 0.447 ± 0.114 |
| affiliation → learning rate off | 0.654 ± 0.090 | 0.447 ± 0.075 |
| activation → tempo/velocity off | 0.640 ± 0.114 | 0.458 ± 0.094 |
| uncertainty/activation → temperature off | 0.638 ± 0.120 | 0.441 ± 0.077 |

Every coupling path costs a little coordination or is neutral; none helps. The engineered
state is therefore demonstrably *influential* (removing paths changes outcomes and the policy
trace shows the changed temperature), but its influence on this task is not beneficial. That
is an honest implementation result, not a failure of the lab: the hypothesis under test is
history dependence, not improved performance.

## 3. History dependence and persistence

| metric (agent A, condition `full`) | value |
|---|---|
| policy shift, receiver, KL(final ‖ initial) | 0.750 ± 0.095 |
| policy shift, sender, KL(final ‖ initial) | 1.263 ± 0.229 |
| activation persistence after a +0.3 pulse (steps to return within 0.05 of baseline) | 11.8 ± 2.4 |
| max policy probability, familiar (stored) motifs | 0.754 ± 0.052 |
| max policy probability, transformed (transposed + tempo-shifted) motifs | 0.754 ± 0.054 |
| generalization score on transposed motifs | 0.631 ± 0.065 |
| activation-trajectory correlation between agents | 0.808 ± 0.060 |

- Policies move far from their initial distributions in every learning condition and not at
  all in `no_history` (KL = 0).
- A state pulse decays in about 12 episodes with the default decay 0.15 and inertia 0.6; this
  is a direct consequence of the configured equations and is reported as a calibration, not a
  finding.
- The receiver treats transposed, tempo-shifted motifs the same as the originals: its
  confidence is unchanged and it still generalizes (0.63 vs 0.61 in training). This follows
  from learning over centred, largely transposition-invariant features. Familiarity in the
  sense of episodic memory does not alter the linear policy at all; a future experiment should
  give retrieved memories a direct route into the decision (section 5).
- Agent state similarity is high in every condition and exactly 1.0 under `symbol`. Both
  agents receive the same score each episode, so their `expected_value` and `uncertainty` are
  identical by construction; only `activation` differs, and only through the acoustic drive
  that the receiver alone gets. Treat `state_similarity` as a sanity check, not a result.

## 4. Presets (seed 7)

| preset | measured comparison | reading |
|---|---|---|
| first_encounter (60 episodes) | first block 0.47 → final 0.49; success 0.43 | At this seed, 60 episodes is too short to show learning; the batch shows the mean curve reaching 0.66 by episode 120. The live UI lets you keep stepping. |
| shared_history (120 episodes, then probe with the motif most often followed by success) | trained receiver: top pattern 2, entropy 1.11; memory-reset clone, same phrase/state/RNG draw: top pattern 0, entropy 1.39 | History changes the decision distribution (lower entropy, different argmax). At this seed the trained choice is *not* the pattern the motif most often preceded; the effect is history dependence, not correctness. |
| same_phrase_different_history (constructed) | P(pattern 2 ‖ motif) = 0.999 after a predictive history vs 0.260 after an uninformative one; entropy 0.007 vs 1.356; activation response +0.093 vs −0.056 | Built by the training schedule, as labelled. It demonstrates that the same phrase produces different behaviour and a different state response depending on outcome history. |

## 5. Caveats and confounds observed

1. `state_fixed` ≡ `state_decoupled` in behaviour, by design. Keep both: one shows the state
   trajectory, the other hides it, which matters for the inspector, not for scores.
2. `expected_value` and `uncertainty` are shared between agents because the score is shared.
   Per-agent asymmetry needs per-agent sensitivity or baselines, which the config allows.
3. The acoustic → activation drive is hand-authored. With it off, history dependence of state
   comes only through prediction errors and outcomes, and coordination is unchanged.
4. Learning-rate scaling by affiliation lowers scores slightly: the learner overshoots when
   affiliation is high. A smaller gain or a ceiling is a reasonable next default.
5. The bundled 0.5 B-parameter local model, when enabled as a receiver, chose pattern 0 in both
   recorded calls regardless of the features (rationales are repetitive). It is a real,
   tested integration and a weak decision-maker; its outputs are labelled generated.

## 6. Next experiments, in priority order

1. **Give memory a route into decisions.** Add a retrieval-weighted prior (k-nearest stored
   outcomes) to the receiver's scores so that familiar motifs can behave differently from
   unfamiliar ones. Then re-run `familiar_vs_unfamiliar`; a difference would be the first
   genuinely memory-based effect this lab could show.
2. **Break the symmetry.** Give agents different sensitivities and baselines, add per-agent
   noise to the outcome signal (e.g. a private cost), and re-measure state similarity and
   affiliation dynamics.
3. **Make expressive modulation informative.** Let the sender's activation carry *task-relevant*
   information (e.g. target-dependent tempo) and test whether the receiver learns to read it;
   compare against `symbol`, which cannot carry it.
4. **Longer horizons and more targets.** Run 6–8 patterns for 300–600 episodes to see whether
   music's graded similarity helps when the message space is larger than the motif bank.
5. **Replace the task.** Use the `Task` protocol to try a shared-navigation task where timing
   matters continuously, so partial credit is graded rather than set-based.
