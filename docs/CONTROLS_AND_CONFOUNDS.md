# Controls and known confounds

Six conditions run on matched seeds (same target sequence, same exploration draws where
the policy is unchanged). Runs, not turns, are the unit of analysis.

| Condition | What is manipulated | What a difference from `full` can show | What it cannot show |
|---|---|---|---|
| `full` | nothing; music, memory, state updates and coupling all on | reference | — |
| `no_history` | learning off, memory capacity 0 | the contribution of learned history (both sender values and receiver weights) | which part of history matters |
| `state_fixed` | state held at baseline every step | whether a *moving* state changes outcomes at all | whether music or outcomes moved it |
| `state_decoupled` | state updates normally but cannot influence temperature, expression or learning rate | whether the state→behaviour path matters, with identical state trajectories available for inspection | — |
| `symbol` | the receiver gets a one-hot of the sender's motif index instead of a phrase | whether the graded, feature-based musical message adds or costs anything relative to an arbitrary, perfectly identifying symbol | — |
| `perturbed(kind)` | a transform applied in the channel to every phrase | separates expressive-structure changes from message-destroying changes (table below) | — |

## The symbol control is deliberately strong

The symbol carries the message *identity* exactly (same cardinality as the motif bank, same
action space, same learner and update rule, same targets and seeds). It is matched on
information and *not* matched on: graded similarity between messages (one-hot has none),
expressive modulation (none), and the hand-authored acoustic drive on activation (none,
there is no acoustic input). Because identity decoding is at least as easy as feature
decoding, `symbol` may equal or beat `full` on coordination. That result is reported as-is.

## Perturbations

| kind | Preserves | Destroys | Classification |
|---|---|---|---|
| `transpose` | intervals, contour, rhythm, dynamics | absolute pitch | expressive-structure change |
| `velocity_flatten` | pitches, rhythm | dynamics | expressive-structure change |
| `tempo_shift` | beat-relative structure | notes-per-second, duration | expressive-structure change (alters density features) |
| `contour_invert` | rhythm, pitch set size | contour, interval signs | partially message-destroying |
| `rhythm_shuffle` | pitch sequence, note count | inter-onset structure, regularity, syncopation | message-destroying for rhythm-coded motifs |
| `pitch_shuffle` | rhythm, pitch set | contour, interval sequence | message-destroying for contour-coded motifs |

A drop in coordination under a message-destroying perturbation is a communication effect,
not evidence of an affective effect. Compare perturbations against `full` *and* against each
other before interpreting any state difference.

## Known confounds

1. **Expressive modulation changes measured features.** When coupling is on, activation
   scales tempo and velocity, which changes `note_density`, `mean_velocity` and `tempo` in
   the receiver's feature vector. The receiver may learn to ignore them, or communication may
   degrade. `state_decoupled` isolates this path.
2. **Hand-authored acoustic drive.** The only direct music→state path is the switchable
   density/velocity → activation term. With it off, history dependence can arise only through
   prediction errors and outcomes. Both settings should be reported.
3. **Partner-similarity nudge on affiliation** is hand-authored and small; it can be set to 0.
4. **Signalling-game equilibria.** With 4 targets and 8 motifs the two learners can settle
   into partial-pooling equilibria (some targets share a motif). Final scores therefore vary
   by seed; that variability is real and is reported per run.
5. **Correlated turns.** Scores within a run are not independent; aggregate over seeds.
6. **Constructed presets.** `same_phrase_different_history` builds its result through the
   training schedule and says so; it demonstrates the mechanism, it does not discover it.
7. **Small local model.** If the model-assisted policy is enabled with the bundled
   0.5B-parameter model, its decisions are a weak baseline and may be worse than the local
   learner. The input modality is symbolic features, never audio.
