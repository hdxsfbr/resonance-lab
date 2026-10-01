# Definitions: state variables, update rules, learned components

Everything on this page is **engineered**. The variables are operational quantities
chosen to make history-dependence observable; none of them is a measurement of an
experience. Where the UI uses a word like "activation", this page is its definition.

## State vector (per agent, bounded)

| Variable | Range | Baseline | Operational definition |
|---|---|---|---|
| `activation` | [0, 1] | 0.5 | Drive-like scalar. Raised by dense/loud input when the hand-authored acoustic influence is on, and by surprising outcomes. Decays toward baseline. When coupling is on it lowers the exploration temperature and raises expressive tempo and velocity. |
| `expected_value` | [0, 1] | 0.5 | Running expectation of the coordination score. |
| `uncertainty` | [0, 1] | 0.5 | Running magnitude of recent prediction error. When coupling is on it raises the exploration temperature. |
| `affiliation` | [−1, 1] | 0.0 | Running credit toward the partner: rises with shared success, falls with shared failure; nudged up by phrases similar to the agent's own (hand-authored). When coupling is on it scales the effective learning rate. |

## Update rule (applied once per episode to both agents unless frozen)

For each dimension *k* with current value *s_k*:

```
pe        = score − expected_value                      # prediction error, computed before the update
drive_k   = sensitivity · ( music_k
                          + pe_gain_k · pe_term_k
                          + outcome_gain_k · (score − 0.5)
                          + partner_similarity_gain · similarity   [affiliation only] )
delta_k   = clip( (1 − inertia_k) · drive_k + decay_k · (baseline_k − s_k),  −max_step_k, +max_step_k )
s_k'      = clip( s_k + delta_k, bounds_k )
```

- `music_k` is non-zero only for `activation`, and only when
  `state.acoustic_activation_enabled` is true. It is
  `acoustic_activation_gain · ((density_norm − 0.5) + (velocity_norm − 0.5))`.
  **This is hand-authored, switchable, and not learned.** Turning it off removes the only
  direct acoustic→state path; any remaining history dependence then comes through
  prediction errors and outcomes.
- `pe_term` is `pe` for `activation`, `(|pe| − uncertainty)` for `uncertainty`, and
  `pe` for `expected_value` (so `expected_value` tracks the score).
- `inertia_k` keeps a fraction of the previous value; `decay_k` is the configurable
  return toward baseline; `max_step_k` bounds every update.
- `frozen` skips the update entirely and records `inputs.frozen = true`.

All gains, inertias, decays and bounds live in `config/default.yaml` under `state:`.

## State → behaviour coupling (switchable, `coupling.enabled`)

| Effect | Formula | Off |
|---|---|---|
| Exploration temperature | `τ = clip(τ_base · (1 + k_u·(U − 0.5)) · (1 + k_a·(0.5 − A)), τ_min, τ_max)` | `τ = τ_base` |
| Expressive tempo | `tempo_multiplier = 1 + k_t·(A − 0.5)` | 1 |
| Expressive velocity | `velocity_offset = k_v·(A − 0.5)` | 0 |
| Learning rate | `η_eff = clip(η · (1 + k_aff·affiliation), 0.01, 1)` | `η` |

The `set_coupling` intervention flips this per agent at runtime, and the `PolicyTrace`
records the base and effective temperature so the change is visible. Note that
expressive tempo changes the receiver's *measured* features (notes per second), so an
activation change can degrade or alter communication. That is a confound to measure,
not an affective effect.

## Learned components

| Component | Shape | Update | Used for |
|---|---|---|---|
| Sender values `Q[target][motif]` | n_patterns × n_motifs | `Q += η_eff · (score − Q)` | which motif to send for a target |
| Receiver weights `W[pattern]`, `b[pattern]` | n_patterns × 16 features (+ bias) | for the chosen pattern *a*: `w_a += η_eff · (score − q_a) · x − λ·w_a` | expected score per pattern from the phrase's feature vector |
| Episodic memory | FIFO, `memory.capacity` items | append per episode | retrieval by cosine similarity; familiarity; model prompts |

The receiver learns over a **feature vector**, not motif identities, so related
phrases share information: a transposed motif has the same intervals, contour and rhythm
features and therefore a similar expected-score profile.

## Reset semantics

| Operation | Episodic memory | Learned weights | State vector |
|---|---|---|---|
| `reset_memory` | cleared | re-initialised | untouched |
| `reset_state` | untouched | untouched | := baseline |
| session reset | fresh agents, same seed and config | | |

## Programmed vs learned

- **Programmed**: motif bank, feature definitions, state equations and gains, coupling
  formulas, the task's scoring rule, the acoustic→activation drive, the partner-similarity
  nudge, the perturbation transforms.
- **Learned**: `Q`, `W`, `b`, and the contents of episodic memory.
- **Generated** (optional): model rationales and narrative text; never read by the above.
