# Prompt templates

Plain-text templates used by the optional model layer (`backend/resonance/models/`).
Code never inlines prompt text; it loads `<name>.txt` from this directory
(`model.prompts_dir`, default `prompts`, resolved against the project root).

Placeholders are `{identifier}` (lowercase, digits, underscore). Other braces are
left untouched, so a literal JSON example needs no escaping. Rendering fails if a
placeholder is not supplied.

Every system prompt states the input modality truthfully: no provider in this build
receives audio. Every prompt asks for JSON only. The receiver prompts
(`choose_pattern.*`) are written without the word "target", and they never receive
the target. A test enforces this.

| Template | Placeholders |
|---|---|
| `choose_pattern.system.txt` | `n_patterns`, `max_pattern_id`, `schema` |
| `choose_pattern.user.txt` | `step`, `channel`, `feature_summary`, `patterns`, `learner_scores`, `state`, `memories` |
| `choose_motif.system.txt` | `max_motif_index`, `schema` |
| `choose_motif.user.txt` | `step`, `partner_id`, `target_id`, `target_name`, `target_onsets`, `motifs`, `learner_values`, `state`, `memories` |
| `propose_phrase.system.txt` | `length_beats`, `tempo_min`, `tempo_max`, `schema` |
| `propose_phrase.user.txt` | `step`, `partner_id`, `target_id`, `target_name`, `target_onsets`, `length_beats`, `motifs`, `state` |
| `narrative.system.txt` | `schema` |
| `narrative.user.txt` | `episode_summary`, `state_before`, `state_after` |

What fills each placeholder:

- `schema`: the compact JSON Schema the answer must match. llama.cpp also enforces it by
  grammar. OpenAI-compatible servers get `json_object` mode. Anthropic gets structured
  outputs with numeric and length limits stripped. The caller re-validates every field.
- `feature_summary`: the `SymbolicFeatures` fields of the received observation, with
  values to 2 decimals, plus the normalised vector φ by feature name. For the symbol
  channel it is `symbol #k`. Keys containing "target" are stripped defensively.
- `state`: the four `StateVector` values, each with its operational definition.
- `memories`: up to `memory.retrieval_k` retrieved memories as
  `step, role, action, score, similarity`.
