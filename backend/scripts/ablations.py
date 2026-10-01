"""Targeted ablations for docs/EXPERIMENT_REPORT.md. Run from backend/: `.venv/bin/python scripts/ablations.py`. Writes JSON+CSV into data/exports."""
import json, sys, time, copy
from resonance import schemas as S
from resonance.config import load_config
from resonance.experiments import run_experiment, to_csv

seeds = list(range(1, 11))
EP = 120
base = load_config()
base.episodes = EP

def run(name, cfg, conds):
    t = time.time()
    res = run_experiment(S.ExperimentRequest(name=name, conditions=conds, seeds=seeds, config=cfg))
    for a in res.aggregates:
        print(f"{name:28s} {a.condition:16s} {a.perturbation:15s} final={a.mean_final_block_score:.3f}±{a.sd_final_block_score:.3f} succ={a.mean_success_rate:.3f}±{a.sd_success_rate:.3f} per-run={[round(x,2) for x in a.per_run_final_block]}")
    open(f"../data/exports/experiment_{name}.json", "w").write(res.model_dump_json(indent=1))
    open(f"../data/exports/experiment_{name}.csv", "w").write(to_csv(res))
    print(f"  ({time.time()-t:.1f}s)")
    return res

full = [S.ConditionSpec(name="full")]
# 1. acoustic drive off (removes the only direct music->state path)
c = copy.deepcopy(base); c.state.acoustic_activation_enabled = False
run("ablation_acoustic_drive_off", c, full)
# 2. coupling components
c = copy.deepcopy(base); c.coupling.affiliation_to_learning_rate = 0.0
run("ablation_no_affiliation_lr_coupling", c, full)
c = copy.deepcopy(base); c.coupling.activation_to_tempo = 0.0; c.coupling.activation_to_velocity = 0.0
run("ablation_no_expressive_modulation", c, full)
c = copy.deepcopy(base); c.coupling.uncertainty_to_temperature = 0.0; c.coupling.activation_to_temperature = 0.0
run("ablation_no_temperature_coupling", c, full)
# 3. all perturbations
perts = [S.ConditionSpec(name="perturbed", perturbation=p) for p in ["transpose","velocity_flatten","tempo_shift","contour_invert","rhythm_shuffle","pitch_shuffle"]]
run("all_perturbations", base, perts)
# 4. longer horizon
c = copy.deepcopy(base); c.episodes = 300
run("full_vs_symbol_300_episodes", c, [S.ConditionSpec(name="full"), S.ConditionSpec(name="symbol"), S.ConditionSpec(name="no_history")])
