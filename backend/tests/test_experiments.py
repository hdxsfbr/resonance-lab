import csv
import io

import numpy as np

from resonance.experiments import run_experiment, to_csv
from resonance.schemas import ConditionSpec, ExperimentRequest


def test_small_batch(config):
    cfg = config.model_copy(update={"episodes": 30})
    req = ExperimentRequest(name="t", conditions=[ConditionSpec(name="full"), ConditionSpec(name="symbol")], seeds=[1, 2, 3], config=cfg)
    res = run_experiment(req)
    assert len(res.runs) == 6 and len(res.aggregates) == 2
    assert any("unit of analysis" in n for n in res.notes) and any("report as-is" in n for n in res.notes)
    for r in res.runs:
        assert r.episodes == 30 and len(r.learning_curve) == 10
        assert set(r.policy_shift_kl) == {"A_receiver", "A_sender", "B_receiver", "B_sender"}
        assert set(r.familiar_vs_unfamiliar) == {"A_familiar", "A_unfamiliar", "B_familiar", "B_unfamiliar"}
        assert all(0 <= v <= 200 for v in r.state_effect_persistence.values())
        assert 0 <= r.generalization_score <= 1
    finals = [r.final_block_score for r in res.runs if r.condition == "full"]
    means = [r.mean_score for r in res.runs if r.condition == "full"]
    assert np.std(finals) > 0 or np.std(means) > 0  # runs vary across seeds
    agg = res.aggregates[0]
    assert agg.n_runs == 3 and agg.per_run_final_block == finals
    assert agg.sd_final_block_score == round(float(np.std(finals, ddof=1)), 6)
    rows = list(csv.DictReader(io.StringIO(to_csv(res))))
    assert [r["row_type"] for r in rows] == ["run"] * 6 + ["aggregate"] * 2
    assert float(rows[0]["final_block_score"]) == res.runs[0].final_block_score


def test_symbol_familiarity_identical(config):
    cfg = config.model_copy(update={"episodes": 20})
    res = run_experiment(ExperimentRequest(conditions=[ConditionSpec(name="symbol")], seeds=[1], config=cfg))
    f = res.runs[0].familiar_vs_unfamiliar
    # one-hot codes cannot represent transposition / tempo change: identical by construction
    assert f["A_familiar"] == f["A_unfamiliar"] and f["B_familiar"] == f["B_unfamiliar"]
    assert res.runs[0].state_similarity is not None
