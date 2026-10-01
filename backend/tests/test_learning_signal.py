"""The task must produce meaningful, variable learning: neither guaranteed nor random."""

import numpy as np

from resonance.env.timing_task import TimingTask
from resonance.experiments import run_single
from resonance.schemas import ConditionSpec

SEEDS = range(1, 7)


def test_full_condition_learns(config):
    cfg = config.model_copy(update={"episodes": 150})
    runs = [run_single(cfg, ConditionSpec(name="full"), s) for s in SEEDS]
    first = np.array([r.first_block_score for r in runs])
    final = np.array([r.final_block_score for r in runs])
    chance = TimingTask(cfg.task.n_patterns, cfg.task.partial_credit).chance_score()  # 1/n exact + partial credit
    assert final.mean() > first.mean() + 0.1
    assert final.mean() > chance + 0.15
    assert final.mean() < 0.95  # not guaranteed
    assert final.std() > 0.02  # varies across seeds


def test_no_history_stays_near_chance(config):
    cfg = config.model_copy(update={"episodes": 150})
    runs = [run_single(cfg, ConditionSpec(name="no_history"), s) for s in SEEDS]
    chance = TimingTask(cfg.task.n_patterns, cfg.task.partial_credit).chance_score()
    assert abs(np.mean([r.final_block_score for r in runs]) - chance) < 0.1
