from resonance.session import Session


def _log(config, condition, seed, n=40):
    s = Session(config, condition, seed=seed)
    s.step(n)
    return [(e.seq, e.step, e.type, e.agent_id, e.visibility, e.payload) for e in s.events]


def test_same_seed_identical_events(config, full):
    assert _log(config, full, 3) == _log(config, full, 3)


def test_different_seeds_differ(config, full):
    assert _log(config, full, 3) != _log(config, full, 4)


def test_perturbed_reproducible(config):
    from resonance.schemas import ConditionSpec

    cond = ConditionSpec(name="perturbed", perturbation="rhythm_shuffle")
    assert _log(config, cond, 9, 20) == _log(config, cond, 9, 20)
