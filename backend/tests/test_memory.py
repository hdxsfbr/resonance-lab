from resonance.agents.memory import EpisodicMemory
from resonance.schemas import MemoryItem, StateVector

S0 = StateVector(activation=0.5, expected_value=0.5, uncertainty=0.5, affiliation=0.0)


def item(step: int, vec: list[float]) -> MemoryItem:
    return MemoryItem(step=step, role="receiver", partner_id="A", phrase_id=f"p{step}", feature_vector=vec,
                      action=0, score=1.0, state_before=S0, state_after=S0)


def test_capacity_and_fifo():
    m = EpisodicMemory(3)
    for i in range(5):
        m.append(item(i, [1.0, float(i)]))
    assert len(m) == 3 and [it.step for it in m.items()] == [2, 3, 4]


def test_retrieval_order_and_familiarity():
    m = EpisodicMemory(10)
    m.append(item(0, [1.0, 0.0]))
    m.append(item(1, [0.0, 1.0]))
    m.append(item(2, [1.0, 1.0]))
    r = m.retrieve([1.0, 0.1], 2)
    assert [x.item.step for x in r] == [0, 2] and r[0].similarity >= r[1].similarity
    assert abs(m.familiarity([0.0, 1.0]) - 1.0) < 1e-9


def test_capacity_zero_never_stores():
    m = EpisodicMemory(0)
    m.append(item(0, [1.0]))
    assert len(m) == 0 and m.retrieve([1.0], 3) == [] and m.familiarity([1.0]) == 0.0


def test_clear():
    m = EpisodicMemory(5)
    m.append(item(0, [1.0]))
    m.clear()
    assert len(m) == 0
