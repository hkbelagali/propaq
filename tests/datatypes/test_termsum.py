import pytest

from propaq.datatypes.termsum import TermSum
from propaq.noise.truncation import TruncationPolicy


class MockTerm:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, MockTerm) and self.name == other.name


class MockNoise:
    def __init__(self, factor):
        self.factor = factor

    def damping_factor(self, weight, active_modes):
        return self.factor


def test_add_and_len_items():
    ts = TermSum()
    t = MockTerm("a", 1)
    ts.add(t, 1+0j)
    ts.add(t, 2+0j)
    assert len(ts) == 1
    items = list(ts.items())
    assert items[0][1] == 3+0j


def test_scale_and_norm_squared():
    ts = TermSum()
    a = MockTerm("a", 1)
    b = MockTerm("b", 2)
    ts.add(a, 1+1j)
    ts.add(b, 0.5+0j)
    orig = ts.norm_squared()
    factor = 2+0j
    ts.scale(factor)
    assert pytest.approx(ts.norm_squared(), rel=1e-9) == (abs(factor)**2) * orig


def test_merge_and_copy_independence():
    a = MockTerm("a", 1)
    b = MockTerm("b", 2)
    ts1 = TermSum()
    ts2 = TermSum()
    ts1.add(a, 1)
    ts2.add(b, 2)
    ts1.merge(ts2)
    assert len(ts1) == 2
    c = ts1.copy()
    ts1.add(MockTerm("c", 3), 3)
    assert len(c) == 2


def test_truncate_removes_terms_safely():
    ts = TermSum()
    heavy = MockTerm("heavy", 5)
    light = MockTerm("light", 1)
    ts.add(heavy, 0.01)
    ts.add(light, 1.0)
    policy = TruncationPolicy(weight_cutoff=2, coeff_cutoff=0.1)
    ts.truncate(policy)
    # heavy should be removed, light should remain
    names = {t.name for t, _ in ts.items()}
    assert "heavy" not in names
    assert "light" in names


def test_apply_damping_uses_noise_model():
    ts = TermSum()
    t = MockTerm("x", 2)
    ts.add(t, 2+0j)
    noise = MockNoise(0.5)
    ts.apply_damping(noise, active_modes=0)
    _, coeff = list(ts.items())[0]
    assert coeff == 2 * 0.5
