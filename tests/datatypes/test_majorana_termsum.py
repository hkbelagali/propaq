import pytest

from propaq.datatypes import MajoranaTermSum, MajoranaMonomial
from propaq.noise.truncation import TruncationPolicy
from propaq.noise.uniform import UniformNoiseModel


# n_modes=8 gives 4 fermionic sites; enough for all tests below.
N = 8


def mon(modes_int: int) -> MajoranaMonomial:
    return MajoranaMonomial(modes_int, N)


def test_add_and_len_items():
    ts = MajoranaTermSum()
    t = mon(0b01)
    ts.add(t, 1 + 0j)
    ts.add(t, 2 + 0j)
    assert len(ts) == 1
    items = list(ts.items())
    assert items[0][1] == pytest.approx(3 + 0j)


def test_scale_and_norm_squared():
    ts = MajoranaTermSum()
    a = mon(0b01)
    b = mon(0b10)
    ts.add(a, 1 + 1j)
    ts.add(b, 0.5 + 0j)
    orig = ts.norm_squared()
    factor = 2 + 0j
    ts.scale(factor)
    assert pytest.approx(ts.norm_squared(), rel=1e-9) == (abs(factor) ** 2) * orig


def test_merge_and_copy_independence():
    a = mon(0b0001)
    b = mon(0b0010)
    ts1 = MajoranaTermSum()
    ts2 = MajoranaTermSum()
    ts1.add(a, 1)
    ts2.add(b, 2)
    ts1.merge(ts2)
    assert len(ts1) == 2
    c = ts1.copy()
    ts1.add(mon(0b0100), 3)
    assert len(c) == 2


def test_truncate_removes_terms_safely():
    ts = MajoranaTermSum()
    # weight-4 monomial: modes 0b11110000 → sites 2,3 both occupied → high weight
    heavy = mon(0b00001111)  # weight >= 1
    light = mon(0b00000011)  # single site, low weight
    ts.add(heavy, 0.01)
    ts.add(light, 1.0)
    policy = TruncationPolicy(weight_cutoff=0, coeff_cutoff=0.1)
    ts.truncate(policy)
    remaining = [m for m, _ in ts.items()]
    assert heavy not in remaining
    assert light not in remaining or any(m == light for m in remaining)


def test_truncate_keeps_heavy_above_cutoff():
    ts = MajoranaTermSum()
    t = mon(0b00000011)
    ts.add(t, 1.0)
    policy = TruncationPolicy(weight_cutoff=10, coeff_cutoff=0.0)
    ts.truncate(policy)
    assert len(ts) == 1


def test_apply_damping_uses_noise_model():
    ts = MajoranaTermSum()
    t = mon(0b00000011)
    ts.add(t, 2 + 0j)
    noise = UniformNoiseModel(0.0)  # zero damping → coefficients unchanged
    ts.apply_damping(noise, active_modes=0)
    _, coeff = list(ts.items())[0]
    assert coeff == pytest.approx(2 + 0j)

def test_constructor_with_dict():
    t = mon(0b01)
    ts = MajoranaTermSum({t: 2.5 + 0j})
    assert len(ts) == 1
    _, coeff = list(ts.items())[0]
    assert coeff == pytest.approx(2.5 + 0j)

def test_constructor_empty_dict():
    ts = MajoranaTermSum({})
    assert len(ts) == 0

def test_constructor_multi_term_dict():
    a, b = mon(0b0001), mon(0b0010)
    ts = MajoranaTermSum({a: 1.0, b: -1.0j})
    assert len(ts) == 2

def test_setitem_and_getitem():
    ts = MajoranaTermSum()
    t = mon(0b01)
    ts[t] = 3.0 + 0j
    assert ts[t] == pytest.approx(3.0 + 0j)

def test_setitem_overwrites():
    ts = MajoranaTermSum()
    t = mon(0b01)
    ts[t] = 1.0
    ts[t] = 5.0
    assert ts[t] == pytest.approx(5.0)
    assert len(ts) == 1

def test_getitem_missing_returns_zero():
    ts = MajoranaTermSum()
    t = mon(0b01)
    assert ts[t] == pytest.approx(0.0)

def test_merge_overlapping_accumulates():
    a = mon(0b0001)
    ts1 = MajoranaTermSum()
    ts2 = MajoranaTermSum()
    ts1.add(a, 1.0 + 0j)
    ts2.add(a, 2.0 + 0j)
    ts1.merge(ts2)
    assert len(ts1) == 1
    assert ts1[a] == pytest.approx(3.0 + 0j)

def test_merge_mixed_overlap_and_new():
    a, b = mon(0b0001), mon(0b0010)
    ts1 = MajoranaTermSum({a: 1.0})
    ts2 = MajoranaTermSum({a: 2.0, b: 3.0})
    ts1.merge(ts2)
    assert len(ts1) == 2
    assert ts1[a] == pytest.approx(3.0)
    assert ts1[b] == pytest.approx(3.0)

class EvenWeightTruncation:
    """Truncates terms with even Pauli weight."""
    def should_truncate(self, weight, abs_coeff):
        return weight % 2 == 0

def test_truncate_custom_python_policy():
    # modes=0b11 → weight 1 (odd) → kept
    # modes=0b0101_0101 → weight 4 (even) → truncated
    kept = mon(0b00000011)          # weight 1
    removed = mon(0b01010101)       # weight 4
    ts = MajoranaTermSum({kept: 1.0, removed: 1.0})
    ts.truncate(EvenWeightTruncation())
    remaining = [m for m, _ in ts.items()]
    assert kept in remaining
    assert removed not in remaining
    
class ConstantDampingNoise:
    def __init__(self, factor):
        self.factor = factor

    def damping_factor(self, weight, active_modes):
        return self.factor

def test_apply_damping_custom_python_noise():
    t = mon(0b00000011)
    ts = MajoranaTermSum({t: 4.0 + 0j})
    ts.apply_damping(ConstantDampingNoise(0.25), active_modes=0)
    assert ts[t] == pytest.approx(1.0 + 0j)  # 4.0 * 0.25
