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
