import math

import pytest

from propaq.datatypes import MajoranaMonomial, MajoranaTermSum
from propaq.noise.uniform import UniformNoiseModel


def mon(modes_int: int, n_modes: int = 8) -> MajoranaMonomial:
    return MajoranaMonomial(modes_int, n_modes)


def test_damping_factor_matches_formula():
    damping = 0.2
    model = UniformNoiseModel(damping)
    for w in (0, 1, 2, 5):
        expected = math.exp(-damping * w)
        assert model.damping_factor(w, active_modes=0) == pytest.approx(expected)


def test_apply_noise_scales_term_sum():
    model = UniformNoiseModel(0.5)
    ts = MajoranaTermSum()
    t = mon(0b00000011)  # one fermionic site -> weight 1
    ts.add(t, 2.0)
    model.apply_noise(ts)
    _, coeff = list(ts.items())[0]
    w = t.weight
    assert coeff == pytest.approx(math.exp(-0.5 * w) * 2.0)
