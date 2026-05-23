import numpy as np

import pytest 

from propaq.noise.uniform import UniformNoiseModel
from propaq.datatypes.majorana_termsum import MajoranaTermSum


class DummyTerm:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, DummyTerm) and self.name == other.name


def test_damping_factor_matches_formula():
    damping = 0.2
    model = UniformNoiseModel(damping)
    for w in (0, 1, 2, 5):
        expected = np.exp(-damping * w)
        assert model.damping_factor(w, active_modes=0) == pytest.approx(expected)


def test_apply_noise_scales_term_sum():
    model = UniformNoiseModel(0.5)
    ts = MajoranaTermSum()
    t = DummyTerm("t", 1)
    ts.add(t, 2.0)
    model.apply_noise(ts)
    _, coeff = list(ts.items())[0]
    assert coeff == np.exp(-0.5) * 2.0
