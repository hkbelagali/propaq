import pytest

from propaq.noise.gate import GateNoiseModel


class DirectNoise(GateNoiseModel):
    def __init__(self, factor: float):
        self.factor = factor

    def damping_factor(self, term_weight, active_modes):
        return self.factor


def test_subclass_damping_factor_is_used_directly():
    model = DirectNoise(0.123)
    assert model.damping_factor(5, active_modes=5) == 0.123


def test_subclass_constructor_args_are_not_mistaken_for_base_state():
    model = DirectNoise(0.5)
    assert model.factor == 0.5


def test_unoverridden_damping_factor_errors_clearly():
    with pytest.raises(NotImplementedError, match="must be overridden by a subclass"):
        GateNoiseModel().damping_factor(1, 1)
