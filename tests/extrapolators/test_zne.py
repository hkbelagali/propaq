import math

import numpy as np
import pytest

from propaq.circuits import MajoranaCircuit
from propaq.circuits.majorana.rotation import MajoranaRotation
from propaq.datatypes import MajoranaMonomial, MajoranaTermSum
from propaq.extrapolators import ZeroNoiseExtrapolator, ZNEResult
from propaq.noise import GateNoiseModel, UniformNoiseModel
from propaq.propagators.majorana import MajoranaPropagator

N = 8


def mon(modes_int):
    return MajoranaMonomial(modes_int, N)


def one_gate_circuit():
    return MajoranaCircuit([MajoranaRotation(mon(0b11), 0.0)], N)


def test_noise_getter_default_none():
    prop = MajoranaPropagator()
    assert prop.noise is None


def test_noise_setter_roundtrip():
    prop = MajoranaPropagator()
    nm = UniformNoiseModel(0.05)
    prop.set_noise(nm)
    assert prop.noise is not None


def test_noise_setter_restore_none():
    prop = MajoranaPropagator(noise=UniformNoiseModel(0.1))
    prop.set_noise(None)
    assert prop.noise is None


def test_zne_result_fields():
    obs = MajoranaTermSum({mon(0b11): 1.0})
    circuit = one_gate_circuit()
    prop = MajoranaPropagator()

    extrapolator = ZeroNoiseExtrapolator(
        fitting_fn=lambda x, a, b: a + b * x,
        noise_values=[0.01, 0.02, 0.03],
    )
    result = extrapolator.run(prop, obs, circuit, initial_state=0)

    assert isinstance(result, ZNEResult)
    assert len(result.noise_values) == 3
    assert len(result.expectation_values) == 3
    assert result.fit_params.shape == (2,)
    assert result.fit_covariance.shape == (2, 2)


def test_zne_linear_extrapolation_close_to_noiseless():
    obs = MajoranaTermSum({mon(0b11): 1.0})
    circuit = one_gate_circuit()
    prop = MajoranaPropagator()

    noiseless = prop.expectation_value(obs, circuit, initial_state=0).expectation_value

    extrapolator = ZeroNoiseExtrapolator(
        fitting_fn=lambda x, a, b: a + b * x,
        noise_values=[0.01, 0.02, 0.03, 0.04],
    )
    result = extrapolator.run(prop, obs, circuit, initial_state=0)

    assert result.zero_noise_value == pytest.approx(noiseless, abs=0.05)


def test_zne_exponential_fit_with_p0():
    obs = MajoranaTermSum({mon(0b11): 1.0})
    circuit = one_gate_circuit()
    prop = MajoranaPropagator()

    noiseless = prop.expectation_value(obs, circuit, initial_state=0).expectation_value

    extrapolator = ZeroNoiseExtrapolator(
        fitting_fn=lambda x, a, b: a * np.exp(-b * x),
        noise_values=[0.01, 0.02, 0.03, 0.04],
    )
    result = extrapolator.run(prop, obs, circuit, initial_state=0, p0=[-1.0, 2.0])

    assert result.zero_noise_value == pytest.approx(noiseless, abs=1e-4)


def test_zne_restores_original_noise_none():
    obs = MajoranaTermSum({mon(0b11): 1.0})
    circuit = one_gate_circuit()
    prop = MajoranaPropagator()

    ZeroNoiseExtrapolator(lambda x, a, b: a + b * x, [0.01, 0.02]).run(prop, obs, circuit)

    assert prop.noise is None


def test_zne_restores_original_noise_model():
    obs = MajoranaTermSum({mon(0b11): 1.0})
    circuit = one_gate_circuit()
    prop = MajoranaPropagator(noise=UniformNoiseModel(0.005))

    ZeroNoiseExtrapolator(lambda x, a, b: a + b * x, [0.01, 0.02]).run(prop, obs, circuit)

    assert prop.noise is not None


def test_build_noise_default_returns_uniform_noise_model():
    extrapolator = ZeroNoiseExtrapolator(lambda x, a, b: a + b * x, [0.01, 0.02])
    built = extrapolator.build_noise(0.05)
    assert isinstance(built, UniformNoiseModel)
    assert built.damping == 0.05


class StretchedExponentialNoise(GateNoiseModel):
    """A custom Python noise model with a fixed `beta`, swept on `gamma`."""

    def __init__(self, gamma: float, beta: float) -> None:
        self.gamma = gamma
        self.beta = beta

    def damping_factor(self, term_weight, active_modes):
        return math.exp(-((self.gamma * term_weight) ** self.beta))


def test_custom_noise_subclass_sweeps_a_custom_model():
    """Overriding `build_noise` lets the sweep drive a custom `GateNoiseModel`
    instead of the hardcoded `UniformNoiseModel`."""

    seen_gammas: list[float] = []

    class StretchedExponentialExtrapolator(ZeroNoiseExtrapolator):
        def build_noise(self, value: float) -> StretchedExponentialNoise:
            seen_gammas.append(value)
            return StretchedExponentialNoise(gamma=value, beta=0.8)

    obs = MajoranaTermSum({mon(0b11): 1.0})
    circuit = one_gate_circuit()
    prop = MajoranaPropagator()

    noiseless = prop.expectation_value(obs, circuit, initial_state=0).expectation_value

    extrapolator = StretchedExponentialExtrapolator(
        fitting_fn=lambda x, a, b: a + b * x,
        noise_values=[0.01, 0.02, 0.03, 0.04],
    )
    result = extrapolator.run(prop, obs, circuit, initial_state=0)

    assert seen_gammas == [0.01, 0.02, 0.03, 0.04]
    assert result.zero_noise_value == pytest.approx(noiseless, abs=0.05)
    assert prop.noise is None, "original (no) noise must be restored after the sweep"
