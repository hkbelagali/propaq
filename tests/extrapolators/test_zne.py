import numpy as np
import pytest

from propaq.circuits import MajoranaCircuit
from propaq.circuits.majorana.rotation import MajoranaRotation
from propaq.datatypes import MajoranaMonomial, MajoranaTermSum
from propaq.noise import UniformNoiseModel
from propaq.propagators.majorana import MajoranaPropagator
from propaq.extrapolators import ZeroNoiseExtrapolator, ZNEResult

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
    result = extrapolator.run(prop, obs, circuit, fock_state=0)

    assert isinstance(result, ZNEResult)
    assert len(result.noise_values) == 3
    assert len(result.expectation_values) == 3
    assert result.fit_params.shape == (2,)
    assert result.fit_covariance.shape == (2, 2)


def test_zne_linear_extrapolation_close_to_noiseless():
    obs = MajoranaTermSum({mon(0b11): 1.0})
    circuit = one_gate_circuit()
    prop = MajoranaPropagator()

    noiseless = prop.expectation_value(obs, circuit, fock_state=0).expectation_value

    extrapolator = ZeroNoiseExtrapolator(
        fitting_fn=lambda x, a, b: a + b * x,
        noise_values=[0.01, 0.02, 0.03, 0.04],
    )
    result = extrapolator.run(prop, obs, circuit, fock_state=0)

    assert result.zero_noise_value == pytest.approx(noiseless, abs=0.05)


def test_zne_exponential_fit_with_p0():
    obs = MajoranaTermSum({mon(0b11): 1.0})
    circuit = one_gate_circuit()
    prop = MajoranaPropagator()

    noiseless = prop.expectation_value(obs, circuit, fock_state=0).expectation_value

    extrapolator = ZeroNoiseExtrapolator(
        fitting_fn=lambda x, a, b: a * np.exp(-b * x),
        noise_values=[0.01, 0.02, 0.03, 0.04],
    )
    result = extrapolator.run(prop, obs, circuit, fock_state=0, p0=[-1.0, 2.0])

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
