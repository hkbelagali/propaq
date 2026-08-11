import pytest

from propaq.circuits import MajoranaCircuit
from propaq.circuits.majorana.rotation import MajoranaRotation
from propaq.datatypes import MajoranaMonomial, MajoranaTermSum
from propaq.extrapolators import CoefficientCutoffExtrapolator, WeightCutoffExtrapolator, ZCEResult
from propaq.noise import TruncationPolicy
from propaq.propagators.majorana import MajoranaPropagator

N = 8


def mon(modes_int):
    return MajoranaMonomial(modes_int, N)


def one_gate_circuit():
    return MajoranaCircuit([MajoranaRotation(mon(0b11), 0.0)], N)


def trunc(weight_cutoff=None, coeff_cutoff=0.0):
    return TruncationPolicy(weight_cutoff=weight_cutoff, coeff_cutoff=coeff_cutoff)


def test_truncation_getter_default_none():
    prop = MajoranaPropagator()
    assert prop.truncators == []


def test_truncation_setter_roundtrip():
    prop = MajoranaPropagator()
    prop.set_truncation(trunc(weight_cutoff=4))
    assert WeightCutoffExtrapolator(lambda w, a, b: a, [])._get_cutoff(prop) == 4


def test_truncation_setter_restore_none():
    prop = MajoranaPropagator(truncation=trunc(coeff_cutoff=0.01))
    prop.set_truncation(None)
    assert prop.truncators == []


def test_zce_coeff_result_fields():
    obs = MajoranaTermSum({mon(0b11): 1.0})
    circuit = one_gate_circuit()
    prop = MajoranaPropagator(truncation=trunc(coeff_cutoff=0.001))

    extrapolator = CoefficientCutoffExtrapolator(
        fitting_fn=lambda eps, a, b: a + b * eps,
        cutoff_values=[0.001, 0.002, 0.003],
    )
    result = extrapolator.run(prop, obs, circuit, initial_state=0)

    assert isinstance(result, ZCEResult)
    assert len(result.cutoff_values) == 3
    assert len(result.expectation_values) == 3
    assert result.fit_params.shape == (2,)
    assert result.fit_covariance.shape == (2, 2)


def test_zce_weight_result_fields():
    obs = MajoranaTermSum({mon(0b11): 1.0})
    circuit = one_gate_circuit()
    prop = MajoranaPropagator(truncation=trunc(weight_cutoff=4))

    extrapolator = WeightCutoffExtrapolator(
        fitting_fn=lambda w, a, b: a + b / (w + 1),
        cutoff_values=[2, 3, 4],
    )
    result = extrapolator.run(prop, obs, circuit, initial_state=0)

    assert isinstance(result, ZCEResult)
    assert len(result.cutoff_values) == 3
    assert len(result.expectation_values) == 3
    assert result.fit_params.shape == (2,)
    assert result.fit_covariance.shape == (2, 2)


def test_zce_coeff_linear_extrapolation_close_to_reference():
    obs = MajoranaTermSum({mon(0b11): 1.0})
    circuit = one_gate_circuit()
    prop = MajoranaPropagator(truncation=trunc(coeff_cutoff=0.001))

    reference = (
        MajoranaPropagator(truncation=trunc(coeff_cutoff=0.0))
        .expectation_value(obs, circuit, initial_state=0)
        .expectation_value
    )

    extrapolator = CoefficientCutoffExtrapolator(
        fitting_fn=lambda eps, a, b: a + b * eps,
        cutoff_values=[0.001, 0.002, 0.003, 0.004],
    )
    result = extrapolator.run(prop, obs, circuit, initial_state=0)

    assert result.zero_cutoff_value == pytest.approx(reference, abs=1e-4)


def test_zce_coeff_truncation_changes_expectation_value():
    obs = MajoranaTermSum({mon(0b11): 1.0, mon(0b1111): 0.05})
    circuit = one_gate_circuit()

    ev_exact = (
        MajoranaPropagator(truncation=trunc(coeff_cutoff=0.0))
        .expectation_value(obs, circuit, initial_state=0)
        .expectation_value
    )

    ev_truncated = (
        MajoranaPropagator(truncation=trunc(coeff_cutoff=0.06))
        .expectation_value(obs, circuit, initial_state=0)
        .expectation_value
    )

    assert ev_exact != pytest.approx(ev_truncated, abs=1e-10)


def test_zce_weight_converges_with_increasing_cutoff():
    obs = MajoranaTermSum({mon(0b11): 1.0, mon(0b1111): 0.5})
    circuit = one_gate_circuit()

    reference = (
        MajoranaPropagator(truncation=trunc(weight_cutoff=None))
        .expectation_value(obs, circuit, initial_state=0)
        .expectation_value
    )

    errors = []
    for w in [2, 3, 4, 5]:
        ev = (
            MajoranaPropagator(truncation=trunc(weight_cutoff=w))
            .expectation_value(obs, circuit, initial_state=0)
            .expectation_value
        )
        errors.append(abs(ev - reference))

    # Error must be non-increasing as weight cutoff grows.
    assert all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))


def test_zce_coeff_restores_original_truncation():
    obs = MajoranaTermSum({mon(0b11): 1.0})
    circuit = one_gate_circuit()
    original = trunc(coeff_cutoff=0.001)
    prop = MajoranaPropagator(truncation=original)

    extrapolator = CoefficientCutoffExtrapolator(lambda eps, a, b: a + b * eps, [0.002, 0.004])
    extrapolator.run(prop, obs, circuit)

    assert extrapolator._get_cutoff(prop) == pytest.approx(0.001)


def test_zce_weight_restores_original_truncation():
    obs = MajoranaTermSum({mon(0b11): 1.0})
    circuit = one_gate_circuit()
    prop = MajoranaPropagator(truncation=trunc(weight_cutoff=6))

    extrapolator = WeightCutoffExtrapolator(lambda w, a, b: a + b / (w + 1), [3, 4, 5])
    extrapolator.run(prop, obs, circuit)

    assert extrapolator._get_cutoff(prop) == 6


def test_zce_coeff_restores_when_original_truncation_is_none():
    obs = MajoranaTermSum({mon(0b11): 1.0})
    circuit = one_gate_circuit()
    prop = MajoranaPropagator(truncation=trunc(coeff_cutoff=0.001))

    extrapolator = CoefficientCutoffExtrapolator(lambda eps, a, b: a + b * eps, [0.002, 0.004])
    extrapolator.run(prop, obs, circuit)
    prop.set_truncation(None)

    with pytest.raises(ValueError):
        extrapolator.run(prop, obs, circuit)

    assert prop.truncators == []


def test_zce_coeff_set_cutoff_raises_when_no_truncation():
    prop = MajoranaPropagator()
    extrapolator = CoefficientCutoffExtrapolator(lambda eps, a, b: a + b * eps, [0.01, 0.02])
    with pytest.raises(ValueError, match="no truncation policy"):
        extrapolator._set_cutoff(prop, 0.01)


def test_zce_weight_set_cutoff_raises_when_no_truncation():
    prop = MajoranaPropagator()
    extrapolator = WeightCutoffExtrapolator(lambda w, a, b: a + b / (w + 1), [3, 4, 5])
    with pytest.raises(ValueError, match="no truncation policy"):
        extrapolator._set_cutoff(prop, 3)
