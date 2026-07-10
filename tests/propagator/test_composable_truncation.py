"""The composable truncator API is shared by the numerical and surrogate propagators."""

import pytest

from propaq import (
    CoefficientTruncator,
    FlushSchedule,
    FrequencyTruncator,
    MonomialBudget,
    TermBudget,
    Truncator,
    WeightTruncator,
)
from propaq.circuits import PauliCircuit
from propaq.circuits.pauli.rotation import PauliRotation
from propaq.datatypes import PauliString, PauliTermSum
from propaq.datatypes._abstract import BitMask
from propaq.noise import TruncationPolicy
from propaq.propagators.pauli import PauliPropagator

N = 4


def ps(x: int, z: int) -> PauliString:
    return PauliString(BitMask(x), BitMask(z), N)


def _obs_and_circuit():
    # Sum of Z strings + a branching X rotation, so weight/coeff truncation bites.
    obs = PauliTermSum({ps(0, 0b0001): 1.0, ps(0, 0b0011): 0.3, ps(0, 0b0111): 0.05})
    circ = PauliCircuit([PauliRotation(ps(0b0001, 0), 0.6), PauliRotation(ps(0b0110, 0), 0.4)])
    return obs, circ


def _ev(prop, obs, circ):
    return prop.expectation_value(obs, circ, initial_state=0).expectation_value


class TestNumericalListAPI:
    def test_list_reproduces_legacy_policy(self):
        obs, circ = _obs_and_circuit()
        legacy = PauliPropagator(
            truncation=TruncationPolicy(weight_cutoff=2, coeff_cutoff=1e-6)
        )
        composed = PauliPropagator(
            truncation=[WeightTruncator(2), CoefficientTruncator(1e-6), TermBudget(max_terms=10_000_000)]
        )
        assert _ev(composed, obs, circ) == pytest.approx(_ev(legacy, obs, circ), rel=1e-12)

    def test_single_truncator_accepted(self):
        obs, circ = _obs_and_circuit()
        prop = PauliPropagator(truncation=WeightTruncator(3))
        assert isinstance(_ev(prop, obs, circ), float)

    def test_none_truncation_is_exact(self):
        obs, circ = _obs_and_circuit()
        a = _ev(PauliPropagator(truncation=None), obs, circ)
        b = _ev(PauliPropagator(), obs, circ)
        assert a == pytest.approx(b, rel=1e-12)

    def test_truncators_getter_returns_truncator_instances(self):
        prop = PauliPropagator(truncation=[WeightTruncator(2), CoefficientTruncator(1e-6)])
        trs = prop.truncators
        assert len(trs) == 2
        assert all(isinstance(t, Truncator) for t in trs)
        assert isinstance(prop.schedule, FlushSchedule)

    def test_rejects_frequency_truncator(self):
        with pytest.raises(ValueError, match="surrogate"):
            PauliPropagator(truncation=[FrequencyTruncator(5)])

    def test_rejects_monomial_budget(self):
        with pytest.raises(ValueError, match="surrogate"):
            PauliPropagator(truncation=[MonomialBudget(max_monomials=100)])

    def test_none_valued_truncators_are_noops(self):
        obs, circ = _obs_and_circuit()
        a = _ev(
            PauliPropagator(truncation=[WeightTruncator(None), CoefficientTruncator(None)]),
            obs,
            circ,
        )
        b = _ev(PauliPropagator(), obs, circ)
        assert a == pytest.approx(b, rel=1e-12)


class TestCrossPropagator:
    """The same truncator objects drive both propagators."""

    def test_shared_weight_and_term_budget_truncators(self):
        from propaq import PauliSurrogatePropagator, SurrogatePauliCircuit

        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        circ = PauliCircuit([PauliRotation(ps(0b0001, 0), 0.3)])
        # `CoefficientTruncator` is left out of the shared set: it's
        # monomial-level and not yet supported by the Phase A surrogate (see
        # `test_surrogate_rejects_coefficient_truncator_in_phase_a` below),
        # even though the numerical propagator honors it fine.
        ops = [WeightTruncator(4), TermBudget(max_terms=5_000_000)]

        # Numerical: honored, runs fine.
        num = PauliPropagator(truncation=ops).expectation_value(obs, circ, initial_state=0)
        assert isinstance(num.expectation_value, float)

        # Surrogate: the same objects are accepted (term-level truncators are
        # a subset it honors too).
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0])
        model = PauliSurrogatePropagator(truncation=ops).build(obs, sc, initial_state=0)
        assert isinstance(model.evaluate([0.3]), float)

    def test_surrogate_rejects_coefficient_truncator_in_phase_a(self):
        """Unlike the numerical propagator, the Phase A surrogate rejects
        `CoefficientTruncator` (monomial-level truncation is deferred to
        Phase B; see `propaq.MD`)."""
        from propaq import PauliSurrogatePropagator, SurrogatePauliCircuit

        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        circ = PauliCircuit([PauliRotation(ps(0b0001, 0), 0.3)])
        ops = [WeightTruncator(4), CoefficientTruncator(1e-9), TermBudget(max_terms=5_000_000)]

        num = PauliPropagator(truncation=ops).expectation_value(obs, circ, initial_state=0)
        assert isinstance(num.expectation_value, float)

        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0])
        with pytest.raises(ValueError, match="not yet supported"):
            PauliSurrogatePropagator(truncation=ops).build(obs, sc, initial_state=0)
