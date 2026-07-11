"""The composable truncator API is shared by the numerical and surrogate propagators."""

import pytest

from propaq import (
    CoefficientTruncator,
    FlushSchedule,
    FrequencyTruncator,
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

    def test_shared_weight_coefficient_and_term_budget_truncators(self):
        from propaq import PauliSurrogatePropagator, SurrogatePauliCircuit

        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        circ = PauliCircuit([PauliRotation(ps(0b0001, 0), 0.3)])
        ops = [WeightTruncator(4), CoefficientTruncator(1e-9), TermBudget(max_terms=5_000_000)]

        # Numerical: honored, runs fine.
        num = PauliPropagator(truncation=ops).expectation_value(obs, circ, initial_state=0)
        assert isinstance(num.expectation_value, float)

        # Surrogate: the same objects are accepted -- `CoefficientTruncator`
        # is decided structurally by `SymbolicCoeff::prune`, no monomial
        # expansion needed (see `propaq.MD`'s "Truncation" section).
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0])
        model = PauliSurrogatePropagator(truncation=ops).build(obs, sc, initial_state=0)
        assert isinstance(model.evaluate([0.3]), float)

    def test_surrogate_and_numerical_coefficient_truncator_boundary_cases_agree(self):
        """Numerical `CoefficientTruncator` filters on the exact runtime
        `|coefficient|`; the surrogate filters on a structural upper bound
        that ignores unresolved trig factors (which can only shrink the
        true value further -- see `SymbolicCoeff::prune`'s doc comment).
        The two aren't the same quantity for intermediate cutoffs, but they
        must still agree at the boundaries: a cutoff near zero prunes
        nothing for either, and a cutoff far above any real coefficient
        prunes everything for both."""
        from propaq import PauliSurrogatePropagator, SurrogatePauliCircuit

        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        circ = PauliCircuit([PauliRotation(ps(0b0001, 0), 0.3)])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0])
        angle = [0.3]

        exact_num = PauliPropagator(truncation=None).expectation_value(obs, circ, initial_state=0).expectation_value
        exact_surr = PauliSurrogatePropagator(truncation=None).build(obs, sc, initial_state=0).evaluate(angle)
        assert exact_num == pytest.approx(exact_surr, rel=1e-9)

        tiny_num = PauliPropagator(
            truncation=[CoefficientTruncator(1e-15)]
        ).expectation_value(obs, circ, initial_state=0).expectation_value
        tiny_surr = PauliSurrogatePropagator(
            truncation=[CoefficientTruncator(1e-15)]
        ).build(obs, sc, initial_state=0).evaluate(angle)
        assert tiny_num == pytest.approx(exact_num, rel=1e-9)
        assert tiny_surr == pytest.approx(exact_surr, rel=1e-9)

        huge_num = PauliPropagator(
            truncation=[CoefficientTruncator(1e9)]
        ).expectation_value(obs, circ, initial_state=0).expectation_value
        huge_surr = PauliSurrogatePropagator(
            truncation=[CoefficientTruncator(1e9)]
        ).build(obs, sc, initial_state=0).evaluate(angle)
        assert huge_num == pytest.approx(0.0, abs=1e-12)
        assert huge_surr == pytest.approx(0.0, abs=1e-12)
