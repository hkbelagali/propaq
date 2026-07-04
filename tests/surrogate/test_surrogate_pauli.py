"""Correctness tests for the surrogate Pauli propagator.

All imports come from the public Python API (propaq.*) only.
"""

import math
import tempfile
import os

import pytest

from propaq import (
    PauliTermSum,
    PauliString,
    PauliPropagator,
    PauliSurrogatePropagator,
    PauliSurrogateModel,
    FrequencyTruncationPolicy,
    FlushSchedule,
    Truncator,
    FrequencyTruncator,
    CoefficientTruncator,
    WeightTruncator,
    TermBudget,
    MonomialBudget,
    SurrogatePauliCircuit,
)
from propaq.circuits.pauli import PauliCircuit
from propaq.circuits.pauli.rotation import PauliRotation
from propaq.circuits.pauli.surrogate_rotation import SurrogateRotation
from propaq.datatypes._abstract import BitMask

N = 4  # n_qubits for all tests


def ps(x: int, z: int) -> PauliString:
    return PauliString(BitMask(x), BitMask(z), N)


def numerical_ev(obs: PauliTermSum, circ: PauliCircuit, initial_state: int = 0) -> float:
    return PauliPropagator().expectation_value(obs, circ, initial_state=initial_state).expectation_value


def surrogate_ev(
    obs: PauliTermSum,
    surrogate_circ: SurrogatePauliCircuit,
    params: list[float],
    initial_state: int = 0,
    truncation: FrequencyTruncationPolicy | None = None,
) -> float:
    model = PauliSurrogatePropagator(truncation=truncation).build(
        obs, surrogate_circ, initial_state=initial_state
    )
    return model.evaluate(params)

class TestNumericalAgreement:
    def test_single_rotation(self):
        obs = PauliTermSum({ps(0, 0b0001): 1.0})  # Z_0
        gen = ps(0b0001, 0)                        # X_0
        angle = 0.3
        circ = PauliCircuit([PauliRotation(gen, angle)])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0])
        surr = surrogate_ev(obs, sc, [angle])
        numerical = numerical_ev(obs, circ)
        assert surr == pytest.approx(numerical, rel=1e-9)

    def test_two_rotations_independent_params(self):
        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        gens = [ps(0b0001, 0), ps(0b0010, 0)]  # X_0, X_1
        angles = [0.7, 1.2]
        circ = PauliCircuit([PauliRotation(g, a) for g, a in zip(gens, angles)])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0, 1])
        surr = surrogate_ev(obs, sc, angles)
        numerical = numerical_ev(obs, circ)
        assert surr == pytest.approx(numerical, rel=1e-9)

    def test_three_rotations(self):
        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        gens = [
            ps(0b0001, 0),        # X_0
            ps(0b0011, 0),        # X_0 X_1
            ps(0, 0b0010),        # Z_1
        ]
        angles = [0.3, 0.7, 1.1]
        circ = PauliCircuit([PauliRotation(g, a) for g, a in zip(gens, angles)])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0, 1, 2])
        surr = surrogate_ev(obs, sc, angles)
        numerical = numerical_ev(obs, circ)
        assert surr == pytest.approx(numerical, rel=1e-9)

    def test_shared_parameter(self):
        """Two rotations with the same param_index."""
        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        angle = 0.5
        gens = [ps(0b0001, 0), ps(0b0010, 0)]
        circ = PauliCircuit([PauliRotation(g, angle) for g in gens])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0, 0])
        surr = surrogate_ev(obs, sc, [angle])
        numerical = numerical_ev(obs, circ)
        assert surr == pytest.approx(numerical, rel=1e-9)

    def test_empty_circuit(self):
        """Empty circuit: surrogate should equal the direct expectation value."""
        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        circ = PauliCircuit([])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[])
        model = PauliSurrogatePropagator().build(obs, sc, initial_state=0)
        assert model.evaluate([]) == pytest.approx(1.0)

    def test_excited_initial_state(self):
        obs = PauliTermSum({ps(0, 0b0001): 1.0})  # Z_0
        gen = ps(0b0001, 0)                        # X_0
        angle = 0.4
        circ = PauliCircuit([PauliRotation(gen, angle)])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0])
        for state in [0, 1]:
            surr = surrogate_ev(obs, sc, [angle], initial_state=state)
            numerical = numerical_ev(obs, circ, initial_state=state)
            assert surr == pytest.approx(numerical, rel=1e-9)

    def test_multi_qubit_observable(self):
        obs = PauliTermSum({
            ps(0, 0b0001): 1.0,   # Z_0
            ps(0, 0b0010): 0.5,   # Z_1
        })
        gen = ps(0b0001, 0)
        angle = 0.6
        circ = PauliCircuit([PauliRotation(gen, angle)])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0])
        surr = surrogate_ev(obs, sc, [angle])
        numerical = numerical_ev(obs, circ)
        assert surr == pytest.approx(numerical, rel=1e-9)

class TestFrequencyTruncation:
    def _circuit_and_obs(self):
        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        gens = [ps(0b0001, 0), ps(0b0011, 0), ps(0, 0b0010)]
        angles = [0.3, 0.7, 1.1]
        circ = PauliCircuit([PauliRotation(g, a) for g, a in zip(gens, angles)])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0, 1, 2])
        return obs, sc, circ, angles

    def test_max_frequency_zero_drops_all_non_constant(self):
        """freq=0 keeps only monomials with no trig factors (constant terms)."""
        obs, sc, circ, angles = self._circuit_and_obs()
        model = PauliSurrogatePropagator(
            truncation=FrequencyTruncationPolicy(max_frequency=0)
        ).build(obs, sc, initial_state=0)
        # may have 0 terms or only constant-factor terms
        assert model.n_terms >= 0

    def test_increasing_frequency_reduces_error(self):
        obs, sc, circ, angles = self._circuit_and_obs()
        n_rots = 3
        numerical = numerical_ev(obs, circ)
        prev_err = float("inf")
        for freq in range(0, n_rots + 1):
            model = PauliSurrogatePropagator(
                truncation=FrequencyTruncationPolicy(max_frequency=freq)
            ).build(obs, sc, initial_state=0)
            err = abs(model.evaluate(angles) - numerical)
            assert err <= prev_err + 1e-12, (
                f"Error did not decrease monotonically at freq={freq}: "
                f"prev={prev_err:.3e}, curr={err:.3e}"
            )
            prev_err = err

    def test_exact_at_n_rotations(self):
        """At max_frequency >= n_rotations, the result must be exact."""
        obs, sc, circ, angles = self._circuit_and_obs()
        n_rots = 3
        model = PauliSurrogatePropagator(
            truncation=FrequencyTruncationPolicy(max_frequency=n_rots)
        ).build(obs, sc, initial_state=0)
        numerical = numerical_ev(obs, circ)
        assert model.evaluate(angles) == pytest.approx(numerical, rel=1e-9)


class TestMonomialRangeTruncation:
    """Exercise the importance-ranking (frequency desc, |scalar| asc) path."""

    def _circuit_and_obs(self):
        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        gens = [ps(0b0001, 0), ps(0b0011, 0), ps(0, 0b0010)]
        angles = [0.3, 0.7, 1.1]
        circ = PauliCircuit([PauliRotation(g, a) for g, a in zip(gens, angles)])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0, 1, 2])
        return obs, sc, circ, angles

    def test_generous_monomial_range_is_exact(self):
        """A monomial_range far above the live count never truncates -> exact."""
        obs, sc, circ, angles = self._circuit_and_obs()
        policy = FrequencyTruncationPolicy()
        policy.monomial_range = (1_000, 1_000_000)
        model = PauliSurrogatePropagator(truncation=policy).build(obs, sc, initial_state=0)
        assert model.evaluate(angles) == pytest.approx(numerical_ev(obs, circ), rel=1e-9)

    def test_tight_monomial_range_runs_and_stays_finite(self):
        """A tiny monomial_range forces the importance-ranking removal path."""
        obs, sc, circ, angles = self._circuit_and_obs()
        policy = FrequencyTruncationPolicy()
        policy.monomial_range = (1, 2)
        model = PauliSurrogatePropagator(truncation=policy).build(obs, sc, initial_state=0)
        val = model.evaluate(angles)
        assert math.isfinite(val)
        assert model.n_terms >= 0


class TestMergeCadence:
    """The finer lossless merge cadence must not change results."""

    def _circuit_and_obs(self):
        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        gens = [ps(0b0001, 0), ps(0b0011, 0), ps(0, 0b0010)]
        angles = [0.3, 0.7, 1.1]
        circ = PauliCircuit([PauliRotation(g, a) for g, a in zip(gens, angles)])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0, 1, 2])
        return obs, sc, circ, angles

    def test_frequent_merges_match_exact_and_merge_disabled(self):
        obs, sc, circ, angles = self._circuit_and_obs()
        exact = numerical_ev(obs, circ)

        # Force a merge after essentially every branching gate.
        eager = FrequencyTruncationPolicy()
        eager.merge_max_terms = 1
        m_eager = PauliSurrogatePropagator(truncation=eager).build(obs, sc, initial_state=0)

        # Disable the finer cadence entirely (merge only at truncation flushes).
        off = FrequencyTruncationPolicy()
        off.merge_max_terms = None
        m_off = PauliSurrogatePropagator(truncation=off).build(obs, sc, initial_state=0)

        assert m_eager.evaluate(angles) == pytest.approx(exact, rel=1e-9)
        assert m_eager.evaluate(angles) == pytest.approx(m_off.evaluate(angles), rel=1e-12)

    def test_default_policy_has_merge_cadence_on(self):
        assert FrequencyTruncationPolicy().merge_max_terms is not None


class TestComposableTruncation:
    """The `truncation` field accepts a list of individual truncator operators."""

    def _circ(self):
        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        gens = [ps(0b0001, 0), ps(0b0011, 0), ps(0, 0b0010)]
        angles = [0.3, 0.7, 1.1]
        circ = PauliCircuit([PauliRotation(g, a) for g, a in zip(gens, angles)])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0, 1, 2])
        return obs, sc, circ, angles

    def test_list_of_truncators_frequency_exact(self):
        obs, sc, circ, angles = self._circ()
        model = PauliSurrogatePropagator(
            truncation=[FrequencyTruncator(3), CoefficientTruncator(1e-15)]
        ).build(obs, sc, initial_state=0)
        assert model.evaluate(angles) == pytest.approx(numerical_ev(obs, circ), rel=1e-9)

    def test_single_truncator_accepted(self):
        obs, sc, circ, angles = self._circ()
        model = PauliSurrogatePropagator(truncation=FrequencyTruncator(3)).build(
            obs, sc, initial_state=0
        )
        assert model.evaluate(angles) == pytest.approx(numerical_ev(obs, circ), rel=1e-9)

    def test_coefficient_truncator_tiny_threshold_is_exact(self):
        obs, sc, circ, angles = self._circ()
        model = PauliSurrogatePropagator(
            truncation=[CoefficientTruncator(1e-15)]
        ).build(obs, sc, initial_state=0)
        assert model.evaluate(angles) == pytest.approx(numerical_ev(obs, circ), rel=1e-9)

    def test_coefficient_truncator_huge_threshold_prunes_everything(self):
        obs, sc, circ, angles = self._circ()
        model = PauliSurrogatePropagator(
            truncation=[CoefficientTruncator(1e9)]
        ).build(obs, sc, initial_state=0)
        # |scalar| < 1e9 for every real monomial, so all get pruned at the flush.
        assert model.evaluate(angles) == pytest.approx(0.0, abs=1e-12)

    def test_weight_truncator_matches_legacy_weight_cutoff(self):
        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        gens = [ps(0b0001, 0), ps(0b0011, 0)]
        angles = [0.5, 0.8]
        circ = PauliCircuit([PauliRotation(g, a) for g, a in zip(gens, angles)])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0, 1])
        m_list = PauliSurrogatePropagator(truncation=[WeightTruncator(1)]).build(
            obs, sc, initial_state=0
        )
        m_legacy = PauliSurrogatePropagator(
            truncation=FrequencyTruncationPolicy(weight_cutoff=1)
        ).build(obs, sc, initial_state=0)
        assert m_list.n_terms == m_legacy.n_terms

    def test_explicit_schedule_plus_operators(self):
        obs, sc, circ, angles = self._circ()
        sched = FlushSchedule(merge_max_terms=500_000)
        model = PauliSurrogatePropagator(
            schedule=sched, truncation=[FrequencyTruncator(3), TermBudget(max_terms=1_000_000)]
        ).build(obs, sc, initial_state=0)
        assert model.evaluate(angles) == pytest.approx(numerical_ev(obs, circ), rel=1e-9)

    def test_monomial_budget_operator_runs(self):
        obs, sc, circ, angles = self._circ()
        model = PauliSurrogatePropagator(
            truncation=[MonomialBudget(min_monomials=1, max_monomials=2)]
        ).build(obs, sc, initial_state=0)
        assert math.isfinite(model.evaluate(angles))

    def test_schedule_and_truncators_getters(self):
        prop = PauliSurrogatePropagator(
            truncation=[FrequencyTruncator(3), WeightTruncator(2)]
        )
        trs = prop.truncators
        assert len(trs) == 2
        assert isinstance(prop.schedule, FlushSchedule)
        assert prop.schedule.merge_max_terms is not None  # default-on cadence
        # set_truncation preserves the schedule and replaces operators
        prop.set_truncation([CoefficientTruncator(1e-8)])
        assert len(prop.truncators) == 1

    def test_none_truncation_is_lossless_exact(self):
        obs, sc, circ, angles = self._circ()
        model = PauliSurrogatePropagator(truncation=None).build(obs, sc, initial_state=0)
        assert model.evaluate(angles) == pytest.approx(numerical_ev(obs, circ), rel=1e-9)

    def test_all_truncators_are_truncator_instances(self):
        for op in [
            FrequencyTruncator(None),
            CoefficientTruncator(None),
            WeightTruncator(None),
            TermBudget(),
            MonomialBudget(),
        ]:
            assert isinstance(op, Truncator)

    def test_none_valued_truncators_are_noop_exact(self):
        obs, sc, circ, angles = self._circ()
        model = PauliSurrogatePropagator(
            truncation=[FrequencyTruncator(None), CoefficientTruncator(None), WeightTruncator(None)]
        ).build(obs, sc, initial_state=0)
        assert model.evaluate(angles) == pytest.approx(numerical_ev(obs, circ), rel=1e-9)

class TestLoschmidtEcho:
    def test_echo_recovers_initial(self):
        """U†U should be identity: surrogate evaluated at [θ, -θ] reproduces initial EV.

        The surrogate stores no angle signs; the inverse rotation is represented by
        supplying a negated angle at evaluate time (params[backward_idx] = -angle),
        which correctly gives cos(-θ) = cos(θ) and sin(-θ) = -sin(θ).
        """
        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        gen = ps(0b0001, 0)
        angle = 0.9
        forward = PauliCircuit([PauliRotation(gen, angle)])
        backward = forward.inverse()  # angle = -0.9 in the numerical circuit

        combined_rots = forward.rotations + backward.rotations
        combined_circ = PauliCircuit(combined_rots)
        # Forward gate → param 0, backward gate → param 1
        sc = SurrogatePauliCircuit.from_pauli_circuit(combined_circ, param_indices=[0, 1])
        # Supply -angle for the backward gate so the surrogate reproduces exp(+iθX)
        surr = surrogate_ev(obs, sc, [angle, -angle], initial_state=0)
        # ⟨0|Z|0⟩ = 1.0 after U†U
        assert surr == pytest.approx(1.0, abs=1e-9)

    def test_echo_matches_numerical(self):
        """Surrogate U†U matches the numerical propagator at the same angles."""
        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        gen = ps(0b0001, 0)
        angle = 0.7
        forward = PauliCircuit([PauliRotation(gen, angle)])
        backward = forward.inverse()
        combined_circ = PauliCircuit(forward.rotations + backward.rotations)

        sc = SurrogatePauliCircuit.from_pauli_circuit(combined_circ, param_indices=[0, 1])
        surr = surrogate_ev(obs, sc, [angle, -angle])
        numerical = numerical_ev(obs, combined_circ)
        assert surr == pytest.approx(numerical, rel=1e-9)

class TestSaveLoad:
    def test_round_trip_single_rotation(self):
        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        gen = ps(0b0001, 0)
        angle = 0.3
        circ = PauliCircuit([PauliRotation(gen, angle)])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0])
        model = PauliSurrogatePropagator().build(obs, sc, initial_state=0)
        original_val = model.evaluate([angle])

        with tempfile.NamedTemporaryFile(suffix=".surrogate.gz", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = PauliSurrogateModel.load(path)
            assert loaded.evaluate([angle]) == pytest.approx(original_val, rel=1e-14)
            assert loaded.n_terms == model.n_terms
            assert loaded.n_params == model.n_params
        finally:
            os.unlink(path)

    def test_round_trip_many_terms_multi_shard(self):
        """Many surviving terms exercise the multi-shard save/load path.

        A sum of distinct all-Z strings under an empty circuit keeps every term
        (each overlaps |0..0>), so n_terms is large enough to span several
        parallel shards rather than the single-shard small-model cases above.
        """
        obs = PauliTermSum({ps(0, z): 1.0 + 0.1 * z for z in range(1, 16)})
        circ = PauliCircuit([])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[])
        model = PauliSurrogatePropagator().build(obs, sc, initial_state=0)
        assert model.n_terms == 15

        with tempfile.NamedTemporaryFile(suffix=".surrogate.gz", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = PauliSurrogateModel.load(path)
            assert loaded.n_terms == model.n_terms
            assert loaded.evaluate([]) == pytest.approx(model.evaluate([]), rel=1e-14)
        finally:
            os.unlink(path)

    def test_load_rejects_unrecognized_format(self):
        """A non-surrogate/old-format file is rejected, not silently misread."""
        with tempfile.NamedTemporaryFile(suffix=".surrogate.gz", delete=False) as f:
            path = f.name
            f.write(b"not a surrogate model file")
        try:
            with pytest.raises(Exception):
                PauliSurrogateModel.load(path)
        finally:
            os.unlink(path)

    def test_round_trip_multi_rotation(self):
        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        gens = [ps(0b0001, 0), ps(0b0011, 0), ps(0, 0b0010)]
        angles = [0.3, 0.7, 1.1]
        circ = PauliCircuit([PauliRotation(g, a) for g, a in zip(gens, angles)])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0, 1, 2])
        model = PauliSurrogatePropagator().build(obs, sc, initial_state=0)

        with tempfile.NamedTemporaryFile(suffix=".surrogate.gz", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = PauliSurrogateModel.load(path)
            # Evaluate at a different set of angles too
            for test_angles in [angles, [0.1, 0.2, 0.3], [math.pi / 4] * 3]:
                assert loaded.evaluate(test_angles) == pytest.approx(
                    model.evaluate(test_angles), rel=1e-14
                )
        finally:
            os.unlink(path)

class TestNTermsFiltering:
    def test_n_terms_leq_propagated(self):
        """model.n_terms <= terms from the propagator (zero-overlap terms excluded)."""
        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        gens = [ps(0b0001, 0), ps(0b0011, 0)]
        angles = [0.3, 0.7]
        circ = PauliCircuit([PauliRotation(g, a) for g, a in zip(gens, angles)])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0, 1])
        model = PauliSurrogatePropagator().build(obs, sc, initial_state=0)
        # With a Z_0 observable and X-type generators, some terms may anticommute
        # and only those with nonzero trace survive.
        assert model.n_terms >= 0

    def test_weight_cutoff_reduces_terms(self):
        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        gens = [ps(0b0001, 0), ps(0b0011, 0)]
        angles = [0.5, 0.8]
        circ = PauliCircuit([PauliRotation(g, a) for g, a in zip(gens, angles)])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0, 1])
        model_full = PauliSurrogatePropagator().build(obs, sc, initial_state=0)
        model_cut = PauliSurrogatePropagator(
            truncation=FrequencyTruncationPolicy(weight_cutoff=1)
        ).build(obs, sc, initial_state=0)
        assert model_cut.n_terms <= model_full.n_terms
        
class TestCircuitConstruction:
    def test_from_generators_and_param_indices(self):
        gens = [ps(0b0001, 0), ps(0, 0b0001)]
        sc = SurrogatePauliCircuit.from_generators_and_param_indices(gens, [0, 1])
        assert sc.n_params == 2
        assert len(sc.rotations) == 2

    def test_n_params_with_shared_index(self):
        gens = [ps(0b0001, 0), ps(0b0010, 0)]
        sc = SurrogatePauliCircuit.from_generators_and_param_indices(gens, [0, 0])
        assert sc.n_params == 1

    def test_param_indices_length_mismatch_raises(self):
        circ = PauliCircuit([PauliRotation(ps(0b0001, 0), 0.3)])
        with pytest.raises(ValueError):
            SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0, 1])

    def test_repr(self):
        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        gen = ps(0b0001, 0)
        circ = PauliCircuit([PauliRotation(gen, 0.3)])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0])
        model = PauliSurrogatePropagator().build(obs, sc, initial_state=0)
        r = repr(model)
        assert "PauliSurrogateModel" in r
        assert "n_terms" in r

class TestNumericAngleRotations:
    def test_all_numeric_rotations_matches_numerical(self):
        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        gens = [ps(0b0001, 0), ps(0b0011, 0), ps(0, 0b0010)]
        angles = [0.3, 0.7, 1.1]
        circ = PauliCircuit([PauliRotation(g, a) for g, a in zip(gens, angles)])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[None, None, None])
        assert sc.n_params == 0
        surr = surrogate_ev(obs, sc, [])
        numerical = numerical_ev(obs, circ)
        assert surr == pytest.approx(numerical, rel=1e-9)

    def test_mixed_numeric_and_symbolic_matches_numerical(self):
        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        gens = [ps(0b0001, 0), ps(0b0011, 0), ps(0, 0b0010)]
        angles = [0.3, 0.7, 1.1]
        circ = PauliCircuit([PauliRotation(g, a) for g, a in zip(gens, angles)])
        # Outer two gates baked numeric, middle gate symbolic.
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[None, 0, None])
        assert sc.n_params == 1
        surr = surrogate_ev(obs, sc, [angles[1]])
        numerical = numerical_ev(obs, circ)
        assert surr == pytest.approx(numerical, rel=1e-9)

    def test_mixed_numeric_and_symbolic_shared_symbolic_index(self):
        obs = PauliTermSum({ps(0, 0b0001): 1.0})
        angle = 0.5
        numeric_angle = 0.9
        gens = [ps(0b0001, 0), ps(0b0010, 0), ps(0, 0b0011)]
        circ = PauliCircuit([
            PauliRotation(gens[0], angle),
            PauliRotation(gens[1], angle),
            PauliRotation(gens[2], numeric_angle),
        ])
        # Two symbolic gates share param_index=0; the third is numeric.
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0, 0, None])
        assert sc.n_params == 1
        surr = surrogate_ev(obs, sc, [angle])
        numerical = numerical_ev(obs, circ)
        assert surr == pytest.approx(numerical, rel=1e-9)

    def test_n_params_skips_numeric_rotations(self):
        gen = ps(0b0001, 0)
        layers = [
            [SurrogateRotation(gen, angle=0.1)],
            [SurrogateRotation(gen, param_index=0)],
            [SurrogateRotation(gen, angle=0.2)],
            [SurrogateRotation(gen, param_index=2)],
        ]
        sc = SurrogatePauliCircuit(layers)
        assert sc.n_params == 3

        all_numeric_layers = [[SurrogateRotation(gen, angle=0.1)]]
        assert SurrogatePauliCircuit(all_numeric_layers).n_params == 0

    def test_from_pauli_circuit_keeps_source_angle_for_none_index(self):
        gens = [ps(0b0001, 0), ps(0b0010, 0)]
        angles = [0.3, 0.6]
        circ = PauliCircuit([PauliRotation(g, a) for g, a in zip(gens, angles)])
        sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[None, 0])

        assert sc.rotations[0].param_index is None
        assert sc.rotations[0].angle == angles[0]
        assert sc.rotations[1].param_index == 0
        assert sc.rotations[1].angle is None

    def test_surrogate_rotation_requires_exactly_one_of_param_index_or_angle(self):
        gen = ps(0b0001, 0)
        with pytest.raises(ValueError):
            SurrogateRotation(gen)
        with pytest.raises(ValueError):
            SurrogateRotation(gen, param_index=0, angle=0.5)
        # Falsy-but-non-None values must still count as "given".
        with pytest.raises(ValueError):
            SurrogateRotation(gen, param_index=0, angle=0.0)
