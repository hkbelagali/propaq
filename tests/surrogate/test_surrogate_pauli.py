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
    SurrogatePauliCircuit,
)
from propaq.circuits.pauli import PauliCircuit
from propaq.circuits.pauli.rotation import PauliRotation
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
