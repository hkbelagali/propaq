"""Targeted unit tests for PauliPropagator, parallel to test_propagator.py for Majorana."""

import math

import pytest

from propaq.circuits import PauliCircuit
from propaq.circuits.pauli.rotation import PauliRotation
from propaq.datatypes import PauliString, PauliTermSum
from propaq.datatypes._abstract import BitMask
from propaq.noise import TruncationPolicy, UniformNoiseModel
from propaq.propagators.pauli import PauliPropagator

N = 4  # n_qubits for all tests


def ps(x: int, z: int) -> PauliString:
    return PauliString(BitMask(x), BitMask(z), N)


def empty_circuit() -> PauliCircuit:
    return PauliCircuit([])


def test_expectation_value_zeros_state():
    # Z on qubit 0: ⟨0|Z|0⟩ = +1.0
    obs = PauliTermSum({ps(0, 0b0001): 1.0})
    prop = PauliPropagator()
    val = prop.expectation_value(obs, empty_circuit(), fock_state=0).expectation_value
    assert val == pytest.approx(1.0)


def test_expectation_value_qubit0_excited():
    # Z on qubit 0: ⟨1|Z|1⟩ = -1.0
    obs = PauliTermSum({ps(0, 0b0001): 1.0})
    prop = PauliPropagator()
    val = prop.expectation_value(obs, empty_circuit(), fock_state=1).expectation_value
    assert val == pytest.approx(-1.0)


def test_expectation_value_qubit1():
    # Z on qubit 1: +1 when qubit 1 = |0⟩, -1 when qubit 1 = |1⟩
    obs = PauliTermSum({ps(0, 0b0010): 1.0})
    prop = PauliPropagator()
    assert prop.expectation_value(obs, empty_circuit(), fock_state=0b00).expectation_value == pytest.approx(1.0)
    assert prop.expectation_value(obs, empty_circuit(), fock_state=0b10).expectation_value == pytest.approx(-1.0)


def test_expectation_value_linear_in_coefficient():
    # Z on qubit 0 with coefficient 3: 3 * ⟨0|Z|0⟩ = 3.0
    obs = PauliTermSum({ps(0, 0b0001): 3.0})
    prop = PauliPropagator()
    val = prop.expectation_value(obs, empty_circuit(), fock_state=0).expectation_value
    assert val == pytest.approx(3.0)


def test_expectation_value_superposition_of_terms():
    # Z_0 (coeff 1.0) + Z_1 (coeff 2.0)
    obs = PauliTermSum({ps(0, 0b0001): 1.0, ps(0, 0b0010): 2.0})
    prop = PauliPropagator()
    # fock_state=0b00: both qubits |0⟩ → 1*(+1) + 2*(+1) = +3
    assert prop.expectation_value(obs, empty_circuit(), fock_state=0b00).expectation_value == pytest.approx(3.0)
    # fock_state=0b11: both qubits |1⟩ → 1*(-1) + 2*(-1) = -3
    assert prop.expectation_value(obs, empty_circuit(), fock_state=0b11).expectation_value == pytest.approx(-3.0)


def test_noise_damps_commuting_term():
    obs_term = ps(0, 0b0001)  # Z_0, weight = 1
    obs = PauliTermSum({obs_term: 1.0})
    # Same generator as obs → commutes → rotation is trivial
    generator = ps(0, 0b0001)  # Z_0
    circuit = PauliCircuit([PauliRotation(generator, 0.5)])
    noise = UniformNoiseModel(damping=0.5)
    prop = PauliPropagator(noise=noise)
    evolved = prop.propagate(obs, circuit)
    expected_coeff = math.exp(-0.5 * obs_term.weight)
    _, coeff = list(evolved.items())[0]
    assert abs(coeff) == pytest.approx(expected_coeff, rel=1e-6)


def test_no_noise_preserves_norm():
    obs = PauliTermSum({ps(0, 0b0001): 1.0, ps(0, 0b0010): 0.5})
    original_norm = obs.norm_squared()
    # X_0 anticommutes with Z_0, commutes with Z_1 → spawns a new term
    generator = ps(0b0001, 0)  # X_0
    circuit = PauliCircuit([PauliRotation(generator, math.pi / 4)])
    prop = PauliPropagator(noise=None)
    evolved = prop.propagate(obs, circuit)
    assert evolved.norm_squared() == pytest.approx(original_norm, rel=1e-9)


def test_noise_strictly_reduces_norm():
    obs = PauliTermSum({ps(0, 0b0001): 1.0, ps(0, 0b0010): 0.5})
    original_norm = obs.norm_squared()
    # Z_0 commutes with both terms → trivial rotation, noise alone reduces norm
    generator = ps(0, 0b0001)  # Z_0
    circuit = PauliCircuit([PauliRotation(generator, math.pi / 4)])
    noise = UniformNoiseModel(damping=0.3)
    prop = PauliPropagator(noise=noise)
    evolved = prop.propagate(obs, circuit)
    assert evolved.norm_squared() < original_norm


def test_truncation_removes_heavy_terms():
    obs = PauliTermSum({ps(0, 0b0001): 1.0})  # Z_0, weight = 1
    # X_0 X_1 anticommutes with Z_0 I → spawns Y_0 X_1 (weight 2)
    generator = ps(0b0011, 0)  # X_0 X_1
    circuit = PauliCircuit([PauliRotation(generator, math.pi / 4)])
    trunc = TruncationPolicy(weight_cutoff=1, coeff_cutoff=0.0)
    prop_trunc = PauliPropagator(truncation=trunc)
    prop_free = PauliPropagator(truncation=None)
    evolved_trunc = prop_trunc.propagate(obs, circuit)
    evolved_free = prop_free.propagate(obs, circuit)
    assert len(evolved_trunc) <= len(evolved_free)
    for term, _ in evolved_trunc.items():
        assert term.weight <= 1


def test_n_threads_single_thread():
    obs = PauliTermSum({ps(0, 0b0001): 1.0, ps(0, 0b0010): 0.5j})
    generator = ps(0b1111, 0)  # X_0 X_1 X_2 X_3
    circuit = PauliCircuit([PauliRotation(generator, 0.3)])
    prop1 = PauliPropagator(n_threads=1)
    prop4 = PauliPropagator(n_threads=4)
    ev1 = prop1.propagate(obs, circuit)
    ev4 = prop4.propagate(obs, circuit)
    for term, c1 in ev1.items():
        c4 = ev4[term]
        assert abs(c1 - c4) < 1e-10, f"Thread count changed result for term {term}"


def test_n_threads_does_not_raise():
    prop = PauliPropagator(n_threads=2)
    obs = PauliTermSum({ps(0, 0b0001): 1.0})
    prop.propagate(obs, empty_circuit())
