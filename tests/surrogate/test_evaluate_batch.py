"""
Batch evaluation of compiled surrogate models
"""

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import XXPlusYYGate

from propaq import (
    PauliString,
    PauliSurrogatePropagator,
    PauliTermSum,
    SurrogatePauliCircuit,
    VariationalSurrogateModel,
)
from propaq.circuits.pauli import PauliCircuit
from propaq.circuits.pauli.rotation import PauliRotation
from propaq.datatypes.abstract import BitMask

N = 4  # n_qubits for all tests


def ps(x: int, z: int) -> PauliString:
    return PauliString(BitMask(x), BitMask(z), N)


def build_model():
    obs = PauliTermSum({ps(0, 0b0001): 1.0})  # Z_0
    rotations = [
        PauliRotation(ps(0b0001, 0), 0.0),  # X_0
        PauliRotation(ps(0b0010, 0b0001), 0.0),  # X_1 Z_0
        PauliRotation(ps(0b0100, 0b0010), 0.0),  # X_2 Z_1
    ]
    circ = PauliCircuit(rotations)
    sc = SurrogatePauliCircuit.from_pauli_circuit(circ, param_indices=[0, 1, 2])
    return PauliSurrogatePropagator().build(obs, sc, initial_state=0)


class TestEvaluateBatch:
    def test_matches_per_assignment_evaluate(self):
        model = build_model()
        param_sets = [[0.1 * (k + 1), 0.2 * (k + 1), 0.3 * (k + 1)] for k in range(5)]
        batch = model.evaluate_batch(param_sets)
        assert batch == pytest.approx([model.evaluate(p) for p in param_sets], rel=1e-12)

    def test_empty_batch(self):
        model = build_model()
        assert model.evaluate_batch([]) == []

    def test_short_param_set_raises(self):
        model = build_model()
        with pytest.raises(ValueError):
            model.evaluate_batch([[0.1, 0.2, 0.3], [0.1]])


class TestVariationalBatch:
    def build_variational(self):
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2)
        qc.rz(theta, 0)
        qc.append(XXPlusYYGate(phi), [0, 1])
        qc.rz(theta, 1)
        sc = SurrogatePauliCircuit.from_qiskit(qc)
        obs = PauliTermSum({PauliString(BitMask(0), BitMask(0b01), 2): 1.0})
        model = PauliSurrogatePropagator().build(obs, sc, initial_state=0b01)
        return (
            VariationalSurrogateModel(model, sc.parameter_sources, sc.qiskit_parameters),
            theta,
            phi,
        )

    def test_batch_matches_evaluate_for_both_binding_forms(self):
        variational, theta, phi = self.build_variational()
        bindings = [
            {theta: 0.3, phi: 1.1},
            {theta: -0.7, phi: 0.4},
            [0.5, 2.0],  # positional, aligned with variational.parameters
        ]
        batch = variational.evaluate_batch(bindings)
        assert batch == pytest.approx([variational.evaluate(b) for b in bindings], rel=1e-12)
        # Sanity: the model actually varies over the batch.
        assert max(batch) - min(batch) > 1e-6
