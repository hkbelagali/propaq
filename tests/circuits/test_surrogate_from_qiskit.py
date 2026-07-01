"""Tests for SurrogatePauliCircuit.from_qiskit / SurrogateMajoranaCircuit.from_qiskit.

Cross-validates the symbolic (parameterized) conversion path against the existing
concrete `PauliCircuit.from_qiskit`/`MajoranaCircuit.from_qiskit` + propagator path,
by binding the same Qiskit circuit to concrete values and comparing results.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import SparsePauliOp

from propaq import (
    MajoranaPropagator,
    MajoranaSurrogatePropagator,
    MajoranaTermSum,
    PauliPropagator,
    PauliSurrogatePropagator,
    PauliTermSum,
    SurrogateMajoranaCircuit,
    SurrogatePauliCircuit,
    VariationalSurrogateModel,
)
from propaq.circuits.majorana.circuit import MajoranaCircuit
from propaq.circuits.pauli.circuit import PauliCircuit

N_QUBITS = 3
N_MODES = 2 * N_QUBITS
OBS = SparsePauliOp("ZZI")


def _pauli_variational_model(qc: QuantumCircuit) -> tuple[VariationalSurrogateModel, SurrogatePauliCircuit]:
    obs = PauliTermSum.from_sparse_pauli_op(OBS)
    circuit = SurrogatePauliCircuit.from_qiskit(qc)
    model = PauliSurrogatePropagator().build(obs, circuit, initial_state=0)
    return VariationalSurrogateModel(model, circuit.parameter_sources, circuit.qiskit_parameters), circuit


def _pauli_concrete_ev(qc: QuantumCircuit, binding: dict) -> float:
    obs = PauliTermSum.from_sparse_pauli_op(OBS)
    bound = qc.assign_parameters(binding)
    circuit = PauliCircuit.from_qiskit(bound)
    return PauliPropagator().expectation_value(obs, circuit, initial_state=0).expectation_value


def _majorana_variational_model(
    qc: QuantumCircuit,
) -> tuple[VariationalSurrogateModel, SurrogateMajoranaCircuit]:
    obs = MajoranaTermSum.from_sparse_pauli_op(OBS)
    circuit = SurrogateMajoranaCircuit.from_qiskit(qc, n_modes=N_MODES)
    model = MajoranaSurrogatePropagator().build(obs, circuit, initial_state=0)
    return VariationalSurrogateModel(model, circuit.parameter_sources, circuit.qiskit_parameters), circuit


def _majorana_concrete_ev(qc: QuantumCircuit, binding: dict) -> float:
    obs = MajoranaTermSum.from_sparse_pauli_op(OBS)
    bound = qc.assign_parameters(binding)
    circuit = MajoranaCircuit.from_qiskit(bound, n_modes=N_MODES)
    return MajoranaPropagator().expectation_value(obs, circuit, initial_state=0).expectation_value


def _variational_model(kind: str, qc: QuantumCircuit):
    return _pauli_variational_model(qc) if kind == "pauli" else _majorana_variational_model(qc)


def _concrete_ev(kind: str, qc: QuantumCircuit, binding: dict) -> float:
    return _pauli_concrete_ev(qc, binding) if kind == "pauli" else _majorana_concrete_ev(qc, binding)


def _from_qiskit(kind: str, qc: QuantumCircuit):
    if kind == "pauli":
        return SurrogatePauliCircuit.from_qiskit(qc)
    return SurrogateMajoranaCircuit.from_qiskit(qc, n_modes=N_MODES)


def _bare_parameter_circuit() -> tuple[QuantumCircuit, Parameter]:
    theta = Parameter("theta")
    qc = QuantumCircuit(N_QUBITS)
    qc.rz(theta, 0)
    qc.p(theta, 1)   # same Parameter, same scale -> shares a param_index with the rz above
    qc.rz(0.4, 2)    # concrete float mixed in -> its own constant slot
    return qc, theta


def _xx_plus_yy_circuit() -> tuple[QuantumCircuit, Parameter, Parameter]:
    theta = Parameter("theta")
    beta = Parameter("beta")
    qc = QuantumCircuit(N_QUBITS)
    qc.append(XXPlusYYGate(theta, beta), [0, 1])
    return qc, theta, beta


def _affine_circuit() -> tuple[QuantumCircuit, Parameter, Parameter]:
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = QuantumCircuit(N_QUBITS)
    qc.rz(2 * theta + phi + 1, 0)
    return qc, theta, phi


def _x_and_swap_circuit() -> tuple[QuantumCircuit, Parameter]:
    # x/swap have no Qiskit gate parameters at all; this exercises the constant-only
    # code path (no free Parameter) mixed with a genuinely parameterized gate.
    theta = Parameter("theta")
    qc = QuantumCircuit(N_QUBITS)
    qc.x(0)
    qc.swap(0, 1)
    qc.rz(theta, 2)
    return qc, theta


@pytest.mark.parametrize("kind", ["pauli", "majorana"])
class TestFromQiskitAgreesWithConcrete:
    def test_bare_parameter(self, kind):
        qc, theta = _bare_parameter_circuit()
        variational, _ = _variational_model(kind, qc)
        rng = np.random.default_rng(0)
        for _ in range(5):
            v = float(rng.uniform(-np.pi, np.pi))
            got = variational.evaluate({theta: v})
            want = _concrete_ev(kind, qc, {theta: v})
            assert got == pytest.approx(want, abs=1e-9)

    def test_shared_parameter_collapses_n_params(self, kind):
        qc, theta = _bare_parameter_circuit()
        variational, circuit = _variational_model(kind, qc)
        # theta shared (bare, scale 1.0) across rz+p -> 1 slot; rz(0.4) -> 1 constant slot.
        assert circuit.n_params == 2
        assert variational.n_params == 2
        assert variational.parameters == (theta,)

    def test_xx_plus_yy_symbolic_beta(self, kind):
        qc, theta, beta = _xx_plus_yy_circuit()
        variational, _ = _variational_model(kind, qc)
        rng = np.random.default_rng(1)
        for _ in range(5):
            binding = {
                theta: float(rng.uniform(-np.pi, np.pi)),
                beta: float(rng.uniform(-np.pi, np.pi)),
            }
            got = variational.evaluate(binding)
            want = _concrete_ev(kind, qc, binding)
            assert got == pytest.approx(want, abs=1e-9)

    def test_affine_multi_parameter_expression(self, kind):
        qc, theta, phi = _affine_circuit()
        variational, _ = _variational_model(kind, qc)
        rng = np.random.default_rng(2)
        for _ in range(5):
            binding = {
                theta: float(rng.uniform(-np.pi, np.pi)),
                phi: float(rng.uniform(-np.pi, np.pi)),
            }
            got = variational.evaluate(binding)
            want = _concrete_ev(kind, qc, binding)
            assert got == pytest.approx(want, abs=1e-9)

    def test_x_and_swap_gates(self, kind):
        qc, theta = _x_and_swap_circuit()
        variational, _ = _variational_model(kind, qc)
        rng = np.random.default_rng(3)
        for _ in range(5):
            binding = {theta: float(rng.uniform(-np.pi, np.pi))}
            got = variational.evaluate(binding)
            want = _concrete_ev(kind, qc, binding)
            assert got == pytest.approx(want, abs=1e-9)

    def test_positional_evaluate_matches_dict(self, kind):
        qc, theta, phi = _affine_circuit()
        variational, _ = _variational_model(kind, qc)
        binding = {theta: 0.7, phi: -0.3}
        by_dict = variational.evaluate(binding)
        # `.parameters` order is implementation-defined (Parameter sets aren't ordered
        # by declaration), so build the positional sequence from it directly.
        by_seq = variational.evaluate([binding[p] for p in variational.parameters])
        assert by_seq == pytest.approx(by_dict)

    def test_non_affine_expression_raises(self, kind):
        theta = Parameter("theta")
        qc = QuantumCircuit(2)
        qc.rz(theta * theta, 0)
        with pytest.raises(ValueError, match="not affine"):
            _from_qiskit(kind, qc)

    def test_unsupported_gate_raises(self, kind):
        qc = QuantumCircuit(1)
        qc.h(0)
        with pytest.raises(ValueError, match="Unsupported gate h"):
            _from_qiskit(kind, qc)
