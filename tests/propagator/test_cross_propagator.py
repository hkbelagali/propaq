"""
Compare MajoranaPropagator and PauliPropagator outputs against each other and Qiskit simulators.
"""

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CPhaseGate, PhaseGate, RZGate, SwapGate, XGate, XXPlusYYGate
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import DensityMatrix, Kraus, SparsePauliOp, Statevector

from propaq.circuits import MajoranaCircuit, PauliCircuit
from propaq.datatypes import MajoranaTermSum, PauliTermSum
from propaq.noise import TruncationPolicy, UniformNoiseModel
from propaq.propagators.majorana import MajoranaPropagator
from propaq.propagators.pauli import PauliPropagator

N_QUBITS = 4
N_MODES = 2 * N_QUBITS
OBSERVABLES = ["ZZZZ", "XXXX", "IIZZ"]
TRUNC = TruncationPolicy(weight_cutoff=10000, coeff_cutoff=0.0)


def _make_circuit(seed: int = 0) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    gates = [
        (lambda: XXPlusYYGate(rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)), 2),
        (lambda: PhaseGate(rng.uniform(0, 2 * np.pi)), 1),
        (lambda: RZGate(rng.uniform(0, 2 * np.pi)), 1),
        (lambda: CPhaseGate(rng.uniform(0, 2 * np.pi)), 2),
        (lambda: SwapGate(), 2),
        (lambda: XGate(), 1),
    ]
    qc = QuantumCircuit(N_QUBITS)
    for _ in range(8):
        factory, nq = gates[rng.integers(len(gates))]
        qc.append(factory(), rng.choice(N_QUBITS, size=nq, replace=False).tolist())
    return qc


def _dm_expectation_noisy(qc: QuantumCircuit, obs: SparsePauliOp, damping: float) -> float:
    """Schrödinger-picture DensityMatrix simulation with per-layer per-qubit depolarizing noise."""
    n = qc.num_qubits
    p = 0.75 * (1.0 - math.exp(-damping))
    s = math.sqrt
    depol = Kraus([
        s(1 - p) * np.eye(2),
        s(p / 3) * np.array([[0, 1], [1, 0]]),
        s(p / 3) * np.array([[0, -1j], [1j, 0]]),
        s(p / 3) * np.array([[1, 0], [0, -1]]),
    ])

    dm = DensityMatrix.from_int(0, 2**n)
    for layer in circuit_to_dag(qc).layers():
        layer_qc = QuantumCircuit(n)
        for node in layer["graph"].topological_op_nodes():
            if node.op.name in ("measure", "barrier"):
                continue
            layer_qc.append(node.op, [qc.find_bit(q).index for q in node.qargs])
        dm = dm.evolve(layer_qc)
        for q in range(n):
            dm = dm.evolve(depol, qargs=[q])

    return dm.expectation_value(obs).real


@pytest.mark.parametrize("obs_str", OBSERVABLES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_noiseless_majorana_pauli_agree_with_statevector(seed, obs_str):
    qc = _make_circuit(seed)
    obs = SparsePauliOp(obs_str)
    sv_ev = Statevector(qc).expectation_value(obs).real

    mc = MajoranaCircuit.from_qiskit(qc, n_modes=N_MODES)
    pc = PauliCircuit.from_qiskit(qc)
    maj_obs = MajoranaTermSum.from_sparse_pauli_op(obs)
    pau_obs = PauliTermSum.from_sparse_pauli_op(obs)

    maj_ev = MajoranaPropagator(None, TRUNC).expectation_value(maj_obs, mc, fock_state=0).expectation_value
    pau_ev = PauliPropagator(None, TRUNC).expectation_value(pau_obs, pc, fock_state=0).expectation_value

    assert np.isclose(maj_ev, sv_ev, atol=1e-6), f"Majorana vs Statevector: {maj_ev} vs {sv_ev}"
    assert np.isclose(pau_ev, sv_ev, atol=1e-6), f"Pauli vs Statevector: {pau_ev} vs {sv_ev}"


@pytest.mark.parametrize("obs_str", OBSERVABLES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_noisy_majorana_pauli_agree(seed, obs_str):
    qc = _make_circuit(seed)
    obs = SparsePauliOp(obs_str)
    noise = UniformNoiseModel(damping=0.05)

    mc = MajoranaCircuit.from_qiskit(qc, n_modes=N_MODES)
    pc = PauliCircuit.from_qiskit(qc)
    maj_obs = MajoranaTermSum.from_sparse_pauli_op(obs)
    pau_obs = PauliTermSum.from_sparse_pauli_op(obs)

    maj_ev = MajoranaPropagator(noise, TRUNC).expectation_value(maj_obs, mc, fock_state=0).expectation_value
    pau_ev = PauliPropagator(noise, TRUNC).expectation_value(pau_obs, pc, fock_state=0).expectation_value

    assert np.isclose(maj_ev, pau_ev, atol=1e-10), f"Majorana vs Pauli (noisy): {maj_ev} vs {pau_ev}"


@pytest.mark.parametrize("obs_str", OBSERVABLES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_noisy_propagators_match_density_matrix(seed, obs_str):
    qc = _make_circuit(seed)
    obs = SparsePauliOp(obs_str)
    damping = 0.1
    noise = UniformNoiseModel(damping=damping)

    mc = MajoranaCircuit.from_qiskit(qc, n_modes=N_MODES)
    pc = PauliCircuit.from_qiskit(qc)
    maj_obs = MajoranaTermSum.from_sparse_pauli_op(obs)
    pau_obs = PauliTermSum.from_sparse_pauli_op(obs)

    maj_ev = MajoranaPropagator(noise, TRUNC).expectation_value(maj_obs, mc, fock_state=0).expectation_value
    pau_ev = PauliPropagator(noise, TRUNC).expectation_value(pau_obs, pc, fock_state=0).expectation_value
    dm_ev = _dm_expectation_noisy(qc, obs, damping)

    assert np.isclose(maj_ev, dm_ev, atol=1e-6), f"Majorana vs DensityMatrix: {maj_ev} vs {dm_ev}"
    assert np.isclose(pau_ev, dm_ev, atol=1e-6), f"Pauli vs DensityMatrix: {pau_ev} vs {dm_ev}"
