"""Tests that from_qiskit accepts arbitrary Qiskit gates via transpile-based decomposition.

Each representation's from_qiskit (PauliCircuit, MajoranaCircuit) is exercised against
gates well outside the native rotation basis (xx_plus_yy, p, rz, rx, ry, cp, x, swap),
including multi-qubit UnitaryGate instances, and cross-checked against Statevector.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    CCXGate,
    CHGate,
    CPhaseGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CSwapGate,
    CXGate,
    CYGate,
    CZGate,
    HGate,
    PhaseGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZXGate,
    RZZGate,
    SdgGate,
    SGate,
    SwapGate,
    SXGate,
    TdgGate,
    TGate,
    UGate,
    UnitaryGate,
    XGate,
    XXPlusYYGate,
    YGate,
    ZGate,
    iSwapGate,
)
from qiskit.quantum_info import SparsePauliOp, Statevector, random_unitary

from propaq.circuits import MajoranaCircuit, PauliCircuit
from propaq.datatypes import MajoranaTermSum, PauliTermSum
from propaq.noise import TruncationPolicy
from propaq.propagators.majorana import MajoranaPropagator
from propaq.propagators.pauli import PauliPropagator

N_QUBITS = 3
TRUNC = TruncationPolicy(weight_cutoff=10000, coeff_cutoff=0.0)
OBSERVABLES = ["ZZZ", "XXX", "IZZ", "XYZ"]
REPS = ["pauli", "majorana"]

_rng = np.random.default_rng(0)

GATE_POOL = [
    ("h", HGate(), 1),
    ("x", XGate(), 1),
    ("y", YGate(), 1),
    ("z", ZGate(), 1),
    ("s", SGate(), 1),
    ("sdg", SdgGate(), 1),
    ("t", TGate(), 1),
    ("tdg", TdgGate(), 1),
    ("sx", SXGate(), 1),
    ("rx", RXGate(_rng.uniform(0, 2 * np.pi)), 1),
    ("ry", RYGate(_rng.uniform(0, 2 * np.pi)), 1),
    ("rz", RZGate(_rng.uniform(0, 2 * np.pi)), 1),
    ("p", PhaseGate(_rng.uniform(0, 2 * np.pi)), 1),
    ("u", UGate(_rng.uniform(0, np.pi), _rng.uniform(0, 2 * np.pi), _rng.uniform(0, 2 * np.pi)), 1),
    ("cx", CXGate(), 2),
    ("cy", CYGate(), 2),
    ("cz", CZGate(), 2),
    ("ch", CHGate(), 2),
    ("crx", CRXGate(_rng.uniform(0, 2 * np.pi)), 2),
    ("cry", CRYGate(_rng.uniform(0, 2 * np.pi)), 2),
    ("crz", CRZGate(_rng.uniform(0, 2 * np.pi)), 2),
    ("cp", CPhaseGate(_rng.uniform(0, 2 * np.pi)), 2),
    ("swap", SwapGate(), 2),
    ("iswap", iSwapGate(), 2),
    ("rxx", RXXGate(_rng.uniform(0, 2 * np.pi)), 2),
    ("ryy", RYYGate(_rng.uniform(0, 2 * np.pi)), 2),
    ("rzz", RZZGate(_rng.uniform(0, 2 * np.pi)), 2),
    ("rzx", RZXGate(_rng.uniform(0, 2 * np.pi)), 2),
    ("xx_plus_yy", XXPlusYYGate(_rng.uniform(0, 2 * np.pi), _rng.uniform(0, 2 * np.pi)), 2),
    ("ccx", CCXGate(), 3),
    ("cswap", CSwapGate(), 3),
    ("unitary2", UnitaryGate(random_unitary(4, seed=1)), 2),
]

RANDOM_CIRCUIT_POOL = [
    (lambda: HGate(), 1),
    (lambda: TGate(), 1),
    (lambda: RXGate(np.random.uniform(0, 2 * np.pi)), 1),
    (lambda: RYGate(np.random.uniform(0, 2 * np.pi)), 1),
    (lambda: RZGate(np.random.uniform(0, 2 * np.pi)), 1),
    (lambda: CXGate(), 2),
    (lambda: CPhaseGate(np.random.uniform(0, 2 * np.pi)), 2),
    (lambda: SwapGate(), 2),
    (lambda: XXPlusYYGate(np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi)), 2),
]


def _observable(rep_name: str, label: str):
    op = SparsePauliOp(label)
    if rep_name == "pauli":
        return PauliTermSum.from_sparse_pauli_op(op)
    return MajoranaTermSum.from_sparse_pauli_op(op)


def _expectation(rep_name: str, qc: QuantumCircuit, label: str) -> float:
    if rep_name == "pauli":
        circuit = PauliCircuit.from_qiskit(qc)
        prop = PauliPropagator(None, TRUNC)
    else:
        circuit = MajoranaCircuit.from_qiskit(qc, n_modes=2 * qc.num_qubits)
        prop = MajoranaPropagator(None, TRUNC)
    obs = _observable(rep_name, label)
    return prop.expectation_value(obs, circuit, initial_state=0).expectation_value


@pytest.mark.parametrize("rep_name", REPS)
@pytest.mark.parametrize("name,gate,arity", GATE_POOL, ids=[g[0] for g in GATE_POOL])
def test_single_gate_matches_statevector(rep_name, name, gate, arity):
    qc = QuantumCircuit(N_QUBITS)
    qc.append(gate, list(range(arity)))
    sv = Statevector(qc)

    for label in OBSERVABLES:
        want = sv.expectation_value(SparsePauliOp(label)).real
        got = _expectation(rep_name, qc, label)
        assert np.isclose(got, want, atol=1e-6), f"{name} ({rep_name}), obs={label}: {got} vs {want}"


@pytest.mark.parametrize("rep_name", REPS)
@pytest.mark.parametrize("seed", range(5))
def test_random_circuit_matches_statevector(rep_name, seed):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # RANDOM_CIRCUIT_POOL factories draw from the global RNG

    qc = QuantumCircuit(N_QUBITS)
    for _ in range(10):
        factory, arity = RANDOM_CIRCUIT_POOL[rng.integers(len(RANDOM_CIRCUIT_POOL))]
        qubits = rng.choice(N_QUBITS, size=arity, replace=False).tolist()
        qc.append(factory(), qubits)
    sv = Statevector(qc)

    for label in OBSERVABLES:
        want = sv.expectation_value(SparsePauliOp(label)).real
        got = _expectation(rep_name, qc, label)
        assert np.isclose(got, want, atol=1e-6), f"seed={seed} ({rep_name}), obs={label}: {got} vs {want}"


@pytest.mark.parametrize("rep_name", REPS)
def test_non_unitary_op_raises(rep_name):
    qc = QuantumCircuit(1)
    qc.reset(0)
    with pytest.raises(ValueError, match="non-unitary"):
        if rep_name == "pauli":
            PauliCircuit.from_qiskit(qc)
        else:
            MajoranaCircuit.from_qiskit(qc, n_modes=2)
