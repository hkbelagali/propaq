from unittest.mock import patch

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

qu = pytest.importorskip("quimb")
qtn = pytest.importorskip("quimb.tensor")

from propaq import hybrid as hybrid_mod  # noqa: E402
from propaq.circuits import PauliCircuit  # noqa: E402
from propaq.datatypes import PauliTermSum  # noqa: E402
from propaq.hybrid import hybrid_expectation_value  # noqa: E402
from propaq.propagators.pauli import PauliPropagator  # noqa: E402


def test_direct_mps_matches_dense_reference():
    n = 4
    c1 = QuantumCircuit(n)
    c1.rx(0.7, 0)
    c1.rz(0.3, 1)
    c1.cx(1, 2)
    c1.rx(0.5, 3)

    c2 = QuantumCircuit(n)
    c2.h(0)
    c2.cx(0, 1)
    c2.cx(1, 2)
    c2.cx(2, 3)
    c2.rz(0.9, 3)

    mps = qtn.MPS_computational_state("0" * n)
    h_gate = qu.hadamard()
    cx_gate = qu.CNOT()
    mps.gate_(h_gate, 0, contract=True)
    mps.gate_(cx_gate, (0, 1), contract="swap+split", max_bond=8)
    mps.gate_(cx_gate, (1, 2), contract="swap+split", max_bond=8)
    mps.gate_(cx_gate, (2, 3), contract="swap+split", max_bond=8)
    mps.gate_(qu.rotation(0.9, "z"), 3, contract=True)

    observable = SparsePauliOp("Z" + "I" * (n - 1))
    pauli_observable = PauliTermSum.from_sparse_pauli_op(observable)
    pc1 = PauliCircuit.from_qiskit(c1)

    theta = PauliPropagator().propagate(pauli_observable, pc1)
    value = hybrid_expectation_value(theta, mps)

    full = c2.compose(c1)
    reference = Statevector(full).expectation_value(observable).real

    assert np.isclose(value, reference, atol=1e-6), (
        f"direct-MPS path diverged from dense reference: {value} vs {reference}"
    )


def test_direct_mps_skips_build_mps():
    n = 2
    pc1 = PauliCircuit.from_qiskit(QuantumCircuit(n))
    observable = PauliTermSum.from_sparse_pauli_op(SparsePauliOp("ZI"))
    mps = qtn.MPS_computational_state("0" * n)

    theta = PauliPropagator().propagate(observable, pc1)
    with patch.object(hybrid_mod, "_build_mps") as mock_build_mps:
        value = hybrid_mod.hybrid_expectation_value(theta, mps)

    assert not mock_build_mps.called
    assert np.isclose(value, 1.0, atol=1e-6)
