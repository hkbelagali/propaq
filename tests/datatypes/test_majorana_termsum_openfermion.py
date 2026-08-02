"""
Tests for MajoranaTermSum.from_openfermion
"""

import numpy as np
import pytest

openfermion = pytest.importorskip("openfermion")

from qiskit.quantum_info import SparsePauliOp  # noqa: E402

from propaq.datatypes import MajoranaTermSum  # noqa: E402


def _openfermion_ground_truth(op, n_qubits: int) -> SparsePauliOp:
    """Convert an OpenFermion FermionOperator to a Qiskit SparsePauliOp via
    OpenFermion's own Jordan-Wigner transform, for use as ground truth."""
    qop = openfermion.transforms.jordan_wigner(op)
    pairs = []
    for term, coeff in qop.terms.items():
        chars = ["I"] * n_qubits
        for idx, action in term:
            chars[idx] = action
        pairs.append(("".join(reversed(chars)), coeff))
    if not pairs:
        pairs = [("I" * n_qubits, 0.0)]
    return SparsePauliOp.from_list(pairs).simplify()


def _assert_matches(term_sum: MajoranaTermSum, expected: SparsePauliOp):
    got = dict(term_sum.to_sparse_pauli_op().simplify().to_list())
    want = dict(expected.to_list())
    assert got.keys() == want.keys()
    for label in got:
        assert got[label] == pytest.approx(want[label], abs=1e-9)


def test_hopping_and_number_number_hamiltonian():
    op = (
        openfermion.FermionOperator("1^ 0", 0.5)
        + openfermion.FermionOperator("0^ 1", 0.5)
        + openfermion.FermionOperator("0^ 0", 1.2)
        + openfermion.FermionOperator("0^ 0 1^ 1", 0.7)
    )
    n_modes = 4
    got = MajoranaTermSum.from_openfermion(op, n_modes)
    _assert_matches(got, _openfermion_ground_truth(op, n_modes // 2))


def test_random_hermitian_two_body_hamiltonian():
    norb = 2
    rng = np.random.default_rng(0)
    h1 = rng.standard_normal((norb, norb))
    h1 = h1 + h1.T
    h2 = rng.standard_normal((norb, norb, norb, norb))

    op = openfermion.FermionOperator()
    for p in range(norb):
        for q in range(norb):
            if h1[p, q]:
                op += openfermion.FermionOperator(f"{p}^ {q}", h1[p, q])
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    coeff = h2[p, q, r, s] + np.conj(h2[s, r, q, p])
                    if coeff:
                        op += openfermion.FermionOperator(f"{p}^ {q} {r}^ {s}", 0.5 * coeff)

    n_modes = 2 * norb
    got = MajoranaTermSum.from_openfermion(op, n_modes)
    _assert_matches(got, _openfermion_ground_truth(op, n_modes // 2))
