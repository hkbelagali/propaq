"""
Tests for MajoranaTermSum.from_ffsim
"""

import numpy as np
import pytest

ffsim = pytest.importorskip("ffsim")

from propaq.datatypes import MajoranaTermSum  # noqa: E402


def _assert_matches(term_sum: MajoranaTermSum, expected):
    got = dict(term_sum.to_sparse_pauli_op().simplify().to_list())
    want = dict(expected.simplify().to_list())
    assert got.keys() == want.keys()
    for label in got:
        assert got[label] == pytest.approx(want[label], abs=1e-9)


def _random_molecular_hamiltonian(norb: int, seed: int) -> ffsim.MolecularHamiltonian:
    rng = np.random.default_rng(seed)
    h1e = rng.standard_normal((norb, norb))
    h1e = h1e + h1e.T
    h2e = rng.standard_normal((norb, norb, norb, norb))
    h2e = h2e + h2e.transpose(1, 0, 3, 2)
    return ffsim.MolecularHamiltonian(one_body_tensor=h1e, two_body_tensor=h2e)


def test_molecular_hamiltonian():
    norb = 2
    ham = _random_molecular_hamiltonian(norb, seed=0)
    got = MajoranaTermSum.from_ffsim(ham)
    expected = ffsim.qiskit.jordan_wigner(ffsim.fermion_operator(ham), norb)
    _assert_matches(got, expected)


def test_diagonal_coulomb_hamiltonian():
    norb = 2
    rng = np.random.default_rng(1)
    one_body = rng.standard_normal((norb, norb))
    one_body = one_body + one_body.conj().T
    mat_aa = rng.standard_normal((norb, norb))
    mat_aa = mat_aa + mat_aa.T
    mat_ab = rng.standard_normal((norb, norb))
    mat_ab = mat_ab + mat_ab.T
    diag_coulomb_mats = np.stack([mat_aa, mat_ab])

    ham = ffsim.DiagonalCoulombHamiltonian(one_body_tensor=one_body, diag_coulomb_mats=diag_coulomb_mats)
    got = MajoranaTermSum.from_ffsim(ham)
    expected = ffsim.qiskit.jordan_wigner(ffsim.fermion_operator(ham), norb)
    _assert_matches(got, expected)


def test_explicit_n_modes_override():
    norb = 2
    ham = _random_molecular_hamiltonian(norb, seed=2)
    got = MajoranaTermSum.from_ffsim(ham, n_modes=4 * norb)
    expected = ffsim.qiskit.jordan_wigner(ffsim.fermion_operator(ham), norb)
    _assert_matches(got, expected)
