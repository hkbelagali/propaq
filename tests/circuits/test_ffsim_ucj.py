"""
Tests for MajoranaCircuit.from_ffsim_ucj
"""

import numpy as np
import pytest

ffsim = pytest.importorskip("ffsim")

from qiskit.quantum_info import Operator, SparsePauliOp  # noqa: E402

from propaq.circuits.majorana.circuit import MajoranaCircuit  # noqa: E402
from propaq.datatypes.majorana.termsum import MajoranaTermSum  # noqa: E402
from propaq.propagators import MajoranaPropagator  # noqa: E402


def _random_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(a)
    return q @ np.diag(np.exp(1j * np.angle(np.diag(r))))


def _hf_int(norb: int, nelec: tuple[int, int]) -> int:
    n_alpha, n_beta = nelec
    return sum(1 << k for k in range(n_alpha)) | sum(1 << (norb + k) for k in range(n_beta))


def _random_observable(n_qubits: int, seed: int) -> SparsePauliOp:
    rng = np.random.default_rng(seed)
    labels = ["".join(rng.choice(list("IXYZ"), n_qubits)) for _ in range(5)]
    coeffs = rng.standard_normal(len(labels))
    return SparsePauliOp.from_list(list(zip(labels, coeffs))).simplify()


def _check_ucj(op, norb, nelec, n_modes, seed):
    hf_vec = ffsim.hartree_fock_state(norb, nelec)
    vec_out = ffsim.apply_unitary(hf_vec, op, norb=norb, nelec=nelec)
    obs_pauli = _random_observable(2 * norb, seed=seed)
    qvec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(vec_out, norb, nelec)
    truth = float(np.real(np.conj(qvec) @ Operator(obs_pauli).data @ qvec))

    mc = MajoranaCircuit.from_ffsim_ucj(op, n_modes)
    obs = MajoranaTermSum.from_sparse_pauli_op(obs_pauli)
    prop = MajoranaPropagator(schedule=None, n_threads=1, progress_bar=False)
    got = prop.expectation_value(obs, mc, initial_state=_hf_int(norb, nelec)).expectation_value

    assert got == pytest.approx(truth, abs=1e-8)


def test_ucj_matches_ffsim_simulation():
    norb = 2
    nelec = (1, 1)
    n_reps = 2
    n_modes = 4 * norb
    rng = np.random.default_rng(11)

    orbital_rotations = np.array([_random_unitary(norb, rng) for _ in range(n_reps)])
    diag_coulomb_mats = np.zeros((n_reps, 2, norb, norb))
    for k in range(n_reps):
        aa = rng.standard_normal((norb, norb))
        aa = aa + aa.T
        ab = rng.standard_normal((norb, norb))
        ab = ab + ab.T
        diag_coulomb_mats[k, 0] = aa
        diag_coulomb_mats[k, 1] = ab
    final_rot = _random_unitary(norb, rng)

    op = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=diag_coulomb_mats,
        orbital_rotations=orbital_rotations,
        final_orbital_rotation=final_rot,
    )
    _check_ucj(op, norb, nelec, n_modes, seed=401)


def test_ucj_no_final_rotation():
    norb = 2
    nelec = (1, 0)
    n_modes = 4 * norb
    rng = np.random.default_rng(15)

    orbital_rotations = np.array([_random_unitary(norb, rng)])
    aa = rng.standard_normal((norb, norb))
    aa = aa + aa.T
    ab = rng.standard_normal((norb, norb))
    ab = ab + ab.T
    diag_coulomb_mats = np.array([[aa, ab]])

    op = ffsim.UCJOpSpinBalanced(diag_coulomb_mats=diag_coulomb_mats, orbital_rotations=orbital_rotations)
    _check_ucj(op, norb, nelec, n_modes, seed=402)
