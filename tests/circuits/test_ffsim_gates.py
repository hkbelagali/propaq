"""
Tests for the ffsim orbital-rotation/diagonal-Coulomb gate primitives
"""

import numpy as np
import pytest

ffsim = pytest.importorskip("ffsim")

from qiskit.quantum_info import Operator, SparsePauliOp  # noqa: E402

from propaq.circuits.majorana._ffsim_gates import (  # noqa: E402
    diag_coulomb_generators,
    orbital_rotation_generators,
)
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


def _ffsim_ground_truth(vec_out, norb, nelec, obs_pauli) -> float:
    qvec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(vec_out, norb, nelec)
    return float(np.real(np.conj(qvec) @ Operator(obs_pauli).data @ qvec))


@pytest.mark.parametrize("norb", [2, 3, 4])
def test_orbital_rotation_matches_ffsim_simulation(norb):
    nelec = (min(2, norb), 0)
    n_modes = 4 * norb
    rng = np.random.default_rng(norb)
    mat = _random_unitary(norb, rng)

    hf_vec = ffsim.hartree_fock_state(norb, nelec)
    vec_out = ffsim.apply_orbital_rotation(hf_vec, (mat, None), norb, nelec)

    obs_pauli = _random_observable(2 * norb, seed=100 + norb)
    truth = _ffsim_ground_truth(vec_out, norb, nelec, obs_pauli)

    terms = orbital_rotation_generators(mat, n_modes, mode_offset=0)
    generators = [g for g, _ in terms]
    angles = [a for _, a in terms]
    mc = MajoranaCircuit.from_generators_and_angles(generators, angles, n_modes)
    obs = MajoranaTermSum.from_sparse_pauli_op(obs_pauli)
    prop = MajoranaPropagator(n_threads=1)
    got = prop.expectation_value(obs, mc, initial_state=_hf_int(norb, nelec)).expectation_value

    assert got == pytest.approx(truth, abs=1e-8)


def test_orbital_rotation_spinful():
    norb = 3
    nelec = (2, 1)
    n_modes = 4 * norb
    rng = np.random.default_rng(7)
    mat_a = _random_unitary(norb, rng)
    mat_b = _random_unitary(norb, rng)

    hf_vec = ffsim.hartree_fock_state(norb, nelec)
    vec_out = ffsim.apply_orbital_rotation(hf_vec, (mat_a, mat_b), norb, nelec)

    obs_pauli = _random_observable(2 * norb, seed=201)
    truth = _ffsim_ground_truth(vec_out, norb, nelec, obs_pauli)

    mc = MajoranaCircuit.from_ffsim_orbital_rotation((mat_a, mat_b), norb, n_modes)
    obs = MajoranaTermSum.from_sparse_pauli_op(obs_pauli)
    prop = MajoranaPropagator(n_threads=1)
    got = prop.expectation_value(obs, mc, initial_state=_hf_int(norb, nelec)).expectation_value

    assert got == pytest.approx(truth, abs=1e-8)


def test_diag_coulomb_evolution_matches_ffsim_simulation():
    norb = 2
    nelec = (1, 1)
    n_modes = 4 * norb
    time = 0.37
    rng = np.random.default_rng(2)
    mat_aa = rng.standard_normal((norb, norb))
    mat_aa = mat_aa + mat_aa.T
    mat_bb = rng.standard_normal((norb, norb))
    mat_bb = mat_bb + mat_bb.T
    mat_ab = rng.standard_normal((norb, norb))

    hf_vec = ffsim.hartree_fock_state(norb, nelec)
    vec_out = ffsim.apply_diag_coulomb_evolution(
        hf_vec, (mat_aa, mat_ab, mat_bb), time, norb, nelec
    )

    obs_pauli = _random_observable(2 * norb, seed=301)
    truth = _ffsim_ground_truth(vec_out, norb, nelec, obs_pauli)

    terms = diag_coulomb_generators(mat_aa, mat_ab, mat_bb, time, norb, n_modes)
    generators = [g for g, _ in terms]
    angles = [a for _, a in terms]
    mc = MajoranaCircuit.from_generators_and_angles(generators, angles, n_modes)
    obs = MajoranaTermSum.from_sparse_pauli_op(obs_pauli)
    prop = MajoranaPropagator(n_threads=1)
    got = prop.expectation_value(obs, mc, initial_state=_hf_int(norb, nelec)).expectation_value

    assert got == pytest.approx(truth, abs=1e-8)


def test_diag_coulomb_evolution_with_orbital_rotation_sandwich():
    norb = 2
    nelec = (1, 1)
    n_modes = 4 * norb
    time = 0.42
    rng = np.random.default_rng(9)
    mat_aa = rng.standard_normal((norb, norb))
    mat_aa = mat_aa + mat_aa.T
    mat_bb = rng.standard_normal((norb, norb))
    mat_bb = mat_bb + mat_bb.T
    mat_ab = rng.standard_normal((norb, norb))
    rot = _random_unitary(norb, rng)

    hf_vec = ffsim.hartree_fock_state(norb, nelec)
    vec_out = ffsim.apply_diag_coulomb_evolution(
        hf_vec, (mat_aa, mat_ab, mat_bb), time, norb, nelec, orbital_rotation=rot
    )

    obs_pauli = _random_observable(2 * norb, seed=302)
    truth = _ffsim_ground_truth(vec_out, norb, nelec, obs_pauli)

    mc = MajoranaCircuit.from_ffsim_diag_coulomb_evolution(
        (mat_aa, mat_ab, mat_bb), time, norb, n_modes, orbital_rotation=rot
    )
    obs = MajoranaTermSum.from_sparse_pauli_op(obs_pauli)
    prop = MajoranaPropagator(n_threads=1)
    got = prop.expectation_value(obs, mc, initial_state=_hf_int(norb, nelec)).expectation_value

    assert got == pytest.approx(truth, abs=1e-8)
