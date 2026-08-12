"""
Tests for MajoranaCircuit.from_ffsim_uccsd
"""

import numpy as np
import pytest

ffsim = pytest.importorskip("ffsim")

from qiskit.quantum_info import Operator, SparsePauliOp  # noqa: E402

from propaq.circuits.majorana.circuit import MajoranaCircuit  # noqa: E402
from propaq.datatypes.majorana.termsum import MajoranaTermSum  # noqa: E402
from propaq.propagators import MajoranaPropagator  # noqa: E402


def _hf_int(norb: int, nelec: tuple[int, int]) -> int:
    n_alpha, n_beta = nelec
    return sum(1 << k for k in range(n_alpha)) | sum(1 << (norb + k) for k in range(n_beta))


def _expectation(op, norb, nelec, n_modes, obs_pauli, use_propagator: bool) -> float:
    hf_vec = ffsim.hartree_fock_state(norb, nelec)
    if use_propagator:
        mc = MajoranaCircuit.from_ffsim_uccsd(op, n_modes)
        obs = MajoranaTermSum.from_sparse_pauli_op(obs_pauli)
        prop = MajoranaPropagator(n_threads=1)
        return prop.expectation_value(obs, mc, initial_state=_hf_int(norb, nelec)).expectation_value
    vec_out = ffsim.apply_unitary(hf_vec, op, norb=norb, nelec=nelec)
    qvec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(vec_out, norb, nelec)
    return float(np.real(np.conj(qvec) @ Operator(obs_pauli).data @ qvec))


def test_trotter_error_vanishes_with_amplitude_scale():
    norb = 3
    nocc, nvrt = 1, 2
    nelec = (nocc, nocc)
    n_modes = 4 * norb
    rng = np.random.default_rng(13)

    t1_dir = rng.standard_normal((nocc, nvrt)) + 1j * rng.standard_normal((nocc, nvrt))
    t2_dir = rng.standard_normal((nocc, nocc, nvrt, nvrt)) + 1j * rng.standard_normal(
        (nocc, nocc, nvrt, nvrt)
    )

    rng2 = np.random.default_rng(500)
    labels = ["".join(rng2.choice(list("IXYZ"), 2 * norb)) for _ in range(5)]
    coeffs = rng2.standard_normal(len(labels))
    obs_pauli = SparsePauliOp.from_list(list(zip(labels, coeffs))).simplify()

    errors = []
    for scale in [0.1, 0.01, 0.001]:
        op = ffsim.UCCSDOpRestricted(t1=scale * t1_dir, t2=scale * t2_dir)
        truth = _expectation(op, norb, nelec, n_modes, obs_pauli, use_propagator=False)
        got = _expectation(op, norb, nelec, n_modes, obs_pauli, use_propagator=True)
        errors.append(abs(got - truth))

    assert errors[0] > errors[1] > errors[2]
    assert errors[2] == pytest.approx(0.0, abs=1e-6)
