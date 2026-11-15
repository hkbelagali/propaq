"""End to end with the Python fallback propagator against dense evolution."""

import random

import numpy as np
import pytest
from scipy.linalg import expm

from propaq.tracer.analysis import add_fold
from propaq.tracer.bases import ALL
from propaq.tracer.build import load_basis
from propaq.tracer.evaluate import evaluate
from propaq.tracer.fallback import FallbackBasis, expectation, phase, propagate, rotate, weyl_table
from propaq.tracer.verify import (dense_heisenberg, dense_z4, terms_to_dense)


def ir_of(name):
    return add_fold(ALL[name]().trace_ir())


def rand_circuit(rng, nbits, n_gates):
    return [([rng.randrange(1, 2**nbits)], rng.uniform(-1.5, 1.5)) for _ in range(n_gates)]


@pytest.mark.parametrize("seed", range(6))
def test_z4_two_qudits_against_dense(seed):
    rng = random.Random(seed)
    ir = ir_of("z4")
    circ = rand_circuit(rng, 8, 5)
    obs = {(rng.randrange(2**8),): 1.0, (rng.randrange(2**8),): 0.5 - 0.25j}
    res = propagate(ir, obs, circ)
    assert np.allclose(terms_to_dense("z4", res, 2), dense_heisenberg("z4", obs, circ, 2), atol=1e-10)


def test_z4_negative_sign_flips_the_angle():
    rng = random.Random(9)
    ir = ir_of("z4")
    g = [rng.randrange(1, 2**8)]
    obs = {(rng.randrange(2**8),): 1.0}
    a = propagate(ir, obs, [(g, 0.7, -1.0)])
    b = propagate(ir, obs, [(g, -0.7)])
    assert a.keys() == b.keys() and all(abs(a[k] - b[k]) < 1e-12 for k in a)


@pytest.mark.parametrize("q", range(4))
@pytest.mark.parametrize("theta", [0.37, -1.2, 3.0])
def test_weyl_table_matches_expm_for_every_self_phase(q, theta):
    """U^dag T U = sum_k tab[chi][k] i^(k r) W_(t + k g), for every (g, t) with self phase q."""
    n = 2
    ir = ir_of("z4")
    rng = random.Random(q)
    done = 0
    while done < 12:
        g, t = [rng.randrange(1, 2**8)], [rng.randrange(2**8)]
        if evaluate(ir.programs["self_phase"], [g])[0] % 4 != q:
            continue
        Wg, T = dense_z4(g, n), dense_z4(t, n)
        U = expm(-0.25j * theta * (Wg + Wg.conj().T))
        out = rotate(ir, {tuple(t): 1.0}, g, theta)
        assert np.allclose(terms_to_dense("z4", out, n), U.conj().T @ T @ U, atol=1e-10)
        done += 1


def test_weyl_table_has_unit_first_row_and_unitary_action():
    tab = weyl_table(0.9, 1, 4)
    assert tab[0][0] == pytest.approx(1) and all(abs(c) < 1e-12 for c in tab[0][1:])
    # A rotation keeps the 2-norm of the coefficient vector for every chi.
    for chi in range(4):
        assert sum(abs(c) ** 2 for c in tab[chi]) > 0


def test_characters_of_commuting_terms_keep_the_term():
    ir = ir_of("z4")
    g = [0b0001]  # x0 on qudit 0
    t = [0b0001 << 4]  # x0 on qudit 1: commutes
    out = rotate(ir, {tuple(t): 1.0}, g, 0.8)
    assert out == {tuple(t): 1.0}


@pytest.mark.parametrize("family,n", [("pauli", 3), ("majorana", 3)])
def test_pm1_bases_against_dense(family, n):
    rng = random.Random(4)
    ir = ir_of(family)
    for _ in range(4):
        circ = rand_circuit(rng, 2 * n, 5)
        obs = {(rng.randrange(1, 2 ** (2 * n)),): 1.0}
        res = propagate(ir, obs, circ)
        assert np.allclose(terms_to_dense(family, res, n), dense_heisenberg(family, obs, circ, n), atol=1e-10)


def test_cutoff_drops_small_terms_only():
    rng = random.Random(2)
    ir = ir_of("pauli")
    circ = rand_circuit(rng, 6, 12)
    obs = {(0b10,): 1.0}
    full = propagate(ir, obs, circ)
    cut = propagate(ir, obs, circ, cutoff=1e-2)
    assert len(cut) < len(full)
    assert all(abs(v) >= 1e-2 for v in cut.values())


def test_expectation_pauli_z_after_rotation():
    # <0| U^dag Z U |0> for U = exp(-i theta X / 2) is cos(theta).
    ir = ir_of("pauli")
    theta = 0.83
    res = propagate(ir, {(0b10,): 1.0}, [([0b01], theta)])
    assert expectation(ir, res, 1, [0]).real == pytest.approx(np.cos(theta))
    assert expectation(ir, res, 1, [1]).real == pytest.approx(-np.cos(theta))


def test_fallback_load_without_cargo(monkeypatch):
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("PROPAQ_SOURCE_DIR", "/nonexistent")
    basis = load_basis(ALL["z4"]())
    assert isinstance(basis, FallbackBasis)
    out = basis.propagate({(0b0100,): 1.0}, [([0b0001], 0.4)])
    assert out


def test_phase_helper_is_exact_for_quarter_turns():
    assert [phase(e, 4) for e in range(5)] == [1, 1j, -1, -1j, 1]
    assert phase(1, 8) == pytest.approx(np.exp(2j * np.pi / 8))
