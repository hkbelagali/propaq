"""Dense-matrix checks of a traced basis (small systems only).

Conventions of the dense models:
  Pauli     qubit q is tensor factor q (qubit 0 is the least significant
            index bit). Term (x, z) = i^(x.z) X^x Z^z.
  Majorana  gamma_{2q} = Z..Z X_q, gamma_{2q+1} = Z..Z Y_q (JW).
            Term = i^r(|a|) * sorted product, r(p) = 1 if p mod 4 in {2, 3}.
  Z4        digit-wise X^x Z^z on d = 4 qudits. Qudit 0 is least significant.
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
from scipy.linalg import expm

from .evaluate import evaluate
from .fallback import phase

I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.diag([1, -1]).astype(complex)


def _kron_le(factors):
    """Kronecker product where factors[0] is the least significant tensor factor."""
    return functools.reduce(np.kron, reversed(factors))


def dense_pauli(words, n):
    bits = [(words[i // 64] >> (i % 64)) & 1 for i in range(2 * n)]
    mats = []
    for q in range(n):
        xb, zb = bits[2 * q], bits[2 * q + 1]
        m = np.linalg.matrix_power(PX, xb) @ np.linalg.matrix_power(PZ, zb)
        mats.append((1j) ** (xb & zb) * m)
    return _kron_le(mats)


def jw_majoranas(n):
    gam = []
    for q in range(n):
        for P in (PX, PY):
            gam.append(_kron_le([PZ] * q + [P] + [I2] * (n - q - 1)))
    return gam


def dense_majorana(words, n):
    gam = jw_majoranas(n)
    idx = [i for i in range(2 * n) if (words[i // 64] >> (i % 64)) & 1]
    m = np.eye(2**n, dtype=complex)
    for i in idx:
        m = m @ gam[i]
    return (1j) ** ((len(idx) >> 1) & 1) * m


def dense_z4(words, n):
    X4 = np.roll(np.eye(4), 1, axis=0).astype(complex)
    Z4 = np.diag([(1j) ** j for j in range(4)])
    mats = []
    for u in range(n):
        b = [(words[(4 * u + i) // 64] >> ((4 * u + i) % 64)) & 1 for i in range(4)]
        x, z = b[0] + 2 * b[1], b[2] + 2 * b[3]
        mats.append(np.linalg.matrix_power(X4, x) @ np.linalg.matrix_power(Z4, z))
    return _kron_le(mats)


DENSE = {"pauli": (dense_pauli, 2, 2), "majorana": (dense_majorana, 2, 2), "z4": (dense_z4, 4, 4)}


def all_keys(bits_per_unit, n_units):
    for v in range(2 ** (bits_per_unit * n_units)):
        yield [v]


def check_algebra(ir, family, n):
    """character and product of every pair of strings against dense matrices.

    Returns the number of pairs checked. Raises AssertionError on a mismatch.
    """
    dense, b, _ = DENSE[family]
    meta = ir.meta
    M, D = meta["char_modulus"], meta["phase_order"]
    keys = list(all_keys(b, n))
    mats = {k[0]: dense(k, n) for k in keys}
    count = 0
    for g, t in itertools.product(keys, repeat=2):
        A, B = mats[g[0]], mats[t[0]]
        chi = evaluate(ir.programs["character"], [g, t])[0] % M
        assert np.allclose(A @ B, phase(chi, M) * (B @ A)), ("character", g, t, chi)
        k, e = evaluate(ir.programs["product"], [g, t])
        assert np.allclose(A @ B, phase(e, D) * mats[k[0]]), ("product", g, t)
        count += 1
    return count


def pauli_weight_of_matrix(mat, n):
    """Weight of the Pauli string if `mat` is proportional to one, else raise."""
    singles = [I2, PX, PY, PZ]
    for combo in itertools.product(range(4), repeat=n):
        P = _kron_le([singles[c] for c in combo])
        c = np.trace(P.conj().T @ mat) / 2**n
        if abs(c) > 1e-9:
            assert np.allclose(mat, c * P), "operator is not a single Pauli string"
            return sum(1 for x in combo if x)
    raise AssertionError("zero operator")


def check_weight_trace(ir, family, n, state_one_is_zero=False):
    """weight and trace against dense matrices (Pauli and Majorana).

    Weight: Pauli weight of the dense operator (Pauli: the string itself,
    Majorana: its Jordan-Wigner image). Trace: <s|O|s> for every
    computational state s, with unit bit q of `s` equal to qubit q.
    `state_one_is_zero` flips the mapping of the diagonal-state bit to |0>/|1>.
    """
    dense, b, _ = DENSE[family]
    count = 0
    for key in all_keys(b, n):
        mat = dense(key, n)
        w = evaluate(ir.programs["weight"], [key], n_units=n)[0]
        if family in ("pauli", "majorana"):
            assert w == pauli_weight_of_matrix(mat, n), ("weight", key, w)
        for s in range(2**n):
            idx = s ^ (2**n - 1) if state_one_is_zero else s
            want = mat[idx, idx]
            nz, e = evaluate(ir.programs["trace"], [key, [s]], n_units=n)
            got = phase(e, ir.meta["phase_order"]).real if nz else 0.0
            assert abs(got - want.real) < 1e-12 and abs(want.imag) < 1e-12 or family == "z4", ("trace", key, s, got, want)
            count += 1
    return count


def dense_rotation(family, g_words, theta, n, sign=1.0):
    """The rotation unitary of the convention in `fallback`."""
    dense = DENSE[family][0]
    W = dense(g_words, n)
    if family == "z4":
        return expm(-0.25j * sign * theta * (W + W.conj().T))
    return expm(-0.5j * sign * theta * W)


def dense_heisenberg(family, observable, circuit, n):
    """Dense U^dag O U for a circuit [(g_words, theta[, sign])]. `observable` is {key: coeff}."""
    dense = DENSE[family][0]
    dim = (2 if family != "z4" else 4) ** n
    O = sum((c * dense(list(k), n) for k, c in observable.items()), np.zeros((dim, dim), dtype=complex))
    U = np.eye(dim, dtype=complex)
    for gate in circuit:
        U = dense_rotation(family, list(gate[0]), gate[1], n, gate[2] if len(gate) > 2 else 1.0) @ U
    return U.conj().T @ O @ U


def terms_to_dense(family, terms, n):
    dense = DENSE[family][0]
    dim = (2 if family != "z4" else 4) ** n
    return sum((c * dense(list(k), n) for k, c in terms.items()), np.zeros((dim, dim), dtype=complex))
