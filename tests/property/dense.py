"""Dense-matrix references for operator bases.

Each reference builds the matrix of one basis element from raw words.
Raw words are a tuple of Python ints. Each int holds 64 bits.
The layouts follow docs/architecture/tracer.md section 4.

Conventions (all checked against the propaq v0.1.8 wheel):

* Qubit q (or unit u) is the q-th bit of the basis-state index.
  The matrix is kron(op_{n-1}, ..., op_0).
* Pauli: bit 2q is x, bit 2q+1 is z. P = i^(x.z) X^x Z^z on each qubit.
* Majorana: bit i is gamma_i. gamma_{2k} = Z_0..Z_{k-1} X_k and
  gamma_{2k+1} = Z_0..Z_{k-1} Y_k. The element for a set S of modes is
  i^r gamma_s1 ... gamma_sm in ascending order. r is 0 when |S| mod 4 is
  0 or 1, else 1. This makes every element Hermitian.
* Z4 Weyl qudit: unit bits x0, x1, z0, z1 (digit = lo + 2 hi).
  W = X^x Z^z with X|j> = |j+1> and Z = diag(i^j). No extra phase.
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
from scipy.linalg import expm

I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)
WORD = (1 << 64) - 1


def to_words(value: int, n_words: int) -> tuple[int, ...]:
    """Split a Python int into `n_words` words of 64 bits."""
    return tuple((value >> (64 * i)) & WORD for i in range(n_words))


def from_words(words) -> int:
    """Join words into one Python int."""
    return sum(int(w) << (64 * i) for i, w in enumerate(words))


def kron_little_endian(ops) -> np.ndarray:
    """kron of per-unit operators. ops[0] acts on the lowest index bit."""
    return functools.reduce(np.kron, reversed(list(ops)))


class DenseBasis:
    """Base class. A subclass fixes the layout and the dense matrices."""

    name = "?"
    bits_per_unit = 1
    char_modulus = 2
    phase_order = 4
    local_dim = 2

    def n_words(self, n_units: int) -> int:
        return max(1, -(-n_units * self.bits_per_unit // 64))

    def dim(self, n_units: int) -> int:
        return self.local_dim**n_units

    def n_labels(self, n_units: int) -> int:
        return 1 << (n_units * self.bits_per_unit)

    def label_words(self, label: int, n_units: int) -> tuple[int, ...]:
        return to_words(label, self.n_words(n_units))

    def all_words(self, n_units: int):
        for label in range(self.n_labels(n_units)):
            yield self.label_words(label, n_units)

    def random_words(self, rng, n_units: int) -> tuple[int, ...]:
        label = int(rng.integers(0, self.n_labels(n_units)))
        return self.label_words(label, n_units)

    def matrix(self, words, n_units: int) -> np.ndarray:
        raise NotImplementedError

    def state_index(self, diag: int, n_units: int) -> int:
        """Index of the basis state whose unit u has digit bit u of `diag`."""
        return sum(((diag >> u) & 1) * self.local_dim**u for u in range(n_units))

    def identity_words(self, n_units: int) -> tuple[int, ...]:
        return (0,) * self.n_words(n_units)

    def support(self, words, n_units: int) -> int:
        """Number of units on which the operator is not the identity (dense)."""
        return support_size(self.matrix(words, n_units), self.local_dim, n_units)


class PauliDense(DenseBasis):
    name = "pauli"
    bits_per_unit = 2
    char_modulus = 2

    def matrix(self, words, n_units):
        v = from_words(words)
        ops = []
        for q in range(n_units):
            x, z = (v >> (2 * q)) & 1, (v >> (2 * q + 1)) & 1
            op = np.linalg.matrix_power(PX, x) @ np.linalg.matrix_power(PZ, z)
            ops.append((1j) ** (x * z) * op)
        return kron_little_endian(ops)


class MajoranaDense(DenseBasis):
    """Majorana monomials. `n_units` is the number of sites (qubits).

    The word holds 2 * n_units bits, one per mode.
    """

    name = "majorana"
    bits_per_unit = 2
    char_modulus = 2

    def gamma(self, i: int, n_units: int) -> np.ndarray:
        k, odd = divmod(i, 2)
        ops = [PZ] * k + [PY if odd else PX] + [I2] * (n_units - k - 1)
        return kron_little_endian(ops)

    def matrix(self, words, n_units):
        v = from_words(words)
        modes = [i for i in range(2 * n_units) if (v >> i) & 1]
        m = np.eye(self.dim(n_units), dtype=complex)
        for i in modes:
            m = m @ self.gamma(i, n_units)
        r = 0 if len(modes) % 4 in (0, 1) else 1
        return (1j) ** r * m


class Z4Dense(DenseBasis):
    name = "z4_qudit"
    bits_per_unit = 4
    char_modulus = 4
    local_dim = 4

    @staticmethod
    def digits(words, n_units):
        v = from_words(words)
        xs, zs = [], []
        for u in range(n_units):
            b = (v >> (4 * u)) & 15
            xs.append((b & 1) + 2 * ((b >> 1) & 1))
            zs.append(((b >> 2) & 1) + 2 * ((b >> 3) & 1))
        return xs, zs

    @staticmethod
    def pack(xs, zs) -> int:
        v = 0
        for u, (x, z) in enumerate(zip(xs, zs)):
            b = (x & 1) | ((x >> 1) << 1) | ((z & 1) << 2) | ((z >> 1) << 3)
            v |= b << (4 * u)
        return v

    def matrix(self, words, n_units):
        xs, zs = self.digits(words, n_units)
        sx = np.roll(np.eye(4), 1, axis=0).astype(complex)
        sz = np.diag([(1j) ** j for j in range(4)])
        ops = [np.linalg.matrix_power(sx, x) @ np.linalg.matrix_power(sz, z) for x, z in zip(xs, zs)]
        return kron_little_endian(ops)


BASES = {b.name: b for b in (PauliDense(), MajoranaDense(), Z4Dense())}


def support_size(m: np.ndarray, local_dim: int, n_units: int) -> int:
    """Count units where `m` is not a tensor factor of the identity."""
    d = local_dim
    t = m.reshape([d] * (2 * n_units))  # axes: row units (high..low), col units
    count = 0
    for u in range(n_units):
        ax_r, ax_c = n_units - 1 - u, 2 * n_units - 1 - u
        reduced = np.trace(t, axis1=ax_r, axis2=ax_c) / d
        # Put the traced unit back as an identity and compare.
        rest = reduced
        full = np.einsum("...,ij->...ij", rest, np.eye(d))
        # Axes of `full` are: remaining axes, then (row u, col u). Move them back.
        order = list(range(2 * n_units))
        rem = [a for a in order if a not in (ax_r, ax_c)]
        target = np.empty_like(t)
        perm = rem + [ax_r, ax_c]
        target = np.transpose(full, np.argsort(perm))
        if not np.allclose(target, t, atol=1e-9):
            count += 1
    return count


def expand(basis: DenseBasis, n_units: int, mat: np.ndarray, tol: float = 1e-12) -> dict:
    """Expand `mat` in the basis: {words: complex coeff}. Uses orthogonality."""
    d = basis.dim(n_units)
    out = {}
    for w in basis.all_words(n_units):
        c = np.trace(basis.matrix(w, n_units).conj().T @ mat) / d
        if abs(c) > tol:
            out[w] = c
    return out


def assemble(basis: DenseBasis, n_units: int, terms: dict) -> np.ndarray:
    """Sum of coeff * basis matrix."""
    m = np.zeros((basis.dim(n_units),) * 2, dtype=complex)
    for w, c in terms.items():
        m += c * basis.matrix(w, n_units)
    return m


def dense_rotation(basis: DenseBasis, gen_words, n_units: int, theta: float) -> np.ndarray:
    """U = exp(-i theta H / 2) with H = (W + W^dag) / 2. For a Hermitian W this is exp(-i theta W / 2)."""
    w = basis.matrix(gen_words, n_units)
    return expm(-0.25j * theta * (w + w.conj().T))


def damp(basis: DenseBasis, n_units: int, mat: np.ndarray, gamma: float) -> np.ndarray:
    """Scale each basis component by exp(-gamma * weight). Weight comes from the dense support."""
    terms = expand(basis, n_units, mat)
    damped = {w: c * np.exp(-gamma * basis.support(w, n_units)) for w, c in terms.items()}
    return assemble(basis, n_units, damped)


def heisenberg(
    basis: DenseBasis,
    n_units: int,
    observable: np.ndarray,
    layers,
    gamma: float = 0.0,
) -> np.ndarray:
    """Heisenberg evolution of `observable` through `layers`.

    `layers` is a list of layers. A layer is a list of (gen_words, theta).
    The circuit applies layer 0 first. The propagator walks the layers in
    reverse. When gamma > 0 it damps once per layer, before the layer is
    conjugated (see tests/property/test_noise.py for how this was verified).
    """
    o = observable.copy()
    for layer in reversed(layers):
        if gamma:
            o = damp(basis, n_units, o, gamma)
        for gen, theta in reversed(layer):
            u = dense_rotation(basis, gen, n_units, theta)
            o = u.conj().T @ o @ u
    return o


def expectation(basis: DenseBasis, n_units: int, mat: np.ndarray, diag: int = 0) -> complex:
    i = basis.state_index(diag, n_units)
    return mat[i, i]


def all_pairs(basis: DenseBasis, n_units: int):
    return itertools.product(list(basis.all_words(n_units)), repeat=2)
