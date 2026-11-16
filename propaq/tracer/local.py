"""Local matrices of a basis, for the hybrid contraction (spec 12.2).

A unit label is the `bits_per_unit` bits of one unit of a term key.
`local_matrix(label)` returns the matrix of that unit as a
`local_dim x local_dim` complex array. Label 0 must be the identity.
The term is the tensor product of the unit matrices.
This file has the matrices of the Pauli and Z4 bases and the table
function the tracer uses.

Majorana has no table. A Majorana monomial is not a tensor product of
unit matrices: the Jordan-Wigner string of a mode depends on the other
units. Majorana sums go through the Pauli path (see `propaq.hybrid`).
"""

from __future__ import annotations

import numpy as np

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


def pauli_local_matrix(label):
    """Pauli unit. Bit 0 of the label is x, bit 1 is z.

    The term for (x, z) is the Hermitian string i^(x z) X^x Z^z.
    So (1, 1) is Y.
    """
    return (_I2, _X, _Z, _Y)[label].copy()


# i^k, exact. This keeps the table free of rounding noise.
_I_POWERS = (1 + 0j, 1j, -1 + 0j, -1j)


def z4_local_matrix(label):
    """Z4 Weyl unit. Bits: x0, x1, z0, z1 (digit = lo + 2 hi).

    The term is X^x Z^z with X|j> = |j+1> and Z|j> = i^j |j>.
    So Z X = i X Z, as in `bases/z4.py`.
    """
    x = (label & 1) + 2 * ((label >> 1) & 1)
    z = ((label >> 2) & 1) + 2 * ((label >> 3) & 1)
    m = np.zeros((4, 4), dtype=complex)
    for j in range(4):
        m[(j + x) % 4, j] = _I_POWERS[(j * z) % 4]
    return m


def _clean(x):
    """Round to 15 digits and remove the sign of zero. The IR hash uses the result."""
    x = round(float(x), 15)
    return 0.0 if x == 0.0 else x


def local_table(spec):
    """The table of a spec as a list of `[re, im]` pairs, labels in order.

    Entry order: label, row, column. Raises ValueError if the table has the
    wrong shape, or if label 0 is not the identity.
    """
    d = spec.local_dim
    if d is None or spec.local_matrix is None:
        raise ValueError("a basis needs both local_dim and local_matrix for the hybrid table")
    b = spec.bits_per_unit
    if b > 8:
        raise ValueError("the hybrid table allows at most 8 bits per unit")
    out = []
    for label in range(1 << b):
        m = np.asarray(spec.local_matrix(label), dtype=complex)
        if m.shape != (d, d):
            raise ValueError(f"local_matrix({label}) has shape {m.shape}, expected {(d, d)}")
        if label == 0 and not np.array_equal(m, np.eye(d)):
            raise ValueError("local_matrix(0) must be the identity")
        out.extend([_clean(v.real), _clean(v.imag)] for v in m.reshape(-1))
    return out


def has_local(spec):
    return spec.local_dim is not None and spec.local_matrix is not None


def table_array(table):
    """The table pairs as a flat complex128 array."""
    a = np.asarray(table, dtype=float)
    return (a[:, 0] + 1j * a[:, 1]).astype(np.complex128)
