"""The Pauli basis in the native layout: bit 2q = x, bit 2q+1 = z.

The term for bits (x, z) is the Hermitian string i^(x.z) X^x Z^z.
This is the convention of crates/pauli/src/algebra.rs.
"""

from ..local import pauli_local_matrix
from ..spec import BasisSpec

EVEN = 0x5555_5555_5555_5555


def _swap_pairs(v):
    return ((v & EVEN) << 1) | ((v >> 1) & EVEN)


def _y_sites(v):
    return (v & (v >> 1) & EVEN).popcount()


def _character(g, t):
    return (_swap_pairs(g) & t).popcount() & 1


def _product(g, t):
    k = g ^ t
    zx = ((g >> 1) & t & EVEN).popcount()
    e = _y_sites(g) + _y_sites(t) - _y_sites(k) + zx * 2
    return k, e & 3


def _weight(t):
    return ((t | (t >> 1)) & EVEN).popcount()


def _trace(t, diag):
    # Any X or Y part makes the term off-diagonal.
    nonzero = (t & EVEN).popcount() == 0
    z_sites = t & diag.expand_units(2, 1)
    return nonzero, (z_sites.popcount() & 1) * 2


def pauli_spec():
    return BasisSpec(
        name="pauli", bits_per_unit=2, branches=2, char_modulus=2, phase_order=4,
        character=_character, product=_product, weight=_weight, trace=_trace,
        intrinsic="pauli", local_dim=2, local_matrix=pauli_local_matrix,
    )
