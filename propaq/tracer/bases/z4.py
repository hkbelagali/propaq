"""Weyl-Heisenberg qudit with d = 4 (test basis for branches > 2).

Unit layout, 4 bits: x0, x1, z0, z1 (bit 4u + 0..3). Digit = lo + 2 hi.
The term is W(x, z) = X^x Z^z with Z X = i X Z.
Rotation: U = exp(-i theta H / 2), H = (W_g + W_g^dag) / 2. It has 4 branches.
"""

from ..local import z4_local_matrix
from ..spec import BasisSpec

P1 = 0x1111_1111_1111_1111  # bit 4u: low bit of the x digit
X_DIGITS = 0x3333_3333_3333_3333  # bits 4u and 4u+1: the x digit
P5 = 0x5555_5555_5555_5555  # bits 4u and 4u+2: low bits of the x and z digits


def _dot(a, b):
    """Sum over units of (digit of a at bits 4u,4u+1) * (digit of b), mod 4."""
    lo = (a & b & P1).popcount()
    cross = (a & (b >> 1) & P1).popcount() + ((a >> 1) & b & P1).popcount()
    return (lo + cross * 2) & 3


def _z_to_x(v):
    """Move the z digits of each unit onto the x digit positions."""
    return v >> 2


def _character(g, t):
    # W_g T = i^s T W_g, s = gz.tx - tz.gx.
    return (_dot(_z_to_x(g), t) - _dot(_z_to_x(t), g)) & 3


def _add(a, b):
    """Digit-wise addition mod 4 of x and z digits."""
    carry = (a & b & P5) << 1
    return a ^ b ^ carry


def _product(g, t):
    # W_g W_t = i^(gz.tx) W_(g+t).
    return _add(g, t), _dot(_z_to_x(g), t)


def _weight(t):
    return ((t | (t >> 1) | (t >> 2) | (t >> 3)) & P1).popcount()


def _trace(t, diag):
    # Only Z^z is diagonal. Diagonal state: one bit (0 or 1) per unit.
    nonzero = (t & X_DIGITS).popcount() == 0
    d = diag.expand_units(4, 2)
    lo = (t & d).popcount()
    hi = ((t >> 1) & d).popcount()
    return nonzero, (lo + hi * 2) & 3


def _self_phase(g):
    return _dot(_z_to_x(g), g)


def _branch(g, t, k):
    # Branch k of U^dag T U: W_(t + k g) with phase i^(k r), r = tz.gx.
    kg = g
    for _ in range(k - 1):
        kg = _add(kg, g)
    r = _dot(_z_to_x(t), g)
    return _add(t, kg), (r * k) & 3


def z4_spec():
    return BasisSpec(
        name="z4", bits_per_unit=4, branches=4, char_modulus=4, phase_order=4,
        character=_character, product=_product, weight=_weight, trace=_trace,
        branch=_branch, self_phase=_self_phase,
        local_dim=4, local_matrix=z4_local_matrix,
    )
