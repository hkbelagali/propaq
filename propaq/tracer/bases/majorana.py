"""The Majorana basis in the native layout: bit i = gamma_i.

A unit is one site (two modes), so bits_per_unit is 2 and n_units is the
number of sites. This matches `n_units` in crates/majorana/src/algebra.rs.
The term for a set of modes is i^r(|a|) times the sorted product of gamma_i,
with r(p) = 1 if p mod 4 is 2 or 3, else 0. The phase of a product follows
the same file. Weight is the Jordan-Wigner qubit weight.
"""

from ..spec import BasisSpec
from ..tracer import active, select

EVEN = 0x5555_5555_5555_5555


def _herm(p):
    """Hermiticity exponent of a string with p modes."""
    return (p >> 1) & 1


def _character(g, t):
    return (g.popcount() * t.popcount() + (g & t).popcount()) & 1


def _product(g, t):
    k = g ^ t
    # Pairs (i in g, j in t, j < i): their parity is the resorting sign.
    sigma = (g & t.prefix_parity()).popcount()
    e = _herm(g.popcount()) + _herm(t.popcount()) - _herm(k.popcount()) + sigma * 2
    return k, e & 3


def _weight(t):
    live = active() & EVEN
    single = (t ^ (t >> 1)) & EVEN
    occupied = (t | (t >> 1)) & EVEN
    # Inclusive prefix XOR of `single` over sites.
    incl = single.prefix_parity() ^ single
    odd = single.popcount() & 1
    string = (incl ^ select(odd, live, 0)) & live
    return (single | (occupied ^ string)).popcount()


def _trace(t, diag):
    live = active() & EVEN
    single = (t ^ (t >> 1)) & live
    paired = (t & (t >> 1) & live).popcount()
    empty = (t & (t >> 1) & live & ~diag.expand_units(2, 0)).popcount()
    return single.popcount() == 0, ((paired >> 1) + empty) * 2


def majorana_spec():
    return BasisSpec(
        name="majorana", bits_per_unit=2, branches=2, char_modulus=2, phase_order=4,
        character=_character, product=_product, weight=_weight, trace=_trace,
        intrinsic="majorana",
    )
