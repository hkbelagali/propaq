"""Algebra laws of the Pauli, Majorana and Z4 bases against dense matrices."""

import numpy as np
import pytest
from hypothesis import given, strategies as st

from .conformance import (
    check_basis_conformance,
    check_pair,
    check_term,
    native_callables,
    z4_reference_callables,
)
from .dense import BASES, from_words
from .helpers import fixed, words_st

# (basis, largest unit count for the property tests)
CASES = [("pauli", 4), ("majorana", 4), ("z4_qudit", 2)]


def callables(name, n):
    return z4_reference_callables(n) if name == "z4_qudit" else native_callables(name, n)


@pytest.mark.parametrize("name,max_n", CASES)
def test_exhaustive_small(name, max_n):
    """Every pair of words, for the smallest sizes."""
    for n in (1, 2) if name != "z4_qudit" else (1,):
        basis = BASES[name]
        counts = check_basis_conformance(basis, n, n_pairs=basis.n_labels(n) ** 2, **callables(name, n))
        assert counts["product"] == basis.n_labels(n) ** 2


@pytest.mark.parametrize("name,max_n", CASES)
def test_sampled_helper(name, max_n):
    """The reusable helper, at the largest size."""
    counts = check_basis_conformance(name, max_n, n_pairs=300, n_terms=60, **callables(name, max_n))
    assert counts["character"] == 300


@pytest.mark.parametrize("name,max_n", CASES)
def test_commutation_and_product(name, max_n):
    @fixed(150)
    @given(data=st.data(), n=st.integers(1, max_n))
    def run(data, n):
        g = data.draw(words_st(name, n))
        t = data.draw(words_st(name, n))
        c = callables(name, n)
        check_pair(BASES[name], n, g, t, c["character"], c["product"])

    run()


@pytest.mark.parametrize("name,max_n", CASES)
def test_weight_and_trace(name, max_n):
    @fixed(150)
    @given(data=st.data(), n=st.integers(1, max_n))
    def run(data, n):
        t = data.draw(words_st(name, n))
        diag = data.draw(st.integers(0, (1 << n) - 1))
        c = callables(name, n)
        check_term(BASES[name], n, t, diag, c["weight"], c["trace"])

    run()


@pytest.mark.parametrize("name,max_n", CASES)
def test_product_is_associative(name, max_n):
    """(a b) c = a (b c), including the phases. Needs no dense matrix."""

    @fixed(150)
    @given(data=st.data(), n=st.integers(1, max_n))
    def run(data, n):
        a, b, c = (data.draw(words_st(name, n)) for _ in range(3))
        prod = callables(name, n)["product"]
        order = BASES[name].phase_order
        ab, e1 = prod(a, b)
        abc, e2 = prod(ab, c)
        bc, e3 = prod(b, c)
        abc2, e4 = prod(a, bc)
        assert tuple(abc) == tuple(abc2)
        assert (e1 + e2 - e3 - e4) % order == 0

    run()


@pytest.mark.parametrize("name,max_n", CASES)
def test_character_is_a_bicharacter(name, max_n):
    """chi(g, a b) = chi(g, a) + chi(g, b) mod M."""

    @fixed(150)
    @given(data=st.data(), n=st.integers(1, max_n))
    def run(data, n):
        g, a, b = (data.draw(words_st(name, n)) for _ in range(3))
        c = callables(name, n)
        ab, _ = c["product"](a, b)
        m = BASES[name].char_modulus
        assert c["character"](g, ab) % m == (c["character"](g, a) + c["character"](g, b)) % m

    run()


def test_conformance_detects_wrong_callables():
    """A wrong callable must make the helper fail."""
    n = 2
    good = native_callables("pauli", n)

    def bad_weight(t):
        return good["weight"](t) + 1

    with pytest.raises(AssertionError, match="weight"):
        check_basis_conformance("pauli", n, **{**good, "weight": bad_weight})

    def bad_phase(g, t):
        out, e = good["product"](g, t)
        return out, (e + 1) % 4

    with pytest.raises(AssertionError, match="product"):
        check_basis_conformance("pauli", n, **{**good, "product": bad_phase})

    with pytest.raises(AssertionError, match="character"):
        check_basis_conformance("pauli", n, **{**good, "character": lambda g, t: 0})

    with pytest.raises(AssertionError, match="trace"):
        check_basis_conformance("pauli", n, **{**good, "trace": lambda t, d: 1.0})


def test_dense_references_are_consistent():
    """Basic facts of the references: Hermitian for +-1 bases, unitary for Z4."""
    for name, n in (("pauli", 2), ("majorana", 3)):
        b = BASES[name]
        for w in b.all_words(n):
            m = b.matrix(w, n)
            assert np.allclose(m, m.conj().T)
            assert np.allclose(m @ m, np.eye(b.dim(n)))
    b = BASES["z4_qudit"]
    for w in list(b.all_words(1)):
        m = b.matrix(w, 1)
        assert np.allclose(m @ m.conj().T, np.eye(4))
    assert from_words((5, 1)) == 5 + (1 << 64)
