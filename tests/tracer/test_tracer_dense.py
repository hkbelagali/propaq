"""The evaluator against dense matrices, and evaluator op semantics."""

import random

import pytest

from propaq.tracer.analysis import add_fold
from propaq.tracer.bases import ALL
from propaq.tracer.evaluate import _expand_units, _pdep, _pext, _prefix_parity, evaluate
from propaq.tracer.tracer import trace_program
from propaq.tracer.verify import check_algebra, check_weight_trace


@pytest.mark.parametrize("family,n", [("pauli", 3), ("majorana", 3), ("z4", 2)])
def test_character_and_product_against_dense(family, n):
    ir = add_fold(ALL[family]().trace_ir())
    pairs = check_algebra(ir, family, n)
    assert pairs == (2 ** (ir.meta["bits_per_unit"] * n)) ** 2


@pytest.mark.parametrize("family", ["pauli", "majorana"])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_weight_and_trace_against_dense(family, n):
    ir = ALL[family]().trace_ir()
    assert check_weight_trace(ir, family, n) == 2 ** (2 * n) * 2**n


def test_majorana_matches_hermiticity_convention():
    # i^r(p) with r = 1 for p mod 4 in {2, 3}: gamma0 gamma1 needs a factor i.
    ir = ALL["majorana"]().trace_ir()
    k, e = evaluate(ir.programs["product"], [[0b01], [0b10]])
    # gamma0 gamma1 = -i M with M = i gamma0 gamma1, so e = 3.
    assert k == [0b11] and e == 3
    k, e = evaluate(ir.programs["product"], [[0b01], [0b01]])
    assert k == [0] and e % 4 == 0


def ref_prefix(words):
    out, carry = [], 0
    for w in words:
        o = 0
        for b in range(64):
            if carry:
                o |= 1 << b
            carry ^= (w >> b) & 1
        out.append(o)
    return out


def test_prefix_parity_crosses_words():
    rng = random.Random(3)
    for n in (1, 2, 5):
        for _ in range(30):
            w = [rng.getrandbits(64) & rng.getrandbits(64) for _ in range(n)]
            assert _prefix_parity(w) == ref_prefix(w)


def test_pext_pdep_are_inverse_on_the_mask():
    rng = random.Random(4)
    for _ in range(50):
        m, x = rng.getrandbits(64), rng.getrandbits(64)
        assert _pdep(_pext(x, m), m) == x & m
    assert _pext(0b101100, 0b111100) == 0b1011


@pytest.mark.parametrize("b,off", [(1, 0), (2, 0), (2, 1), (4, 2), (8, 3), (64, 5)])
def test_expand_units(b, off):
    rng = random.Random(b)
    words = [rng.getrandbits(64), rng.getrandbits(64)]
    out = _expand_units(words, b, off)
    nbits = 128
    for u in range(nbits // b):
        bit = (words[u // 64] >> (u % 64)) & 1
        pos = u * b + off
        assert (out[pos // 64] >> (pos % 64)) & 1 == bit
    # no other bit is set
    assert sum(w.bit_count() for w in out) == sum(((words[u // 64] >> (u % 64)) & 1) for u in range(nbits // b))


def test_evaluator_n_words_follow_inputs():
    p = trace_program(lambda a: a.prefix_parity(), ["bits"])
    for n in (1, 2, 4, 8):
        out = evaluate(p, [[1] + [0] * (n - 1)])[0]
        assert len(out) == n
        assert out == ref_prefix([1] + [0] * (n - 1))
