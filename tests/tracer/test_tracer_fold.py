"""Linear-fold discovery, locality and the generated fold program."""

import random

import pytest

from propaq.tracer import BasisSpec
from propaq.tracer.analysis import (add_fold, discover_fold, eval_fold, fold_chi, locality_report)
from propaq.tracer.bases import ALL
from propaq.tracer.evaluate import evaluate

EVEN = 0x5555_5555_5555_5555


def swap_pairs(w):
    return ((w & EVEN) << 1) | ((w >> 1) & EVEN)


def rand_words(rng, n, dens=0.4):
    return [sum(1 << k for k in range(64) if rng.random() < dens) for _ in range(n)]


@pytest.mark.parametrize("name", sorted(ALL))
def test_fold_found_and_equals_character(name):
    ir = add_fold(ALL[name]().trace_ir())
    assert ir.meta["fold"] is not None and "fold" in ir.programs
    rng = random.Random(8)
    for n in (1, 2, 4, 8):
        for _ in range(60):
            g = rand_words(rng, n, rng.choice([0.05, 0.3, 0.6, 0.95]))
            t = rand_words(rng, n, rng.choice([0.05, 0.3, 0.6, 0.95]))
            m, masks, par = eval_fold(ir, g)
            want = evaluate(ir.programs["character"], [g, t])[0] % ir.meta["char_modulus"]
            assert fold_chi(m, masks, par, t) == want


def test_pauli_fold_is_the_pair_swapped_generator():
    ir = add_fold(ALL["pauli"]().trace_ir())
    rng = random.Random(1)
    for _ in range(20):
        g = rand_words(rng, 4)
        m, masks, par = eval_fold(ir, g)
        assert m == 2 and masks[0] == [swap_pairs(w) for w in g] and not par
        assert masks[1] == [0] * 4 and masks[2] == [0] * 4


def test_majorana_fold_is_generator_with_parity_for_odd_length():
    ir = add_fold(ALL["majorana"]().trace_ir())
    assert ir.meta["fold"]["parity_all"] is True
    rng = random.Random(2)
    for _ in range(20):
        g = rand_words(rng, 2)
        m, masks, par = eval_fold(ir, g)
        assert masks[0] == g
        assert par == bool(sum(w.bit_count() for w in g) & 1)


def test_z4_fold_has_modulus_four_with_three_masks():
    ir = add_fold(ALL["z4"]().trace_ir())
    assert ir.meta["fold"]["modulus"] == 4
    rng = random.Random(3)
    seen = [False] * 3
    for _ in range(20):
        _, masks, par = eval_fold(ir, rand_words(rng, 2))
        assert not par
        # masks are disjoint
        assert all(a & b == 0 for x, y in ((0, 1), (0, 2), (1, 2)) for a, b in zip(masks[x], masks[y]))
        for i in range(3):
            seen[i] |= any(masks[i])
    assert all(seen)


def spec_with(character, **kw):
    base = dict(name="t", bits_per_unit=2, character=character,
                product=lambda g, t: (g ^ t, 0), weight=lambda t: t.popcount(), trace=lambda t, d: (1, 0))
    base.update(kw)
    return BasisSpec(**base)


def test_nonlinear_character_has_no_fold():
    # chi = |g & t| * |t| mod 2 is quadratic in t.
    ir = spec_with(lambda g, t: ((g & t).popcount() * t.popcount()) & 1).trace_ir()
    assert discover_fold(ir) is None
    assert add_fold(ir).meta["fold"] is None and "fold" not in ir.programs


def test_character_with_constant_offset_has_no_fold():
    ir = spec_with(lambda g, t: ((g & t).popcount() + 1) & 1).trace_ir()
    assert discover_fold(ir) is None


def test_unit_dependent_character_is_rejected():
    # Weights depend on the unit index: not translation invariant.
    ir = spec_with(lambda g, t: (g & t & 0x0000_0000_FFFF_FFFF).popcount() & 1).trace_ir()
    assert discover_fold(ir) is None


def test_dense_complement_with_parity():
    # chi = |g| |t| + |g & t|: the probe of every bit is dense when |g| is odd.
    ir = spec_with(lambda g, t: (g.popcount() * t.popcount() + (g & t).popcount()) & 1).trace_ir()
    model = discover_fold(ir)
    assert model is not None and model.parity_all
    add_fold(ir)
    rng = random.Random(4)
    for _ in range(40):
        g, t = rand_words(rng, 2, 0.1), rand_words(rng, 2, 0.5)
        m, masks, par = eval_fold(ir, g)
        assert fold_chi(m, masks, par, t) == evaluate(ir.programs["character"], [g, t])[0] % 2


def test_partial_parity_goes_into_the_mask():
    # Only the even bit of each unit carries the parity term.
    ir = spec_with(lambda g, t: (g.popcount() * (t & EVEN).popcount()) & 1).trace_ir()
    model = discover_fold(ir)
    assert model is not None and not model.parity_all and model.a == [1, 0]
    add_fold(ir)
    rng = random.Random(5)
    for _ in range(40):
        g, t = rand_words(rng, 2, 0.1), rand_words(rng, 2, 0.5)
        m, masks, par = eval_fold(ir, g)
        assert not par and fold_chi(m, masks, par, t) == evaluate(ir.programs["character"], [g, t])[0] % 2


def test_modulus_three_is_not_supported():
    ir = spec_with(lambda g, t: (g & t).popcount() % 3, char_modulus=3).trace_ir()
    assert discover_fold(ir) is None


def test_locality_report():
    rep = locality_report(add_fold(ALL["pauli"]().trace_ir()))
    assert rep["fold"]["cross_word"] == [] and rep["fold"]["runtime"] == []  # Pauli fold is word-local
    assert locality_report(add_fold(ALL["majorana"]().trace_ir()))["fold"]["cross_word"] == ["popcount"]
    maj = locality_report(ALL["majorana"]().trace_ir())
    assert "prefix_parity" in maj["product"]["cross_word"]
    assert maj["weight"]["runtime"] == ["active"]
    assert "prefix_parity" not in locality_report(ALL["pauli"]().trace_ir())["product"]["cross_word"]
