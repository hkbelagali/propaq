"""Partial evaluation: the GenContext holds only generator-dependent nodes."""

import random

import pytest

from propaq.tracer.analysis import add_fold, classify, split_program
from propaq.tracer.bases import ALL
from propaq.tracer.emit_rust import emit_rust, hoisted_fields
from propaq.tracer.evaluate import evaluate_all


def ir_of(name):
    return add_fold(ALL[name]().trace_ir())


def mentions_term(key):
    """True if a structural key reads the term input."""
    if key[0] == "t":
        return True
    if key[0] == "g":
        return False
    return any(mentions_term(k) for k in key[3])


@pytest.mark.parametrize("name", sorted(ALL))
def test_hoisted_nodes_depend_only_on_g(name):
    ir = ir_of(name)
    fields = hoisted_fields(ir)
    assert fields
    for h in fields:
        prog = ir.programs[h.program]
        assert classify(prog)[h.node] == "gen"
        assert not mentions_term(h.key)
        assert prog.nodes[h.node].op not in ("input", "const", "const_word")


@pytest.mark.parametrize("name", sorted(ALL))
def test_hoisted_values_do_not_change_with_the_term(name):
    ir = ir_of(name)
    rng = random.Random(1)
    g = [rng.getrandbits(64) for _ in range(2)]
    for h in hoisted_fields(ir):
        prog = ir.programs[h.program]
        seen = []
        for _ in range(5):
            t = [rng.getrandbits(64) for _ in range(2)]
            v = evaluate_all(prog, [g, t])[h.node]
            seen.append(tuple(v) if isinstance(v, list) else v)
        assert len(set(seen)) == 1


def test_late_nodes_are_exactly_those_reading_the_term():
    ir = ir_of("pauli")
    for name in ("character", "product"):
        prog = ir.programs[name]
        sp = split_program(prog)
        for i in sp.late_needed:
            n = prog.nodes[i]
            if n.op in ("const", "const_word"):
                continue
            assert mentions_term(sp.keys[i])
        for i in sp.gen_needed:
            assert not mentions_term(sp.keys[i])


def test_pauli_hoists_the_swapped_generator_and_y_count():
    ir = ir_of("pauli")
    fields = hoisted_fields(ir)
    ops = [ir.programs[h.program].nodes[h.node].op for h in fields]
    assert "or" in ops  # the pair-swapped generator, read by the character
    assert [h.kind for h in fields].count("int") == 1  # the Y-site count of g
    assert len(fields) == 3


def test_majorana_hoists_the_length_terms():
    ir = ir_of("majorana")
    kinds = sorted(h.kind for h in hoisted_fields(ir))
    assert kinds == ["int", "int"]  # |g| and the hermiticity part of the product phase


def test_z4_hoists_multiples_of_the_generator_and_self_phase():
    ir = ir_of("z4")
    fields = hoisted_fields(ir)
    progs = {h.program for h in fields}
    assert {"character", "self_phase", "branch_2", "branch_3"} <= progs
    bits_fields = [h for h in fields if h.kind == "bits"]
    assert len(bits_fields) >= 3  # shifted g planes and the multiples k*g


def test_fields_are_shared_between_programs():
    ir = ir_of("z4")
    src = emit_rust(ir)
    n_fields = src.count(": [u64; W],\n") + src.count(": i64,\n")
    keys = {h.key for h in hoisted_fields(ir)}
    assert len(keys) == len(hoisted_fields(ir))  # no key twice
    assert n_fields >= len(keys)


def test_programs_without_g_have_no_context_fields():
    ir = ir_of("majorana")
    for name in ("weight", "trace"):
        sp = split_program(ir.programs[name], gen_inputs=())
        assert sp.frontier == [] and sp.gen_needed == []
