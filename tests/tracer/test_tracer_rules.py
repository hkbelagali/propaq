"""Tracing rules: ops, constants, sharing, and the branch error."""

import pytest

from propaq.tracer import BasisSpec, TracerBranchError, active, const_word, n_units, select
from propaq.tracer.evaluate import evaluate
from propaq.tracer.tracer import trace_program

B2 = ["bits", "bits"]


def ops(prog):
    return [n.op for n in prog.nodes]


def test_bits_ops_make_nodes():
    p = trace_program(lambda a, b: ((a & b) | (a ^ b), ~a), B2)
    assert ops(p) == ["input", "input", "and", "xor", "or", "not"]
    assert p.output_kinds() == ["bits", "bits"]


def test_word_shift_operators_are_word_local():
    p = trace_program(lambda a: (a << 3, a >> 5, a.shl_w(1)), ["bits"])
    assert [n.op for n in p.nodes[1:]] == ["shl_w", "shr_w", "shl_w"]
    out = evaluate(p, [[1 << 63, 1]])
    assert out[0] == [0, 8] and out[1] == [1 << 58, 0]


def test_zero_shift_is_identity_and_bad_shift_fails():
    p = trace_program(lambda a: a << 0, ["bits"])
    assert ops(p) == ["input"]
    with pytest.raises(ValueError):
        trace_program(lambda a: a << 64, ["bits"])
    with pytest.raises(TracerBranchError):
        trace_program(lambda a, c: a << c, ["bits", "int"])


def test_python_int_becomes_const():
    p = trace_program(lambda a: (a & 0xFF).popcount() + 3, ["bits"])
    assert "const_word" in ops(p) and "const" in ops(p)
    assert evaluate(p, [[0x1F1]])[0] == 5 + 3


def test_constant_folding_and_neutral_words():
    p = trace_program(lambda a: ((a ^ 0) | 0) & 0xFFFF_FFFF_FFFF_FFFF, ["bits"])
    assert ops(p) == ["input"]
    q = trace_program(lambda a: a.popcount() * (2 + 3), ["bits"])
    assert [n.imm for n in q.nodes if n.op == "const"] == [5]


def test_common_subexpressions_are_shared():
    p = trace_program(lambda a, b: (a & b).popcount() + (a & b).popcount(), B2)
    assert ops(p).count("and") == 1 and ops(p).count("popcount") == 1


def test_int_ops_and_comparisons():
    def f(a):
        x = a.popcount()
        return (x + 1, x - 1, x * 3, x & 6, x | 1, x ^ 5, -x, x << 2, x >> 1, x % 4,
                x == 3, x != 3, x < 4, x > 4, x <= 3, x >= 3, 2 - x)

    p = trace_program(f, ["bits"])
    out = evaluate(p, [[0b1011]])
    x = 3
    assert out == [x + 1, x - 1, x * 3, x & 6, x | 1, x ^ 5, -x, x << 2, x >> 1, x % 4,
                   1, 0, 1, 0, 1, 1, 2 - x]


def test_rem_is_euclidean():
    p = trace_program(lambda a: (a.popcount() - 10) % 4, ["bits"])
    assert evaluate(p, [[1]])[0] == (1 - 10) % 4
    with pytest.raises(ValueError):
        trace_program(lambda a: a.popcount() % 0, ["bits"])


def test_int_wraps_at_64_bits():
    p = trace_program(lambda a: (a.popcount() + (2**63 - 1)) , ["bits"])
    assert evaluate(p, [[7]])[0] == 3 + (2**63 - 1) - 2**64


def test_select_for_bits_and_ints():
    p = trace_program(lambda a, b: (select(a.popcount() & 1, a, b), select(b.popcount(), 7, 9)), B2)
    assert evaluate(p, [[1], [2]]) == [[1], 7]
    assert evaluate(p, [[3], [0]]) == [[0], 9]


def test_active_and_n_units():
    p = trace_program(lambda t: (t & active()).popcount() + n_units(), ["bits"], bits_per_unit=2)
    assert evaluate(p, [[0xFF]], n_units=2)[0] == 4 + 2
    with pytest.raises(ValueError):
        evaluate(p, [[0xFF]])


def test_free_functions_need_a_trace():
    with pytest.raises(RuntimeError):
        const_word(1)


def test_pext_pdep_prefix_expand_nodes():
    p = trace_program(lambda a: (a.pext_w(0xF0), a.pdep_w(0x0F), a.prefix_parity(), a.expand_units(2, 1)), ["bits"])
    assert ops(p)[1:] == ["pext_w", "pdep_w", "prefix_parity", "expand_units"]
    assert p.nodes[-1].imm == (2, 1)


@pytest.mark.parametrize("use", [
    lambda a: 1 if a.popcount() else 0,
    lambda a: 1 if a else 0,
    lambda a: int(a.popcount()),
    lambda a: [0, 1][a.popcount()],
    lambda a: list(a),
    lambda a: a.popcount() and 1,
    lambda a: float(a.popcount()),
])
def test_branching_raises(use):
    with pytest.raises(TracerBranchError):
        trace_program(use, ["bits"])


def test_branch_error_is_a_type_error():
    assert issubclass(TracerBranchError, TypeError)


def test_mixed_traces_are_rejected():
    holder = []
    trace_program(lambda a: holder.append(a) or a, ["bits"])
    with pytest.raises(ValueError):
        trace_program(lambda a: a & holder[0], ["bits"])


def test_bad_output_type():
    with pytest.raises(TypeError):
        trace_program(lambda a: "x", ["bits"])
    with pytest.raises(TypeError):
        trace_program(lambda a: 1.5, ["bits"])


def test_python_int_output_is_a_const():
    p = trace_program(lambda a: (a, 4), ["bits"])
    assert p.output_kinds() == ["bits", "int"]


def _spec(**kw):
    base = dict(name="t", bits_per_unit=1, character=lambda g, t: (g & t).popcount() & 1,
                product=lambda g, t: (g ^ t, 0), weight=lambda t: t.popcount(),
                trace=lambda t, d: (1, 0))
    base.update(kw)
    return BasisSpec(**base)


def test_spec_traces_all_programs():
    ir = _spec().trace_ir()
    assert set(ir.programs) == {"character", "product", "weight", "trace"}


def test_spec_rejects_bad_definitions():
    with pytest.raises(ValueError):
        _spec(name="Bad Name").trace_ir()
    with pytest.raises(ValueError):
        _spec(bits_per_unit=3).trace_ir()
    with pytest.raises(ValueError):
        _spec(branches=3, char_modulus=3).trace_ir()  # needs branch and self_phase
    with pytest.raises(ValueError):
        _spec(character=lambda g, t: (g ^ t)).trace_ir()  # wrong output kind
    with pytest.raises(ValueError):
        _spec(character=lambda g, t: (g & active()).popcount()).trace_ir()  # runtime op in g,t program
    with pytest.raises(ValueError):
        _spec(intrinsic="fermi").trace_ir()
