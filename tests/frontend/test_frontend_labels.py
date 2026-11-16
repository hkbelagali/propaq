"""Labels, term sums and circuits of the front-end. No build needed."""

import pytest

from propaq.tracer.bases import ALL
from propaq.tracer.frontend import TracedBasis, TracedCircuit, int_to_words, lower_pauli_gate, words_to_int


class Term:
    def __init__(self, words, n_units):
        self.words, self.n_units = list(words), n_units

    def __eq__(self, o):
        return (self.words, self.n_units) == (o.words, o.n_units)

    def __hash__(self):
        return hash((tuple(self.words), self.n_units))


class TermSum(dict):
    def __init__(self, terms, **kw):
        super().__init__(terms)
        self.kw = kw


class Mod:
    Term, TermSum = Term, TermSum


def basis(name, **kw):
    return TracedBasis(ALL[name](), module=Mod, **kw)


def test_pauli_label_string_sets_the_native_bits():
    b = basis("pauli")
    t = b.term("XZIY")
    # unit u uses bits 2u (x) and 2u+1 (z)
    assert t.n_units == 4
    assert words_to_int(t.words) == 1 | (2 << 2) | (3 << 6)
    assert b.label(t) == "XZIY"


def test_label_forms_agree():
    b = basis("pauli")
    want = b.words("XZIY")
    assert b.words(["X", "Z", "I", "Y"]) == want
    assert b.words(["X", 2, 0, "Y"]) == want
    assert b.words({0: "X", 1: "Z", 3: "Y"}, 4) == want
    assert b.words(words_to_int(want), 4) == want
    assert b.words("X Z I Y") == want


def test_sparse_term_needs_n_units_and_checks_range():
    b = basis("pauli")
    with pytest.raises(ValueError):
        b.words({0: "X"})
    with pytest.raises(ValueError):
        b.words({5: "X"}, 3)
    with pytest.raises(ValueError, match="unknown unit label"):
        b.words("XQ")


def test_many_units_use_many_words():
    b = basis("pauli")
    t = b.term({40: "Y"}, 41)
    assert len(t.words) == 2
    assert words_to_int(t.words) == 3 << 80


def test_z4_labels_cover_all_sixteen_patterns():
    b = basis("z4")
    assert sorted(set(b.unit_labels.values())) == list(range(16))
    assert b.unit_pattern("X") == 1 and b.unit_pattern("X3") == 3
    assert b.unit_pattern("Z2") == 2 << 2
    assert b.unit_pattern("X1Z3") == 1 | (3 << 2)
    t = b.term("X2 Z I X1Z1")
    assert b.label(t) == "X2 Z I X1Z1" or b.label(t) == "X2 Z I X"  # X1Z1 has its own pattern
    assert t.n_units == 4


def test_termsum_from_dict_pairs_and_bare_terms():
    b = basis("pauli", n_units=3)
    ts = b.termsum({"XII": 0.5, "ZZI": -1})
    assert len(ts) == 2 and ts[b.term("XII")] == 0.5
    ts = b.termsum([("XII", 1.0), ("XII", 2.0), "IZI"])
    assert ts[b.term("XII")] == 3.0 and ts[b.term("IZI")] == 1.0


def test_termsum_infers_n_units_from_labels():
    b = basis("pauli")
    ts = b.termsum({"XZ": 1.0})
    assert next(iter(ts)).n_units == 2


def test_complex_basis_makes_complex_coefficients():
    b = basis("z4", n_units=2)
    ts = b.termsum({"X I": 1.0})
    assert isinstance(ts[b.term("X I")], complex)


def test_circuit_from_pairs_and_layers():
    b = basis("pauli", n_units=2)
    c = b.circuit([("XI", 0.1), ("ZZ", 0.2)])
    assert isinstance(c, TracedCircuit) and [len(l) for l in c.layers] == [1, 1]
    c = b.circuit([[("XI", 0.1), ("IX", 0.2)], [("ZZ", 0.3)]])
    assert [len(l) for l in c.layers] == [2, 1]
    assert c.layers[1][0].angle == 0.3 and c.layers[0][0].generator == b.term("XI")


def test_lowering_hook_is_required_for_qiskit():
    b = basis("pauli")
    with pytest.raises(NotImplementedError):
        b.lower("rx", [0], [0.1], 2)


def test_lower_pauli_gate_words():
    (w, a), = lower_pauli_gate("rzz", (0, 2), (0.5,), 3)
    assert words_to_int(w) == (2 << 0) | (2 << 4) and a == 0.5
    with pytest.raises(NotImplementedError):
        lower_pauli_gate("h", (0,), (), 1)


def test_int_to_words_rejects_high_bits():
    assert int_to_words(5, 1, 4) == [5]
    with pytest.raises(ValueError):
        int_to_words(1 << 8, 2, 4)


def test_unbound_qiskit_parameter_is_reported():
    qiskit = pytest.importorskip("qiskit")
    from dataclasses import replace

    from qiskit.circuit import Parameter

    qc = qiskit.QuantumCircuit(1)
    qc.rx(Parameter("a"), 0)
    b = TracedBasis(replace(ALL["pauli"](), lower_gate=lower_pauli_gate), module=Mod)
    with pytest.raises(ValueError, match="unbound"):
        b.from_qiskit(qc)
