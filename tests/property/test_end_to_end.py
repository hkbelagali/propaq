"""Random small circuits through the propagators against exact dense Heisenberg evolution.

No truncation and no noise. The observable is a random real combination of
basis elements. Each case checks the expectation value in a basis state and
the full propagated operator.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st

from .dense import BASES, assemble
from .helpers import (
    dense_run,
    fixed,
    layers_st,
    make_circuit,
    make_propagator,
    make_termsum,
    termsum_to_dict,
)

MAX_N = {"pauli": 4, "majorana": 4}  # qubits / sites (8 Majorana modes)


@pytest.mark.parametrize("name", ["pauli", "majorana"])
@pytest.mark.parametrize("max_gates", [1, 3])
def test_circuit_vs_dense(name, max_gates):
    from .helpers import terms_st

    @fixed(40)
    @given(data=st.data(), n=st.integers(1, MAX_N[name]))
    def run(data, n):
        terms = data.draw(terms_st(name, n))
        layers = data.draw(layers_st(name, n, max_layers=5, max_gates=max_gates))
        diag = data.draw(st.integers(0, (1 << n) - 1))
        prop = make_propagator(name, n_threads=1)
        circuit = make_circuit(name, n, layers)
        res = prop.expectation_value(make_termsum(name, n, terms), circuit, initial_state=diag)
        evolved, ev = dense_run(name, n, terms, layers, diag)
        assert abs(ev.imag) < 1e-9
        assert res.expectation_value == pytest.approx(ev.real, abs=1e-9)
        out = prop.propagate(make_termsum(name, n, terms), circuit)
        got = assemble(BASES[name], n, termsum_to_dict(name, n, out))
        assert np.allclose(got, evolved, atol=1e-9)

    run()


@pytest.mark.parametrize("name", ["pauli", "majorana"])
def test_identity_circuit_keeps_observable(name):
    n = 3
    b = BASES[name]
    terms = {b.label_words(5, n): 0.5, b.label_words(17, n): -1.25}
    prop = make_propagator(name, n_threads=1)
    out = prop.propagate(make_termsum(name, n, terms), make_circuit(name, n, [[(b.label_words(3, n), 0.0)]]))
    assert termsum_to_dict(name, n, out) == pytest.approx(terms)
