"""UniformNoiseModel against a dense reference.

Convention, found by experiment on v0.1.8 and fixed here (exact to 1e-9):

* The propagator walks the layers last to first.
* Before it conjugates a layer, it multiplies every term by exp(-gamma * w).
  w is the Pauli weight. For Majorana it is the Jordan-Wigner qubit weight.
* So there is one damping per layer, and the first layer of the circuit is
  damped too. Gates inside one layer must commute.

The dense reference expands the operator in the basis, scales each element
and rebuilds the matrix. The reference does not use propaq's weight code.
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
    terms_st,
    termsum_to_dict,
)

MAX_N = 3  # the dense damping expands in all 4^n elements


@pytest.mark.parametrize("name", ["pauli", "majorana"])
def test_uniform_noise_vs_dense(name):
    from propaq.noise import UniformNoiseModel

    @fixed(25)
    @given(
        data=st.data(),
        n=st.integers(1, MAX_N),
        gamma=st.floats(0.0, 0.5, allow_nan=False),
    )
    def run(data, n, gamma):
        terms = data.draw(terms_st(name, n))
        layers = data.draw(layers_st(name, n, max_layers=4, max_gates=3))
        diag = data.draw(st.integers(0, (1 << n) - 1))
        prop = make_propagator(name, n_threads=1, noise=UniformNoiseModel(damping=gamma))
        circuit = make_circuit(name, n, layers)
        res = prop.expectation_value(make_termsum(name, n, terms), circuit, initial_state=diag)
        evolved, ev = dense_run(name, n, terms, layers, diag, gamma=gamma)
        assert res.expectation_value == pytest.approx(ev.real, abs=1e-9)
        out = prop.propagate(make_termsum(name, n, terms), circuit)
        got = assemble(BASES[name], n, termsum_to_dict(name, n, out))
        assert np.allclose(got, evolved, atol=1e-9)

    run()


@pytest.mark.parametrize("name", ["pauli", "majorana"])
def test_noise_without_gates_damps_once_per_layer(name):
    """Exact hand check: one layer, one damping of the observable."""
    from propaq.noise import UniformNoiseModel

    n, gamma = 3, 0.2
    b = BASES[name]
    w = b.label_words(0b110101, n)
    weight = b.support(w, n)
    prop = make_propagator(name, n_threads=1, noise=UniformNoiseModel(damping=gamma))
    # A rotation by angle 0 is the identity channel, so only the damping acts.
    out = prop.propagate(make_termsum(name, n, {w: 1.0}), make_circuit(name, n, [[(b.label_words(1, n), 0.0)]]))
    got = termsum_to_dict(name, n, out)
    assert got[w] == pytest.approx(np.exp(-gamma * weight))
