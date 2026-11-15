"""Thread invariance: n_threads in {1, 2, 4} give the same counts and values."""

import numpy as np
import pytest
from hypothesis import given, strategies as st

from .dense import BASES
from .helpers import (
    fixed,
    layers_st,
    make_circuit,
    make_propagator,
    make_termsum,
    terms_st,
    termsum_to_dict,
)

THREADS = (1, 2, 4)


def run_all(name, n, terms, layers, diag, noise=None, truncation=None):
    out = {}
    for t in THREADS:
        prop = make_propagator(name, n_threads=t, noise=noise, truncation=truncation)
        circuit = make_circuit(name, n, layers)
        res = prop.expectation_value(make_termsum(name, n, terms), circuit, initial_state=diag)
        ts = prop.propagate(make_termsum(name, n, terms), circuit)
        out[t] = (res, termsum_to_dict(name, n, ts))
    return out


def assert_same(out):
    ref_res, ref_terms = out[1]
    for t in THREADS[1:]:
        res, terms = out[t]
        assert list(res.n_terms) == list(ref_res.n_terms), f"term counts differ at {t} threads"
        assert res.terms_below_cutoff == ref_res.terms_below_cutoff
        assert res.expectation_value == pytest.approx(ref_res.expectation_value, abs=1e-12, rel=1e-12)
        assert set(terms) == set(ref_terms), f"term sets differ at {t} threads"
        for k, v in ref_terms.items():
            assert terms[k] == pytest.approx(v, abs=1e-12, rel=1e-12)


@pytest.mark.parametrize("name", ["pauli", "majorana"])
def test_threads_small_random(name):
    from propaq.noise import UniformNoiseModel
    from propaq.truncation import TruncationPolicy

    @fixed(15)
    @given(data=st.data(), n=st.integers(3, 8), use_noise=st.booleans(), use_trunc=st.booleans())
    def run(data, n, use_noise, use_trunc):
        terms = data.draw(terms_st(name, n))
        layers = data.draw(layers_st(name, n, max_layers=8, max_gates=1))
        diag = data.draw(st.integers(0, (1 << n) - 1))
        noise = UniformNoiseModel(damping=0.03) if use_noise else None
        trunc = TruncationPolicy(coeff_cutoff=1e-3) if use_trunc else None
        assert_same(run_all(name, n, terms, layers, diag, noise, trunc))

    run()


def _fixed_case(name, n, n_gates, seed):
    rng = np.random.default_rng(seed)
    b = BASES[name]
    layers = []
    for _ in range(n_gates):
        w = b.random_words(rng, n)
        while not any(w):
            w = b.random_words(rng, n)
        layers.append([(w, float(rng.uniform(0.2, 1.4)))])
    terms = {b.label_words(1 << (2 * (n // 2 + 1) - 1), n): 1.0, b.label_words(1, n): 0.5}
    return terms, layers


@pytest.mark.parametrize("name", ["pauli", "majorana"])
def test_threads_larger_case(name):
    """A case with enough terms that every thread has work."""
    from propaq.truncation import TruncationPolicy

    n = 12
    terms, layers = _fixed_case(name, n, 30, seed=11)
    out = run_all(name, n, terms, layers, 0, truncation=TruncationPolicy(coeff_cutoff=1e-4))
    assert max(out[1][0].n_terms) > 500  # guard: the case is big enough to split
    assert_same(out)
