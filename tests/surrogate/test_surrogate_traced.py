"""Surrogate propagation of compiled traced bases against dense evaluation.

The bases are built with `surrogate=True` (cargo build, cached in
PROPAQ_CACHE_DIR). The tests skip when there is no Rust toolchain.
"""

import random

import numpy as np
import pytest

from propaq.propagators.surrogate_traced import (
    TracedSurrogateCircuit,
    TracedSurrogateRotation,
    load_surrogate_basis,
)
from property.dense import BASES, assemble, expectation, heisenberg, to_words

build = pytest.importorskip("propaq.tracer.build")
from propaq.tracer.bases import ALL  # noqa: E402


def _load(name):
    spec = ALL[name]()
    try:
        return load_surrogate_basis(spec, jobs=2)
    except build.BuildUnavailable as exc:  # pragma: no cover
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def pauli():
    return _load("pauli")


@pytest.fixture(scope="module")
def majorana():
    return _load("majorana")


@pytest.fixture(scope="module")
def z4():
    return _load("z4")


def _dense_value(basis, n, obs, gates, params, state):
    layers = []
    for gate in gates:
        theta = gate.angle if gate.angle is not None else params[gate.param_index]
        layers.append([(to_words(gate.generator.words[0], 1), theta)])
    mat = heisenberg(basis, n, assemble(basis, n, obs), layers)
    return expectation(basis, n, mat, state).real


@pytest.mark.parametrize("family,bits_per_unit,n", [("pauli", 2, 3), ("majorana", 2, 3)])
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("threads", [1, 2])
def test_surrogate_matches_dense(request, family, bits_per_unit, n, seed, threads):
    mod = request.getfixturevalue(family)
    basis = BASES[family]
    rng = random.Random(100 * seed + n)
    bits = bits_per_unit * n
    n_params = 3
    gens = [rng.randrange(1, 2**bits) for _ in range(6)]
    gates = [
        TracedSurrogateRotation(mod.Term([g], n), angle=rng.uniform(-1.5, 1.5))
        if i < 2
        else TracedSurrogateRotation(mod.Term([g], n), param_index=rng.randrange(n_params))
        for i, g in enumerate(gens)
    ]
    circ = TracedSurrogateCircuit.from_gates(gates, layer_size=2)
    obs = {to_words(rng.randrange(1, 2**bits), 1): rng.uniform(-1, 1) for _ in range(3)}
    ts = mod.TermSum({mod.Term(list(w), n): c for w, c in obs.items()})
    state = rng.randrange(2**n)
    model = mod.SurrogatePropagator(n_threads=threads).build(ts, circ, initial_state=state)
    assert model.n_params == circ.n_params
    sets = [[rng.uniform(-3, 3) for _ in range(n_params)] for _ in range(5)]
    got = model.evaluate_batch(sets)
    for params, g in zip(sets, got):
        want = _dense_value(basis, n, obs, gates, params, state)
        assert abs(g - want) < 1e-9
        assert abs(model.evaluate(params) - want) < 1e-9


def test_surrogate_save_load_roundtrip(pauli, tmp_path):
    mod, n = pauli, 3
    rng = random.Random(5)
    gates = [TracedSurrogateRotation(mod.Term([rng.randrange(1, 64)], n), param_index=i) for i in range(4)]
    circ = TracedSurrogateCircuit.from_gates(gates)
    ts = mod.TermSum({mod.Term([rng.randrange(1, 64)], n): 1.0})
    model = mod.SurrogatePropagator().build(ts, circ)
    path = str(tmp_path / "m.pqsm")
    model.save(path)
    loaded = mod.SurrogateModel.load(path)
    p = [0.3, -0.7, 1.1, 2.0]
    assert abs(loaded.evaluate(p) - model.evaluate(p)) < 1e-14
    assert loaded.n_terms == model.n_terms
    with pytest.raises(ValueError):
        model.evaluate([0.1])


def test_surrogate_truncation_is_accepted(pauli):
    mod, n = pauli, 3
    rng = random.Random(8)
    gates = [TracedSurrogateRotation(mod.Term([rng.randrange(1, 64)], n), param_index=i % 2) for i in range(8)]
    circ = TracedSurrogateCircuit.from_gates(gates, layer_size=3)
    ts = mod.TermSum({mod.Term([rng.randrange(1, 64)], n): 1.0})
    full = mod.SurrogatePropagator().build(ts, circ)
    cut = mod.SurrogatePropagator(truncation=mod.FrequencyTruncator(2)).build(ts, circ)
    assert cut.n_monomials <= full.n_monomials


def test_z4_surrogate_is_gated_with_clear_error(z4):
    mod, n = z4, 2
    ts = mod.TermSum({mod.Term([0x11], n): 1.0})
    circ = TracedSurrogateCircuit.from_gates([TracedSurrogateRotation(mod.Term([0x01], n), param_index=0)])
    with pytest.raises(ValueError, match="2 branches"):
        mod.SurrogatePropagator().build(ts, circ)


def test_rotation_needs_exactly_one_angle_source():
    with pytest.raises(ValueError):
        TracedSurrogateRotation(object())
    with pytest.raises(ValueError):
        TracedSurrogateRotation(object(), param_index=0, angle=0.1)
