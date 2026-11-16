"""Compiled traced bases, end to end: generate crate, cargo build, import, propagate.

Two bases are built (and cached in `~/.cache/propaq/bases`, or in
`PROPAQ_CACHE_DIR`):

* the Z4 Weyl qudit, compared with the Python fallback propagator and with
  dense Heisenberg evolution;
* the traced Pauli and Majorana programs, built WITHOUT their intrinsic
  (generic emit), compared with dense Heisenberg evolution.

The tests skip when there is no Rust toolchain or no source tree.
"""

import random

import numpy as np
import pytest

from .conformance import check_pair, z4_reference_callables
from .dense import BASES, assemble, expand, expectation, from_words, heisenberg, to_words

build = pytest.importorskip("propaq.tracer.build")
from propaq.tracer.bases import ALL  # noqa: E402
from propaq.tracer.fallback import expectation as fb_expectation, propagate as fb_propagate  # noqa: E402


class Rot:
    """A rotation as the engine reads it: generator, angle, qiskit index."""

    def __init__(self, generator, angle):
        self.generator, self.angle, self.qiskit_gate_idx = generator, angle, None


class Circ:
    def __init__(self, layers):
        self.layers = layers


def _load(name):
    spec = ALL[name]()
    try:
        ir = build.compile_spec(spec)
        res = build.build_basis(spec, ir=ir, jobs=8)
    except (build.BuildUnavailable, build.BuildError) as exc:  # pragma: no cover
        if isinstance(exc, build.BuildError):
            raise
        pytest.skip(str(exc))
    return build.import_extension(res.so_path, build.module_name(ir)), ir


@pytest.fixture(scope="module")
def z4():
    return _load("z4")


@pytest.fixture(scope="module")
def pauli_generic():
    return _load("pauli")


def _dict(ts):
    return {tuple(t.words): c for t, c in ts.items()}


# ------------------------------------------------------------------- Z4


def _z4_case(seed, n=2, n_gates=5):
    rng = random.Random(seed)
    nbits = 4 * n
    circ = [([rng.randrange(1, 2**nbits)], rng.uniform(-1.5, 1.5)) for _ in range(n_gates)]
    obs = {(rng.randrange(1, 2**nbits),): complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(3)}
    return circ, obs


def _layers(mod, circ, n, rng):
    """Cut the gate list into random layers (order is kept)."""
    gates = [Rot(mod.Term(list(g), n), th) for g, th in circ]
    cuts = sorted(rng.sample(range(1, len(gates)), min(2, len(gates) - 1))) if len(gates) > 1 else []
    parts, last = [], 0
    for c in cuts + [len(gates)]:
        parts.append(gates[last:c])
        last = c
    return Circ(parts)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("threads", [1, 4])
def test_z4_compiled_vs_fallback_and_dense(z4, seed, threads):
    mod, ir = z4
    n = 2
    circ, obs = _z4_case(seed)
    ts = mod.TermSum({mod.Term(list(k), n): v for k, v in obs.items()})
    assert ts.dtype == "complex128"
    out = mod.Propagator(n_threads=threads, pin_threads=False).propagate(ts, _layers(mod, circ, n, random.Random(seed)))
    got = _dict(out)
    want = fb_propagate(ir, obs, circ)
    # Same support (term counts exact) and same values.
    live = {k: v for k, v in got.items() if abs(v) > 1e-14}
    assert set(live) == {k for k, v in want.items() if abs(v) > 1e-14}
    assert max(abs(got.get(k, 0) - want.get(k, 0)) for k in set(got) | set(want)) < 1e-12
    # Dense Heisenberg evolution.
    basis = BASES["z4_qudit"]
    dense_obs = assemble(basis, n, {to_words(k[0], 1): v for k, v in obs.items()})
    dense = heisenberg(basis, n, dense_obs, [[(to_words(g[0], 1), th)] for g, th in circ])
    assert np.allclose(assemble(basis, n, {to_words(k[0], 1): v for k, v in got.items()}), dense, atol=1e-10)


@pytest.mark.parametrize("seed", range(4))
def test_z4_compiled_expectation(z4, seed):
    mod, ir = z4
    n = 2
    circ, obs = _z4_case(100 + seed)
    ts = mod.TermSum({mod.Term(list(k), n): v for k, v in obs.items()})
    state = random.Random(seed).randrange(2**n)
    res = mod.Propagator(n_threads=1, pin_threads=False).expectation_value(
        ts, _layers(mod, circ, n, random.Random(seed)), initial_state=state
    )
    want = fb_expectation(ir, fb_propagate(ir, obs, circ), n, [state])
    # The compiled value is the real part of the complex sum.
    assert abs(res.expectation_value - want.real) < 1e-12


def test_z4_complex_sum_api(z4, tmp_path):
    mod, _ = z4
    t = mod.Term([0x1], 2)
    ts = mod.TermSum()
    ts.add(t, 1 + 2j)
    ts.add(t, 0.5)
    assert ts[t] == 1.5 + 2j and len(ts) == 1
    assert abs(ts.norm_squared() - (1.5**2 + 4)) < 1e-12
    assert ts.copy().items() == [(t, 1.5 + 2j)]
    with pytest.raises(ValueError):
        mod.TermSum(dtype="float32")
    # Term file format v2 holds complex coefficients, so a complex sum saves and loads.
    path = str(tmp_path / "z4.bin")
    ts.save(path)
    assert mod.TermSum.from_file(path).items() == [(t, 1.5 + 2j)]


def test_z4_raw_callables_conform(z4):
    """The module's raw functions agree with the dense matrices and the integer reference."""
    mod, _ = z4
    n = 2
    basis, ref = BASES["z4_qudit"], z4_reference_callables(n)
    rng = random.Random(3)
    for _ in range(300):
        g, t = rng.randrange(1, 2**8), rng.randrange(2**8)
        gw, tw = to_words(g, 1), to_words(t, 1)
        check_pair(
            basis, n, gw, tw,
            lambda g_, t_: mod.character(list(g_), list(t_)),
            lambda g_, t_: (lambda out, e: (tuple(out), e))(*mod.product(list(g_), list(t_))),
        )
        assert mod.character(list(gw), list(tw)) == ref["character"](gw, tw)
        assert mod.weight(list(tw), n) == ref["weight"](tw)
        diag = rng.randrange(2**n)
        nonzero, e = ref["trace"](tw, diag)
        # The Rust trace is the real part of the diagonal element.
        assert abs(mod.trace(list(tw), [diag], n) - (np.exp(2j * np.pi * e / 4).real if nonzero else 0.0)) < 1e-12


# ----------------------------------------------------- traced Pauli, generic


def _pauli_case(seed, n=4, n_gates=7):
    rng = random.Random(seed)
    nw = 1
    wl = lambda: to_words(rng.randrange(1, 4**n), nw)  # noqa: E731
    layers = [[(wl(), rng.uniform(-2, 2))] for _ in range(n_gates)]
    obs = {wl(): rng.uniform(-1, 1) for _ in range(3)}
    return n, layers, obs


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("threads", [1, 3])
def test_traced_pauli_generic_matches_dense(pauli_generic, seed, threads):
    mod, _ = pauli_generic
    n, layers, obs = _pauli_case(seed)
    ts = mod.TermSum({mod.Term(list(w), n): c for w, c in obs.items()})
    assert ts.dtype == "float64"
    circ = Circ([[Rot(mod.Term(list(w), n), th) for w, th in layer] for layer in layers])
    got = _dict(mod.Propagator(n_threads=threads, pin_threads=False).propagate(ts, circ))
    got = {tuple(to_words(from_words(k), 1)): v for k, v in got.items()}
    basis = BASES["pauli"]
    dense = heisenberg(basis, n, assemble(basis, n, obs), layers)
    assert np.allclose(assemble(basis, n, got), dense, atol=1e-10)
    # Every dense component is present (term support is exact).
    want = {w: c for w, c in expand(basis, n, dense).items()}
    assert {k for k, v in got.items() if abs(v) > 1e-12} == {k for k, v in want.items() if abs(v) > 1e-12}


@pytest.mark.parametrize("seed", range(4))
def test_traced_pauli_generic_expectation_matches_dense(pauli_generic, seed):
    mod, _ = pauli_generic
    n, layers, obs = _pauli_case(50 + seed)
    state = random.Random(seed).randrange(2**n)
    ts = mod.TermSum({mod.Term(list(w), n): c for w, c in obs.items()})
    circ = Circ([[Rot(mod.Term(list(w), n), th) for w, th in layer] for layer in layers])
    got = mod.Propagator(n_threads=1, pin_threads=False).expectation_value(ts, circ, initial_state=state)
    basis = BASES["pauli"]
    dense = heisenberg(basis, n, assemble(basis, n, obs), layers)
    assert abs(got.expectation_value - expectation(basis, n, dense, state).real) < 1e-10


def test_traced_pauli_generic_truncation_and_files(pauli_generic, tmp_path):
    """Truncators of the module work, and term files round-trip."""
    mod, _ = pauli_generic
    n, layers, obs = _pauli_case(77)
    ts = mod.TermSum({mod.Term(list(w), n): c for w, c in obs.items()})
    circ = Circ([[Rot(mod.Term(list(w), n), th) for w, th in layer] for layer in layers])
    path = str(tmp_path / "terms.bin")
    full = mod.Propagator(n_threads=1, pin_threads=False).propagate(ts, circ, filename=path)
    again = mod.TermSum.from_file(path)
    assert _dict(again) == _dict(full)
    streamed = mod.TermSum()
    streamed.merge_from_file(mod.TermStreamer.from_file(path))
    assert _dict(streamed) == _dict(full)
    cut = mod.Propagator(n_threads=1, pin_threads=False, truncation=mod.WeightTruncator(2)).propagate(ts, circ)
    assert all(mod.weight(list(k), n) <= 2 for k in _dict(cut))
    assert len(cut) <= len(full)


def test_traced_pauli_generic_raw_callables_match_dense(pauli_generic):
    mod, _ = pauli_generic
    from .conformance import check_basis_conformance

    n = 2
    check_basis_conformance(
        "pauli", n,
        character=lambda g, t: mod.character(list(g), list(t)),
        product=lambda g, t: (lambda out, e: (tuple(out), e))(*mod.product(list(g), list(t))),
        weight=lambda t: mod.weight(list(t), n),
        trace=lambda t, diag: (mod.trace(list(t), [diag], n)),
    )


# --------------------------------------------- traced Majorana, generic emit


@pytest.fixture(scope="module")
def majorana_generic():
    return _load("majorana")


@pytest.mark.parametrize("seed", range(6))
def test_traced_majorana_generic_matches_dense(majorana_generic, seed):
    mod, _ = majorana_generic
    n = 3  # sites; 6 modes
    rng = random.Random(seed)
    wl = lambda: to_words(rng.randrange(1, 4**n), 1)  # noqa: E731
    layers = [[(wl(), rng.uniform(-2, 2))] for _ in range(6)]
    obs = {wl(): rng.uniform(-1, 1) for _ in range(3)}
    ts = mod.TermSum({mod.Term(list(w), n): c for w, c in obs.items()})
    circ = Circ([[Rot(mod.Term(list(w), n), th) for w, th in layer] for layer in layers])
    got = _dict(mod.Propagator(n_threads=1, pin_threads=False).propagate(ts, circ))
    got = {tuple(to_words(from_words(k), 1)): v for k, v in got.items()}
    basis = BASES["majorana"]
    dense = heisenberg(basis, n, assemble(basis, n, obs), layers)
    assert np.allclose(assemble(basis, n, got), dense, atol=1e-10)
    live = {k for k, v in got.items() if abs(v) > 1e-12}
    assert live == {k for k, v in expand(basis, n, dense).items() if abs(v) > 1e-12}
