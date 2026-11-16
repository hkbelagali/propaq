"""Compiled traced bases with `noise_key` and `keep` programs, and plain-data settings.

Three bases are built: Pauli, Majorana and Z4, each with a traced
`noise_key` and `keep`. Each result is compared with a pure-Python
reference (`reference_propagate`) and with hand-written Python functions of
the term words. The tests skip when there is no Rust toolchain.
"""

import random
from dataclasses import replace

import numpy as np
import pytest

from propaq.tracer.bases import ALL
from propaq.tracer.frontend import (
    TracedBasis,
    lower_pauli_gate,
    reference_keep,
    reference_noise_key,
    reference_propagate,
    words_to_int,
)

build = pytest.importorskip("propaq.tracer.build")

EVEN = 0x5555_5555_5555_5555
P1 = 0x1111_1111_1111_1111


# ---- traced programs (one per basis) ----


def _pauli_key(t):  # number of Y sites
    return (t & (t >> 1) & EVEN).popcount()


def _pauli_keep(t):  # weight < 4
    return ((t | (t >> 1)) & EVEN).popcount() < 4


def _maj_key(t):  # number of modes
    return t.popcount()


def _maj_keep(t):  # fewer than 5 modes
    return t.popcount() < 5


def _z4_key(t):  # units with a z digit
    return (((t >> 2) | (t >> 3)) & P1).popcount()


def _z4_keep(t):  # at most 2 units with any digit
    return ((t | (t >> 1) | (t >> 2) | (t >> 3)) & P1).popcount() < 3


# Plain-Python references over the words (a list of ints), written by hand.
def _units(words, b, n):
    v = words_to_int(words)
    return [(v >> (b * u)) & ((1 << b) - 1) for u in range(n)]


def ref_key(name, words, n):
    if name == "pauli":
        return sum(1 for p in _units(words, 2, n) if p == 3)
    if name == "majorana":
        return sum(bin(p).count("1") for p in _units(words, 2, n))
    return sum(1 for p in _units(words, 4, n) if p >> 2)


def ref_keep(name, words, n):
    if name == "pauli":
        return int(sum(1 for p in _units(words, 2, n) if p) < 4)
    if name == "majorana":
        return int(sum(bin(p).count("1") for p in _units(words, 2, n)) < 5)
    return int(sum(1 for p in _units(words, 4, n) if p) < 3)


SPECS = {
    "pauli": lambda: replace(ALL["pauli"](), noise_key=_pauli_key, keep=_pauli_keep, lower_gate=lower_pauli_gate),
    "majorana": lambda: replace(ALL["majorana"](), noise_key=_maj_key, keep=_maj_keep),
    "z4": lambda: replace(ALL["z4"](), noise_key=_z4_key, keep=_z4_keep),
}
BITS = {"pauli": 2, "majorana": 2, "z4": 4}
SIZES = {"pauli": 4, "majorana": 3, "z4": 3}


def _load(spec):
    try:
        ir = build.compile_spec(spec)
        res = build.build_basis(spec, ir=ir, jobs=6)
    except build.BuildUnavailable as exc:  # pragma: no cover
        pytest.skip(str(exc))
    return build.import_extension(res.so_path, build.module_name(ir)), ir


@pytest.fixture(scope="module", params=["pauli", "majorana", "z4"])
def kb(request):
    name = request.param
    spec = SPECS[name]()
    mod, ir = _load(spec)
    return name, TracedBasis(spec, module=mod, n_units=SIZES[name]), ir


@pytest.fixture(scope="module")
def plain_pauli():
    spec = replace(ALL["pauli"](), lower_gate=lower_pauli_gate)
    mod, ir = _load(spec)
    return TracedBasis(spec, module=mod), ir


def rand_words(rng, name, n):
    b = BITS[name]
    return [rng.randrange(1 << (b * n))]


def case(name, seed, n_layers=4, per_layer=2):
    rng = random.Random(seed)
    n, b = SIZES[name], BITS[name]
    layers = [
        [([rng.randrange(1, 1 << (b * n))], rng.uniform(-1.4, 1.4)) for _ in range(rng.randint(1, per_layer))]
        for _ in range(n_layers)
    ]
    cx = name == "z4"
    obs = {
        (rng.randrange(1, 1 << (b * n)),): (complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) if cx else rng.uniform(-1, 1))
        for _ in range(4)
    }
    return n, layers, obs


def run(basis, obs, layers, n, **settings):
    prop = basis.propagator(n_threads=1, pin_threads=False, **settings)
    ts = basis.module.TermSum({basis.module.Term(list(k), n): v for k, v in obs.items()})
    circ = basis.circuit([[(basis.module.Term(list(g), n), a) for g, a in layer] for layer in layers])
    return basis.to_dict(prop.propagate(ts, circ))


def close(got, want, tol=1e-11):
    keys = set(got) | set(want)
    assert max((abs(got.get(k, 0) - want.get(k, 0)) for k in keys), default=0) < tol


# ---- the IR and the raw functions ----


def test_programs_enter_the_ir_hash_only_when_defined():
    for name in ("pauli", "majorana", "z4"):
        base = build.compile_spec(ALL[name]())
        withp = build.compile_spec(SPECS[name]())
        assert "noise_key" not in base.programs and "keep" not in base.programs
        assert {"noise_key", "keep"} <= set(withp.programs)
        assert base.hash16() != withp.hash16()
    # Same spec twice: same hash.
    assert build.compile_spec(SPECS["z4"]()).hash16() == build.compile_spec(SPECS["z4"]()).hash16()


def test_intrinsic_with_keep_is_built_not_taken_from_the_builtin(monkeypatch):
    calls = []

    def fake(spec, **kw):
        calls.append(spec.name)
        raise build.BuildUnavailable("stop")

    monkeypatch.setattr(build, "build_basis", fake)
    build.load_basis(SPECS["pauli"]())
    assert calls == ["pauli"]


def test_raw_noise_key_and_keep_match_python(kb):
    name, basis, ir = kb
    mod, rng, n = basis.module, random.Random(5), SIZES[name]
    assert mod.HAS_NOISE_KEY and mod.HAS_KEEP
    for _ in range(60):
        w = rand_words(rng, name, n)
        assert mod.noise_key(w, n) == ref_key(name, w, n) == reference_noise_key(ir, w, n)
        assert mod.keep(w, n) == ref_keep(name, w, n) == reference_keep(ir, w, n)


def test_raw_functions_on_many_words(kb):
    name, basis, ir = kb
    n, b = 70, BITS[name]
    rng = random.Random(8)
    for _ in range(20):
        v = rng.getrandbits(n * b) & rng.getrandbits(n * b) & rng.getrandbits(n * b)
        w = [(v >> (64 * i)) & ((1 << 64) - 1) for i in range(-(-n * b // 64))]
        assert basis.module.noise_key(w, n) == ref_key(name, w, n)
        assert basis.module.keep(w, n) == ref_keep(name, w, n)


def test_plain_basis_has_default_programs(plain_pauli):
    basis, _ = plain_pauli
    assert not basis.module.HAS_NOISE_KEY and not basis.module.HAS_KEEP
    assert basis.module.noise_key([5], 3) == 0 and basis.module.keep([5], 3) == 1


# ---- run: keep ----


@pytest.mark.parametrize("seed", range(5))
def test_keep_drops_terms_like_the_reference(kb, seed):
    name, basis, ir = kb
    n, layers, obs = case(name, seed)
    got = run(basis, obs, layers, n)
    want = reference_propagate(ir, obs, layers, n, keep=lambda k: ref_keep(name, k, n))
    assert all(ref_keep(name, k, n) for k in got)
    close(got, want)


def test_keep_actually_changes_the_result(kb):
    name, basis, ir = kb
    changed = 0
    for seed in range(8):
        n, layers, obs = case(name, seed, n_layers=6)
        free = reference_propagate(ir, obs, layers, n)
        kept = reference_propagate(ir, obs, layers, n, keep=lambda k: ref_keep(name, k, n))
        changed += len(free) != len(kept)
    assert changed > 0


@pytest.mark.parametrize("seed", range(3))
def test_term_budget_floor_suppresses_keep(kb, seed):
    name, basis, ir = kb
    n, layers, obs = case(name, seed)
    got = run(basis, obs, layers, n, truncation={"min_terms": 10**6})
    # The observable is filtered at the start. Later rotations are not.
    start = {k: v for k, v in obs.items() if ref_keep(name, k, n)}
    close(got, reference_propagate(ir, start, layers, n))


# ---- run: keyed noise ----


def _factors(name):
    n = SIZES[name]
    return {k: [round(1 - 0.04 * (k + 1) - 0.01 * j, 4) for j in range(3)] for k in range(0, n + 2)}


@pytest.mark.parametrize("seed", range(5))
def test_keyed_noise_matches_the_reference(kb, seed):
    name, basis, ir = kb
    n, layers, obs = case(name, seed)
    fac = _factors(name)
    fac.pop(1)  # key 1 falls to the default
    default = 0.9

    def noise(k, idx):
        f = fac.get(ref_key(name, k, n))
        return default if f is None else f[min(idx, len(f) - 1)]

    want = reference_propagate(ir, obs, layers, n, noise=noise, keep=lambda k: ref_keep(name, k, n))
    got = run(basis, obs, layers, n, noise={"key_factors": fac, "default": default})
    close(got, want)
    # The class form gives the same result.
    got2 = run(basis, obs, layers, n, noise=basis.keyed_noise(fac, default))
    close(got2, got, 1e-15)


def test_keyed_noise_from_a_function(kb):
    name, basis, ir = kb
    n, layers, obs = case(name, 11)
    f = lambda key, layer: 1.0 - 0.05 * key - 0.02 * layer
    model = basis.keyed_noise({"keys": range(0, 2 * n + 2), "n_layers": len(layers), "f": f})
    want = reference_propagate(
        ir, obs, layers, n, noise=lambda k, idx: f(ref_key(name, k, n), idx), keep=lambda k: ref_keep(name, k, n)
    )
    close(run(basis, obs, layers, n, noise=model), want)


def test_keyed_noise_model_object(kb):
    _, basis, _ = kb
    m = basis.keyed_noise({0: 0.5, 3: [0.1, 0.2]}, 0.7)
    assert m.default == 0.7
    assert m.factor(0) == 0.5 and m.factor(3, 0) == 0.1 and m.factor(3, 5) == 0.2 and m.factor(9) == 0.7
    assert m.factors == {0: [0.5], 3: [0.1, 0.2]}


def test_keyed_noise_with_wide_key_span_uses_the_sparse_table(kb):
    _, basis, _ = kb
    m = basis.keyed_noise({0: 0.5, 10**9: 0.25}, 1.0)
    assert m.factor(0) == 0.5 and m.factor(10**9) == 0.25 and m.factor(5) == 1.0


def test_keyed_noise_needs_a_noise_key_program(plain_pauli):
    basis, _ = plain_pauli
    obs = {(0b10,): 1.0}
    with pytest.raises(RuntimeError, match="noise_key"):
        run(basis, obs, [[([0b01], 0.3)]], 1, noise={"key_factors": {0: 0.5}})


# ---- plain-data settings ----


def test_damping_forms_agree(kb):
    name, basis, ir = kb
    n, layers, obs = case(name, 3)
    from propaq import UniformNoiseModel

    g = 0.03
    from propaq.tracer.fallback import weight_of

    want = reference_propagate(
        ir, obs, layers, n, noise=lambda k, idx: float(np.exp(-g * weight_of(ir, k, n))), keep=lambda k: ref_keep(name, k, n)
    )
    for setting in (g, {"damping": g}, UniformNoiseModel(g), basis.module.UniformNoiseModel(g)):
        close(run(basis, obs, layers, n, noise=setting), want)


def test_weight_truncation_forms_agree(kb):
    name, basis, ir = kb
    n, layers, obs = case(name, 4, n_layers=6)
    import propaq

    want = reference_propagate(ir, obs, layers, n, keep=lambda k: ref_keep(name, k, n), max_weight=2)
    forms = [
        {"weight": 2},
        {"weight_cutoff": 2},
        basis.module.WeightTruncator(2),
        propaq.WeightTruncator(2),
        [propaq.WeightTruncator(2)],
        propaq.TruncationPolicy(weight_cutoff=2),
    ]
    for t in forms:
        close(run(basis, obs, layers, n, truncation=t), want)


def test_coefficient_and_budget_forms_agree(kb):
    name, basis, ir = kb
    n, layers, obs = case(name, 6, n_layers=6)
    import propaq

    a = run(basis, obs, layers, n, truncation={"weight": 3, "coefficient": 1e-2, "min_terms": 2})
    b = run(
        basis,
        obs,
        layers,
        n,
        truncation=[propaq.WeightTruncator(3), propaq.CoefficientTruncator(1e-2), propaq.TermBudget(2)],
    )
    c = run(basis, obs, layers, n, truncation=propaq.TruncationPolicy(3, 1e-2, 2))
    close(a, b, 1e-15)
    close(a, c, 1e-15)


def test_bad_settings_are_rejected(plain_pauli):
    basis, _ = plain_pauli
    for kw in ({"truncation": {"weigth": 2}}, {"noise": {"dampening": 1.0}}, {"noise": {"damping": 1.0, "key_factors": {}}}):
        with pytest.raises(ValueError):
            basis.propagator(**kw)


# ---- defaults keep the old results ----


@pytest.mark.parametrize("seed", range(4))
def test_plain_basis_matches_native_pauli(plain_pauli, seed):
    basis, _ = plain_pauli
    from propaq import PauliPropagator, PauliString, PauliTermSum, UniformNoiseModel, WeightTruncator
    from propaq.circuits import PauliCircuit, PauliRotation

    rng = random.Random(seed)
    n = 4
    layers = [[(rng.randrange(1, 4**n), rng.uniform(-1.3, 1.3)) for _ in range(2)] for _ in range(4)]
    obs = {rng.randrange(1, 4**n): rng.uniform(-1, 1) for _ in range(3)}

    def ps(v):
        x = sum(((v >> (2 * q)) & 1) << q for q in range(n))
        z = sum(((v >> (2 * q + 1)) & 1) << q for q in range(n))
        return PauliString(x, z, n)

    ts = PauliTermSum()
    for v, c in obs.items():
        ts.add(ps(v), c)
    circ = PauliCircuit([[PauliRotation(ps(g), a) for g, a in layer] for layer in layers])
    nat = PauliPropagator(noise=UniformNoiseModel(0.02), truncation=WeightTruncator(3), n_threads=1, pin_threads=False)
    want = {}
    for t, c in nat.propagate(ts, circ).items():
        v = sum((((int(t.x) >> q) & 1) << (2 * q)) | (((int(t.z) >> q) & 1) << (2 * q + 1)) for q in range(n))
        want[(v,)] = c
    got = run(
        basis,
        {(k,): c for k, c in obs.items()},
        [[([g], a) for g, a in layer] for layer in layers],
        n,
        noise=UniformNoiseModel(0.02),
        truncation=WeightTruncator(3),
    )
    assert set(got) == set(want)
    close(got, want, 1e-12)


# ---- Qiskit and Cirq ----


def _native_pauli_dict(circ, obs_ps, n, ts_cls):
    from propaq import PauliPropagator

    out = {}
    for t, c in PauliPropagator(n_threads=1, pin_threads=False).propagate(obs_ps, circ).items():
        v = sum((((int(t.x) >> q) & 1) << (2 * q)) | (((int(t.z) >> q) & 1) << (2 * q + 1)) for q in range(n))
        out[(v,)] = c
    return out


def _obs(n):
    from propaq import PauliString, PauliTermSum

    ts = PauliTermSum()
    ts.add(PauliString(0, 0b101, n), 1.0)
    ts.add(PauliString(0b010, 0b010, n), 0.5)
    return ts, {(2 << 0 | 2 << 4,): 1.0, (3 << 2,): 0.5}


def test_qiskit_circuit_lowers_like_the_native_circuit(plain_pauli):
    qiskit = pytest.importorskip("qiskit")
    from propaq.circuits import PauliCircuit

    basis, _ = plain_pauli
    n = 3
    qc = qiskit.QuantumCircuit(n)
    qc.rx(0.3, 0)
    qc.ry(-0.7, 1)
    qc.rzz(0.45, 0, 2)
    qc.barrier()
    qc.rz(1.1, 2)
    qc.rxx(0.2, 1, 2)
    qc.ryy(-0.6, 0, 1)
    ts, obs = _obs(n)
    want = _native_pauli_dict(PauliCircuit.from_qiskit(qc), ts, n, None)
    circ = basis.from_qiskit(qc)
    assert circ.layers and all(r.generator.n_units == n for r in circ.rotations)
    mod = basis.module
    got = basis.to_dict(
        basis.propagator(n_threads=1, pin_threads=False).propagate(
            mod.TermSum({mod.Term(list(k), n): v for k, v in obs.items()}), circ
        )
    )
    close(got, want, 1e-12)
    assert got


def test_cirq_circuit_lowers_like_the_native_circuit(plain_pauli):
    cirq = pytest.importorskip("cirq")
    from propaq.circuits import PauliCircuit

    basis, _ = plain_pauli
    n = 3
    q = cirq.LineQubit.range(n)
    circuit = cirq.Circuit(
        [cirq.rx(0.3).on(q[0]), cirq.ry(-0.7).on(q[1])],
        [cirq.rz(1.1).on(q[2])],
        [cirq.rx(0.5).on(q[2]), cirq.rz(0.25).on(q[0])],
    )
    ts, obs = _obs(n)
    want = _native_pauli_dict(PauliCircuit.from_cirq(circuit), ts, n, None)
    circ = basis.from_cirq(circuit, n_units=n)
    assert [len(l) for l in circ.layers] == [len(m) for m in circuit]
    mod = basis.module
    got = basis.to_dict(
        basis.propagator(n_threads=1, pin_threads=False).propagate(
            mod.TermSum({mod.Term(list(k), n): v for k, v in obs.items()}), circ
        )
    )
    close(got, want, 1e-12)


def test_labels_build_the_same_circuit_run(plain_pauli):
    basis, _ = plain_pauli
    ts = basis.termsum({"ZIZ": 1.0, "IYI": 0.5})
    circ = basis.circuit([[("XII", 0.3)], [("IYI", 0.2), ("ZZI", 0.4)]])
    out = basis.propagator(n_threads=1, pin_threads=False).propagate(ts, circ)
    assert len(out) > 2
    assert all(isinstance(basis.label(t), str) for t, _ in out.items())
