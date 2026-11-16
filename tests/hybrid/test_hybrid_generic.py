"""Hybrid contraction for any basis, through per-unit local matrices (spec 12.2)."""

import random

import numpy as np
import pytest

from propaq.tracer.bases import ALL
from propaq.tracer.build import BuildUnavailable, compile_spec, load_basis
from propaq.tracer.evaluate import evaluate
from propaq.tracer.local import local_table, pauli_local_matrix, table_array, z4_local_matrix
from propaq.tracer.spec import BasisSpec
from propaq.tracer.verify import _kron_le, dense_pauli, dense_z4


def _phase(e, d):
    return np.exp(2j * np.pi * e / d)


def _unit_labels(word, bits, n):
    return [(word >> (bits * u)) & ((1 << bits) - 1) for u in range(n)]


def _kron_of(spec, word, n):
    mats = [np.asarray(spec.local_matrix(lab)) for lab in _unit_labels(word, spec.bits_per_unit, n)]
    return _kron_le(mats)


def _kron_sites(spec, word, n):
    """Tensor product with unit 0 as the first (most significant) factor, like the MPS state."""
    import functools

    mats = [np.asarray(spec.local_matrix(lab)) for lab in _unit_labels(word, spec.bits_per_unit, n)]
    return functools.reduce(np.kron, mats)


# ---------------------------------------------------------------- tables


@pytest.mark.parametrize("name,dense,n", [("pauli", dense_pauli, 3), ("z4", dense_z4, 2)])
def test_local_matrices_match_the_dense_model(name, dense, n):
    spec = ALL[name]()
    rng = random.Random(1)
    for _ in range(40):
        word = rng.randrange(2 ** (spec.bits_per_unit * n))
        assert np.allclose(_kron_of(spec, word, n), dense([word], n), atol=1e-14)


@pytest.mark.parametrize("name,n", [("pauli", 2), ("z4", 2)])
def test_local_matrices_follow_the_traced_product(name, n):
    """W_g W_t = exp(2 pi i e / D) W_key, with (key, e) from the traced product."""
    spec = ALL[name]()
    ir = compile_spec(spec)
    D = spec.phase_order
    rng = random.Random(2)
    for _ in range(60):
        g, t = rng.randrange(2 ** (spec.bits_per_unit * n)), rng.randrange(2 ** (spec.bits_per_unit * n))
        (key, e) = evaluate(ir.programs["product"], [[g], [t]])
        lhs = _kron_of(spec, g, n) @ _kron_of(spec, t, n)
        rhs = _phase(e, D) * _kron_of(spec, key[0], n)
        assert np.allclose(lhs, rhs, atol=1e-12)


def test_label_zero_is_the_identity():
    assert np.array_equal(pauli_local_matrix(0), np.eye(2))
    assert np.array_equal(z4_local_matrix(0), np.eye(4))


def test_pauli_y_is_the_hermitian_string():
    y = pauli_local_matrix(3)
    assert np.allclose(y, np.array([[0, -1j], [1j, 0]]))


def test_z4_matrices_are_unitary_and_obey_zx_equals_i_xz():
    x, z = z4_local_matrix(0b0001), z4_local_matrix(0b0100)
    assert np.allclose(z @ x, 1j * x @ z)
    for label in range(16):
        m = z4_local_matrix(label)
        assert np.allclose(m @ m.conj().T, np.eye(4))


def test_table_is_exact_and_has_the_expected_shape():
    tab = local_table(ALL["z4"]())
    assert len(tab) == 16 * 4 * 4
    flat = {v for pair in tab for v in pair}
    assert flat <= {0.0, 1.0, -1.0}


def test_table_enters_the_ir_hash():
    base = ALL["pauli"]()
    ir0 = compile_spec(base)
    assert ir0.meta["local"]["dim"] == 2

    def flipped(label):
        m = pauli_local_matrix(label)
        return -m if label == 2 else m

    other = ALL["pauli"]()
    other.local_matrix = flipped
    assert compile_spec(other).hash16() != ir0.hash16()
    plain = ALL["pauli"]()
    plain.local_dim, plain.local_matrix = None, None
    assert "local" not in compile_spec(plain).meta
    assert compile_spec(plain).hash16() != ir0.hash16()


def test_bad_tables_are_rejected():
    spec = ALL["pauli"]()
    spec.local_matrix = lambda label: np.zeros((3, 3))
    with pytest.raises(ValueError):
        local_table(spec)
    spec.local_matrix = lambda label: 2 * np.eye(2)
    with pytest.raises(ValueError, match="identity"):
        local_table(spec)
    spec.local_matrix = None
    with pytest.raises(ValueError):
        local_table(spec)


# ---------------------------------------------------------------- Rust


def _random_mps(rng, n, d, chi):
    bonds = [1] + [chi] * (n - 1) + [1]
    return [
        (rng.standard_normal((bonds[k], d, bonds[k + 1])) + 1j * rng.standard_normal((bonds[k], d, bonds[k + 1]))).astype(
            np.complex128
        )
        for k in range(n)
    ]


def _dense_state(arrays):
    psi = np.ones((1, 1), dtype=complex)  # (physical so far, bond)
    for a in arrays:
        psi = np.einsum("pl,lsr->psr", psi, a).reshape(-1, a.shape[2])
    return psi[:, 0]


@pytest.fixture(scope="module")
def z4_module(tmp_path_factory):
    import os

    os.environ.setdefault("PROPAQ_CACHE_DIR", str(tmp_path_factory.mktemp("bases")))
    try:
        return load_basis(ALL["z4"]())
    except BuildUnavailable:
        pytest.skip("no Rust toolchain or source tree")


@pytest.mark.parametrize("n,chi,seed", [(2, 3, 0), (3, 4, 1), (4, 2, 2)])
def test_z4_hybrid_matches_dense(z4_module, n, chi, seed):
    from propaq._rust_core import hybrid_expectation_raw  # noqa: F401
    from propaq.hybrid import _generic_expectation

    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    spec = ALL["z4"]()
    mod = z4_module
    terms = {}
    for _ in range(25):
        w = pyrng.randrange(2 ** (4 * n))
        terms[w] = complex(rng.standard_normal(), rng.standard_normal())
    ts = mod.TermSum({mod.Term([w], n): c for w, c in terms.items()}) if hasattr(mod, "Term") else None
    arrays = _random_mps(rng, n, 4, chi)
    got = _generic_expectation(ts, mod, arrays)
    psi = _dense_state(arrays)
    op = sum(c * _kron_sites(spec, w, n) for w, c in terms.items())
    want = (psi.conj() @ op @ psi) / (psi.conj() @ psi)
    assert abs(got - want) < 1e-9 * max(1.0, abs(want))


def test_z4_identity_term_gives_its_coefficient(z4_module):
    from propaq.hybrid import _generic_expectation

    mod = z4_module
    rng = np.random.default_rng(3)
    ts = mod.TermSum({mod.Term([0], 3): 2.5 - 1.0j})
    assert abs(_generic_expectation(ts, mod, _random_mps(rng, 3, 4, 3)) - (2.5 - 1.0j)) < 1e-10


def test_z4_rejects_a_qubit_mps_and_a_too_wide_observable(z4_module):
    from propaq.hybrid import _generic_expectation

    mod = z4_module
    rng = np.random.default_rng(4)
    ts = mod.TermSum({mod.Term([0b0001], 2): 1.0})
    with pytest.raises(ValueError):
        _generic_expectation(ts, mod, _random_mps(rng, 2, 2, 2))
    wide = mod.TermSum({mod.Term([1 << 8], 3): 1.0})
    with pytest.raises(ValueError, match="more than"):
        _generic_expectation(wide, mod, _random_mps(rng, 2, 4, 2))


def test_generic_pauli_equals_native_pauli(tmp_path, monkeypatch):
    """The compiled traced Pauli and the native path give the same number."""
    monkeypatch.setenv("PROPAQ_CACHE_DIR", str(tmp_path))
    try:
        gen = load_basis(ALL["pauli"](), force_generic=True)
    except BuildUnavailable:
        pytest.skip("no Rust toolchain or source tree")
    from propaq._rust_core import hybrid_expectation
    from propaq.datatypes import PauliString, PauliTermSum
    from propaq.hybrid import _generic_expectation

    n = 4
    rng = np.random.default_rng(5)
    pyrng = random.Random(5)
    words = {pyrng.randrange(2 ** (2 * n)): float(rng.standard_normal()) for _ in range(30)}
    nat = PauliTermSum()
    for w, c in words.items():
        x = sum(((w >> (2 * q)) & 1) << q for q in range(n))
        z = sum(((w >> (2 * q + 1)) & 1) << q for q in range(n))
        nat.add(PauliString(x, z, n), c)
    ts = gen.TermSum({gen.Term([w], n): c for w, c in words.items()})
    arrays = _random_mps(rng, n, 2, 3)
    want = hybrid_expectation(nat, arrays)
    got = _generic_expectation(ts, gen, arrays)
    assert got.real == pytest.approx(want, rel=1e-10, abs=1e-12)
