"""An intrinsic tag resolves to the built-in classes, with no build."""

import pytest

from propaq.tracer import build
from propaq.tracer.bases import ALL


def _no_build(*a, **k):
    raise AssertionError("an intrinsic must not trigger a build")


@pytest.mark.parametrize("name", ["pauli", "majorana"])
def test_intrinsic_resolves_to_builtin_classes(monkeypatch, name):
    import propaq._rust_core as rc

    monkeypatch.setattr(build, "build_basis", _no_build)
    basis = build.load_basis(ALL[name]())
    cls = {"pauli": "Pauli", "majorana": "Majorana"}[name]
    assert basis.Propagator is getattr(rc, f"{cls}Propagator")
    assert basis.TermSum is getattr(rc, f"{cls}TermSum")
    assert basis.TermStreamer is getattr(rc, f"{cls}TermStreamer")
    assert basis.Term is getattr(rc, "PauliString" if name == "pauli" else "MajoranaMonomial")
    assert basis.intrinsic == name


def test_force_generic_skips_the_intrinsic(monkeypatch):
    calls = []

    def fake(spec, **kw):
        calls.append(spec.name)
        raise build.BuildUnavailable("stop here")

    monkeypatch.setattr(build, "build_basis", fake)
    out = build.load_basis(ALL["pauli"](), force_generic=True)
    assert calls == ["pauli"]
    assert type(out).__name__ == "FallbackBasis"


def test_unknown_intrinsic_is_rejected():
    with pytest.raises(ValueError):
        build.resolve_intrinsic("fermion")


def test_builtin_docstrings_keep_their_basis_text():
    import propaq._rust_core as rc

    assert "The Pauli observable to back-propagate." in rc.PauliPropagator.propagate.__doc__
    assert "A MajoranaCircuit applied to the reference state." in rc.MajoranaPropagator.expectation_value.__doc__
    assert "Load a PauliTermSum from" in rc.PauliTermSum.from_file.__doc__
    assert "monomials already present" in rc.MajoranaTermSum.merge_from_file.__doc__


def test_generated_builtins_keep_their_basis_kind():
    """The built-in modules use the Pauli and Majorana kinds, not Custom."""
    from propaq.tracer import regen

    files = regen.generate()
    assert "BasisKind::Pauli;" in files["crates/traced/src/pauli.rs"]
    assert "BasisKind::Majorana;" in files["crates/traced/src/majorana.rs"]
    assert "BasisKind::Custom;" in files["crates/traced/src/z4.rs"]
    for name in ("pauli", "majorana"):
        assert "BasisKind::Custom" not in files[f"crates/traced/src/{name}.rs"]


def test_generated_traced_crate_is_in_sync():
    """The checked-in Rust of the built-ins equals what the tracer emits now."""
    from propaq.tracer import regen
    from propaq.tracer.build import tree_dir

    try:
        root = tree_dir()
    except Exception as exc:  # no source tree (wheel only)
        pytest.skip(str(exc))
    for rel, text in regen.generate().items():
        path = root / rel
        if not path.is_file():
            pytest.skip(f"no source file {rel}")
        assert path.read_text() == text, f"{rel} differs; run python -m propaq.tracer.regen"


def _dense_pauli(x, z, n):
    import numpy as np

    mats = {(0, 0): np.eye(2), (1, 0): np.array([[0, 1], [1, 0]]), (0, 1): np.diag([1, -1]),
            (1, 1): np.array([[0, -1j], [1j, 0]])}
    out = np.eye(1)
    for q in reversed(range(n)):
        out = np.kron(out, mats[((x >> q) & 1, (z >> q) & 1)])
    return out


def test_builtin_pauli_propagator_matches_dense_evolution():
    """The built-in PauliPropagator (traced program) against exact matrices."""
    import random

    import numpy as np
    from scipy.linalg import expm

    from propaq.circuits import PauliCircuit, PauliRotation
    from propaq.datatypes import PauliString, PauliTermSum
    from propaq.propagators import PauliPropagator

    n, rng = 3, random.Random(5)
    for _ in range(5):
        gates = [(rng.randrange(1, 8), rng.randrange(0, 8), rng.uniform(-2, 2)) for _ in range(5)]
        gates = [(x, z, th) for x, z, th in gates if x or z]
        obs = [(rng.randrange(0, 8), rng.randrange(0, 8), rng.uniform(-1, 1)) for _ in range(3)]
        ts = PauliTermSum()
        dense = np.zeros((2**n, 2**n), dtype=complex)
        for x, z, c in obs:
            ts.add(PauliString(x, z, n), c)
            dense += c * _dense_pauli(x, z, n)
        circ = PauliCircuit([[PauliRotation(PauliString(x, z, n), th)] for x, z, th in gates])
        for x, z, th in reversed(gates):
            u = expm(-0.5j * th * _dense_pauli(x, z, n))
            dense = u.conj().T @ dense @ u
        out = PauliPropagator(pin_threads=False).propagate(ts, circ)
        got = sum((c * _dense_pauli(int(p.x), int(p.z), n) for p, c in out.items()), np.zeros_like(dense))
        assert np.allclose(got, dense, atol=1e-10)
