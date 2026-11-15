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
