"""The vendored core copy, source_dir() order, and the built-in IR hashes in term_io.rs."""

import importlib.util
import re
from pathlib import Path

import pytest

build = pytest.importorskip("propaq.tracer.build")

REPO = Path(__file__).resolve().parents[2]
HAVE_TREE = (REPO / "crates" / "core" / "Cargo.toml").is_file()
needs_tree = pytest.mark.skipif(not HAVE_TREE, reason="no source tree")


def test_source_dir_prefers_env(monkeypatch, tmp_path):
    (tmp_path / "crates" / "core").mkdir(parents=True)
    (tmp_path / "crates" / "core" / "Cargo.toml").write_text("")
    monkeypatch.setenv("PROPAQ_SOURCE_DIR", str(tmp_path))
    assert build.source_dir() == tmp_path
    assert build.core_path(tmp_path) == tmp_path / "crates" / "core"


def test_source_dir_uses_vendored_copy_without_env(monkeypatch):
    monkeypatch.delenv("PROPAQ_SOURCE_DIR", raising=False)
    if not (build.VENDOR_DIR / "propaq-core" / "Cargo.toml").is_file():
        pytest.skip("no vendored copy in this install")
    src = build.source_dir()
    assert src == build.VENDOR_DIR
    assert build.core_path(src) == build.VENDOR_DIR / "propaq-core"
    assert (build.core_path(src) / "src" / "lib.rs").is_file()


def test_generated_crate_depends_on_the_vendored_path(monkeypatch):
    from propaq.tracer.bases import ALL
    from propaq.tracer.build import compile_spec, crate_files

    monkeypatch.delenv("PROPAQ_SOURCE_DIR", raising=False)
    if not (build.VENDOR_DIR / "propaq-core" / "Cargo.toml").is_file():
        pytest.skip("no vendored copy in this install")
    files = crate_files(compile_spec(ALL["z4"]()), build.source_dir(), with_module=True)
    assert f'path = "{(build.VENDOR_DIR / "propaq-core").as_posix()}"' in files["Cargo.toml"]


@needs_tree
def test_vendored_copy_is_in_step_with_crates_core():
    spec = importlib.util.spec_from_file_location("vendor_core", REPO / "scripts" / "vendor_core.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.stale() == [], "run: python scripts/vendor_core.py"


def test_surrogate_path_prefers_vendored_copy(tmp_path):
    (tmp_path / "propaq-surrogate").mkdir()
    (tmp_path / "propaq-surrogate" / "Cargo.toml").write_text("")
    assert build.surrogate_path(tmp_path) == tmp_path / "propaq-surrogate"
    assert build.surrogate_path(tmp_path / "x") == tmp_path / "x" / "crates" / "surrogate"


def test_surrogate_crate_depends_on_surrogate_path(monkeypatch):
    from propaq.tracer.bases import ALL
    from propaq.tracer.build import compile_spec, crate_files

    monkeypatch.delenv("PROPAQ_SOURCE_DIR", raising=False)
    if not (build.VENDOR_DIR / "propaq-surrogate" / "Cargo.toml").is_file():
        pytest.skip("no vendored surrogate in this install")
    files = crate_files(compile_spec(ALL["pauli"]()), build.source_dir(), with_module=True, surrogate=True)
    assert f'path = "{(build.VENDOR_DIR / "propaq-surrogate").as_posix()}"' in files["Cargo.toml"]


@needs_tree
def test_intrinsic_hashes_in_term_io_match_the_programs():
    from propaq.tracer.bases import ALL
    from propaq.tracer.build import compile_spec

    text = (REPO / "crates" / "core" / "src" / "interface" / "term_io.rs").read_text()
    table = dict(re.findall(r'\("([0-9a-f]{16})", BasisKind::(\w+)\)', text))
    want = {compile_spec(ALL[n]()).hash16(): k for n, k in (("pauli", "Pauli"), ("majorana", "Majorana"))}
    assert table == want
