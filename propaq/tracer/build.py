"""Build pipeline: spec -> IR -> Rust crate in the cache -> cargo -> import.

The crate lives in `~/.cache/propaq/bases/<hash16>/` and depends on
`propaq-core` by path. The path comes from the env var PROPAQ_SOURCE_DIR,
else from the repo root that holds this file.

The core macro `propaq_core::define_raw_basis_module!` makes the PyO3
module (spec section 8, item 5). Terms of the module are raw words
(`Term(words, n_units)`), so the crate needs no term type.
`build_basis(..., check_only=True)` runs `cargo check` on the crate without
the macro call.

A spec with `intrinsic="pauli"` or `"majorana"` needs no build. `load_basis`
returns the built-in classes for it (see `resolve_intrinsic`). Pass
`force_generic=True` to compile the generic code instead.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .analysis import add_fold
from .emit_rust import camel, emit_rust
from .fallback import FallbackBasis
from .ir import BasisIR
from .spec import BasisSpec


class BuildUnavailable(RuntimeError):
    """No Rust toolchain, or no core source tree."""


class BuildError(RuntimeError):
    """cargo failed."""


def source_dir() -> Path:
    """Directory that holds `crates/core` (the propaq source tree).

    Search order: `PROPAQ_SOURCE_DIR`, the parents of this file, the working
    directory and its parents.
    """
    env = os.environ.get("PROPAQ_SOURCE_DIR")
    if env:
        cands = [Path(env)]
    else:
        cwd = Path.cwd().resolve()
        cands = [*Path(__file__).resolve().parents, cwd, *cwd.parents]
    for c in cands:
        if (c / "crates" / "core" / "Cargo.toml").is_file():
            return c
    raise BuildUnavailable("propaq source tree not found; set PROPAQ_SOURCE_DIR")


def cache_root() -> Path:
    env = os.environ.get("PROPAQ_CACHE_DIR")
    return Path(env) if env else Path.home() / ".cache" / "propaq" / "bases"


def compile_spec(spec: BasisSpec) -> BasisIR:
    """Trace all programs and add the discovered fold program."""
    return add_fold(spec.trace_ir())


def uses_complex(ir: BasisIR) -> bool:
    """True if the compiled module holds complex coefficients (dtype "complex128").

    A basis with more than two branches or a character modulus above 2 has
    complex rotation tables. Other bases keep real coefficients.
    """
    return ir.meta["branches"] > 2 or ir.meta["char_modulus"] > 2


def module_name(ir: BasisIR) -> str:
    return f"propaq_basis_{ir.hash16()}"


def crate_files(ir: BasisIR, source: Path, with_module: bool) -> dict:
    """The files of the generated crate: relative path -> text."""
    h = ir.hash16()
    ty = camel(ir.meta["name"])
    pyo3_features = '"num-complex", "extension-module"' if with_module else '"num-complex"'
    cargo = f"""[package]
name = "propaq-basis-{h}"
version = "0.0.0"
edition = "2021"

[lib]
name = "propaq_basis_{h}"
crate-type = ["cdylib", "rlib"]

[dependencies]
propaq-core = {{ path = "{(source / 'crates' / 'core').as_posix()}" }}
num-complex = "0.4"
pyo3 = {{ version = "0.27", features = [{pyo3_features}] }}
pyo3-stub-gen = "0.23"

[workspace]

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 16
"""
    lib = emit_rust(ir)
    if with_module:
        b = ir.meta["bits_per_unit"]
        coeff = "propaq_core::basis_module::Cx" if uses_complex(ir) else "f64"
        lib += (
            f"\npropaq_core::define_raw_basis_module! {{\n    module: propaq_basis_{h},\n    algebra: {ty},\n"
            f"    bits_per_unit: {b},\n    coeff: {coeff},\n}}\n"
        )
    return {"Cargo.toml": cargo, "src/lib.rs": lib, "ir.json": ir.to_json()}


@dataclass
class BuildResult:
    hash16: str
    crate_dir: Path
    so_path: Path | None
    cached: bool
    checked_only: bool


def core_stamp(source: Path) -> str:
    """SHA-256 of the core sources. A change of core invalidates cached builds."""
    import hashlib

    h = hashlib.sha256()
    root = source / "crates" / "core"
    for p in sorted([root / "Cargo.toml", *root.joinpath("src").rglob("*.rs")]):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _target_dir() -> Path:
    """One shared target dir for all generated crates, so that deps build once."""
    env = os.environ.get("CARGO_TARGET_DIR")
    return Path(env) if env else cache_root() / "target"


def _cargo(args, crate_dir, jobs):
    cargo = shutil.which("cargo")
    if cargo is None:
        raise BuildUnavailable("cargo not found on PATH")
    cmd = [cargo, *args, "-j", str(jobs)]
    env = dict(os.environ, CARGO_TARGET_DIR=str(_target_dir()))
    r = subprocess.run(cmd, cwd=crate_dir, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise BuildError(f"{' '.join(cmd)} failed:\n{r.stderr[-4000:]}")
    return r


def _write_crate(files, crate_dir: Path, source: Path):
    changed = False
    for rel, text in files.items():
        p = crate_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists() or p.read_text() != text:
            p.write_text(text)
            changed = True
    lock = source / "Cargo.lock"
    if lock.is_file() and not (crate_dir / "Cargo.lock").exists():
        shutil.copy(lock, crate_dir / "Cargo.lock")
    return changed


def build_basis(spec: BasisSpec, *, check_only=False, force=False, jobs=4, with_module=None, ir: BasisIR | None = None) -> BuildResult:
    """Compile `spec` to a Python extension in the cache.

    The cache key is the SHA-256 of the canonical IR. A cached .so is reused
    while the IR, the generated crate text and the core sources are the same
    (`build.stamp` holds the check value). `force=True` rebuilds.
    With `check_only`, run `cargo check` on a crate without the module macro.
    """
    ir = ir or compile_spec(spec)
    source = source_dir()
    h = ir.hash16()
    crate_dir = cache_root() / h
    with_module = (not check_only) if with_module is None else with_module
    so = crate_dir / f"propaq_basis_{h}.so"
    files = crate_files(ir, source, with_module)
    import hashlib

    stamp = hashlib.sha256((core_stamp(source) + files["src/lib.rs"] + files["Cargo.toml"]).encode()).hexdigest()
    stamp_file = crate_dir / "build.stamp"
    if not check_only and not force and so.is_file() and stamp_file.is_file() and stamp_file.read_text() == stamp:
        return BuildResult(h, crate_dir, so, True, False)
    _write_crate(files, crate_dir, source)
    if check_only:
        _cargo(["check", "--release"], crate_dir, jobs)
        return BuildResult(h, crate_dir, None, False, True)
    _cargo(["build", "--release"], crate_dir, jobs)
    target = _target_dir()
    built = target / "release" / f"libpropaq_basis_{h}.so"
    if not built.is_file():
        raise BuildError(f"expected library not found: {built}")
    shutil.copy(built, so)
    stamp_file.write_text(stamp)
    return BuildResult(h, crate_dir, so, False, False)


def import_extension(so_path: Path, name: str):
    """Import a built extension module by path."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class BuiltinBasis:
    """The built-in classes of a native intrinsic, with the module's names.

    Attributes: `Term`, `TermSum`, `Propagator`, `TermStreamer`, `intrinsic`.
    """

    def __init__(self, intrinsic, Term, TermSum, Propagator, TermStreamer):
        self.intrinsic = intrinsic
        self.Term, self.TermSum = Term, TermSum
        self.Propagator, self.TermStreamer = Propagator, TermStreamer

    def __repr__(self):
        return f"<built-in basis {self.intrinsic!r}>"


def resolve_intrinsic(name: str) -> BuiltinBasis:
    """The built-in classes for `intrinsic="pauli"` or `"majorana"`. No build."""
    from .. import _rust_core as rc

    if name == "pauli":
        return BuiltinBasis(name, rc.PauliString, rc.PauliTermSum, rc.PauliPropagator, rc.PauliTermStreamer)
    if name == "majorana":
        return BuiltinBasis(name, rc.MajoranaMonomial, rc.MajoranaTermSum, rc.MajoranaPropagator, rc.MajoranaTermStreamer)
    raise ValueError(f"unknown intrinsic {name!r}")


def load_basis(spec: BasisSpec, *, force_generic: bool = False, **kwargs):
    """The classes of a basis, as an object with `Term`, `TermSum`, `Propagator`.

    1. A spec with an intrinsic gives the built-in classes, with no build.
       `force_generic=True` skips this and compiles the traced program.
    2. Otherwise the basis is built (or taken from the cache) and imported.
    3. With no Rust toolchain, the pure-Python `FallbackBasis` is returned.
    """
    if spec.intrinsic and not force_generic:
        return resolve_intrinsic(spec.intrinsic)
    ir = compile_spec(spec)
    try:
        res = build_basis(spec, ir=ir, **kwargs)
        return import_extension(res.so_path, module_name(ir))
    except BuildUnavailable:
        return FallbackBasis(ir)
