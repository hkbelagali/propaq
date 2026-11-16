"""Build pipeline: spec -> IR -> Rust crate in the cache -> cargo -> import.

The crate lives in `~/.cache/propaq/bases/<hash16>/` and depends on
`propaq-core` by path. The path comes from `source_dir()`: the env var
PROPAQ_SOURCE_DIR, else the copy vendored in the wheel, else the repo root.

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


VENDOR_DIR = Path(__file__).resolve().parent.parent / "_vendor"
"""Where the wheel keeps the vendored `propaq-core` (sources and `Cargo.toml`)."""


def tree_dir() -> Path:
    """Directory of a propaq source tree (it holds `crates/core`).

    Search order: `PROPAQ_SOURCE_DIR`, the parents of this file, the working
    directory and its parents. Tools that write into the tree use this.
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
    raise BuildUnavailable("propaq-core sources not found; set PROPAQ_SOURCE_DIR")


def source_dir() -> Path:
    """Directory that holds the `propaq-core` sources for a user basis.

    Search order:

    1. `PROPAQ_SOURCE_DIR`, if it is set. This is an explicit override.
       (Spec 12.4 lists it after the vendored copy. It is first here so that
       a set variable is never ignored silently.)
    2. The copy in the package (`propaq/_vendor/propaq-core`). Wheels ship it.
    3. A source tree: the parents of this file, the working directory and
       its parents.

    Use `core_path(source)` to get the core crate directory.
    """
    if os.environ.get("PROPAQ_SOURCE_DIR"):
        return tree_dir()
    if (VENDOR_DIR / "propaq-core" / "Cargo.toml").is_file():
        return VENDOR_DIR
    return tree_dir()


def core_path(source: Path) -> Path:
    """The `propaq-core` crate directory under a `source_dir()` result."""
    vendored = source / "propaq-core"
    if (vendored / "Cargo.toml").is_file():
        return vendored
    return source / "crates" / "core"


def surrogate_path(source: Path) -> Path:
    """The `propaq-surrogate` crate directory under a `source_dir()` result.

    The vendored copy has no `builtin` feature and no Pauli or Majorana classes.
    """
    vendored = source / "propaq-surrogate"
    if (vendored / "Cargo.toml").is_file():
        return vendored
    return source / "crates" / "surrogate"


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


def module_name(ir: BasisIR, surrogate: bool = False) -> str:
    """Python module name. A build with surrogate classes has its own name and cache entry."""
    return f"propaq_basis_{ir.hash16()}{'s' if surrogate else ''}"


def crate_files(ir: BasisIR, source: Path, with_module: bool, surrogate: bool = False) -> dict:
    """The files of the generated crate: relative path -> text.

    With `surrogate`, the crate also depends on `propaq-surrogate` and its
    module holds `SurrogatePropagator` and `SurrogateModel`.
    """
    h = ir.hash16() + ("s" if surrogate else "")
    ty = camel(ir.meta["name"])
    surrogate_dep = (
        f'propaq-surrogate = {{ path = "{surrogate_path(source).as_posix()}", default-features = false }}\n'
        if surrogate
        else ""
    )
    pyo3_features = '"num-complex", "extension-module"' if with_module else '"num-complex"'
    cargo = f"""[package]
name = "propaq-basis-{h}"
version = "0.0.0"
edition = "2021"

[lib]
name = "propaq_basis_{h}"
crate-type = ["cdylib", "rlib"]

[dependencies]
propaq-core = {{ path = "{core_path(source).as_posix()}" }}
{surrogate_dep}num-complex = "0.4"
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
        extra = "    extra: [register_surrogate],\n" if surrogate else ""
        local = "    local_dim: LOCAL_DIM,\n    local_table: &LOCAL_TABLE,\n" if "local" in ir.meta else ""
        if surrogate:
            lib += "\npropaq_surrogate::define_raw_surrogate! {\n    codec: Codec,\n    termsum: TermSum,\n}\n"
        lib += (
            f"\npropaq_core::define_raw_basis_module! {{\n    module: propaq_basis_{h},\n    algebra: {ty},\n"
            f"    bits_per_unit: {b},\n    coeff: {coeff},\n{local}{extra}}}\n"
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
    root = core_path(source)
    for p in sorted([root / "Cargo.toml", *root.joinpath("src").rglob("*.rs")]):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _tree_stamp(root: Path) -> str:
    """SHA-256 of the Rust sources of one crate."""
    import hashlib

    h = hashlib.sha256()
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


def build_basis(spec: BasisSpec, *, check_only=False, force=False, jobs=4, with_module=None, ir: BasisIR | None = None, surrogate: bool = False) -> BuildResult:
    """Compile `spec` to a Python extension in the cache.

    The cache key is the SHA-256 of the canonical IR. A cached .so is reused
    while the IR, the generated crate text and the core sources are the same
    (`build.stamp` holds the check value). `force=True` rebuilds.
    With `check_only`, run `cargo check` on a crate without the module macro.
    With `surrogate=True`, the module also holds `SurrogatePropagator` and
    `SurrogateModel`. It needs `propaq-surrogate` (the vendored copy or `crates/surrogate`), and it
    has its own cache entry and module name (`module_name(ir, True)`).
    """
    ir = ir or compile_spec(spec)
    source = source_dir()
    h = ir.hash16() + ("s" if surrogate else "")
    if surrogate and not (surrogate_path(source) / "Cargo.toml").is_file():
        raise BuildUnavailable("surrogate build needs propaq-surrogate (vendored copy or crates/surrogate)")
    crate_dir = cache_root() / h
    with_module = (not check_only) if with_module is None else with_module
    so = crate_dir / f"propaq_basis_{h}.so"
    files = crate_files(ir, source, with_module, surrogate)
    import hashlib

    extra_stamp = _tree_stamp(surrogate_path(source)) if surrogate else ""
    stamp = hashlib.sha256((core_stamp(source) + extra_stamp + files["src/lib.rs"] + files["Cargo.toml"]).encode()).hexdigest()
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
    if spec.intrinsic and not force_generic and spec.keep is None and spec.noise_key is None:
        return resolve_intrinsic(spec.intrinsic)
    ir = compile_spec(spec)
    try:
        res = build_basis(spec, ir=ir, **kwargs)
        return import_extension(res.so_path, module_name(ir, bool(kwargs.get("surrogate"))))
    except BuildUnavailable:
        return FallbackBasis(ir)
