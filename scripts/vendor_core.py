"""Copy the sources of `crates/core` and `crates/surrogate` into `propaq/_vendor/`.

The wheel ships these copies. `propaq.tracer.build` compiles user bases
against them, so a user needs only cargo. Run this script after every change
in `crates/core` or `crates/surrogate`. The test `tests/io/test_vendor.py`
fails if a copy is stale.

* `propaq/_vendor/propaq-core/`: `Cargo.toml` and `src/` of `crates/core`, as is.
* `propaq/_vendor/propaq-surrogate/`: `src/` of `crates/surrogate` without
  `builtin.rs` (the Pauli and Majorana classes), and a reduced `Cargo.toml`.
  The reduced manifest has no `builtin` feature, no Pauli or Majorana
  dependency and no dev-dependency. Its core path is `../propaq-core`.
* `propaq/_vendor/Cargo.lock`: the workspace lock file.

Usage: python scripts/vendor_core.py [--check]
"""

from __future__ import annotations

import filecmp
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "propaq" / "_vendor"
LOCK_DST = VENDOR / "Cargo.lock"
# (name in the vendor dir, source crate dir, source files to leave out)
CRATES = {
    "propaq-core": (ROOT / "crates" / "core", set()),
    "propaq-surrogate": (ROOT / "crates" / "surrogate", {"src/builtin.rs"}),
}


def surrogate_manifest(text: str) -> str:
    """The reduced `Cargo.toml` of the vendored surrogate crate."""
    out = []
    skip = False
    for line in text.splitlines():
        if line.startswith("["):
            skip = line.strip() in ("[features]", "[dev-dependencies]")
        if skip:
            continue
        if line.startswith("propaq-pauli") or line.startswith("propaq-majorana"):
            continue
        if line.startswith("propaq-core"):
            line = 'propaq-core = { path = "../propaq-core" }'
        out.append(line)
    out.append("")
    out.append("[lints.rust]")
    out.append('unexpected_cfgs = { level = "allow" }')
    return "\n".join(out).replace("\n\n\n", "\n\n") + "\n"


def expected(name: str) -> dict[str, bytes]:
    """Relative path -> content that the copy of crate `name` must hold."""
    src, skip = CRATES[name]
    out: dict[str, bytes] = {}
    manifest = (src / "Cargo.toml").read_text()
    if name == "propaq-surrogate":
        manifest = surrogate_manifest(manifest)
    out["Cargo.toml"] = manifest.encode()
    for p in sorted((src / "src").rglob("*")):
        rel = p.relative_to(src).as_posix()
        if p.is_file() and rel not in skip:
            out[rel] = p.read_bytes()
    return out


def stale() -> list[str]:
    """Paths where a copy differs from its source (prefixed by the crate name)."""
    bad = []
    for name in CRATES:
        want = expected(name)
        dst = VENDOR / name
        for rel, data in want.items():
            d = dst / rel
            if not d.is_file() or d.read_bytes() != data:
                bad.append(f"{name}/{rel}")
        if dst.is_dir():
            have = {p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file()}
            bad += [f"{name}/{r}" for r in sorted(have - set(want))]
    lock = ROOT / "Cargo.lock"
    if lock.is_file() and (not LOCK_DST.is_file() or not filecmp.cmp(lock, LOCK_DST, shallow=False)):
        bad.append("Cargo.lock")
    return bad


def main(argv: list[str]) -> int:
    if "--check" in argv:
        bad = stale()
        for b in bad:
            print("stale:", b)
        return 1 if bad else 0
    for name in CRATES:
        for rel, data in expected(name).items():
            d = VENDOR / name / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            d.write_bytes(data)
    if (ROOT / "Cargo.lock").is_file():
        shutil.copyfile(ROOT / "Cargo.lock", LOCK_DST)
    # Files that exist only in a copy are reported. A person removes them.
    for b in stale():
        print("needs manual removal or review:", b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
