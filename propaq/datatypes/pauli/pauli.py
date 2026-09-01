"""Pauli monomial datatype for Pauli propagation."""

try:
    from propaq._rust_core import PauliString as PauliString
except ImportError as exc:
    raise ImportError(
        "PauliString requires the compiled Rust extension (_rust_core). "
        "Run `maturin develop` (or install a binary wheel) to build it."
    ) from exc
