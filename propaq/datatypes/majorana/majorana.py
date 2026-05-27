"""Majorana monomial datatype for Majorana Propagation."""

try:
    from propaq.stubs import MajoranaMonomial as MajoranaMonomial
except ImportError as exc:
    raise ImportError(
        "MajoranaMonomial requires the compiled Rust extension (_rust_core). "
        "Run `maturin develop` (or install a binary wheel) to build it."
    ) from exc


def _hermiticity_exp(length: int) -> int:
    """Power of i needed to make the Majorana monomial with the given length Hermitian."""
    return 0 if length % 4 in (0, 1) else 1


def _resorting_parity(a: int, b: int) -> int:
    """Fermionic sign from resorting b's creation operators past a's."""
    count = 0
    remaining = b
    while remaining:
        lowest_bit = remaining & (-remaining)
        pos = lowest_bit.bit_length() - 1
        count += (a >> (pos + 1)).bit_count()
        remaining ^= lowest_bit
    return count & 1
