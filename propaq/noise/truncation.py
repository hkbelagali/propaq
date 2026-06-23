"""Truncation policy for noise models."""

from propaq._rust_core import TruncationPolicy as _RustTruncationPolicy


class TruncationPolicy(_RustTruncationPolicy):
    """
    Controls when and how terms are discarded during propagation.
    """
    pass