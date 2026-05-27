"""Truncation policy for noise models."""

from propaq.stubs import TruncationPolicy as _RustTruncationPolicy


class TruncationPolicy(_RustTruncationPolicy):
    def error_bound(self, noise_rate: float, circuit_depth: int) -> None:
        """Calculate the error bound for the truncation policy (stub)."""
        return None
