"""Composable truncation operators shared by the numerical and surrogate propagators."""

from propaq._rust_core import CoefficientTruncator as _RustCoefficientTruncator
from propaq._rust_core import FlushSchedule as FlushSchedule
from propaq._rust_core import FrequencyTruncator as _RustFrequencyTruncator
from propaq._rust_core import TermBudget as _RustTermBudget
from propaq._rust_core import WeightTruncator as _RustWeightTruncator
from propaq.truncation.base import Truncator as Truncator


class FrequencyTruncator(_RustFrequencyTruncator):
    """Drop monomials whose frequency (trig-factor count) exceeds ``frequency``. Surrogate-only."""


class CoefficientTruncator(_RustCoefficientTruncator):
    """Drop contributions with coefficient magnitude below ``coefficient``."""


class WeightTruncator(_RustWeightTruncator):
    """Drop terms with operator weight exceeding ``weight``."""


class TermBudget(_RustTermBudget):
    """Term-count budget: ``max_terms`` triggers a flush; ``min_terms`` gates the lossy ops."""


# Register the Rust base classes so both the Python wrappers above (real
# subclasses) and the bare Rust instances returned by a propagator's
# ``truncators`` getter satisfy ``isinstance(_, Truncator)``.
for _base in (
    _RustFrequencyTruncator,
    _RustCoefficientTruncator,
    _RustWeightTruncator,
    _RustTermBudget,
):
    Truncator.register(_base)

__all__ = [
    "Truncator",
    "FrequencyTruncator",
    "CoefficientTruncator",
    "WeightTruncator",
    "TermBudget",
    "FlushSchedule",
]
