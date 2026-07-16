"""Composable truncation operators shared by the numerical and surrogate propagators."""

from propaq._rust_core import CoefficientTruncator as _RustCoefficientTruncator
from propaq._rust_core import FlushSchedule as FlushSchedule
from propaq._rust_core import FrequencyTruncator as _RustFrequencyTruncator
from propaq._rust_core import MonomialBudget as _RustMonomialBudget
from propaq._rust_core import NativeTruncator as _RustNativeTruncator
from propaq._rust_core import Simplify as _RustSimplify
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


class MonomialBudget(_RustMonomialBudget):
    """Monomial-count budget: ``max_monomials`` triggers a flush; ``min_monomials`` gates the
    lossy ops. Structurally identical to ``TermBudget``, keyed on monomial count. Surrogate-only.
    """


class Simplify(_RustSimplify):
    """Real (lossless) algebraic simplification, surrogate-only.

    At every flush, collapses monomials sharing the same canonical
    trig-factor run into one, summing their scalars.
    """


class NativeTruncator(_RustNativeTruncator):
    """Truncation policy backed by a dynamically loaded C, Rust, or
    AOT-compiled Julia shared library, called directly per term with no
    GIL and no Python call overhead. Numerical-only (the surrogate
    propagators reject it): see ``propaq.MD`` / ``examples/plugins/``
    for the ABI contract and example plugins.
    """


# Register the Rust base classes so both the Python wrappers above (real
# subclasses) and the bare Rust instances returned by a propagator's
# ``truncators`` getter satisfy ``isinstance(_, Truncator)``.
for _base in (
    _RustFrequencyTruncator,
    _RustCoefficientTruncator,
    _RustWeightTruncator,
    _RustTermBudget,
    _RustMonomialBudget,
    _RustSimplify,
    _RustNativeTruncator,
):
    Truncator.register(_base)

__all__ = [
    "Truncator",
    "FrequencyTruncator",
    "CoefficientTruncator",
    "WeightTruncator",
    "TermBudget",
    "MonomialBudget",
    "Simplify",
    "NativeTruncator",
    "FlushSchedule",
]
