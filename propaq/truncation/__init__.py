"""Composable truncation operators shared by the numerical and surrogate propagators."""

from propaq._rust_core import CoefficientTruncator as _RustCoefficientTruncator
from propaq._rust_core import FrequencyTruncationPolicy as FrequencyTruncationPolicy
from propaq._rust_core import FrequencyTruncator as _RustFrequencyTruncator
from propaq._rust_core import NativeTruncator as _RustNativeTruncator
from propaq._rust_core import Simplify as _RustSimplify
from propaq._rust_core import TermBudget as _RustTermBudget
from propaq._rust_core import TruncationPolicy as TruncationPolicy
from propaq._rust_core import WeightTruncator as _RustWeightTruncator
from propaq.truncation._apply import ResolvedTruncation as ResolvedTruncation
from propaq.truncation._apply import resolve_truncation as resolve_truncation
from propaq.truncation.base import Truncator as Truncator


class _RustBackedMeta(type):
    """Metaclass making ``isinstance()`` also match the bare Rust instance"""

    def __instancecheck__(cls, instance: object) -> bool:
        return type.__instancecheck__(cls, instance) or isinstance(instance, cls._rust_base)  # type: ignore[attr-defined]


class FrequencyTruncator(_RustFrequencyTruncator, metaclass=_RustBackedMeta):
    """Drop monomials whose frequency (trig-factor count) exceeds ``frequency``. Surrogate-only."""

    _rust_base = _RustFrequencyTruncator


class CoefficientTruncator(_RustCoefficientTruncator, metaclass=_RustBackedMeta):
    """Drop contributions with coefficient magnitude below ``coefficient``."""

    _rust_base = _RustCoefficientTruncator


class WeightTruncator(_RustWeightTruncator, metaclass=_RustBackedMeta):
    """Drop whole Pauli/Majorana terms whose operator weight exceeds ``weight``.
    Applies to both propagators.

    A term with weight ``w`` is exponentially unlikely in ``w`` to contribute to
    the final state, which is why this is a useful truncation criterion for
    larger circuits.
    """

    _rust_base = _RustWeightTruncator


class TermBudget(_RustTermBudget, metaclass=_RustBackedMeta):
    """Live-term floor: below ``min_terms`` terms, all other lossy truncators
    (``WeightTruncator``/``CoefficientTruncator``/``NativeTruncator``) are
    suppressed, keeping propagation exact until the operator has had room to
    grow. ``None`` disables the floor.
    """

    _rust_base = _RustTermBudget


class Simplify(_RustSimplify, metaclass=_RustBackedMeta):
    """Real (lossless) algebraic simplification, surrogate-only.

    At every flush, collapses monomials sharing the same canonical
    trig-factor run into one, summing their scalars.
    """

    _rust_base = _RustSimplify


class NativeTruncator(_RustNativeTruncator, metaclass=_RustBackedMeta):
    """Truncation policy backed by a dynamically loaded C, Rust, or
    AOT-compiled Julia shared library
    """

    _rust_base = _RustNativeTruncator


# Register the Rust base classes so both the Python wrappers above
# and the bare Rust instances returned by a propagator's
# ``truncators`` getter satisfy ``isinstance(_, Truncator)``.
for _base in (
    _RustFrequencyTruncator,
    _RustCoefficientTruncator,
    _RustWeightTruncator,
    _RustTermBudget,
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
    "Simplify",
    "NativeTruncator",
    "TruncationPolicy",
    "FrequencyTruncationPolicy",
    "resolve_truncation",
    "ResolvedTruncation",
]
