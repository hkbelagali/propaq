"""Interpret a truncation pipeline generically, for a Python-defined propagator."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypeAlias, cast

from propaq._rust_core import CoefficientTruncator as _RustCoefficientTruncator
from propaq._rust_core import FrequencyTruncator as _RustFrequencyTruncator
from propaq._rust_core import NativeTruncator as _RustNativeTruncator
from propaq._rust_core import Simplify as _RustSimplify
from propaq._rust_core import TermBudget as _RustTermBudget
from propaq._rust_core import TruncationPolicy
from propaq._rust_core import WeightTruncator as _RustWeightTruncator

if TYPE_CHECKING:
    from collections.abc import Sequence

_SURROGATE_ONLY = (_RustFrequencyTruncator, _RustSimplify)

_NUMERICAL_ONLY = (_RustNativeTruncator,)

_SUPPORTED = (_RustWeightTruncator, _RustCoefficientTruncator, _RustTermBudget)

_Truncator: TypeAlias = "_RustWeightTruncator | _RustCoefficientTruncator | _RustTermBudget"


def resolve_truncation(
    truncation: object | Sequence[object] | TruncationPolicy | None = None,
) -> list[_Truncator]:
    """Normalize the flexible ``truncation`` argument into a truncator list.

    Arguments:
        truncation: The pipeline, in any of the accepted forms.

    Returns:
        The truncators, in application order.

    Raises:
        TypeError: If ``truncation`` (or one of its elements) is not one of the
            accepted forms, or is a surrogate-only (`FrequencyTruncator`/
            `Simplify`) or engine-only (`NativeTruncator`) truncator
    """
    if truncation is None:
        ops: list[object] = []
    elif isinstance(truncation, TruncationPolicy):
        ops = []
        if truncation.weight_cutoff is not None:
            ops.append(_RustWeightTruncator(weight=truncation.weight_cutoff))
        if truncation.coeff_cutoff > 0.0:
            ops.append(_RustCoefficientTruncator(coefficient=truncation.coeff_cutoff))
        if truncation.min_terms is not None:
            ops.append(_RustTermBudget(min_terms=truncation.min_terms))
    elif isinstance(truncation, list | tuple):
        ops = list(truncation)
    else:
        ops = [truncation]

    for op in ops:
        if isinstance(op, _SURROGATE_ONLY):
            raise TypeError(
                f"{type(op).__name__} only applies to surrogate propagation. Use "
                "WeightTruncator / CoefficientTruncator / TermBudget with a "
                "numerical propagator"
            )
        if isinstance(op, _NUMERICAL_ONLY):
            raise TypeError(
                "NativeTruncator is applied by the Rust engine, which a "
                "pure-Python propagator (AbstractPropagator) does not run. Use "
                "WeightTruncator / CoefficientTruncator / TermBudget instead"
            )
        if not isinstance(op, _SUPPORTED):
            raise TypeError(
                "truncation must be a truncator (WeightTruncator/CoefficientTruncator/"
                "TermBudget/FrequencyTruncator/Simplify/NativeTruncator), a sequence of "
                f"truncators, a TruncationPolicy, or None, got {type(op).__name__}"
            )

    return cast("list[_Truncator]", ops)


@dataclass(frozen=True, slots=True)
class ResolvedTruncation:
    """
    A truncator pipeline collapsed to the values the emit gate compares against.
    """

    weight_cutoff: int | None = None
    coeff_cutoff: float | None = None
    min_terms: int | None = None

    @classmethod
    def from_truncators(cls, truncators: Sequence[_Truncator]) -> ResolvedTruncation:
        """Collapse a truncator list into one `ResolvedTruncation`, last-wins per type.

        Arguments:
            truncators: The truncators, as returned by `resolve_truncation`.
        """
        weight_cutoff = coeff_cutoff = min_terms = None
        for op in truncators:
            if isinstance(op, _RustWeightTruncator):
                weight_cutoff = op.weight
            elif isinstance(op, _RustCoefficientTruncator):
                coeff_cutoff = op.coefficient
            elif isinstance(op, _RustTermBudget):
                min_terms = op.min_terms
        return cls(weight_cutoff=weight_cutoff, coeff_cutoff=coeff_cutoff, min_terms=min_terms)

    def at_size(self, n_live: int) -> ResolvedTruncation:
        """The cutoff to use when emitting children of a term sum with `n_live` live terms.

        Arguments:
            n_live: The number of terms live before this emission.

        Returns:
            ``self`` unchanged, unless `min_terms` is set and ``n_live`` is
            below it, in which case `weight_cutoff` and `coeff_cutoff` are
            suppressed (returned as `None`) and `min_terms` is kept.
        """
        if self.min_terms is not None and n_live < self.min_terms:
            return replace(self, weight_cutoff=None, coeff_cutoff=None)
        return self

    def admits(self, weight: int, coeff: complex) -> bool:
        """True if a term of this weight and coefficient belongs in the store.

        Arguments:
            weight: The term's weight.
            coeff: The term's coefficient.
        """
        if self.weight_cutoff is not None and weight > self.weight_cutoff:
            return False
        return not (self.coeff_cutoff is not None and abs(coeff) < self.coeff_cutoff)
