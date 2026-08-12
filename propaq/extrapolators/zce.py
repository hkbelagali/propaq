"""
Zero cutoff extrapolation via curve fitting.

Similar to zero-noise extrapolation, but instead of sweeping over
noise levels, we sweep over cutoff values in the propagator.
This could be either a weight cutoff or a coefficient cutoff,
both of which are individual truncators in the propagator's pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.optimize import curve_fit

# The propagator's ``truncators`` getter returns bare Rust instances, so match
# against the Rust base classes (the Python wrappers subclass them).
from propaq._rust_core import CoefficientTruncator as _RustCoefficientTruncator
from propaq._rust_core import WeightTruncator as _RustWeightTruncator
from propaq.circuits.majorana.circuit import MajoranaCircuit
from propaq.circuits.pauli.circuit import PauliCircuit
from propaq.datatypes import AbstractTermSum
from propaq.propagators import AbstractPropagator
from propaq.truncation import CoefficientTruncator, WeightTruncator


@dataclass
class ZCEResult:
    """Result of a zero-cutoff extrapolation run, including the fitted parameters and covariance."""

    zero_cutoff_value: float
    """Extrapolated expectation value at zero cutoff."""

    cutoff_values: list[float]
    """Cutoff values used in the extrapolation."""
    expectation_values: list[float]
    """Expected values at each cutoff value."""
    fit_params: np.ndarray
    """Fitted parameters."""
    fit_covariance: np.ndarray
    """Covariance matrix of the fitted parameters."""


class ZeroCutoffExtrapolator(ABC):
    """
    Zero cutoff extrapolation via curve fitting.
    """

    fitting_fn: Callable
    """Function to fit to the cutoff vs. expectation value data, passed to scipy.optimize.curve_fit."""

    cutoff_values: list[float]
    """Cutoff values to sweep over for the extrapolation."""

    def __init__(self, fitting_fn: Callable, cutoff_values: list[float]) -> None:
        """Construct a ZeroCutoffExtrapolator with a fitting function and cutoff values."""
        self.fitting_fn = fitting_fn
        self.cutoff_values = list(cutoff_values)

    @abstractmethod
    def _rust_cls(self) -> type[object]:
        """The Rust truncator base class this extrapolator sweeps."""
        ...

    @abstractmethod
    def _read(self, truncator: object) -> float | int | None:
        """Read the cutoff value out of a matching truncator."""
        ...

    @abstractmethod
    def _build(self, cutoff: float | int | None) -> object:
        """Build a fresh truncator carrying the given cutoff (None = no cutoff)."""
        ...

    def _find(self, propagator: AbstractPropagator) -> object | None:
        """The first matching truncator in the pipeline, or None."""
        for t in propagator.truncators:
            if isinstance(t, self._rust_cls()):
                return t
        return None

    def _get_cutoff(self, propagator: AbstractPropagator) -> float | int | None:
        """The current cutoff value, or None if no matching truncator is present."""
        t = self._find(propagator)
        return self._read(t) if t is not None else None

    def _set_cutoff(self, propagator: AbstractPropagator, cutoff: float | int | None) -> None:
        """Replace the matching truncator's cutoff value in place.

        Raises ValueError if the propagator has no truncation policy for this
        cutoff (no matching truncator in its pipeline).
        """
        truncators = list(propagator.truncators)
        idx = next((i for i, t in enumerate(truncators) if isinstance(t, self._rust_cls())), None)
        if idx is None:
            raise ValueError("Propagator has no truncation policy for this cutoff.")
        truncators[idx] = self._build(cutoff)
        propagator.set_truncation(truncators)

    def run(
        self,
        propagator: AbstractPropagator,
        observable: AbstractTermSum,
        circuit: MajoranaCircuit | PauliCircuit,
        initial_state: int = 0,
        **curve_fit_kwargs,
    ) -> ZCEResult:
        """Sweep cutoff values, fit, and extrapolate to zero cutoff.

        Arguments:
            propagator: A MajoranaPropagator or PauliPropagator instance.
            observable: The observable to measure.
            circuit: The circuit to propagate.
            initial_state: Initial state index (default 0).
            **curve_fit_kwargs: Forwarded to scipy.optimize.curve_fit (e.g. p0=).

        Returns:
            A ZCEResult containing the extrapolated zero-cutoff value and fit details.
        """
        # Save the scalar cutoff, not a truncator reference,  _set_cutoff rebuilds
        # the pipeline, so a reference would go stale.
        original_cutoff = self._get_cutoff(propagator)

        try:
            expectation_values = []
            for cutoff in self.cutoff_values:
                self._set_cutoff(propagator, cutoff)
                result = propagator.expectation_value(
                    observable, circuit, initial_state=initial_state
                )
                expectation_values.append(result.expectation_value)
        finally:
            # Restore only if a matching truncator is still present (nothing to
            # restore into otherwise, e.g. the first _set_cutoff already raised).
            if self._find(propagator) is not None:
                self._set_cutoff(propagator, original_cutoff)

        popt, pcov = curve_fit(
            self.fitting_fn, self.cutoff_values, expectation_values, **curve_fit_kwargs
        )
        zero_cutoff_value = float(self.fitting_fn(0.0, *popt))

        return ZCEResult(
            zero_cutoff_value=zero_cutoff_value,
            cutoff_values=self.cutoff_values,
            expectation_values=expectation_values,
            fit_params=popt,
            fit_covariance=pcov,
        )


class WeightCutoffExtrapolator(ZeroCutoffExtrapolator):
    """Zero weight-cutoff extrapolation via curve fitting."""

    def _rust_cls(self) -> type[object]:
        return _RustWeightTruncator

    def _read(self, truncator: object) -> int | None:
        return cast(int | None, getattr(truncator, "weight"))

    def _build(self, cutoff: float | int | None) -> WeightTruncator:
        return WeightTruncator(int(cutoff) if cutoff is not None else None)


class CoefficientCutoffExtrapolator(ZeroCutoffExtrapolator):
    """Zero coefficient-cutoff extrapolation via curve fitting."""

    def _rust_cls(self) -> type[object]:
        return _RustCoefficientTruncator

    def _read(self, truncator: object) -> float | None:
        return cast(float | None, getattr(truncator, "coefficient"))

    def _build(self, cutoff: float | int | None) -> CoefficientTruncator:
        return CoefficientTruncator(float(cutoff) if cutoff is not None else None)
