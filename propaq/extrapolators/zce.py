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

import numpy as np
from scipy.optimize import curve_fit

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
    def truncator_cls(self) -> type[object]:
        """The truncator class this extrapolator sweeps, used to find its slot
        in the propagator's truncation pipeline."""
        ...

    @abstractmethod
    def build_truncator(self, cutoff: float | int | None) -> object:
        """Build a fresh truncator carrying the given cutoff (None = no cutoff)."""
        ...

    def _set_cutoff(self, propagator: AbstractPropagator, cutoff: float | int | None) -> None:
        """Replace the matching truncator's cutoff value in place.

        Raises ValueError if the propagator has no truncation policy for this
        cutoff (no matching truncator in its pipeline).
        """
        truncators = list(propagator.truncators)
        idx = next(
            (i for i, t in enumerate(truncators) if isinstance(t, self.truncator_cls())), None
        )
        if idx is None:
            raise ValueError("Propagator has no truncation policy for this cutoff.")
        truncators[idx] = self.build_truncator(cutoff)
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
        # Truncators are immutable value objects, so the whole pipeline can be
        # snapshotted up front and restored wholesale afterwards
        original_truncators = list(propagator.truncators)

        try:
            expectation_values = []
            for cutoff in self.cutoff_values:
                self._set_cutoff(propagator, cutoff)
                result = propagator.expectation_value(
                    observable, circuit, initial_state=initial_state
                )
                expectation_values.append(result.expectation_value)
        finally:
            propagator.set_truncation(original_truncators)

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

    def truncator_cls(self) -> type[object]:
        """The truncator class this extrapolator sweeps: `WeightTruncator`."""
        return WeightTruncator

    def build_truncator(self, cutoff: float | int | None) -> WeightTruncator:
        """Build a fresh `WeightTruncator` carrying the given weight cutoff."""
        return WeightTruncator(int(cutoff) if cutoff is not None else None)


class CoefficientCutoffExtrapolator(ZeroCutoffExtrapolator):
    """Zero coefficient-cutoff extrapolation via curve fitting."""

    def truncator_cls(self) -> type[object]:
        """The truncator class this extrapolator sweeps: `CoefficientTruncator`."""
        return CoefficientTruncator

    def build_truncator(self, cutoff: float | int | None) -> CoefficientTruncator:
        """Build a fresh `CoefficientTruncator` carrying the given coefficient cutoff."""
        return CoefficientTruncator(float(cutoff) if cutoff is not None else None)
