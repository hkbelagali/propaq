""" 
Zero cutoff extrapolation via curve fitting. 

Similar to zero-noise extrapolation, but instead of sweeping over 
noise levels, we sweep over cutoff values in the propagator. 
This could be either a weight cutoff or a coefficient cutoff,
which both inherit from the same base class. 
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


@dataclass
class ZCEResult:
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
        self.fitting_fn = fitting_fn 
        self.cutoff_values = list(cutoff_values)

    @abstractmethod 
    def _get_cutoff(self, propagator: AbstractPropagator) -> float | int | None: 
        """Get the cutoff value from the propagator."""
        pass 

    @abstractmethod
    def _set_cutoff(self, propagator: AbstractPropagator, cutoff: float | int | None) -> None:
        """Set the cutoff value in the propagator.  A value of None restores the field to its unset state."""
        pass

    def run(self, propagator: AbstractPropagator, observable: AbstractTermSum, circuit: MajoranaCircuit | PauliCircuit, initial_state: int = 0, **curve_fit_kwargs) -> ZCEResult:
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
        # Save the scalar value, not the object reference — _set_cutoff mutates
        # the TruncationPolicy in place, so a reference would capture the mutated state.
        original_cutoff = self._get_cutoff(propagator)

        try:
            expectation_values = []
            for cutoff in self.cutoff_values:
                self._set_cutoff(propagator, cutoff)
                result = propagator.expectation_value(observable, circuit, initial_state=initial_state)
                expectation_values.append(result.expectation_value)
        finally:
            if propagator.truncation is not None:
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
    """
    Zero weight cutoff extrapolation via curve fitting.
    """

    fitting_fn: Callable
    """Function to fit to the weight cutoff vs. expectation value data, passed to scipy.optimize.curve_fit."""

    cutoff_values: list[float]
    """Weight cutoff values to sweep over for the extrapolation."""

    def __init__(self, fitting_fn: Callable, cutoff_values: list[float]) -> None:
        super().__init__(fitting_fn, cutoff_values)

    def _get_cutoff(self, propagator: AbstractPropagator) -> int | None:
        """Get the weight cutoff value from the propagator."""
        t = propagator.truncation
        return t.weight_cutoff if t is not None else None
    
    def _set_cutoff(self, propagator: AbstractPropagator, cutoff: int | None) -> None:
        """Set the weight cutoff value in the propagator.  None removes the weight cutoff."""
        t = propagator.truncation
        if t is None:
            raise ValueError("Propagator has no truncation policy set.")
        t.weight_cutoff = cutoff

class CoefficientCutoffExtrapolator(ZeroCutoffExtrapolator):
    """
    Zero coefficient cutoff extrapolation via curve fitting.
    """

    fitting_fn: Callable
    """Function to fit to the coefficient cutoff vs. expectation value data, passed to scipy.optimize.curve_fit."""

    cutoff_values: list[float]
    """Coefficient cutoff values to sweep over for the extrapolation."""

    def __init__(self, fitting_fn: Callable, cutoff_values: list[float]) -> None:
        super().__init__(fitting_fn, cutoff_values)

    def _get_cutoff(self, propagator: AbstractPropagator) -> float | None:
        """Get the coefficient cutoff value from the propagator."""
        t = propagator.truncation
        return t.coeff_cutoff if t is not None else None
    
    def _set_cutoff(self, propagator: AbstractPropagator, cutoff: float) -> None:
        """Set the coefficient cutoff value in the propagator."""
        t = propagator.truncation
        if t is None:
            raise ValueError("Propagator has no truncation policy set.")
        t.coeff_cutoff = cutoff