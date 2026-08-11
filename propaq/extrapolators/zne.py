from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from propaq.circuits.majorana.circuit import MajoranaCircuit
from propaq.circuits.pauli.circuit import PauliCircuit
from propaq.datatypes._abstract import AbstractTermSum
from propaq.noise import UniformNoiseModel
from propaq.propagators._abstract import AbstractPropagator


@dataclass
class ZNEResult:
    """Result of a zero-noise extrapolation run, including the fitted parameters and covariance."""

    zero_noise_value: float
    """Extrapolated expectation value at zero noise."""

    noise_values: list[float]
    """Noise values used in the extrapolation."""
    expectation_values: list[float]
    """Expected values at each noise level."""
    fit_params: np.ndarray
    """Fitted parameters."""
    fit_covariance: np.ndarray
    """Covariance matrix of the fitted parameters."""


class ZeroNoiseExtrapolator:
    """
    Zero-noise extrapolation via curve fitting.
    """

    fitting_fn: Callable
    """Function to fit to the noise vs. expectation value data, passed to scipy.optimize.curve_fit."""

    noise_values: list[float]
    """Noise values to sweep over for the extrapolation."""

    def __init__(self, fitting_fn: Callable, noise_values: list[float]) -> None:
        """Construct a ZeroNoiseExtrapolator with a fitting function and noise values."""
        self.fitting_fn = fitting_fn
        self.noise_values = list(noise_values)

    def run(
        self,
        propagator: AbstractPropagator,
        observable: AbstractTermSum,
        circuit: MajoranaCircuit | PauliCircuit,
        initial_state: int = 0,
        **curve_fit_kwargs,
    ) -> ZNEResult:
        """Sweep noise levels, fit, and extrapolate to zero noise.

        Arguments:
            propagator: A MajoranaPropagator or PauliPropagator instance.
            observable: The observable to measure.
            circuit: The circuit to propagate.
            initial_state: Initial state index (default 0).
            **curve_fit_kwargs: Forwarded to scipy.optimize.curve_fit (e.g. p0=).

        Returns:
            A ZNEResult containing the extrapolated zero-noise value and fit details.
        """
        original_noise = propagator.noise
        try:
            expectation_values = []
            for val in self.noise_values:
                propagator.set_noise(UniformNoiseModel(val))
                result = propagator.expectation_value(
                    observable, circuit, initial_state=initial_state
                )
                expectation_values.append(result.expectation_value)
        finally:
            propagator.set_noise(original_noise)

        popt, pcov = curve_fit(
            self.fitting_fn, self.noise_values, expectation_values, **curve_fit_kwargs
        )
        zero_noise_value = float(self.fitting_fn(0.0, *popt))

        return ZNEResult(
            zero_noise_value=zero_noise_value,
            noise_values=self.noise_values,
            expectation_values=expectation_values,
            fit_params=popt,
            fit_covariance=pcov,
        )
