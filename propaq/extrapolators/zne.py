from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from scipy.optimize import curve_fit

from propaq.datatypes.abstract import AbstractTerm, AbstractTermSum, FockState
from propaq.noise import UniformNoiseModel
from propaq.propagators.abstract import AbstractPropagator, CircuitLike

if TYPE_CHECKING:
    from propaq.noise import GateNoiseModel, NativeNoiseModel

TermT = TypeVar("TermT", bound=AbstractTerm)
RotationT = TypeVar("RotationT")


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

    def build_noise(self, value: float) -> UniformNoiseModel | GateNoiseModel | NativeNoiseModel:
        """Build a fresh noise model carrying the given sweep value.

        Defaults to `UniformNoiseModel(value)`. Override to sweep a different
        single-parameter noise model, e.g. one parameter of a custom
        `GateNoiseModel` subclass with the rest held fixed.
        """
        return UniformNoiseModel(value)

    def run(
        self,
        propagator: AbstractPropagator[TermT, RotationT],
        observable: AbstractTermSum[TermT],
        circuit: CircuitLike[RotationT],
        initial_state: FockState = 0,
        **curve_fit_kwargs,
    ) -> ZNEResult:
        """Sweep noise levels, fit, and extrapolate to zero noise.

        Works with any `AbstractPropagator`

        Arguments:
            propagator: The propagator to sweep noise on.
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
                propagator.set_noise(self.build_noise(val))
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
