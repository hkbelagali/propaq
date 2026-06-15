from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from scipy.optimize import curve_fit

from propaq.noise import UniformNoiseModel


@dataclass
class ZNEResult:
    zero_noise_value: float
    noise_values: List[float]
    expectation_values: List[float]
    fit_params: np.ndarray
    fit_covariance: np.ndarray


class ZeroNoiseExtrapolator:
    """Zero-noise extrapolation via curve fitting.

    Args:
        fitting_fn: Model function f(x, *params) passed to scipy.optimize.curve_fit.
        noise_values: Noise (damping) values to evaluate the propagator at.
    """

    def __init__(self, fitting_fn: Callable, noise_values: List[float]) -> None:
        self.fitting_fn = fitting_fn
        self.noise_values = list(noise_values)

    def run(
        self,
        propagator,
        observable,
        circuit,
        fock_state: int = 0,
        **curve_fit_kwargs,
    ) -> ZNEResult:
        """Sweep noise levels, fit, and extrapolate to zero noise.

        Args:
            propagator: A MajoranaPropagator or PauliPropagator instance.
            observable: The observable to measure.
            circuit: The circuit to propagate.
            fock_state: Initial state index (default 0).
            **curve_fit_kwargs: Forwarded to scipy.optimize.curve_fit (e.g. p0=).
        """
        original_noise = propagator.noise
        try:
            expectation_values = []
            for val in self.noise_values:
                propagator.set_noise(UniformNoiseModel(val))
                result = propagator.expectation_value(observable, circuit, fock_state=fock_state)
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
