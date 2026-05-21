"""Uniform noise model."""

from .base import NoiseModel
import numpy as np 

class UniformNoiseModel(NoiseModel):
    """Uniform noise model that applies a constant damping factor regardless of the number of active modes."""
    
    def __init__(self, damping: float):
        self.damping = damping

    def apply_noise(self, term_sum):
        """Apply uniform noise to the term sum by scaling all coefficients by the damping factor."""
        term_sum.scale(self.damping)

    def damping_factor(self, term_weight: float, active_modes: int) -> float:
        """Return the constant damping factor for any number of active modes."""
        return np.exp(-self.damping * term_weight)