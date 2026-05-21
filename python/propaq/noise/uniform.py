"""Uniform noise model."""

from .base import NoiseModel
import numpy as np 

class UniformNoiseModel(NoiseModel):
    """Uniform noise model that applies a constant damping factor regardless of the number of active modes."""
    
    def __init__(self, damping: float):
        self.damping = damping

    def apply_noise(self, term_sum):
        """Apply uniform noise to the term sum by scaling all coefficients by the damping factor."""
        for term, coeff in term_sum.items(): 
            weight = term.weight
            damping_factor = self.damping_factor(weight, active_modes=0)
            term_sum[term] = coeff * damping_factor

    def damping_factor(self, term_weight: float, active_modes: int) -> float:
        """Return the constant damping factor for any number of active modes."""
        return np.exp(-self.damping * term_weight)