"""Noiseless model."""

from .base import NoiseModel
import numpy as np 

class NoiselessModel(NoiseModel):
    """Noiseless model that applies no noise."""

    def __init__(self):
        pass

    def apply_noise(self, term_sum):
        """Apply no noise to the term sum."""
        pass

    def damping_factor(self, term_weight: float, active_modes: int) -> float:
        """Return the constant damping factor for any number of active modes."""
        return 1.0