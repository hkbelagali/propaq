"""Gate-based noise model."""

from .base import NoiseModel

class GateNoiseModel(NoiseModel):
    """Gate-based noise model that applies noise after each gate operation."""
    
    def __init__(self, gate_noise: NoiseModel):
        self.gate_noise = gate_noise

    def apply_noise(self, term_sum):
        """Apply gate noise to the term sum by invoking the noise model's apply_noise method."""
        self.gate_noise.apply_noise(term_sum)

    def damping_factor(self, term_weight: float, active_modes: int) -> float:
        """Calculate the damping factor based on the gate noise model."""
        return self.gate_noise.damping_factor(term_weight, active_modes)
