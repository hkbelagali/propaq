"""Gate-based noise model."""

from .base import NoiseModel


class GateNoiseModel(NoiseModel):
    """Delegates noise application and damping to an inner noise model."""

    def __init__(self, inner: NoiseModel):
        self.inner = inner

    def apply_noise(self, term_sum):
        self.inner.apply_noise(term_sum)

    def damping_factor(self, term_weight: float, active_modes: int) -> float:
        return self.inner.damping_factor(term_weight, active_modes)
