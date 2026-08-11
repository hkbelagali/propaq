from abc import ABC, abstractmethod


class NoiseModel(ABC):
    """
    Abstract base class for noise models. Specific noise models should inherit from this class and implement the necessary methods.
    """

    @abstractmethod
    def apply_noise(self, term_sum):
        """Apply noise to the given term sum based on the active modes."""
        pass

    @abstractmethod
    def damping_factor(self, term_weight: float, active_modes: int) -> float:
        """Calculate the damping factor based on the number of active modes."""
        pass
