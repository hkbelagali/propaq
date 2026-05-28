from abc import ABC, abstractmethod


class AbstractPropagator(ABC):
    """Abstract propagator interface.  Concrete examples: MajoranaPropagator, PauliPropagator."""

    @abstractmethod
    def propagate(self, observable, circuit):
        """Back-propagate *circuit* through *observable* in the Heisenberg picture."""
        pass

    @abstractmethod
    def expectation_value(self, observable, circuit, initial_state=0):
        """Compute the expectation value of *observable* after evolving through *circuit*."""
        pass
