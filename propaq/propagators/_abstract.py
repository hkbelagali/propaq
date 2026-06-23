from abc import ABC, abstractmethod


class AbstractPropagator(ABC):
    """Abstract propagator interface.  Concrete examples: MajoranaPropagator, PauliPropagator."""

    @abstractmethod
    def propagate(self, observable, circuit, filename=None):
        """Back-propagate *circuit* through *observable* in the Heisenberg picture.

        If *filename* is given, the final term sum is saved to a gzip-compressed
        binary file at that path.
        """
        pass

    @abstractmethod
    def expectation_value(self, observable, circuit, initial_state=0, filename=None):
        """Compute the expectation value of *observable* after evolving through *circuit*.

        If *filename* is given, the final term sum is saved to a gzip-compressed
        binary file at that path.
        """
        pass
