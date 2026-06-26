from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from propaq._rust_core import (
        GateNoiseModel,
        PropagationResult,
        TruncationPolicy,
        UniformNoiseModel,
    )


class AbstractPropagator(ABC):
    """Abstract propagator interface.  Concrete examples: MajoranaPropagator, PauliPropagator."""

    @property
    @abstractmethod
    def noise(self) -> UniformNoiseModel | GateNoiseModel | None:
        """The current noise model, or None."""

    @abstractmethod
    def set_noise(self, noise: UniformNoiseModel | GateNoiseModel | None = None) -> None:
        """Replace the noise model."""

    @property
    @abstractmethod
    def truncation(self) -> TruncationPolicy | None:
        """The current truncation policy, or None."""

    @abstractmethod
    def set_truncation(self, truncation: TruncationPolicy | None = None) -> None:
        """Replace the truncation policy."""

    @abstractmethod
    def propagate(self, observable, circuit, filename=None):
        """Back-propagate *circuit* through *observable* in the Heisenberg picture.

        If *filename* is given, the final term sum is saved to a gzip-compressed
        binary file at that path.
        """

    @abstractmethod
    def expectation_value(self, observable, circuit, initial_state: int = 0, filename=None) -> PropagationResult:
        """Compute the expectation value of *observable* after evolving through *circuit*.

        If *filename* is given, the final term sum is saved to a gzip-compressed
        binary file at that path.
        """
