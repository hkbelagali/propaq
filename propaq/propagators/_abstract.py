from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from propaq._rust_core import (
        GateNoiseModel,
        NativeNoiseModel,
        PropagationResult,
        TruncationPolicy,
        UniformNoiseModel,
    )


class AbstractPropagator(ABC):
    """Abstract propagator interface.  Concrete examples: MajoranaPropagator, PauliPropagator."""

    @property
    @abstractmethod
    def noise(self) -> UniformNoiseModel | GateNoiseModel | NativeNoiseModel | None:
        """The current noise model, or None."""

    @abstractmethod
    def set_noise(
        self, noise: UniformNoiseModel | GateNoiseModel | NativeNoiseModel | None = None
    ) -> None:
        """Replace the noise model."""

    @property
    @abstractmethod
    def truncators(self) -> list[object]:
        """The current truncation pipeline."""

    @abstractmethod
    def set_truncation(
        self,
        truncation: object | Sequence[object] | TruncationPolicy | None = None,
    ) -> None:
        """Replace the truncation pipeline."""

    @abstractmethod
    def propagate(self, observable, circuit, filename=None):
        """Back-propagate *circuit* through *observable* in the Heisenberg picture.

        If *filename* is given, the final term sum is saved to a gzip-compressed
        binary file at that path.
        """

    @abstractmethod
    def expectation_value(
        self, observable, circuit, initial_state: int = 0, filename=None
    ) -> PropagationResult:
        """Compute the expectation value of *observable* after evolving through *circuit*.

        If *filename* is given, the final term sum is saved to a gzip-compressed
        binary file at that path.
        """
