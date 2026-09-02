from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from propaq._rust_core import PropagationResult
    from propaq.datatypes import AbstractTermSum
    from propaq.noise import GateNoiseModel, NativeNoiseModel, UniformNoiseModel
    from propaq.truncation import TruncationPolicy


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
    def propagate(
        self, observable: AbstractTermSum, circuit: Any, filename: str | None = None
    ) -> AbstractTermSum:
        """Back-propagate *circuit* through *observable* in the Heisenberg picture.

        If *filename* is given, the final term sum is saved to a gzip-compressed
        binary file at that path.

        Arguments:
            observable: The term sum to back-propagate (a `PauliTermSum` or
                `MajoranaTermSum`, matching the propagator's representation).
            circuit: The circuit to propagate through (a `PauliCircuit`/
                `MajoranaCircuit` for a numerical propagator, or a
                `SurrogatePauliCircuit`/`SurrogateMajoranaCircuit` for a surrogate
                propagator).
            filename: Optional path to save the evolved term sum to, gzip-compressed.

        Returns:
            The evolved term sum.
        """

    @abstractmethod
    def expectation_value(
        self,
        observable: AbstractTermSum,
        circuit: Any,
        initial_state: int = 0,
        filename: str | None = None,
    ) -> PropagationResult:
        """Compute the expectation value of *observable* after evolving through *circuit*.

        If *filename* is given, the final term sum is saved to a gzip-compressed
        binary file at that path.

        Arguments:
            observable: The term sum whose expectation value is computed (a
                `PauliTermSum` or `MajoranaTermSum`, matching the propagator's
                representation).
            circuit: The circuit to propagate through (a `PauliCircuit`/
                `MajoranaCircuit` for a numerical propagator, or a
                `SurrogatePauliCircuit`/`SurrogateMajoranaCircuit` for a surrogate
                propagator).
            initial_state: The computational basis state to evaluate the
                expectation value in, as an integer bitmask.
            filename: Optional path to save the evolved term sum to, gzip-compressed.

        Returns:
            The expectation value, plus any diagnostics collected during propagation.
        """
