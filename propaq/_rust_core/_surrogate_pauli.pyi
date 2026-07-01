from __future__ import annotations

from typing import TYPE_CHECKING

from ._logger import Logger
from ._pauli_term_sum import PauliTermSum
from ._surrogate_truncation_policy import FrequencyTruncationPolicy

if TYPE_CHECKING:
    from propaq.circuits.pauli.surrogate_circuit import SurrogatePauliCircuit


class PauliSurrogateModel:
    """
    Compiled surrogate model for Pauli observables.

    Produced by `PauliSurrogatePropagator.build`. Call `evaluate(params)` to
    obtain the expectation value for any parameter assignment without re-running
    propagation. Use `save` / `load` for disk persistence.
    """

    @property
    def n_params(self) -> int:
        """Number of distinct parameter indices used by this model."""
        ...

    @property
    def n_terms(self) -> int:
        """Number of compiled terms (zero-overlap terms excluded)."""
        ...

    def evaluate(self, params: list[float]) -> float:
        """
        Evaluate the expectation value.

        Arguments:
            params: Angles in radians. ``params[i]`` is the angle for parameter index ``i``.
                Length must be at least ``n_params``.
        """
        ...

    def save(self, path: str) -> None:
        """Save to a gzip-compressed binary file."""
        ...

    @staticmethod
    def load(path: str) -> PauliSurrogateModel:
        """Load a model from a file produced by `save`."""
        ...

    def __repr__(self) -> str: ...


class PauliSurrogatePropagator:
    """
    Back-propagates Pauli observables symbolically, producing a compiled model
    that can be re-evaluated cheaply for any parameter assignment.

    Arguments:
        truncation: Optional FrequencyTruncationPolicy (frequency + weight cutoffs).
        n_threads: Number of worker threads. Defaults to the system thread count.
        progress_bar: Display a tqdm progress bar during propagation.
        logger: Optional Logger for verbose JSON Lines event logging.
    """

    def __init__(
        self,
        truncation: FrequencyTruncationPolicy | None = None,
        n_threads: int | None = None,
        progress_bar: bool = False,
        logger: Logger | None = None,
    ) -> None: ...

    @property
    def truncation(self) -> FrequencyTruncationPolicy | None: ...
    def set_truncation(self, truncation: FrequencyTruncationPolicy | None = None) -> None: ...

    def build(
        self,
        observable: PauliTermSum,
        circuit: SurrogatePauliCircuit,
        initial_state: int = 0,
    ) -> PauliSurrogateModel:
        """
        Compile the observable back-propagated through the circuit.

        Arguments:
            observable: The Pauli observable to back-propagate.
            circuit: A SurrogatePauliCircuit (generators + parameter indices).
            initial_state: Fock state as a bitstring integer (default 0).

        Returns:
            A PauliSurrogateModel ready for parameter-free evaluation.
        """
        ...
