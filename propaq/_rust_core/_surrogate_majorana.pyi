from __future__ import annotations

from typing import TYPE_CHECKING

from ._logger import Logger
from ._majorana_term_sum import MajoranaTermSum
from ._surrogate_truncation_policy import FrequencyTruncationPolicy
from ._truncation_policy import TruncationPolicy
from ._truncators import (
    CoefficientTruncator,
    FlushSchedule,
    FrequencyTruncator,
    TermBudget,
    WeightTruncator,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from propaq.circuits.majorana.surrogate_circuit import SurrogateMajoranaCircuit

# The surrogate honors every truncator, including the monomial-level
# FrequencyTruncator/CoefficientTruncator: both are decided structurally
# from the symbolic DAG's cached per-node bounds, with no monomial expansion
# needed (see propaq.MD's "Truncation" section).
_Truncator = FrequencyTruncator | CoefficientTruncator | WeightTruncator | TermBudget
_Truncation = (
    _Truncator | Sequence[_Truncator] | FrequencyTruncationPolicy | TruncationPolicy | None
)


class MajoranaSurrogateModel:
    """
    Compiled surrogate model for Majorana observables.

    Produced by `MajoranaSurrogatePropagator.build`. Call `evaluate(params)` to
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

    @property
    def n_monomials(self) -> int:
        """
        Total pre-dedup monomial-instance count across every surviving term
        (an upper bound, not deduplicated). `n_terms` alone doesn't say how
        much underlying computation a term represents.
        """
        ...

    def evaluate(self, params: list[float]) -> float:
        """
        Evaluate the expectation value.

        Arguments:
            params: Angles in radians. ``params[i]`` is the angle for parameter index ``i``.
                Length must be at least ``n_params``.
        """
        ...

    def evaluate_batch(self, param_sets: list[list[float]]) -> list[float]:
        """
        Evaluate many parameter assignments at once (parallelized across
        assignments). Each entry follows the same convention as `evaluate`.
        """
        ...

    def save(self, path: str) -> None:
        """Save to a gzip-compressed binary file."""
        ...

    @staticmethod
    def load(path: str) -> MajoranaSurrogateModel:
        """Load a model from a file produced by `save`."""
        ...

    def __repr__(self) -> str: ...


class MajoranaSurrogatePropagator:
    """
    Back-propagates Majorana observables symbolically, producing a compiled model
    that can be re-evaluated cheaply for any parameter assignment.

    Arguments:
        truncation: A list of truncators (FrequencyTruncator/CoefficientTruncator/
            WeightTruncator/TermBudget), a single truncator, a legacy
            FrequencyTruncationPolicy or TruncationPolicy (decomposed), or None.
        schedule: Optional FlushSchedule controlling the lossless merge cadence.
        n_threads: Number of worker threads. Defaults to the system thread count.
        progress_bar: Display a tqdm progress bar during propagation.
        logger: Optional Logger for verbose JSON Lines event logging.
    """

    def __init__(
        self,
        truncation: _Truncation = None,
        schedule: FlushSchedule | None = None,
        n_threads: int | None = None,
        progress_bar: bool = False,
        logger: Logger | None = None,
    ) -> None: ...

    @property
    def truncators(self) -> list[_Truncator]: ...
    @property
    def schedule(self) -> FlushSchedule: ...
    @schedule.setter
    def schedule(self, schedule: FlushSchedule) -> None: ...
    def set_truncation(self, truncation: _Truncation = None) -> None: ...

    def build(
        self,
        observable: MajoranaTermSum,
        circuit: SurrogateMajoranaCircuit,
        initial_state: int = 0,
    ) -> MajoranaSurrogateModel:
        """
        Compile the observable back-propagated through the circuit.

        Arguments:
            observable: The Majorana observable to back-propagate.
            circuit: A SurrogateMajoranaCircuit (generators + parameter indices).
            initial_state: Fock state as a bitstring integer (default 0).

        Returns:
            A MajoranaSurrogateModel ready for parameter-free evaluation.
        """
        ...
