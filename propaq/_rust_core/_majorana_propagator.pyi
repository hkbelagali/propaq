from __future__ import annotations

from typing import TYPE_CHECKING

from ._logger import Logger
from ._majorana_term_sum import MajoranaTermSum
from ._noise import GateNoiseModel, UniformNoiseModel
from ._truncation_policy import TruncationPolicy
from ._truncators import CoefficientTruncator, FlushSchedule, TermBudget, WeightTruncator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from propaq.circuits import MajoranaCircuit

# Truncators the numerical propagator honors (symbolic-only ones are rejected).
_NumericalTruncator = WeightTruncator | CoefficientTruncator | TermBudget

class PropagationResult:
    n_terms: list[int]
    expectation_value: float

class MajoranaPropagator:

    def __init__(
        self,
        noise: UniformNoiseModel | GateNoiseModel | None = None,
        truncation: _NumericalTruncator
        | Sequence[_NumericalTruncator]
        | TruncationPolicy
        | None = None,
        schedule: FlushSchedule | None = None,
        n_threads: int | None = None,
        progress_bar: bool = False,
        logger: Logger | None = None,
    ) -> None:
        """
        Initialize the Majorana propagator.

        Arguments:
            noise: Optional noise model. Use UniformNoiseModel for depolarising noise, or
                wrap a custom duck-typed model in GateNoiseModel.
            truncation: The truncation pipeline, a list of truncators
                (WeightTruncator/CoefficientTruncator/TermBudget), a single such
                truncator, a legacy TruncationPolicy (decomposed), or None. The
                symbolic-only FrequencyTruncator/MonomialBudget are rejected.
            schedule: Optional FlushSchedule controlling the lossless merge cadence.
            n_threads: Number of worker threads. Defaults to the number of logical CPU cores.
            progress_bar: If True, display a tqdm progress bar over circuit gates.
            logger: Optional Logger instance for JSONL event logging.
        """
        ...

    @property
    def noise(self) -> UniformNoiseModel | GateNoiseModel | None: ...
    def set_noise(self, noise: UniformNoiseModel | GateNoiseModel | None = None) -> None: ...

    @property
    def truncators(self) -> list[_NumericalTruncator]: ...
    @property
    def schedule(self) -> FlushSchedule: ...
    @schedule.setter
    def schedule(self, schedule: FlushSchedule) -> None: ...
    def set_truncation(
        self,
        truncation: _NumericalTruncator
        | Sequence[_NumericalTruncator]
        | TruncationPolicy
        | None = None,
    ) -> None: ...

    def propagate(self, observable: MajoranaTermSum, circuit: MajoranaCircuit, filename: str | None = None) -> MajoranaTermSum:
        """
        Back-propagate *circuit* through *observable* in the Heisenberg picture.

        For each parameterized gate, the observable is evolved batch-wise over its terms
        using multithreading. Noise and truncation policies are applied after each gate.
        Intermediate parameterizations that do not preserve particle number are not truncated.

        Arguments:
            observable: The observable to propagate, represented in the Majorana term sum format.
            circuit: The quantum circuit to propagate through.
            filename: Optional filename to write the propagated observable to disk as a compressed gzip file.

        Returns:
            The propagated observable, represented in the Majorana term sum format.
        """
        ...

    def expectation_value(
        self,
        observable: MajoranaTermSum,
        circuit: MajoranaCircuit,
        initial_state: int = 0,
        filename: str | None = None,
    ) -> PropagationResult:
        """
        Calculate the expectation value of an observable with respect to a Fock state.

        Arguments:
            observable: The observable to calculate the expectation value for.
            circuit: The quantum circuit to propagate through.
            initial_state: The initial Fock state as a bitstring integer.
            filename: Optional filename to write the propagated observable to disk as a compressed gzip file.

        Returns:
            PropagationResult containing the number of terms in the propagated observable at each step and the final expectation value.
        """
        ...
