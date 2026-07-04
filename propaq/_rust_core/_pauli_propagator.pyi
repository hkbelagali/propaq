from __future__ import annotations

from typing import TYPE_CHECKING

from ._logger import Logger
from ._majorana_propagator import PropagationResult
from ._noise import GateNoiseModel, UniformNoiseModel
from ._pauli_term_sum import PauliTermSum
from ._truncation_policy import TruncationPolicy
from ._truncators import CoefficientTruncator, FlushSchedule, TermBudget, WeightTruncator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from propaq.circuits import PauliCircuit

# Truncators the numerical propagator honors (symbolic-only ones are rejected).
_NumericalTruncator = WeightTruncator | CoefficientTruncator | TermBudget

class PauliPropagator:

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
        Initialize the Pauli propagator.

        Arguments:
            noise: Optional noise model. Use UniformNoiseModel for depolarising noise, or
                wrap a custom duck-typed model in GateNoiseModel.
            truncation: The truncation pipeline — a list of truncators
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
    def set_schedule(self, schedule: FlushSchedule) -> None: ...
    def set_truncation(
        self,
        truncation: _NumericalTruncator
        | Sequence[_NumericalTruncator]
        | TruncationPolicy
        | None = None,
    ) -> None: ...

    def propagate(self, observable: PauliTermSum, circuit: PauliCircuit) -> PauliTermSum:
        r"""
        Back-propagate *circuit* through *observable* in the Heisenberg picture.

        For each parameterized Pauli rotation $\exp{(-i \theta P)}$ the observable
        is evolved via the BCH formula:
            $$\mathcal{O} -> \cos(\theta)\mathcal{O} + i*\sin(\theta)*[P, \mathcal{O}]$$

        Arguments:
            observable: The observable to propagate, as a PauliTermSum.
            circuit: The quantum circuit (PauliCircuit) to propagate through.

        Returns:
            The propagated observable as a PauliTermSum.
        """
        ...

    def expectation_value(
        self,
        observable: PauliTermSum,
        circuit: PauliCircuit,
        initial_state: int = 0,
    ) -> PropagationResult:
        """
        Compute the expectation value of *observable* after back-propagating *circuit*.

        Arguments:
            observable: The observable to propagate.
            circuit: The quantum circuit to propagate through.
            initial_state: Computational basis state (bitstring integer) for the trace.

        Returns:
            PropagationResult with ``expectation_value`` and per-gate ``n_terms``.
        """
        ...
