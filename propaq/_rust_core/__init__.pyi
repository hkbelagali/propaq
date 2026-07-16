from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._logger import Logger as Logger
    from ._majorana_monomial import MajoranaMonomial as MajoranaMonomial
    from ._majorana_propagator import MajoranaPropagator as MajoranaPropagator
    from ._majorana_propagator import PropagationResult as PropagationResult
    from ._majorana_term_streamer import MajoranaTermStreamer as MajoranaTermStreamer
    from ._majorana_term_sum import MajoranaTermSum as MajoranaTermSum
    from ._noise import GateNoiseModel as GateNoiseModel
    from ._noise import NativeNoiseModel as NativeNoiseModel
    from ._noise import UniformNoiseModel as UniformNoiseModel
    from ._pauli_propagator import PauliPropagator as PauliPropagator
    from ._pauli_string import PauliString as PauliString
    from ._pauli_term_streamer import PauliTermStreamer as PauliTermStreamer
    from ._pauli_term_sum import PauliTermSum as PauliTermSum
    from ._surrogate_majorana import MajoranaSurrogateModel as MajoranaSurrogateModel
    from ._surrogate_majorana import MajoranaSurrogatePropagator as MajoranaSurrogatePropagator
    from ._surrogate_pauli import PauliSurrogateModel as PauliSurrogateModel
    from ._surrogate_pauli import PauliSurrogatePropagator as PauliSurrogatePropagator
    from ._surrogate_truncation_policy import FrequencyTruncationPolicy as FrequencyTruncationPolicy
    from ._truncation_policy import TruncationPolicy as TruncationPolicy
    from ._truncators import CoefficientTruncator as CoefficientTruncator
    from ._truncators import FlushSchedule as FlushSchedule
    from ._truncators import FrequencyTruncator as FrequencyTruncator
    from ._truncators import MonomialBudget as MonomialBudget
    from ._truncators import NativeTruncator as NativeTruncator
    from ._truncators import Simplify as Simplify
    from ._truncators import TermBudget as TermBudget
    from ._truncators import WeightTruncator as WeightTruncator

def rust_available() -> bool: ...
