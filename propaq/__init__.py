"""
Fast Heisenberg-picture propagation for quantum circuit simulation.
"""

from importlib.metadata import PackageNotFoundError, version
from importlib.metadata import metadata as _metadata

__version__ = version("propaq")

try:
    _readme = _metadata("propaq").json.get("description")
except PackageNotFoundError:  # running from an uninstalled source tree
    _readme = None
if isinstance(_readme, str) and _readme:
    __doc__ = _readme


from ._rust_core import Logger as Logger
from .circuits import AbstractCircuit as AbstractCircuit
from .circuits import AbstractRotation as AbstractRotation
from .circuits import GateDecompositionWarning as GateDecompositionWarning
from .circuits import GateRep as GateRep
from .circuits import GateValidationError as GateValidationError
from .circuits import MajoranaCircuit as MajoranaCircuit
from .circuits import MajoranaRotation as MajoranaRotation
from .circuits import ParamSource as ParamSource
from .circuits import PauliCircuit as PauliCircuit
from .circuits import PauliRotation as PauliRotation
from .circuits import SurrogateMajoranaCircuit as SurrogateMajoranaCircuit
from .circuits import SurrogateMajoranaRotation as SurrogateMajoranaRotation
from .circuits import SurrogatePauliCircuit as SurrogatePauliCircuit
from .circuits import SurrogateRotation as SurrogateRotation
from .circuits import pauli_rotation_generator as pauli_rotation_generator
from .circuits import register_cirq_gate as register_cirq_gate
from .circuits import register_qiskit_gate as register_qiskit_gate
from .datatypes import AbstractTerm as AbstractTerm
from .datatypes import AbstractTermSum as AbstractTermSum
from .datatypes import DictTermSum as DictTermSum
from .datatypes import MajoranaMonomial as MajoranaMonomial
from .datatypes import MajoranaTermStreamer as MajoranaTermStreamer
from .datatypes import MajoranaTermSum as MajoranaTermSum
from .datatypes import PauliString as PauliString
from .datatypes import PauliTermStreamer as PauliTermStreamer
from .datatypes import PauliTermSum as PauliTermSum
from .extrapolators import CoefficientCutoffExtrapolator as CoefficientCutoffExtrapolator
from .extrapolators import WeightCutoffExtrapolator as WeightCutoffExtrapolator
from .extrapolators import ZCEResult as ZCEResult
from .extrapolators import ZeroCutoffExtrapolator as ZeroCutoffExtrapolator
from .extrapolators import ZeroNoiseExtrapolator as ZeroNoiseExtrapolator
from .extrapolators import ZNEResult as ZNEResult
from .hybrid import hybrid_expectation_value as hybrid_expectation_value
from .log_parser import EnginePhasesEvent as EnginePhasesEvent
from .log_parser import GateEvent as GateEvent
from .log_parser import LogParser as LogParser
from .log_parser import SurrogateMergeEvent as SurrogateMergeEvent
from .log_parser import TruncationEvent as TruncationEvent
from .models import MajoranaSurrogateModel as MajoranaSurrogateModel
from .models import PauliSurrogateModel as PauliSurrogateModel
from .models import VariationalSurrogateModel as VariationalSurrogateModel
from .noise import GateNoiseModel as GateNoiseModel
from .noise import NativeNoiseModel as NativeNoiseModel
from .noise import NoiseModel as NoiseModel
from .noise import UniformNoiseModel as UniformNoiseModel
from .propagators import AbstractPropagator as AbstractPropagator
from .propagators import CircuitLike as CircuitLike
from .propagators import MajoranaPropagator as MajoranaPropagator
from .propagators import MajoranaSurrogatePropagator as MajoranaSurrogatePropagator
from .propagators import PauliPropagator as PauliPropagator
from .propagators import PauliSurrogatePropagator as PauliSurrogatePropagator
from .propagators import PropagationResult as PropagationResult
from .truncation import CoefficientTruncator as CoefficientTruncator
from .truncation import FrequencyTruncationPolicy as FrequencyTruncationPolicy
from .truncation import FrequencyTruncator as FrequencyTruncator
from .truncation import NativeTruncator as NativeTruncator
from .truncation import ResolvedTruncation as ResolvedTruncation
from .truncation import Simplify as Simplify
from .truncation import TermBudget as TermBudget
from .truncation import TruncationPolicy as TruncationPolicy
from .truncation import Truncator as Truncator
from .truncation import WeightTruncator as WeightTruncator
from .truncation import resolve_truncation as resolve_truncation
