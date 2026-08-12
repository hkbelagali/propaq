"""
Fast Heisenberg-picture propagation for quantum circuit simulation.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import metadata as _metadata

try:
    _readme = _metadata("propaq").json.get("description")
except PackageNotFoundError:  # running from an uninstalled source tree
    _readme = None
if isinstance(_readme, str) and _readme:
    __doc__ = _readme


from ._rust_core import Logger as Logger
from .circuits import (
    MajoranaCircuit as MajoranaCircuit,
)
from .circuits import (
    MajoranaRotation as MajoranaRotation,
)
from .circuits import (
    PauliCircuit as PauliCircuit,
)
from .circuits import (
    PauliRotation as PauliRotation,
)
from .circuits import (
    SurrogateMajoranaCircuit as SurrogateMajoranaCircuit,
)
from .circuits import (
    SurrogatePauliCircuit as SurrogatePauliCircuit,
)
from .datatypes import (
    MajoranaMonomial as MajoranaMonomial,
)
from .datatypes import (
    MajoranaTermSum as MajoranaTermSum,
)
from .datatypes import (
    PauliString as PauliString,
)
from .datatypes import (
    PauliTermSum as PauliTermSum,
)
from .extrapolators import ZeroCutoffExtrapolator as ZeroCutoffExtrapolator
from .extrapolators import ZeroNoiseExtrapolator as ZeroNoiseExtrapolator
from .extrapolators import ZNEResult as ZNEResult
from .log_parser import EnginePhasesEvent as EnginePhasesEvent
from .log_parser import GateEvent as GateEvent
from .log_parser import LogParser as LogParser
from .log_parser import SurrogateFlushDeferredEvent as SurrogateFlushDeferredEvent
from .log_parser import SurrogateFlushEvent as SurrogateFlushEvent
from .log_parser import TruncationEvent as TruncationEvent
from .models import (
    MajoranaSurrogateModel as MajoranaSurrogateModel,
)
from .models import (
    ParamSource as ParamSource,
)
from .models import (
    PauliSurrogateModel as PauliSurrogateModel,
)
from .models import (
    VariationalSurrogateModel as VariationalSurrogateModel,
)
from .noise import (
    FrequencyTruncationPolicy as FrequencyTruncationPolicy,
)
from .propagators import (
    MajoranaPropagator as MajoranaPropagator,
)
from .propagators import (
    MajoranaSurrogatePropagator as MajoranaSurrogatePropagator,
)
from .propagators import (
    PauliPropagator as PauliPropagator,
)
from .propagators import (
    PauliSurrogatePropagator as PauliSurrogatePropagator,
)
from .truncation import (
    CoefficientTruncator as CoefficientTruncator,
)
from .truncation import (
    FrequencyTruncator as FrequencyTruncator,
)
from .truncation import (
    MonomialBudget as MonomialBudget,
)
from .truncation import (
    NativeTruncator as NativeTruncator,
)
from .truncation import (
    Simplify as Simplify,
)
from .truncation import (
    TermBudget as TermBudget,
)
from .truncation import (
    Truncator as Truncator,
)
from .truncation import (
    WeightTruncator as WeightTruncator,
)
