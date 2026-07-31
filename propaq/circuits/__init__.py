"""Circuit representations and gate parameterizations for propaq."""

from ._gates import GateDecompositionWarning as GateDecompositionWarning
from .majorana.circuit import MajoranaCircuit as MajoranaCircuit
from .majorana.rotation import MajoranaRotation as MajoranaRotation
from .majorana.surrogate_circuit import SurrogateMajoranaCircuit as SurrogateMajoranaCircuit
from .majorana.surrogate_rotation import SurrogateMajoranaRotation as SurrogateMajoranaRotation
from .pauli.circuit import PauliCircuit as PauliCircuit
from .pauli.rotation import PauliRotation as PauliRotation
from .pauli.surrogate_circuit import SurrogatePauliCircuit as SurrogatePauliCircuit
from .pauli.surrogate_rotation import SurrogateRotation as SurrogateRotation

__all__ = [
    "GateDecompositionWarning",
    "MajoranaRotation", "MajoranaCircuit",
    "SurrogateMajoranaRotation", "SurrogateMajoranaCircuit",
    "PauliRotation", "PauliCircuit",
    "SurrogateRotation", "SurrogatePauliCircuit",
]