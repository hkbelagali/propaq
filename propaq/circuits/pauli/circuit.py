"""Circuit representation for Pauli propagation."""

from typing import List, Union

from ...datatypes.pauli.pauli import PauliMonomial
from .rotation import PauliRotation


def _compound_gate_reversed(layer: List[PauliRotation]) -> List[PauliRotation]:
    """Reverse a layer's rotations for the inverse circuit, preserving compound-gate grouping."""
    compound_gates: List[List[PauliRotation]] = []
    current: List[PauliRotation] = []
    for rot in layer:
        current.append(rot)
        if not rot.is_intermediate:
            compound_gates.append(current)
            current = []
    if current:
        compound_gates.append(current)

    result: List[PauliRotation] = []
    for gate in reversed(compound_gates):
        reversed_gate = list(reversed(gate))
        for i, rot in enumerate(reversed_gate):
            result.append(PauliRotation(rot.generator, -rot.angle, i < len(reversed_gate) - 1))
    return result


class PauliCircuit:
    """A quantum circuit expressed as a sequence of Pauli-string rotations.

    Unlike MajoranaCircuit, no Jordan-Wigner transform is required — generators
    are Pauli strings (PauliMonomial) supplied directly by the caller.
    """

    def __init__(
        self,
        rotations_or_layers: Union[List[PauliRotation], List[List[PauliRotation]]],
    ):
        if rotations_or_layers and isinstance(rotations_or_layers[0], list):
            self._layers: List[List[PauliRotation]] = rotations_or_layers
        else:
            self._layers = [[r] for r in rotations_or_layers]  # type: ignore[arg-type]

    @property
    def layers(self) -> List[List[PauliRotation]]:
        return self._layers

    @property
    def rotations(self) -> List[PauliRotation]:
        return [r for layer in self._layers for r in layer]

    @classmethod
    def from_generators_and_angles(
        cls,
        generators: List[PauliMonomial],
        angles: List[float],
    ) -> "PauliCircuit":
        """Construct a PauliCircuit from lists of Pauli generators and rotation angles."""
        rotations = [PauliRotation(gen, angle) for gen, angle in zip(generators, angles)]
        return cls(rotations)

    def inverse(self) -> "PauliCircuit":
        """Return a new PauliCircuit representing the adjoint (U†) of this circuit."""
        reversed_layers = [_compound_gate_reversed(layer) for layer in reversed(self._layers)]
        circ = PauliCircuit.__new__(PauliCircuit)
        circ._layers = reversed_layers
        return circ
