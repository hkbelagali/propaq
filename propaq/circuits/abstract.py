"""Reusable base classes for a custom basis's gate and circuit representation."""

from __future__ import annotations

from typing import Generic, TypeVar, cast

from ._utils import compound_gate_reversed as _compound_gate_reversed

GeneratorT = TypeVar("GeneratorT")
RotationT = TypeVar("RotationT", bound="AbstractRotation")


class AbstractRotation(Generic[GeneratorT]):
    r"""
    A gate parameterized by a basis element and a rotation angle:

    ```math
    G = e^{-i \theta g / 2}
    ```

    where \(g\) is the generator and \(\theta\) is the angle in radians.
    """

    generator: GeneratorT
    """The generator \\(g\\) that parameterizes the rotation."""

    angle: float
    r"""The rotation angle \(\theta\) in radians."""

    is_intermediate: bool
    """
    Whether this rotation is an intermediate parameterization that should not 
    be truncated after. For example, if a particle number-conserving gate is divided 
    into two rotations, the first rotation is intermediate and should not be truncated
    after, since this would potentially break the symmetry.
    """

    qiskit_gate_idx: int | None
    """Index of the originating Qiskit gate in the source circuit, or None for non-Qiskit circuits."""

    def __init__(
        self,
        generator: GeneratorT,
        angle: float,
        is_intermediate: bool = False,
        qiskit_gate_idx: int | None = None,
    ) -> None:
        """Construct a rotation from a generator and an angle."""
        self.generator = generator
        self.angle = angle
        self.is_intermediate = is_intermediate
        self.qiskit_gate_idx = qiskit_gate_idx

    def __repr__(self) -> str:
        """A short representation naming the generator and angle."""
        return f"{type(self).__name__}({self.generator!r}, {self.angle!r})"


class AbstractCircuit(Generic[RotationT]):
    """
    A circuit as a list of layers of rotations, where a layer's rotations can be applied in parallel.
    """

    def __init__(self, rotations_or_layers: list[RotationT] | list[list[RotationT]]) -> None:
        """Construct a circuit from a list of rotations or a list of layers of rotations."""
        if rotations_or_layers and isinstance(rotations_or_layers[0], list):
            self._layers: list[list[RotationT]] = cast("list[list[RotationT]]", rotations_or_layers)
        else:
            self._layers = [[r] for r in cast("list[RotationT]", rotations_or_layers)]

    @property
    def layers(self) -> list[list[RotationT]]:
        """The layers of the circuit, where each layer's gates can be applied in parallel."""
        return self._layers

    @property
    def rotations(self) -> list[RotationT]:
        """The flat list of all rotations in the circuit, in the order they are applied."""
        return [r for layer in self._layers for r in layer]

    def append(self, rotation: RotationT, *, new_layer: bool = True) -> None:
        """Append a rotation to the circuit.

        Arguments:
            rotation: The rotation to append.
            new_layer: If True (default), the rotation starts its own layer;
                if False, it is added to the last existing layer.
        """
        if new_layer or not self._layers:
            self._layers.append([rotation])
        else:
            self._layers[-1].append(rotation)

    def inverse(self) -> AbstractCircuit[RotationT]:
        """Return a new circuit with reversed layer order and negated angles (U-dagger)."""
        return type(self)([_compound_gate_reversed(layer) for layer in reversed(self._layers)])
