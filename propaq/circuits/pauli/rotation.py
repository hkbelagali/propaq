"""Gate parameterization for Pauli propagation circuits."""

from ...datatypes.pauli.pauli import PauliMonomial


class PauliRotation:
    """A single rotation gate exp(-i angle/2 * generator) where *generator* is a Pauli string."""

    def __init__(self, generator: PauliMonomial, angle: float, is_intermediate: bool = False):
        self.generator = generator
        self.angle = angle
        self.is_intermediate = is_intermediate
