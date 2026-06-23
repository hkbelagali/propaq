"""Gate parameterization for circuits in the Pauli representation."""

from ...datatypes.pauli.pauli import PauliString


class PauliRotation:
    r"""
    Class representing a gate in the Pauli representation, parameterized by a Pauli string and a rotation angle.

    The gate parameterization is given by:

    $$
    G = e^{-i \theta P}
    $$
    
    where $P$ is a Pauli string and $\theta$ is the rotation angle in radians.
    """

    generator: PauliString
    """The Pauli string $P$ that generates the rotation."""

    angle: float
    r"""The rotation angle $\theta$ in radians."""

    is_intermediate: bool
    """
    Whether this rotation is an intermediate parameterization that does not preserve particle number. This is used to control
    truncation policies during propagation.
    """

    def __init__(self, generator: PauliString, angle: float, is_intermediate: bool = False):
        self.generator = generator
        self.angle = angle
        self.is_intermediate = is_intermediate
