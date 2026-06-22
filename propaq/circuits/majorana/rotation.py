"""Gate parameterization for circuits in the Majorana representation."""

from ...datatypes import MajoranaMonomial


class MajoranaRotation:
    r"""
    Class representing a gate in the Majorana representation, parameterized by a Majorana monomial and a rotation angle.

    The gate parameterization is given by:

    $$
    G = e^{-i \theta M}
    $$
    
    where $M$ is a Majorana monomial and $\theta$ is the rotation angle in radians.
    """

    generator: MajoranaMonomial
    """The Majorana monomial $M$ that generates the rotation."""

    angle: float
    r"""The rotation angle $\theta$ in radians."""

    is_intermediate: bool
    """
    Whether this rotation is an intermediate parameterization that does not preserve particle number. This is used to control
    truncation policies during propagation.
    """

    def __init__(self, generator: MajoranaMonomial, angle: float, is_intermediate: bool = False):
        self.generator = generator
        self.angle = angle
        self.is_intermediate = is_intermediate