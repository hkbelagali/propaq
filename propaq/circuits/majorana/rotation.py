"""Gate parameterization for fermionic circuits in the Majorana representation."""

from ...datatypes import MajoranaMonomial


class MajoranaRotation:
    """Class representing a rotation in the Majorana representation."""
    def __init__(self, generator: MajoranaMonomial, angle: float, is_intermediate: bool = False):
        self.generator = generator
        self.angle = angle
        self.is_intermediate = is_intermediate