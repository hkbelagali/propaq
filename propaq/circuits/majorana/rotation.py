"""Gate parameterization for circuits in the Majorana representation."""

from ...datatypes import MajoranaMonomial
from ..abstract import AbstractRotation


class MajoranaRotation(AbstractRotation[MajoranaMonomial]):
    r"""
    Class representing a gate in the Majorana representation, parameterized by a Majorana monomial and a rotation angle.

    The gate parameterization is given by:

    ```math
    G = e^{-i \theta M / 2}
    ```

    where \(M\) is a Majorana monomial and \(\theta\) is the rotation angle in radians.
    """

    generator: MajoranaMonomial
    """The Majorana monomial \\(M\\) that generates the rotation."""
