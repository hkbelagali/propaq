"""Gate parameterization for circuits in the Majorana representation."""

from ...datatypes import MajoranaMonomial


class MajoranaRotation:
    r"""
    Class representing a gate in the Majorana representation, parameterized by a Majorana monomial and a rotation angle.

    The gate parameterization is given by:

    \[
    G = e^{-i \theta M / 2}
    \]

    where \(M\) is a Majorana monomial and \(\theta\) is the rotation angle in radians.
    """

    generator: MajoranaMonomial
    """The Majorana monomial \\(M\\) that generates the rotation."""

    angle: float
    r"""The rotation angle \(\theta\) in radians."""

    is_intermediate: bool
    """
    Whether this rotation is an intermediate parameterization that does not preserve particle number. This is used to control
    truncation policies during propagation.
    """

    qiskit_gate_idx: int | None
    """Index of the originating Qiskit gate in the source circuit, or None for non-Qiskit circuits."""

    def __init__(
        self,
        generator: MajoranaMonomial,
        angle: float,
        is_intermediate: bool = False,
        qiskit_gate_idx: int | None = None,
    ):
        """
        Construct a MajoranaRotation from a generator and an angle.
        """
        self.generator = generator
        self.angle = angle
        self.is_intermediate = is_intermediate
        self.qiskit_gate_idx = qiskit_gate_idx
