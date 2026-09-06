"""Gate parameterization for circuits in the Pauli representation."""

from ...datatypes.pauli.pauli import PauliString
from ..abstract import AbstractRotation


class PauliRotation(AbstractRotation[PauliString]):
    r"""
    Class representing a gate in the Pauli representation, parameterized by a Pauli string and a rotation angle.

    The gate parameterization is given by:

    ```math
    G = e^{-i \theta P / 2}
    ```

    where \(P\) is a Pauli string and \(\theta\) is the rotation angle in radians
    (the same half-angle convention as Qiskit's RZ/CP gate parameters, enforced by
    the propagator regardless of what a naive dense-matrix exponential of \(P\) alone
    would suggest).
    """

    generator: PauliString
    """The Pauli string \\(P\\) that generates the rotation."""
