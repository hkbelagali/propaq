"""Surrogate gate parameterization for circuits in the Majorana representation."""

from ...datatypes import MajoranaMonomial


class SurrogateMajoranaRotation:
    """
    A Majorana rotation gate parameterized by a symbolic parameter index.

    Attributes:
        generator: The Majorana monomial generating the rotation.
        param_index: Index into the parameter vector supplied at evaluate time.
        is_intermediate: Whether this gate is intermediate (controls truncation).
        qiskit_gate_idx: Index of the originating Qiskit gate, or None.
    """

    generator: MajoranaMonomial
    param_index: int
    is_intermediate: bool
    qiskit_gate_idx: int | None

    def __init__(
        self,
        generator: MajoranaMonomial,
        param_index: int,
        is_intermediate: bool = False,
        qiskit_gate_idx: int | None = None,
    ):
        self.generator = generator
        self.param_index = param_index
        self.is_intermediate = is_intermediate
        self.qiskit_gate_idx = qiskit_gate_idx
