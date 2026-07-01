"""Surrogate gate parameterization for circuits in the Pauli representation."""

from ...datatypes.pauli.pauli import PauliString


class SurrogateRotation:
    """
    A Pauli rotation gate parameterized by a symbolic parameter index.

    Unlike `PauliRotation` (which stores a concrete angle), this class stores
    a `param_index` so that the surrogate propagator can record which trig
    factors each term accumulates without binding to specific angle values.

    Attributes:
        generator: The Pauli string generating the rotation.
        param_index: Index into the parameter vector supplied at evaluate time.
        is_intermediate: Whether this gate is intermediate (controls truncation).
        qiskit_gate_idx: Index of the originating Qiskit gate, or None.
    """

    generator: PauliString
    param_index: int
    is_intermediate: bool
    qiskit_gate_idx: int | None

    def __init__(
        self,
        generator: PauliString,
        param_index: int,
        is_intermediate: bool = False,
        qiskit_gate_idx: int | None = None,
    ):
        self.generator = generator
        self.param_index = param_index
        self.is_intermediate = is_intermediate
        self.qiskit_gate_idx = qiskit_gate_idx
