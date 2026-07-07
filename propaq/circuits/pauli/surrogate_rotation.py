"""Surrogate gate parameterization for circuits in the Pauli representation."""

from ...datatypes.pauli.pauli import PauliString


class SurrogateRotation:
    """
    A Pauli rotation gate parameterized by either a symbolic parameter index
    or a concrete numeric angle, depending on which one is given.

    If `param_index` is not None, the surrogate propagator records which
    trig factors each term accumulates without binding to a specific angle
    value, resolved later at `SurrogateModel.evaluate` time. If `angle` is
    not None, `cos`/`sin` of the angle are computed immediately during
    propagation and folded directly into each term's scalar, exactly like
    `PauliRotation`'s numeric propagation. Exactly one of the two must be
    given.

    Attributes:
        generator: The Pauli string generating the rotation.
        param_index: Index into the parameter vector supplied at evaluate
            time, or None if `angle` is given instead.
        angle: A concrete rotation angle baked in at build time, or None if
            `param_index` is given instead.
        is_intermediate: Whether this gate is intermediate (controls truncation).
        qiskit_gate_idx: Index of the originating Qiskit gate, or None.
    """

    generator: PauliString
    param_index: int | None
    angle: float | None
    is_intermediate: bool
    qiskit_gate_idx: int | None

    def __init__(
        self,
        generator: PauliString,
        param_index: int | None = None,
        angle: float | None = None,
        is_intermediate: bool = False,
        qiskit_gate_idx: int | None = None,
    ):
        if (param_index is None) == (angle is None):
            raise ValueError(
                "SurrogateRotation requires exactly one of `param_index` "
                "(symbolic) or `angle` (numeric, baked in at build time), got "
                f"param_index={param_index!r}, angle={angle!r}"
            )
        self.generator = generator
        self.param_index = param_index
        self.angle = angle
        self.is_intermediate = is_intermediate
        self.qiskit_gate_idx = qiskit_gate_idx
