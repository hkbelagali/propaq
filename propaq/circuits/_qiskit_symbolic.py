"""
Helpers for qiskit conversion
"""

from dataclasses import dataclass
from typing import Any

from qiskit.circuit import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression


@dataclass(frozen=True)
class ParamSource:
    """Describes how one propaq surrogate parameter slot is evaluated.

    If `parameter` is not None, the slot's value at evaluate time is
    `scale * value_of(parameter)`. If `parameter` is None, the slot is a fixed
    constant equal to `scale`, independent of any Qiskit Parameter.

    Attributes:
        parameter: The Qiskit Parameter this slot tracks, or None for a fixed constant.
        scale: The affine coefficient applied to `parameter`'s value (or the constant itself).
    """

    parameter: "Parameter | None"
    scale: float


def affine_components(
    expr: "ParameterExpression | float",
) -> tuple[list[tuple[Parameter, float]], float]:
    """
    Decompose a gate angle into its affine components.

    Arguments:
        expr: A concrete float/int, or a Qiskit ParameterExpression that is
            affine (real-linear) in each of its free Parameters.

    Returns:
        (terms, offset): `terms` holds one `(Parameter, slope)` pair per free
        parameter with nonzero slope; `offset` is the residual constant term.

    Raises:
        ValueError: If `expr` is not affine in one of its free parameters.
    """
    if isinstance(expr, int | float):
        return [], float(expr)

    terms: list[tuple[Parameter, float]] = []
    residual = expr
    for p in expr.parameters:
        grad = expr.gradient(p)
        if isinstance(grad, ParameterExpression) and grad.parameters:
            raise ValueError(
                f"Gate angle is not affine in parameter '{p.name}'; propaq's surrogate "
                "propagator only supports rotation angles that are affine (real-linear) "
                "combinations of Parameters."
            )
        slope = float(grad)
        if slope != 0.0:
            terms.append((p, slope))
        residual = residual.assign(p, 0)
    offset = float(residual)
    return terms, offset


class ParamIndexPool:
    """Allocates/reuses surrogate `param_index` values for `(Parameter | None, scale)` keys.

    Used once per `from_qiskit` conversion so that identical angle expressions
    (e.g. an ansatz parameter reused verbatim across several gates) collapse to
    a single parameter index.
    """

    def __init__(self) -> None:
        self._index_of: dict[tuple[Parameter | None, float], int] = {}
        self.sources: list[ParamSource] = []

    def index_for(self, parameter: "Parameter | None", scale: float) -> int:
        key = (parameter, scale)
        idx = self._index_of.get(key)
        if idx is None:
            idx = len(self.sources)
            self._index_of[key] = idx
            self.sources.append(ParamSource(parameter, scale))
        return idx

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """Distinct Qiskit Parameters actually used, in first-allocated order."""
        seen: dict[Parameter, None] = {}
        for src in self.sources:
            if src.parameter is not None and src.parameter not in seen:
                seen[src.parameter] = None
        return tuple(seen)


def expand_affine_rotation(
    generator: Any,
    raw_angle: "ParameterExpression | float",
    pool: ParamIndexPool,
    rotation_cls: type,
    qiskit_gate_idx: "int | None",
) -> list[Any]:
    """
    Expand one (generator, angle) gate term into a chain of rotations about the
    same generator, one per affine component of `raw_angle`.

    All but the last rotation in the chain are marked `is_intermediate=True`.
    Returns an empty list if `raw_angle` is the exact constant zero (mirrors
    skipping an identity gate).
    """
    terms, offset = affine_components(raw_angle)
    if not terms and offset == 0.0:
        return []

    specs: list[tuple[int | None, float | None]] = [
        (pool.index_for(parameter, slope), None) for parameter, slope in terms
    ]
    if offset != 0.0 or not specs:
        specs.append((None, offset))

    return [
        rotation_cls(
            generator=generator,
            param_index=param_index,
            angle=angle,
            is_intermediate=i < len(specs) - 1,
            qiskit_gate_idx=qiskit_gate_idx,
        )
        for i, (param_index, angle) in enumerate(specs)
    ]
