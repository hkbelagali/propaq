"""
Helpers for Cirq conversion.

Mirrors propaq/circuits/_qiskit_symbolic.py, but built on sympy instead of
Qiskit's ParameterExpression: Cirq parameterizes gates with plain
sympy.Symbol/sympy.Expr objects rather than a bespoke Parameter class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._qiskit_symbolic import ParamSource as ParamSource

if TYPE_CHECKING:
    import sympy


def affine_components(expr: sympy.Expr | float) -> tuple[list[tuple[sympy.Symbol, float]], float]:
    """
    Decompose a gate angle into its affine components.

    Arguments:
        expr: A concrete float/int, or a sympy expression that is affine
            (real-linear) in each of its free symbols.

    Returns:
        (terms, offset): `terms` holds one `(Symbol, slope)` pair per free
        symbol with nonzero slope; `offset` is the residual constant term.

    Raises:
        ValueError: If `expr` is not affine in one of its free symbols.
    """
    if isinstance(expr, int | float):
        return [], float(expr)

    import sympy

    terms: list[tuple[sympy.Symbol, float]] = []
    residual = expr
    for p in expr.free_symbols:
        grad = sympy.diff(expr, p)
        if isinstance(grad, sympy.Expr) and grad.free_symbols:
            raise ValueError(
                f"Gate angle is not affine in symbol '{p.name}'; propaq's surrogate "
                "propagator only supports rotation angles that are affine (real-linear) "
                "combinations of symbols."
            )
        slope = float(grad)
        if slope != 0.0:
            terms.append((p, slope))
        residual = residual.subs(p, 0)
    offset = float(residual)
    return terms, offset


class ParamIndexPool:
    """Allocates/reuses surrogate `param_index` values for `(Symbol | None, scale)` keys.

    Used once per `from_cirq` conversion so that identical angle expressions
    (e.g. an ansatz symbol reused verbatim across several gates) collapse to a
    single parameter index.
    """

    def __init__(self) -> None:
        self._index_of: dict[tuple[sympy.Symbol | None, float], int] = {}
        self.sources: list[ParamSource] = []

    def index_for(self, symbol: sympy.Symbol | None, scale: float) -> int:
        key = (symbol, scale)
        idx = self._index_of.get(key)
        if idx is None:
            idx = len(self.sources)
            self._index_of[key] = idx
            self.sources.append(ParamSource(symbol, scale))
        return idx

    @property
    def parameters(self) -> tuple[sympy.Symbol, ...]:
        """Distinct sympy Symbols actually used, in first-allocated order."""
        seen: dict[sympy.Symbol, None] = {}
        for src in self.sources:
            if src.parameter is not None and src.parameter not in seen:
                seen[src.parameter] = None
        return tuple(seen)


def expand_affine_rotation(
    generator: Any,
    raw_angle: sympy.Expr | float,
    pool: ParamIndexPool,
    rotation_cls: type,
    qiskit_gate_idx: int | None,
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
        (pool.index_for(symbol, slope), None) for symbol, slope in terms
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
