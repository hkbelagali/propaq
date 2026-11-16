"""Surrogate propagation for compiled traced bases.

A traced basis that compiles with `surrogate=True` has the classes
`SurrogatePropagator` and `SurrogateModel`. This file gives the loader and
a small circuit type for them. The circuit holds terms of that module.

Only bases with 2 branches per rotation work (Pauli, Majorana, and other
bases of the +-1 class). A basis with more branches, such as the Z4 Weyl
basis, raises `ValueError` in `SurrogatePropagator.build`. A symbolic
coefficient is a real sum of cos and sin factors. More branches need
complex trigonometric polynomials with several frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class TracedSurrogateRotation:
    """One rotation. Give `param_index` for a symbolic angle, or `angle` for a fixed one."""

    generator: Any
    param_index: int | None = None
    angle: float | None = None
    qiskit_gate_idx: int | None = None

    def __post_init__(self):
        if (self.param_index is None) == (self.angle is None):
            raise ValueError("give exactly one of param_index and angle")


@dataclass
class TracedSurrogateCircuit:
    """Layers of rotations in circuit order. The first gate acts first."""

    layers: list[list[TracedSurrogateRotation]] = field(default_factory=list)

    @property
    def n_params(self) -> int:
        idx = [g.param_index for layer in self.layers for g in layer if g.param_index is not None]
        return max(idx) + 1 if idx else 0

    @classmethod
    def from_gates(cls, gates: Sequence[TracedSurrogateRotation], layer_size: int | None = None):
        """Cut a flat gate list into layers of `layer_size` (default: one layer)."""
        gates = list(gates)
        size = layer_size or max(len(gates), 1)
        return cls([gates[i : i + size] for i in range(0, len(gates), size)])


def load_surrogate_basis(spec, **kwargs):
    """Build (or take from the cache) the module of `spec` with surrogate classes.

    The generic traced program is compiled even for the built-in bases.
    """
    from ..tracer.build import load_basis

    return load_basis(spec, force_generic=True, surrogate=True, **kwargs)
