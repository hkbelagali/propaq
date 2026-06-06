"""Shared utilities for circuit classes."""

from typing import Any, List


def compound_gate_reversed(rotations: List[Any]) -> List[Any]:
    """Reverse a layer's rotations for the inverse circuit, preserving compound-gate grouping.

    Rotations within a compound gate are contiguous; a gate ends at each is_intermediate=False
    boundary. After reversal, all positions except the new last become intermediate.
    Works for any rotation type whose constructor accepts (generator, angle, is_intermediate).
    """
    compound_gates: List[List[Any]] = []
    current: List[Any] = []
    for rot in rotations:
        current.append(rot)
        if not rot.is_intermediate:
            compound_gates.append(current)
            current = []
    if current:
        compound_gates.append(current)

    result: List[Any] = []
    for gate in reversed(compound_gates):
        reversed_gate = list(reversed(gate))
        for i, rot in enumerate(reversed_gate):
            result.append(type(rot)(rot.generator, -rot.angle, i < len(reversed_gate) - 1))
    return result
