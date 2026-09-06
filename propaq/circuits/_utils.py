"""Shared utilities for circuit classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .abstract import AbstractRotation

_R = TypeVar("_R", bound="AbstractRotation")


def compound_gate_reversed(rotations: list[_R]) -> list[_R]:
    """
    Reverse a layer's rotations for the inverse circuit, preserving compound-gate grouping.

    Arguments:
        rotations: One layer's rotations, in application order.

    Returns:
        The same rotations, regrouped and angle-negated for the inverse circuit.
    """
    compound_gates: list[list[_R]] = []
    current: list[_R] = []
    for rot in rotations:
        current.append(rot)
        if not rot.is_intermediate:
            compound_gates.append(current)
            current = []
    if current:
        compound_gates.append(current)

    result: list[_R] = []
    for gate in reversed(compound_gates):
        reversed_gate = list(reversed(gate))
        for i, rot in enumerate(reversed_gate):
            result.append(type(rot)(rot.generator, -rot.angle, i < len(reversed_gate) - 1))
    return result
