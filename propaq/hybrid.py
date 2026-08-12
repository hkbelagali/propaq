"""
Hybrid Schrodinger-Heisenberg expectation values.

For a circuit C = C1 . C2, <Psi_0|C^dagger O C|Psi_0> splits into a
Heisenberg half and a Schrodinger half (|Psi> = C2|Psi_0>, as an MPS via quimb).
`hybrid_expectation_value` contracts every term of theta against |Psi>
in one native call.

Requires the `quimb` optional dependency (`pip install propaq[hybrid]`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from propaq._rust_core import MajoranaTermSum as _RustMajoranaTermSum
from propaq._rust_core import PauliTermSum as _RustPauliTermSum
from propaq._rust_core import hybrid_expectation
from propaq.datatypes import MajoranaTermSum, PauliTermSum

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from quimb.tensor import MatrixProductState

    from propaq.datatypes import AbstractTermSum

__all__ = ["hybrid_expectation_value"]


def _build_mps(circuit2: QuantumCircuit, initial_state: int):
    """
    Builds the MPS |Psi> = C2|Psi_0> via quimb.
    """
    import qiskit.qasm2 as qasm2
    import quimb.tensor as qtn
    from qiskit import QuantumCircuit

    n = circuit2.num_qubits
    full = QuantumCircuit(n)
    for q in range(n):
        if (initial_state >> q) & 1:
            full.x(q)
    full.compose(circuit2, inplace=True)

    circ = qtn.CircuitMPS.from_openqasm2_str(qasm2.dumps(full))
    return circ.psi


def _is_mps(obj) -> bool:
    """
    True if *obj* is already a quimb `MatrixProductState`
    """
    try:
        import quimb.tensor as qtn
    except ImportError:
        return False
    return isinstance(obj, qtn.MatrixProductState)


def _to_pauli_term_sum(term_sum: AbstractTermSum) -> PauliTermSum:
    """
    Converts *term_sum* to an equivalent `PauliTermSum` if it's a Majorana
    term sum, passing Pauli term sums through unchanged.
    """
    if isinstance(term_sum, _RustPauliTermSum):
        return term_sum
    if not isinstance(term_sum, _RustMajoranaTermSum):
        raise TypeError(f"Unsupported term sum type: {type(term_sum)!r}")
    wrapped: MajoranaTermSum = MajoranaTermSum(dtype=term_sum.dtype)
    wrapped.merge(term_sum)
    return PauliTermSum.from_sparse_pauli_op(wrapped.to_sparse_pauli_op())


def _normalize_mps_arrays(mps) -> list[np.ndarray]:
    """
    Extracts *mps*'s site tensors as rank-3 `(bond_l, phys, bond_r)`
    complex128 arrays, with dummy size-1 bonds inserted at the open
    boundaries
    """
    n = mps.L
    arrays = []
    for i in range(n):
        left = mps.bond(i - 1, i) if i > 0 else None
        right = mps.bond(i, i + 1) if i < n - 1 else None
        phys = mps.site_ind(i)
        want = tuple(ix for ix in (left, phys, right) if ix is not None)
        arr = mps[i].transpose(*want).data
        if left is None:
            arr = arr.reshape(1, *arr.shape)
        if right is None:
            arr = arr.reshape(*arr.shape, 1)
        arrays.append(np.ascontiguousarray(arr, dtype=np.complex128))
    return arrays


def hybrid_expectation_value(
    propagated_observable: AbstractTermSum,
    circuit2: QuantumCircuit | MatrixProductState,
    initial_state: int = 0,
) -> float:
    """
    Computes `<Psi|theta|Psi>` for an already Heisenberg-propagated
    observable `theta` and `|Psi> = C2|Psi_0>`.

    Arguments:
        propagated_observable: theta = C1^dagger O C1, already propagated
        circuit2: The Schrodinger half. Either a plain Qiskit `QuantumCircuit`, or an
            already-built quimb `MatrixProductState` representing `|Psi> =
            C2|Psi_0>` directly
        initial_state: Computational basis reference state as a bitstring
            integer (bit q = qubit q), matching `AbstractPropagator.expectation_value`'s
            convention. Only used when *circuit2* is a `QuantumCircuit`.

    Returns:
        The real expectation value.
    """
    theta = _to_pauli_term_sum(propagated_observable)
    mps = circuit2 if _is_mps(circuit2) else _build_mps(circuit2, initial_state)
    mps_arrays = _normalize_mps_arrays(mps)
    return hybrid_expectation(theta, mps_arrays)
