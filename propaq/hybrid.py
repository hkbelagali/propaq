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

from propaq._rust_core import hybrid_expectation, hybrid_expectation_raw
from propaq.datatypes import MajoranaTermSum, PauliTermSum
from propaq.datatypes.majorana.termsum import is_majorana_term_sum
from propaq.datatypes.pauli.termsum import is_pauli_term_sum

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
    if is_pauli_term_sum(term_sum):
        return term_sum
    if not is_majorana_term_sum(term_sum):
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


def _generic_expectation(terms, basis, mps_arrays) -> complex:
    """
    Contracts a term sum of any compiled basis against the MPS.

    *basis* is a module made by `load_basis`. It gives the matrix of each
    unit label (`local_table`) and the terms as raw words (`export_terms`).
    The basis must have a local table: its spec sets `local_dim` and `local_matrix`.
    """
    local_dim, flat = basis.local_table()
    if local_dim == 0:
        raise ValueError("this basis has no local matrices; set local_dim and local_matrix in its BasisSpec")
    flat = np.asarray(flat, dtype=np.float64)
    table = np.ascontiguousarray(flat[0::2] + 1j * flat[1::2], dtype=np.complex128)
    n_labels = table.size // (local_dim * local_dim)
    bits = n_labels.bit_length() - 1
    words_b, coeffs_b, stride = basis.export_terms(terms)
    words = np.frombuffer(words_b, dtype="<u8").astype(np.uint64).reshape(-1, stride)
    coeffs = np.frombuffer(coeffs_b, dtype="<c16").astype(np.complex128)
    n_sites = len(mps_arrays)
    for a in mps_arrays:
        if a.shape[1] != local_dim:
            raise ValueError(f"MPS physical dimension {a.shape[1]} does not match the basis local dimension {local_dim}")
    # A term must not act on units beyond the MPS.
    used = n_sites * bits
    if words.size and stride * 64 > used:
        for w in range(stride):
            lo = w * 64
            if lo + 64 <= used:
                continue
            keep = max(0, used - lo)
            mask = np.uint64((1 << keep) - 1)
            if np.any(words[:, w] & ~mask):
                raise ValueError(f"the observable acts on more than the {n_sites} units of the MPS")
    return complex(hybrid_expectation_raw(words, coeffs, bits, local_dim, table, mps_arrays))


def hybrid_expectation_value(
    propagated_observable: AbstractTermSum,
    circuit2: QuantumCircuit | MatrixProductState,
    initial_state: int = 0,
    *,
    basis=None,
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
        basis: A module made by `propaq.tracer.load_basis` for a basis with
            local matrices. Pass it when *propagated_observable* is a term sum of that
            module. The MPS then needs one physical index of dimension
            `local_dim` for each unit, so *circuit2* must be an MPS. Leave it
            `None` for Pauli and Majorana term sums.

    Returns:
        The real expectation value. For a basis given by *basis* with
        a non-Hermitian observable the imaginary part is dropped; use
        `_generic_expectation` for the complex value.
    """
    if basis is not None:
        if not _is_mps(circuit2):
            raise TypeError("a generic basis needs an MPS for the Schrodinger half")
        value = _generic_expectation(propagated_observable, basis, _normalize_mps_arrays(circuit2))
        return float(value.real)
    theta = _to_pauli_term_sum(propagated_observable)
    mps = circuit2 if _is_mps(circuit2) else _build_mps(circuit2, initial_state)
    mps_arrays = _normalize_mps_arrays(mps)
    return hybrid_expectation(theta, mps_arrays)
