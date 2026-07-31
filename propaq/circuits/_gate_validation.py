"""
Validates a custom-registered gate's terms_fn against propaq's own decomposition path for the same gate.
"""

from __future__ import annotations

import itertools
import math
import random
import zlib
from typing import TYPE_CHECKING, Any

from qiskit.circuit import Instruction
from qiskit.quantum_info import SparsePauliOp

from ..propagators import MajoranaPropagator, PauliPropagator
from ._gates import PAULI, _dispatch_native, _Rep
from .majorana.circuit import MajoranaCircuit
from .pauli.circuit import PauliCircuit

if TYPE_CHECKING:
    import cirq

    from ._registry import CirqTermsFn, QiskitTermsFn

_ATOL = 1e-9
_MAX_MISMATCHES = 5
_N_RANDOM_PARAM_SAMPLES = 2


class GateValidationError(Exception):
    """Raised when a custom-registered gate's terms_fn disagrees with propaq's own
    decomposition path for the same gate."""


def _rep_name(rep: _Rep) -> str:
    return "pauli" if rep is PAULI else "majorana"


def _nontrivial_pauli_labels(n: int) -> list[str]:
    return [
        "".join(chars)
        for chars in itertools.product("IXYZ", repeat=n)
        if any(c != "I" for c in chars)
    ]


def _embed_label(local_label: str, q_indices: list[int], n_qubits_total: int) -> str:
    label = ["I"] * n_qubits_total
    for axis, q in zip(local_label, q_indices):
        label[n_qubits_total - 1 - q] = axis
    return "".join(label)


def _build_circuit(rep: _Rep, groups: list[list[tuple[Any, Any]]], width: int) -> Any:
    gens: list[Any] = []
    angles: list[float] = []
    for group in groups:
        for gen, angle in group:
            gens.append(gen)
            angles.append(float(angle))
    if rep is PAULI:
        return PauliCircuit.from_generators_and_angles(gens, angles)
    return MajoranaCircuit.from_generators_and_angles(gens, angles, n_modes=width)


def _propagator_for(rep: _Rep) -> PauliPropagator | MajoranaPropagator:
    return PauliPropagator() if rep is PAULI else MajoranaPropagator()


def _compare_against_ground_truth(
    rep: _Rep,
    groups_registered: list[list[tuple[Any, Any]]],
    groups_ground_truth: list[list[tuple[Any, Any]]],
    q_indices: list[int],
    width: int,
    param_label: str,
    mismatches_out: list[str],
) -> None:
    """Propagates a spanning set of Pauli observables through both the registered and
    ground-truth decompositions and compares the resulting term sums.
    """
    n = len(q_indices)
    n_qubits_total = rep.qubits_in_width(width)
    circuit_registered = _build_circuit(rep, groups_registered, width)
    circuit_ground_truth = _build_circuit(rep, groups_ground_truth, width)
    propagator = _propagator_for(rep)

    for local_label in _nontrivial_pauli_labels(n):
        full_label = _embed_label(local_label, q_indices, n_qubits_total)
        obs = rep.termsum_cls.from_sparse_pauli_op(SparsePauliOp(full_label))
        ts_registered = propagator.propagate(obs, circuit_registered)  # type: ignore[arg-type]
        ts_ground_truth = propagator.propagate(obs, circuit_ground_truth)  # type: ignore[arg-type]

        coeffs_registered = dict(ts_registered.items())
        coeffs_ground_truth = dict(ts_ground_truth.items())
        for gen in set(coeffs_registered) | set(coeffs_ground_truth):
            got = coeffs_registered.get(gen, 0j)
            expected = coeffs_ground_truth.get(gen, 0j)
            if abs(got - expected) > _ATOL:
                mismatches_out.append(
                    f"params={param_label} observable={local_label!r} generator={gen!r}: "
                    f"expected {expected!r}, got {got!r}"
                )
                if len(mismatches_out) >= _MAX_MISMATCHES:
                    return


def _raise_if_mismatched(key: object, rep: _Rep, kind: str, mismatches: list[str]) -> None:
    if not mismatches:
        return
    details = "\n".join(mismatches)
    raise GateValidationError(
        f"propaq: custom terms_fn registered for {kind} gate {key!r} ({_rep_name(rep)} "
        f"representation) disagrees with propaq's own decomposition:\n{details}"
    )


def validate_qiskit_gate(
    key: str,
    terms_fn: QiskitTermsFn,
    instr: Instruction,
    q_indices: list[int],
    width: int,
    rep: _Rep,
) -> None:
    """Validates a registered Qiskit terms_fn against `_dispatch_native` for the actual
    dispatched instance, plus a couple of randomly-sampled parameter values (deterministic
    per gate name) when all of `instr.params` are plain numbers.

    Raises `GateValidationError` on any mismatch.
    """
    param_sets: list[tuple[Instruction, str]] = [(instr, "actual")]
    if instr.params and all(isinstance(p, int | float) for p in instr.params):
        rng = random.Random(zlib.crc32(repr(key).encode()))
        for i in range(_N_RANDOM_PARAM_SAMPLES):
            resampled = instr.copy()
            resampled.params = [rng.uniform(0, 2 * math.pi) for _ in instr.params]
            param_sets.append((resampled, f"random-{i}"))

    mismatches: list[str] = []
    for sample_instr, label in param_sets:
        groups_registered = terms_fn(sample_instr, q_indices, width, rep)
        groups_ground_truth = _dispatch_native(sample_instr, q_indices, width, rep)
        _compare_against_ground_truth(
            rep, groups_registered, groups_ground_truth, q_indices, width, label, mismatches
        )
        if mismatches:
            break

    _raise_if_mismatched(key, rep, "Qiskit", mismatches)


def validate_cirq_gate(
    key: type,
    terms_fn: CirqTermsFn,
    op: cirq.Operation,
    q_indices: list[int],
    width: int,
    rep: _Rep,
) -> None:
    """Validates a registered Cirq terms_fn against `_dispatch_native_cirq` for the actual
    dispatched instance. Unlike the Qiskit path, only the actual parameters are checked -
    there is no generic way to resample an arbitrary custom `cirq.Gate` subclass's
    parameters.

    Raises `GateValidationError` on any mismatch.
    """
    from ._cirq_gates import _dispatch_native_cirq

    mismatches: list[str] = []
    groups_registered = terms_fn(op, q_indices, width, rep)
    groups_ground_truth = _dispatch_native_cirq(op, q_indices, width, rep)
    _compare_against_ground_truth(
        rep, groups_registered, groups_ground_truth, q_indices, width, "actual", mismatches
    )

    _raise_if_mismatched(key, rep, "Cirq", mismatches)
