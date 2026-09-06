"""A pure-Python propagator built on `AbstractPropagator` must reproduce `PauliPropagator`."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    CPhaseGate,
    PhaseGate,
    RZGate,
    SwapGate,
    XGate,
    XXPlusYYGate,
)
from qiskit.quantum_info import SparsePauliOp

from propaq.circuits import AbstractCircuit, PauliCircuit, PauliRotation
from propaq.datatypes import PauliString, PauliTermSum
from propaq.noise import UniformNoiseModel
from propaq.propagators import AbstractPropagator, PauliPropagator
from propaq.truncation import (
    CoefficientTruncator,
    TermBudget,
    TruncationPolicy,
    WeightTruncator,
)

GATES = [
    (
        lambda: XXPlusYYGate(np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi)),
        2,
    ),
    (lambda: PhaseGate(np.random.uniform(0, 2 * np.pi)), 1),
    (lambda: RZGate(np.random.uniform(0, 2 * np.pi)), 1),
    (lambda: CPhaseGate(np.random.uniform(0, 2 * np.pi)), 2),
    (lambda: SwapGate(), 2),
    (lambda: XGate(), 1),
]


class ToyPauliPropagator(AbstractPropagator[PauliString, PauliRotation]):
    """A from-scratch Pauli propagator; `apply_gate` is its only algebra-specific code."""

    def apply_gate(
        self, term: PauliString, coeff: complex, rotation: PauliRotation
    ) -> Iterable[tuple[PauliString, complex]]:
        """The standard rotation branching rule: unchanged if it commutes, else cos/sin split."""
        generator = rotation.generator
        if term.commutes_with(generator):
            yield term, coeff
            return
        sin_t, cos_t = math.sin(rotation.angle), math.cos(rotation.angle)
        phase, child = generator @ term
        yield term, coeff * cos_t
        yield child, coeff * sin_t * (-phase.imag)


def _random_circuit(
    n_qubits: int = 4, n_gates: int = 10, seed: int | None = None
) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)
    for _ in range(n_gates):
        factory, nq = GATES[rng.integers(len(GATES))]
        gate = factory()
        qubits = rng.choice(n_qubits, size=nq, replace=False).tolist()
        qc.append(gate, qubits)
    return qc


def _assert_matches(
    qc, observable, *, truncation=None, noise=None, initial_state=0, strict_counts=False
) -> None:
    pc = PauliCircuit.from_qiskit(qc)
    ref = PauliPropagator(noise=noise, truncation=truncation)
    toy = ToyPauliPropagator(noise=noise, truncation=truncation)

    ref_result = ref.expectation_value(observable, pc, initial_state=initial_state)
    toy_result = toy.expectation_value(observable, pc, initial_state=initial_state)

    assert np.isclose(ref_result.expectation_value, toy_result.expectation_value, atol=1e-9), (
        f"{ref_result.expectation_value} vs {toy_result.expectation_value}"
    )
    assert len(toy_result.n_terms) == len(ref_result.n_terms)
    if strict_counts:
        assert toy_result.n_terms == ref_result.n_terms, (
            f"per-gate term counts diverged: {ref_result.n_terms} vs {toy_result.n_terms}"
        )


_OBSERVABLE = SparsePauliOp("ZZZZ")


@pytest.mark.parametrize("seed", range(5))
def test_matches_with_no_policy(seed):
    qc = _random_circuit(seed=seed)
    observable = PauliTermSum.from_sparse_pauli_op(_OBSERVABLE)
    _assert_matches(qc, observable)


@pytest.mark.parametrize("seed", range(5))
def test_matches_with_weight_truncation(seed):
    qc = _random_circuit(seed=seed)
    observable = PauliTermSum.from_sparse_pauli_op(_OBSERVABLE)
    _assert_matches(qc, observable, truncation=WeightTruncator(weight=2), strict_counts=True)


@pytest.mark.parametrize("seed", range(5))
def test_matches_with_coefficient_truncation(seed):
    qc = _random_circuit(seed=seed)
    observable = PauliTermSum.from_sparse_pauli_op(_OBSERVABLE)
    _assert_matches(
        qc,
        observable,
        truncation=CoefficientTruncator(coefficient=1e-3),
        strict_counts=True,
    )


@pytest.mark.parametrize("seed", range(5))
def test_matches_with_term_budget(seed):
    qc = _random_circuit(seed=seed, n_gates=14)
    observable = PauliTermSum.from_sparse_pauli_op(_OBSERVABLE)
    _assert_matches(
        qc,
        observable,
        truncation=[WeightTruncator(weight=2), TermBudget(min_terms=8)],
        strict_counts=True,
    )


@pytest.mark.parametrize("seed", range(5))
def test_matches_with_noise(seed):
    qc = _random_circuit(seed=seed)
    observable = PauliTermSum.from_sparse_pauli_op(_OBSERVABLE)
    _assert_matches(qc, observable, noise=UniformNoiseModel(damping=0.05))


@pytest.mark.parametrize("seed", range(5))
def test_matches_with_truncation_policy(seed):
    qc = _random_circuit(seed=seed)
    observable = PauliTermSum.from_sparse_pauli_op(_OBSERVABLE)
    _assert_matches(
        qc,
        observable,
        truncation=TruncationPolicy(weight_cutoff=3, coeff_cutoff=1e-4),
        strict_counts=True,
    )


@pytest.mark.parametrize("seed", range(3))
def test_matches_with_noise_and_weight_truncation(seed):
    qc = _random_circuit(seed=seed, n_gates=12)
    observable = PauliTermSum.from_sparse_pauli_op(_OBSERVABLE)
    _assert_matches(
        qc,
        observable,
        truncation=WeightTruncator(weight=2),
        noise=UniformNoiseModel(damping=0.05),
    )


def test_gate_ordering_matches_reversed_layer_and_reversed_gates():
    """A two-gate, non-commuting layer must be applied in reverse *within* the layer too."""
    qc = QuantumCircuit(2)
    qc.rz(0.3, 0)
    qc.cx(0, 1)
    pc = PauliCircuit.from_qiskit(qc)
    # Force both rotations into one layer to exercise the inner reversal.
    flat = pc.rotations
    combined = PauliCircuit([flat])
    observable = PauliTermSum.from_sparse_pauli_op(SparsePauliOp("ZZ"))
    _assert_matches(qc, observable)  # sanity: from_qiskit's own layering
    ref = PauliPropagator()
    toy = ToyPauliPropagator()
    ref_val = ref.expectation_value(observable, combined).expectation_value
    toy_val = toy.expectation_value(observable, combined).expectation_value
    assert np.isclose(ref_val, toy_val, atol=1e-9)


def test_layers_of_accepts_flat_list_and_abstract_circuit():
    qc = QuantumCircuit(2)
    qc.rz(0.3, 0)
    qc.cx(0, 1)
    pc = PauliCircuit.from_qiskit(qc)
    flat = pc.rotations
    as_abstract = AbstractCircuit(pc.layers)

    toy = ToyPauliPropagator()
    observable = PauliTermSum.from_sparse_pauli_op(SparsePauliOp("ZZ"))

    a = toy.expectation_value(observable, flat).expectation_value
    b = toy.expectation_value(observable, as_abstract).expectation_value
    c = toy.expectation_value(observable, pc).expectation_value
    assert np.isclose(a, b, atol=1e-12)
    assert np.isclose(a, c, atol=1e-12)


def test_n_terms_has_one_entry_per_gate():
    qc = _random_circuit(seed=0, n_gates=7)
    pc = PauliCircuit.from_qiskit(qc)
    observable = PauliTermSum.from_sparse_pauli_op(_OBSERVABLE)
    toy = ToyPauliPropagator()
    result = toy.expectation_value(observable, pc)
    assert len(result.n_terms) == len(pc.rotations)


def test_apply_gate_is_the_only_abstract_method():
    with pytest.raises(TypeError):
        AbstractPropagator()  # type: ignore[abstract]

    class _OnlyApplyGate(AbstractPropagator):
        def apply_gate(self, term, coeff, rotation):
            yield term, coeff

    _OnlyApplyGate()  # must not raise


def test_expectation_value_returns_real_propagation_result():
    from propaq._rust_core import PropagationResult

    qc = _random_circuit(seed=1, n_gates=6)
    pc = PauliCircuit.from_qiskit(qc)
    observable = PauliTermSum.from_sparse_pauli_op(_OBSERVABLE)
    result = ToyPauliPropagator().expectation_value(observable, pc)
    assert isinstance(result, PropagationResult)


def test_propagate_and_filename_round_trip(tmp_path):
    qc = _random_circuit(seed=2, n_gates=6)
    pc = PauliCircuit.from_qiskit(qc)
    observable = PauliTermSum.from_sparse_pauli_op(_OBSERVABLE)
    toy = ToyPauliPropagator()

    path = str(tmp_path / "terms.bin.gz")
    evolved = toy.propagate(observable, pc, filename=path)
    reloaded = PauliTermSum.from_file(path)

    a = dict(evolved.items())
    b = dict(reloaded.items())
    assert a.keys() == b.keys()
    for key in a:
        assert np.isclose(a[key], b[key], atol=1e-12)
