# Quickstart

This page walks through one complete propagation: a Qiskit circuit, a Pauli
observable, a truncation pipeline, and an expectation value.

## 1. Build a circuit

propaq consumes Qiskit (and optionally Cirq) circuits. Convert one into the
representation matching the basis you want to propagate in:

=== "Majorana"

    ```python
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate, RZGate, SwapGate

    from propaq.circuits import MajoranaCircuit

    qc = QuantumCircuit(4)
    qc.append(XXPlusYYGate(np.pi / 4, 0.0), [0, 1])
    qc.append(RZGate(np.pi / 3), [2])
    qc.append(SwapGate(), [2, 3])

    circuit = MajoranaCircuit.from_qiskit(qc, n_modes=2 * qc.num_qubits)
    ```

=== "Pauli"

    ```python
    import numpy as np
    from qiskit import QuantumCircuit

    from propaq.circuits import PauliCircuit

    qc = QuantumCircuit(4)
    qc.rx(np.pi / 4, 0)
    qc.cx(0, 1)
    qc.rz(np.pi / 3, 2)

    circuit = PauliCircuit.from_qiskit(qc)
    ```

Gates outside propaq's native basis are transpiled automatically. If a gate you
use often is being decomposed into many native rotations, register a fast path
for it. See [Circuits and gates](../guides/circuits.md).

## 2. Choose an observable

An observable is a *term sum*: a weighted sum of Pauli strings or Majorana
monomials. The easiest way in is from a Qiskit `SparsePauliOp`:

```python
from qiskit.quantum_info import SparsePauliOp
from propaq.datatypes import MajoranaTermSum

observable = SparsePauliOp.from_list([
    ("XIII", 1.0),
    ("IXII", 1.0),
    ("IIXI", 1.0),
    ("IIIX", 1.0),
])

mts = MajoranaTermSum.from_sparse_pauli_op(observable)
```

## 3. Pick a truncation pipeline
 A propagator takes a **list of truncators** that run together
 after each gate, or according to a [`TermBudget`][propaq.truncation.TermBudget] if specified.

```python
from propaq.truncation import WeightTruncator, CoefficientTruncator, TermBudget

truncation = [
    WeightTruncator(weight=10),            # drop terms of operator weight > 10
    CoefficientTruncator(coefficient=1e-5),  # drop |coeff| < 1e-5
    TermBudget(max_terms=10_000_000),      # fire truncation at 10M live terms
]
```

!!! note "`TermBudget` arguments are keyword-only."


The [truncation guide](../guides/truncation.md) explains what each truncator
costs and when it fires.

## 4. Propagate

```python
from propaq.noise import UniformNoiseModel
from propaq.propagators import MajoranaPropagator

prop = MajoranaPropagator(
    noise=UniformNoiseModel(damping=0.001),
    truncation=truncation,
)

result = prop.expectation_value(mts, circuit, initial_state=0)

print("expectation value:", result.expectation_value)
print("terms at each gate:", result.n_terms)      # per-gate trace, not a total
print("final term count: ", result.n_terms[-1])
print("below cutoff:     ", result.terms_below_cutoff)
```

!!! note "`n_terms` is a trace, not a number"

    [`PropagationResult.n_terms`][propaq._rust_core.PropagationResult] is a
    `list[int]`. It represents the live term count after each gate, so you can see where
    branching took off. Use `n_terms[-1]` for the final count.

[`expectation_value`][propaq.propagators.MajoranaPropagator.expectation_value]
back-propagates the observable through the circuit and contracts the result
against the computational-basis state given by `initial_state`, represented as a bitstring. It returns a
[`PropagationResult`][propaq._rust_core.PropagationResult].

If you want the propagated operator itself rather than a number, use
[`propagate`][propaq.propagators.MajoranaPropagator.propagate], which returns
the evolved `MajoranaTermSum`:

```python
theta = prop.propagate(mts, circuit)
print(theta.norm_squared())
```

Both methods accept `filename=`, which writes the final terms to a
gzip-compressed binary file you can reload lazily later. See
[Streaming and I/O](../guides/streaming.md).

## Where to go next

<div class="grid cards" markdown>

-   __Understand the model__

    [:octicons-arrow-right-24: Core concepts](concepts.md)

-   __Keep the term count bounded__

    [:octicons-arrow-right-24: Truncation](../guides/truncation.md)

-   __Sweep circuit parameters cheaply__

    [:octicons-arrow-right-24: Surrogate propagation](../guides/surrogate.md)

-   __See it all working end to end__

    [:octicons-arrow-right-24: Example notebooks](../examples/index.md)

</div>
