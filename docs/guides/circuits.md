# Circuits and gates

propaq exposes tools to build circuits natively, either in terms of [Pauli][propaq.circuits.PauliCircuit] or [Majorana][propaq.circuits.MajoranaCircuit] generators.
However, it's much easier to build them in Qiskit (or
Cirq) and convert them into the representation that matches the basis you want
to propagate in. For the case of Majorana circuits, where one generally does not want 
a spin-qubit representation, we also provide functionality to build circuits from
ffsim and openfermion.

!!! note "Pennylane and Catalyst"

    We plan to support Pennylane and qchem in the future. 

| Numerical | Surrogate (parameterized) |
| --- | --- |
| [`PauliCircuit`][propaq.circuits.PauliCircuit] | [`SurrogatePauliCircuit`][propaq.circuits.SurrogatePauliCircuit] |
| [`MajoranaCircuit`][propaq.circuits.MajoranaCircuit] | [`SurrogateMajoranaCircuit`][propaq.circuits.SurrogateMajoranaCircuit] |

```python
from qiskit import QuantumCircuit
from propaq.circuits import MajoranaCircuit, PauliCircuit

qc = QuantumCircuit(4)
...

pc = PauliCircuit.from_qiskit(qc)
mc = MajoranaCircuit.from_qiskit(qc, n_modes=2 * qc.num_qubits)
```

A Majorana circuit is defined over **modes**, not qubits: under Jordan–Wigner,
\(n\) qubits carry \(2n\) Majorana modes, hence `n_modes=2 * qc.num_qubits`.

## The native gate basis

Internally a circuit is a sequence of **rotations** - a
[`PauliRotation`][propaq.circuits.PauliRotation] or
[`MajoranaRotation`][propaq.circuits.MajoranaRotation], each a generator and an
angle. Gates that propaq recognises directly are converted into rotations with
no intermediate step:

=== "Qiskit"

    `xx_plus_yy`, `p`, `rz`, `rx`, `ry`, `cp`, `x`, `swap`

=== "Cirq"

    The `*PowGate` family

## The transpile fallback

Any gate outside that basis is decomposed automatically, via Qiskit's
transpiler, or `cirq.decompose` into gates that are recognized by propaq. This always
works, but it is not always cheap, as a gate that decomposes into many native
rotations multiplies the branching at that point in the circuit. 

When that happens propaq emits a
[`GateDecompositionWarning`][propaq.circuits.GateDecompositionWarning]. This is 
only exposed if you have the appropriate warnings filter set.

## Registering a custom gate

[`register_qiskit_gate`][propaq.circuits.register_qiskit_gate] and
[`register_cirq_gate`][propaq.circuits.register_cirq_gate] install a
generator-based decomposition for a gate, bypassing the transpile fallback:

CNOT is a good worked example.

```python
import math

from propaq.circuits import pauli_rotation_generator, register_qiskit_gate


def cnot_terms(instr_or_op, q_indices, width, rep):
    i, j = q_indices  # control, target
    n_qubits = rep.qubits_in_width(width)

    def label(axis_i, axis_j):
        chars = ["I"] * n_qubits
        if axis_i:
            chars[n_qubits - 1 - i] = axis_i
        if axis_j:
            chars[n_qubits - 1 - j] = axis_j
        return "".join(chars)

    terms = []
    for axis_i, axis_j, coeff in (
        ("Z", None, math.pi / 2),
        (None, "X", math.pi / 2),
        ("Z", "X", -math.pi / 2),
    ):
        gen, unit = pauli_rotation_generator(rep, label(axis_i, axis_j))
        terms.append((gen, coeff * unit))
    return [terms]


register_qiskit_gate("cx", cnot_terms)
```

The contract for `terms_fn`:

- **Signature.** `terms_fn(instr, q_indices, width, rep)` for Qiskit,
  `terms_fn(op, q_indices, width, rep)` for Cirq.
- **Return shape.** A `list[list[tuple[generator, angle]]]`, the same shape
  propaq's own built-in dispatch branches produce (see the `cp` case in
  `propaq.circuits._gates.gate_terms`, or the `ZZPowGate` case in
  `propaq.circuits._cirq_gates.cirq_gate_terms`).
- **Build generators with the helper.**
  [`pauli_rotation_generator`][propaq.circuits.pauli_rotation_generator] turns an
  \(n\)-qubit Pauli label such as `"XIZ"` into a `(generator, unit coefficient)`
  pair, and works for both the Pauli and Majorana representations. Pass through
  the same `rep` your `terms_fn` was handed.
- **Mind the label ordering.** Pauli labels follow Qiskit's convention, so qubit
  `i` is at string position `n_qubits - 1 - i`. Note also that `width` is the
  representation's internal width, not the qubit count - use
  `rep.qubits_in_width(width)` for the latter.
- **Stay parametric.** The function must be correct for whatever `width` and
  `q_indices` it is given, and must not hardcode absolute qubit positions.
  Recursive decomposition of sub-instructions relies on that.

Registration is validated by default (`validate=True`), which checks the
decomposition you supplied against the gate's actual unitary, by comparing to propaq's native decomposition pathway. Keep it on unless
you have a specific reason not to.

!!! note "Cirq matching is exact"

    `register_cirq_gate` matches on `type(op.gate)` exactly, not by
    `isinstance`. Registering a base class does **not** also cover its
    subclasses. Register each concrete type you care about.

A malformed registration raises
[`GateValidationError`][propaq.circuits.GateValidationError].

## Parameterized circuits

For a circuit carrying Qiskit `Parameter` objects, convert it to a surrogate
circuit instead, and feed that to a surrogate propagator:

```python
from propaq.circuits import SurrogatePauliCircuit

sc = SurrogatePauliCircuit.from_qiskit(parameterized_qc)
```

See the [surrogate guide](surrogate.md).

## Worked examples

- [Notebook 01 - Getting started](../examples/usage/01_getting_started.ipynb)
- [Notebook 09 - Custom gate registration](../examples/usage/09_custom_gate_registration.ipynb)
