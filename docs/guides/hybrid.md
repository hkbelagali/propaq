For a circuit \(C = C_1 C_2\), the expectation value can be split into a Heisenberg
half and a Schrödinger half:

\[
\langle \Psi_0 | C^\dagger O C | \Psi_0 \rangle
= \langle \Psi | \left( C_1^\dagger O C_1 \right) | \Psi \rangle,
\qquad | \Psi \rangle = C_2 | \Psi_0 \rangle .
\]

Propagate \(O\) through \(C_1\) with propaq, build \(|\Psi\rangle\) as a matrix
product state with `quimb`, and contract the two in one native call:

```python
from propaq.hybrid import hybrid_expectation_value

theta = prop.propagate(observable, circuit1)
value = hybrid_expectation_value(theta, circuit2, initial_state=0)
```

`circuit2` may be a plain Qiskit `QuantumCircuit`, in which case propaq builds
the MPS for you from `initial_state`. It can also be an already-built `quimb`
`MatrixProductState` representing \(|\Psi\rangle\) directly, if you want control
over the bond dimension and compression.

This requires the `hybrid` extra:

```bash
pip install "propaq[hybrid]"
```

For certain circuits with naturally bipartite structure, this can be a more efficient and accurate way to compute 
expectation values than in either picture individually.

## Worked examples
- [Notebook 10 - Hybrid Schrödinger–Heisenberg](../examples/usage/10_hybrid_schrodinger_heisenberg.ipynb)

