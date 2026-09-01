---
title: propaq
hide:
  - navigation
  - toc
---

<div class="pq-hero" markdown>

# propaq { .pq-hero-title }

### Fast Heisenberg-picture propagation for quantum circuit simulation. { .pq-hero-tagline }

[Get started :material-arrow-right:](getting-started/index.md){ .md-button .md-button--primary }
[Browse examples](examples/index.md){ .md-button }
[API reference](reference/index.md){ .md-button }

<!-- Badges are set flat-square in a single neutral/violet range so the masthead
     keeps one palette; shields.io's defaults are green/yellow/blue per badge. -->
<p class="pq-badges">
<a href="https://github.com/hkbelagali/propaq/actions/workflows/workflow.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/hkbelagali/propaq/workflow.yml?style=flat-square&label=CI&color=6d28d9"></a>
<a href="https://pypi.org/project/propaq/"><img alt="PyPI" src="https://img.shields.io/pypi/v/propaq?style=flat-square&color=6d28d9"></a>
<a href="https://pypi.org/project/propaq/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/propaq?style=flat-square&color=52525b"></a>
<a href="https://github.com/hkbelagali/propaq/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-52525b?style=flat-square"></a>
</p>

</div>

---

```python title="Back-propagate an observable through a fermionic circuit"
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate, RZGate, SwapGate
from qiskit.quantum_info import SparsePauliOp

from propaq.circuits import MajoranaCircuit
from propaq.datatypes import MajoranaTermSum
from propaq.noise import UniformNoiseModel
from propaq.propagators import MajoranaPropagator
from propaq.truncation import WeightTruncator, CoefficientTruncator, TermBudget

qc = QuantumCircuit(4)
qc.append(XXPlusYYGate(np.pi / 4, 0.0), [0, 1])
qc.append(RZGate(np.pi / 3), [2])
qc.append(SwapGate(), [2, 3])

observable = SparsePauliOp.from_list([("XIII", 1.0), ("IXII", 1.0)])

prop = MajoranaPropagator(
    noise=UniformNoiseModel(damping=0.001),
    truncation=[
        WeightTruncator(weight=10),
        CoefficientTruncator(coefficient=1e-5),
        TermBudget(min_terms=1_000_000),
    ],
)

result = prop.expectation_value(
    MajoranaTermSum.from_sparse_pauli_op(observable),
    MajoranaCircuit.from_qiskit(qc, n_modes=2 * qc.num_qubits),
    initial_state=0,
)
print(result.expectation_value)
```

## What propaq does

<div class="grid cards" markdown>

-   __Pauli and Majorana propagation__

    ---

    Back-propagate an observable through a circuit in the Heisenberg picture,
    in either the Pauli or the Majorana basis. Implement custom bases by subclassing 
    propaq's Python abstract classes.

-   __Composable truncation__

    ---

    Weight, coefficient and term-budget truncators can be composed 
    for flexible control over accuracy and memory usage.

-   __Surrogate propagation__

    ---

    Compile a parameterized circuit into a symbolic model once, then
    evaluate expectation values for any parameter assignment at significantly lower cost, designed for variational algorithms.

-   __Native plugin ABI__

    ---

    Write custom noise models and truncation policies in C, Rust or
    AOT-compiled Julia and load them as shared libraries, allowing for 
    low-overhead customization of the propagation engine.

-   __Extrapolation__

    ---

    Zero-noise and zero-cutoff extrapolators recover an estimate of the
    noiseless, untruncated expectation value from a sweep of runs.

-   __I/O, Storage & Logging__

    ---
    Save and load propagated observables, lazy iteration of 
    terms from disk, and logging of truncation statistics and 
    key performance metrics.


</div>

## Install

<div class="pq-install" markdown>

=== "pip"

    ```bash
    pip install propaq
    ```

=== "with extras"

    ```bash
    pip install "propaq[cirq,openfermion,hybrid]"
    ```

=== "from source"

    ```bash
    git clone https://github.com/hkbelagali/propaq
    cd propaq
    pip install -e ".[dev]"
    ```

</div>

Requires Python 3.10 or newer. Pre-built wheels are published for Linux x86-64,
macOS and Windows. See [Installation](getting-started/installation.md) for more 
information.

## References

propaq implements the algorithms described in:

!!! quote "Pauli propagation"

    M. S. Rudolph, T. Jones, Y. Teng, A. Angrisani, and Z. Holmes,
    *"Pauli Propagation: A Computational Framework for Simulating Quantum Systems,"*
    May 27, 2025. [arXiv:2505.21606](https://arxiv.org/abs/2505.21606)

!!! quote "Majorana propagation"

    A. Miller et al.,
    *"Simulation of Fermionic circuits using Majorana Propagation,"*
    Dec. 16, 2025. [arXiv:2503.18939](https://arxiv.org/abs/2503.18939)
