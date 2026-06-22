"""
# propaq

[![CI](https://github.com/hkbelagali/propaq/actions/workflows/workflow.yml/badge.svg)](https://github.com/hkbelagali/propaq/actions/workflows/workflow.yml)
[![PyPI](https://img.shields.io/pypi/v/propaq)](https://pypi.org/project/propaq/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/hkbelagali/propaq/blob/main/LICENSE)

Fast Heisenberg-picture propagation for quantum circuit simulation, with a parallel Rust backend!

Implements the algorithms described in: 

>  M. S. Rudolph, T. Jones, Y. Teng, A. Angrisani, and Z. Holmes, “Pauli Propagation: A Computational Framework for Simulating Quantum Systems,” May 27, 2025, arXiv: arXiv:2505.21606. doi: 10.48550/arXiv.2505.21606.

>  A. Miller et al., "Simulation of Fermionic circuits using Majorana Propagation," Dec. 16, 2025, arXiv: arXiv:2503.18939. doi: 10.48550/arXiv.2503.18939.

## Installation

```bash
pip install propaq
```

Requires Python 3.10+ and a pre-built wheel for your platform (Linux x86-64, macOS, Windows).

For development, clone the repo and install with: 
```bash
pip install -e .[dev]   
```
which will install the Rust toolchain and build the Rust backend from source via `maturin`. You can also build the Rust backend manually with 
`maturin develop` or `maturin build` and then install the resulting wheel with `pip install dist/propaq-*.whl`.

## Quick start

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate, RZGate, SwapGate
from qiskit.quantum_info import SparsePauliOp

from propaq.circuits import MajoranaCircuit
from propaq.datatypes import MajoranaTermSum
from propaq.noise import UniformNoiseModel, TruncationPolicy
from propaq.propagators import MajoranaPropagator

# Build a random 4-qubit circuit
qc = QuantumCircuit(4)
qc.append(XXPlusYYGate(np.pi / 4, 0.0), [0, 1])
qc.append(RZGate(np.pi / 3), [2])
qc.append(SwapGate(), [2, 3])

# Observable: sum of single-site X operators
observable = SparsePauliOp.from_list([
    ("XIII", 1.0), ("IXII", 1.0), ("IIXI", 1.0), ("IIIX", 1.0)
])

# Convert to propaq types
mc = MajoranaCircuit.from_qiskit(qc, n_modes=2 * qc.num_qubits)
mts = MajoranaTermSum.from_sparse_pauli_op(observable)

# Build propagator with noise and truncation
prop = MajoranaPropagator(
    noise=UniformNoiseModel(damping=0.001),
    truncation=TruncationPolicy(weight_cutoff=10, coeff_cutoff=1e-5, truncation_range=(None, 10_000_000)),
)

# Back-propagate and evaluate
result = prop.expectation_value(mts, mc, fock_state=0)
print("Expectation value:", result.expectation_value)
```

For a more detailed introduction, see the example notebooks in the documentation.

## Documentation

Full API reference and example notebooks: https://hkbelagali.github.io/propaq

## Citation

If you use propaq in your research, please cite:

```bibtex
```

## License
```md
MIT License

Copyright (c) 2026 Hrishikesh Belagali

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
"""

from ._rust_core import Logger as Logger
from .circuits import (
    MajoranaCircuit as MajoranaCircuit,
)
from .circuits import (
    MajoranaRotation as MajoranaRotation,
)
from .circuits import (
    PauliCircuit as PauliCircuit,
)
from .circuits import (
    PauliRotation as PauliRotation,
)
from .datatypes import (
    MajoranaMonomial as MajoranaMonomial,
)
from .datatypes import (
    MajoranaTermSum as MajoranaTermSum,
)
from .datatypes import (
    PauliString as PauliString,
)
from .datatypes import (
    PauliTermSum as PauliTermSum,
)
from .extrapolators import ZeroNoiseExtrapolator as ZeroNoiseExtrapolator
from .extrapolators import ZNEResult as ZNEResult
from .log_parser import GateEvent as GateEvent
from .log_parser import LogParser as LogParser
from .log_parser import TruncationEvent as TruncationEvent
from .propagators import (
    MajoranaPropagator as MajoranaPropagator,
)
from .propagators import (
    PauliPropagator as PauliPropagator,
)
