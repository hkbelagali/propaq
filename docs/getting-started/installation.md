# Installation

propaq requires **Python 3.10 or newer**.

## From PyPI

```bash
pip install propaq
```

Pre-built wheels are published for Linux x86-64, macOS and Windows, for CPython
3.10, 3.11 and 3.12. Installing from a wheel needs no Rust toolchain.

## Optional extras

The core install pulls in `numpy`, `scipy`, `qiskit`, `ffsim` and `tqdm`.
Everything else is opt-in:

| Extra | Install | Enables |
| --- | --- | --- |
| `cirq` | `pip install "propaq[cirq]"` | Building circuits from Cirq, and [`register_cirq_gate`][propaq.circuits.register_cirq_gate] |
| `openfermion` | `pip install "propaq[openfermion]"` | Converting OpenFermion fermionic operators into propaq observables |
| `hybrid` | `pip install "propaq[hybrid]"` | [`propaq.hybrid`][propaq.hybrid] - hybrid Schrödinger–Heisenberg expectation values against a `quimb` MPS |
| `dev` | `pip install "propaq[dev]"` | `pytest`, `ruff`, `mypy`, `coverage`, `maturin` |
| `docs` | `pip install "propaq[docs]"` | This documentation site |

Extras compose, so:

```bash
pip install "propaq[cirq,openfermion,hybrid]"
```

## From source

Building from source requires a [Rust toolchain](https://rustup.rs); the
extension module is compiled by [maturin](https://www.maturin.rs).

```bash
git clone https://github.com/hkbelagali/propaq
cd propaq
pip install -e ".[dev]"
```

To rebuild the Rust backend after changing Rust sources:

=== "Development build"

    ```bash
    maturin develop
    ```

=== "Optimised build"

    ```bash
    maturin develop --release
    ```

=== "Wheel"

    ```bash
    maturin build --release
    pip install target/wheels/propaq-*.whl
    ```

!!! warning "Use `--release` for anything you intend to time"

    A debug build of the Rust core is roughly an order of magnitude slower than
    a release build. Performance benchmarks and production runs should always use `--release`. 

## Verifying the install

```python
import propaq
from propaq.propagators import PauliPropagator

print(PauliPropagator())
```

## Thread count and BLAS

propaq's propagators are multi-threaded and, by default, **pin each worker
thread to its own CPU**. This is to maintain good cache performance. 

Pinning interacts badly with threaded BLAS. A pinned propaq worker
cannot step around a spinning OpenBLAS thread, and importing `qiskit` starts one
spinner per core. If you run propaq in the same process as heavy NumPy/Qiskit
linear algebra, either set

```python
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # before numpy is imported
```

or disable pinning per propagator:

```python
PauliPropagator(pin_threads=False)
```

Setting `OPENBLAS_NUM_THREADS=1` is the better fix. See
[`PauliPropagator`][propaq.propagators.PauliPropagator] for the details.
