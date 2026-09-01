# Installation

propaq requires **Python 3.10 or newer**.

## From PyPI

```bash
pip install propaq
```

Pre-built wheels are published for Linux x86-64, macOS and Windows, for CPython
3.10, 3.11 and 3.12. A Rust toolchain is not required for this install.

## Optional extras

The core install pulls in `numpy`, `scipy`, `qiskit` and `tqdm`. A number of optional extras are available 
for framework integration and certain features.

| Extra | Install | Enables |
| --- | --- | --- |
| `cirq` | `pip install "propaq[cirq]"` | Building circuits from Cirq, and [`register_cirq_gate`][propaq.circuits.register_cirq_gate] |
| `ffsim` | `pip install "propaq[ffsim]"` | The `from_ffsim_*` circuit constructors and [`MajoranaTermSum.from_ffsim`][propaq.datatypes.MajoranaTermSum.from_ffsim] |
| `openfermion` | `pip install "propaq[openfermion]"` | Converting OpenFermion fermionic operators into propaq observables |
| `hybrid` | `pip install "propaq[hybrid]"` | [`propaq.hybrid`][propaq.hybrid] - hybrid Schrödinger–Heisenberg expectation values against a `quimb` MPS |
| `examples` | `pip install "propaq[examples]"` | Everything needed to run the [example notebooks](../examples/index.md): `cirq`, `ffsim`, `hybrid`, plus `matplotlib`, `qiskit-nature`, `jupyter`, `ipywidgets` |
| `dev` | `pip install "propaq[dev]"` | `pytest`, `ruff`, `mypy`, `coverage`, `maturin` |
| `docs` | `pip install "propaq[docs]"` | This documentation site |

```bash
pip install "propaq[cirq,ffsim,openfermion,hybrid]"
```
## From source

Building from source requires a [Rust toolchain](https://rustup.rs). The
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

!!! warning "Performance-critical builds"

    A debug build of the Rust core is roughly an order of magnitude slower than
    a release build. Performance benchmarks and production runs should always use `--release`. 
    Additionally, building from source will compile the Rust backend for your current CPU architecture.
    In HPC environments, this necessitates building on the target machine directly, as the package 
    will not run on a different CPU architecture than the one it was built for.

## Verifying the install

```python
import propaq

print(propaq.__version__)
```

## Thread count, BLAS, and CPU Pinning

We have observed that the default OpenBLAS thread count can 
cause performance regressions in some environments. If you observe that 
the propagation engine is running slower than expected, try pinning OpenBLAS to a single thread: 

```python
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # before numpy is imported
```

Additionally, the Rust backend pins each thread to a single CPU core by default. This is to 
maintain good cache locality and avoid performance loss due to thread migration. If you wish to disable this 
behavior, disable `pin_threads` in [`PauliPropagator`][propaq.propagators.PauliPropagator] or [`MajoranaPropagator`][propaq.propagators.MajoranaPropagator].