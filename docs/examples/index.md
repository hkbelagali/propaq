# Examples

Every page in this section is a Jupyter notebook from the
[`examples/`](https://github.com/hkbelagali/propaq/tree/main/examples) directory
of the repository, rendered here with its committed outputs.

## Usage

Start at the top, the notebooks build on each other!

<div class="grid cards" markdown>

-   __[1. Getting started](usage/01_getting_started.ipynb)__

    ---

    The basics end to end: build a circuit, choose an observable, propagate,
    read off an expectation value.

-   __[2. Pauli vs Majorana propagation](usage/02_pauli_vs_majorana.ipynb)__

    ---

    The same unitary cluster Jastrow (UCJ) ansatz for $\text{H}_2$ propagated in both
    bases, compared directly.

-   __[3. Truncation pipelines](usage/03_truncation_pipelines.ipynb)__

    ---

    Merging versus truncation, and how to compose weight, coefficient and
    budget truncators to keep branching under control.

-   __[4. Variational quantum circuits](usage/04_variational_surrogate.ipynb)__

    ---

    Compile a parameterized ansatz into a surrogate model once, then drive it
    with a SciPy optimiser as a cost function.

-   __[5. Surrogate model persistence](usage/05_surrogate_persistence.ipynb)__

    ---

    Save a compiled surrogate to disk and reload it in another process, paying
    the build cost only once.

-   __[6. Term streaming](usage/06_term_streaming.ipynb)__

    ---

    Write a propagated term sum to disk and read it back lazily, one term at a
    time, when it is too large to hold in memory.

-   __[7. Zero cutoff extrapolation](usage/07_zero_cutoff_extrapolation.ipynb)__

    ---

    Sweep a truncation cutoff and extrapolate to zero to estimate the
    untruncated expectation value.

-   __[8. Zero noise extrapolation](usage/08_zero_noise_extrapolation.ipynb)__

    ---

    The same idea applied to the damping rate, recovering the noiseless
    expectation value from a sweep of noisy runs.

-   __[9. Custom gate registration](usage/09_custom_gate_registration.ipynb)__

    ---

    Replace the transpile-based fallback for a gate with a hand-written
    generator decomposition, and let propaq validate it.

-   __[10. Hybrid Schrödinger–Heisenberg](usage/10_hybrid_schrodinger_heisenberg.ipynb)__

    ---

    Split a circuit in two, propagate one half and build the other as an MPS,
    then contract them in one native call.

-   __[11. Propagating in a custom basis](usage/11_custom_basis_qudit_weyl.ipynb)__

    ---

    Implement the qudit Weyl–Heisenberg basis from scratch against propaq's
    abstract interfaces, and check it against exact dense simulation, a
    closed-form solution, and propaq's own Pauli propagator.

</div>

## Native plugins

Writing noise models and truncation policies against the
[plugin ABI](../guides/plugins.md), in each supported language.

<div class="grid cards" markdown>

-   __[C plugins](plugins/01_c_plugins.ipynb)__

    ---

    Build every C plugin under `examples/plugins/c/`, load each through
    `NativeNoiseModel` / `NativeTruncator`, and check it against its built-in
    equivalent.

-   __[Rust plugins](plugins/02_rust_plugins.ipynb)__

    ---

    The Rust counterpart: each plugin is a standalone `cdylib` crate.

-   __[Julia AOT plugins](plugins/03_julia_aot_plugins.ipynb)__

    ---

    Julia plugins AOT-compiled with `PackageCompiler.jl`, loaded by propaq as
    plain C ABI libraries.

-   __[Batch ABI benchmark](plugins/04_batch_abi_benchmark.ipynb)__

    ---

    Measures the benefit of amortizing FFI overhead by batching calls to a plugin.

</div>

!!! info "These notebooks are not executed at build time"

    The documentation renders each notebook's **committed** outputs rather than
    re-running it.
