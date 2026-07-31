# propaq Plugin ABI
propaq can load custom noise models and truncation policies from 
dynamically loaded C, Rust, or AOT-compiled Julia shared libraries. 
Functionality exists for Python implementations as well. However, this is not recommended, since both noise and truncation are hot in the propagator's inner loop and incur GIL overhead. The plugin ABI is 
designed to be as fast as possible, while allowing for maximum
flexibility in the implementation of noise and truncation policies.
See `examples/plugins` for example implementations in C, Rust, and Julia.

Each language directory is split into `noise/` and `truncation/`
subdirectories. Rust plugins are each a standalone cdylib crate under
`rust/{noise,truncation}/<name>/` (own `Cargo.toml`); Julia plugins are
AOT-compiled via `PackageCompiler.jl`, and the resulting shared object is
treated as a C ABI library by propaq.

`uniform_noise` and `weight_truncator` each implement the same formula as
a built-in model, so their output is directly diffable against it:

| Kind      | Built-in equivalent | C                                  | Rust                             | Julia                                 |
|-----------|----------------------|-------------------------------------|-----------------------------------|------------------------------------------|
| Noise     | `UniformNoiseModel`  | `c/noise/uniform_noise.c`          | `rust/noise/uniform_noise/`      | `julia/noise/uniform_noise.jl`         |
| Truncator | `WeightTruncator`    | `c/truncation/weight_truncator.c`  | `rust/truncation/weight_truncator/` | `julia/truncation/weight_truncator.jl` |

The remaining plugins are custom policies with no built-in
equivalent: each one is implemented identically across all three
languages (config keys and all), so they're cross-language diffable
instead: given the same config and the same call sequence, C, Rust, and
Julia produce bit-identical output.

| Kind      | Policy                 | Formula                                                                         | Config keys                     | C                                        | Rust                                        | Julia                                          |
|-----------|-------------------------|----------------------------------------------------------------------------------|----------------------------------|--------------------------------------------|------------------------------------------------|---------------------------------------------------|
| Noise     | `thermal_decay_noise`  | `exp(-(gamma * weight)^beta)` (stretched exponential; `beta=1` = plain exponential) | `gamma`, `beta`                 | `c/noise/thermal_decay_noise.c`          | `rust/noise/thermal_decay_noise/`             | `julia/noise/thermal_decay_noise.jl`             |
| Noise     | `drifting_noise`       | `exp(-damping * weight * (1 + drift_rate * call_index))` (grows across calls)     | `damping`, `drift_rate`         | `c/noise/drifting_noise.c`               | `rust/noise/drifting_noise/`                  | `julia/noise/drifting_noise.jl`                  |
| Truncator | `pareto_truncator`     | keep iff `coeff_magnitude * exp(-alpha * weight) > threshold` (joint score)        | `threshold`, `alpha`            | `c/truncation/pareto_truncator.c`        | `rust/truncation/pareto_truncator/`           | `julia/truncation/pareto_truncator.jl`           |
| Truncator | `stochastic_truncator` | keep with probability `min(1, coeff_magnitude / threshold)` (importance sampling) | `threshold`, `seed`              | `c/truncation/stochastic_truncator.c`    | `rust/truncation/stochastic_truncator/`       | `julia/truncation/stochastic_truncator.jl`       |

`drifting_noise` and `stochastic_truncator` both hold real mutable ctx
state (a call counter, reserved atomically rather than behind a lock); see
[Safety](#safety) below for why that matters here specifically.

`drifting_noise`'s `call_index` counts per-term damping calls, not gate
layers, `drift_rate` needs to be picked
accordingly small (the plugin's default is `1e-5`, deliberately tiny);
tuning it as if `call_index` tracked circuit depth saturates
`damping_factor` to a fully-underflowed `0.0` well before a real circuit
finishes.

## Noise ABI

Fixed `extern "C"` symbol names **(C-stable types only, u32/f64/pointers/usize)**:

```c
uint32_t propaq_noise_abi_version(void);                 // required, must equal 1
void*    propaq_noise_create(const char* config_json);   // optional
void     propaq_noise_destroy(void* ctx);                // required iff create is present
double   propaq_noise_damping_factor(void* ctx, uint32_t term_weight, uint32_t active_modes);  // required
int32_t  propaq_noise_damping_batch(void* ctx, const uint32_t* term_weights,
                                     const uint32_t* active_modes, double* out, size_t n);      // optional, faster
```

```python
from propaq.noise import NativeNoiseModel

model = NativeNoiseModel("examples/plugins/c/noise/uniform_noise.so", config='{"damping": 0.001}')
propagator.set_noise(model)
```

## Truncator ABI

```c
uint32_t propaq_truncator_abi_version(void);                 // required, must equal 1
void*    propaq_truncator_create(const char* config_json);   // optional
void     propaq_truncator_destroy(void* ctx);                // required iff create is present
int32_t  propaq_truncator_keep(void* ctx, uint32_t term_weight, double coeff_magnitude,
                                uint32_t active_modes);       // required; nonzero = keep
int32_t  propaq_truncator_keep_batch(void* ctx, const uint32_t* term_weights,
                                      const double* coeff_magnitudes, const uint32_t* active_modes,
                                      uint8_t* out_keep, size_t n);  // optional, faster
```

`coeff_magnitude` is the term's real coefficient magnitude
(`f64::abs`). A native truncator fully replaces the built-in
weight/coefficient cutoff comparison for the run rather than composing
with it.

```python
from propaq.truncation import NativeTruncator

trunc = NativeTruncator("examples/plugins/c/truncation/weight_truncator.so", config='{"max_weight": 4}')
propagator.set_truncation(trunc)
```

## The batch entry points

`propaq_noise_damping_batch` / `propaq_truncator_keep_batch` are an
optional performance path: propaq calls them once per parallel chunk
of terms instead of once per term when present, amortizing the FFI
boundary cost.

## Safety

Loading a plugin runs unsandboxed native code in-process. Only load
libraries you trust, the same way you'd trust any other compiled
dependency you link against. Plugin functions are called concurrently
from arbitrary worker threads sharing one `ctx` pointer. As a result,
 plugin state must tolerate concurrent reads, and plugin code must never panic,
unwind, or `longjmp` across the call boundary.

This is more than a theoretical concern once a plugin holds *mutable*
state, not just read-only config: `drifting_noise` and
`stochastic_truncator` both need a per-call counter that every concurrent caller advances. Each implements
it as a single lock-free atomic counter (`_Atomic uint64_t` in C,
`AtomicU64` in Rust, `Threads.Atomic{UInt64}` in Julia) reserved via one
fetch-add per call, never a plain non-atomic field, and never a full mutable
RNG state guarded by nothing. A plain increment here would race under
concurrent worker threads and silently corrupt the sequence.

**Race-free is not the same as order-deterministic.** The engine splits a
term array into `rayon::current_num_threads()` chunks and calls the batch
entry point once per chunk; which chunk's call actually reaches the
atomic fetch-add first depends on rayon's work-stealing scheduler, not on
chunk index. Measured directly (`examples/plugins/notebooks/01_c_plugins.ipynb`):
`drifting_noise`'s output changes not just across `n_threads`
values but from run to run at the *same* `n_threads > 1`, because which
physical term ends up with the smallest `call_index` isn't fixed. `stochastic_truncator` happened to come out
stable on that notebook's test circuit, but that's a property of Monte
Carlo aggregation smoothing out per-term reassignment, not a guarantee.