# propaq Plugin ABI
propaq can load custom noise models and truncation policies from 
dynamically loaded C, Rust, or AOT-compiled Julia shared libraries. 
Functionality exists for Python implementations as well. However, this is not recommended, since both noise and truncation are hot in the propagator's inner loop and incur GIL overhead. The plugin ABI is 
designed to be as fast as possible, while allowing for maximum
flexibility in the implementation of noise and truncation policies.
See `examples/plugins` for example implementations in C, Rust, and Julia.

Each subdirectory implements the same formula as a built-in model, so
its output is directly diffable against it:

| Kind       | Built-in equivalent | C                       | Rust               | Julia                     |
|------------|----------------------|-------------------------|---------------------|----------------------------|
| Noise      | `UniformNoiseModel`  | `c/uniform_noise.c`     | `rust/`             | `julia/uniform_noise.jl`  |
| Truncator  | `WeightTruncator`    | `c/weight_truncator.c`  | `rust_truncator/`   | `julia/weight_truncator.jl` |

Rust plugins are each a standalone cdylib crate (own `Cargo.toml`);
Julia plugins are AOT-compiled via `PackageCompiler.jl`, and the resulting 
shared object is treated as a C ABI library by propaq.

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

model = NativeNoiseModel("examples/plugins/c/uniform_noise.so", config='{"damping": 0.001}')
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

trunc = NativeTruncator("examples/plugins/c/weight_truncator.so", config='{"max_weight": 4}')
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
