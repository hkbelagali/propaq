# propaq Plugin ABI

propaq can load custom noise models and truncation policies from
dynamically loaded C, Rust, or AOT-compiled Julia shared libraries.
Functionality exists for Python implementations as well. However, this is not recommended, since both noise and truncation are hot in the propagator's inner loop and incur GIL overhead.
See `examples/plugins` for example implementations in C, Rust, and Julia.

Each language directory is split into `noise/` and `truncation/`
subdirectories. Rust plugins are each a standalone cdylib crate under
`rust/{noise,truncation}/<name>/` (own `Cargo.toml`); Julia plugins are
AOT-compiled via `PackageCompiler.jl`, and the resulting shared object is
treated as a C ABI library by propaq.

## Dependencies

A plugin specifies what information it reads from the engine using dependencies.
By default, a plugin is able to read only the term's weight and coefficient magnitude. If it needs to read the term's entire bitmask, or the layer index, it must declare those 
keys in its `depends` bitmask. This is meant to allow the engine to optimize the plugin's execution path and avoid unnecessary overhead.

```c
#define PROPAQ_DEPENDS_KEY    (1u << 0)   /* reads `words`                    */
#define PROPAQ_DEPENDS_LAYER  (1u << 1)   /* reads `layer_index` / `n_layers` */
```

Export `propaq_noise_depends` / `propaq_truncator_depends` returning the OR of
the bits you need. **Omitting the symbol means `0`**, a function of term weight.

### Dependency permissions and costs

**Noise:**

| `depends` | Strategy | `words` | Clifford deferral |
|---|---|---|---|
| `0` | tabulated once at setup (`n_units + 1` calls, total) | `NULL` | kept |
| `LAYER` | tabulated, rebuilt at each layer boundary | `NULL` | kept |
| `KEY` | called per term, every layer | real | **off** |
| `KEY \| LAYER` | called per term, with the layer index | real | **off** |

Truncation is never tabulated, so we only toggle deferral to provide key access.

| `depends` | `words` | Clifford deferral |
|---|---|---|
| `0` | `NULL` | kept |
| `LAYER` | `NULL` | kept |
| `KEY` (w or w/o `LAYER`) | real | **off** |

Note that without declaring the appropriate permissions, `words` is `NULL` and the plugin
will crash immediately.

## Example plugins

`uniform_noise` and `weight_truncator` are just re-implementations of the built-in
`UniformNoiseModel` and `WeightTruncator` policies, respectively, so they can be
used to verify that the ABI is working correctly.

| Kind      | Built-in equivalent | C                                  | Rust                             | Julia                                 |
|-----------|----------------------|-------------------------------------|-----------------------------------|------------------------------------------|
| Noise     | `UniformNoiseModel`  | `c/noise/uniform_noise.c`          | `rust/noise/uniform_noise/`      | `julia/noise/uniform_noise.jl`         |
| Truncator | `WeightTruncator`    | `c/truncation/weight_truncator.c`  | `rust/truncation/weight_truncator/` | `julia/truncation/weight_truncator.jl` |

We also provide custom policies without built-in equivalents to demonstrate plugin
syntax and semantics.

| Kind | Policy | `depends` | Formula | Config keys | C | Rust | Julia |
|---|---|---|---|---|---|---|---|
| Noise | `thermal_decay_noise` | `0` | `exp(-(gamma * weight)^beta)` (stretched exponential; `beta=1` = plain exponential) | `gamma`, `beta` | `c/noise/thermal_decay_noise.c` | `rust/noise/thermal_decay_noise/` | `julia/noise/thermal_decay_noise.jl` |
| Noise | `drifting_noise` | `0` | `exp(-damping * weight * (1 + drift_rate * call_index))` | `damping`, `drift_rate` | `c/noise/drifting_noise.c` | `rust/noise/drifting_noise/` | `julia/noise/drifting_noise.jl` |
| Noise | `depth_dependent_noise` | `LAYER` | `exp(-damping * weight * (1 + rate * layer_index / n_layers))` | `damping`, `rate` | `c/noise/depth_dependent_noise.c` | `rust/noise/depth_dependent_noise/` | `julia/noise/depth_dependent_noise.jl` |
| Noise | `qubit_local_noise` | `KEY` | `exp(-damping * \|support(term) & mask\|)` (only the masked units are noisy) | `damping`, `mask` | `c/noise/qubit_local_noise.c` | `rust/noise/qubit_local_noise/` | - |
| Truncator | `pareto_truncator` | `0` | keep iff `coeff_magnitude * exp(-alpha * weight) > threshold` (joint score) | `threshold`, `alpha` | `c/truncation/pareto_truncator.c` | `rust/truncation/pareto_truncator/` | `julia/truncation/pareto_truncator.jl` |
| Truncator | `stochastic_truncator` | `KEY` | keep with probability `min(1, coeff_magnitude / threshold)` (importance sampling) | `threshold`, `seed` | `c/truncation/stochastic_truncator.c` | `rust/truncation/stochastic_truncator/` | `julia/truncation/stochastic_truncator.jl` |
| Truncator | `support_truncator` | `KEY` | keep iff `coeff_magnitude * exp(-alpha * \|support(term) \\ mask\|) > threshold` | `threshold`, `alpha`, `mask` | `c/truncation/support_truncator.c` | `rust/truncation/support_truncator/` | - |

`mask` is an unsigned integer over unit indices, bit `q` selecting
qubit/site `q`, and addresses the first 64 units.

## Noise ABI

Fixed `extern "C"` symbol names **(C-stable types only, u32/f64/pointers/usize)**:

```c
uint32_t propaq_noise_abi_version(void);                 // required, must equal 1
uint32_t propaq_noise_depends(void);                     // optional; absent => 0
void*    propaq_noise_create(const char* config_json);   // optional
void     propaq_noise_destroy(void* ctx);                // required iff create is present

double   propaq_noise_factor(void* ctx, uint32_t basis_kind,
                             const uint64_t* words, size_t n_words,
                             uint32_t n_units, uint32_t weight,
                             uint32_t layer_index, uint32_t n_layers);   // required

int32_t  propaq_noise_factor_batch(void* ctx, uint32_t basis_kind,
                                   const uint64_t* words, size_t n_words_per_term,
                                   uint32_t n_units, const uint32_t* weights,
                                   uint32_t layer_index, uint32_t n_layers,
                                   double* out, size_t n_terms);         // optional, faster
```

```python
from propaq.noise import NativeNoiseModel

model = NativeNoiseModel("examples/plugins/c/noise/uniform_noise.so", config='{"damping": 0.001}')
propagator.set_noise(model)
```

## Truncator ABI

```c
uint32_t propaq_truncator_abi_version(void);                 // required, must equal 1
uint32_t propaq_truncator_depends(void);                     // optional; absent => 0
void*    propaq_truncator_create(const char* config_json);   // optional
void     propaq_truncator_destroy(void* ctx);                // required iff create is present

int32_t  propaq_truncator_keep(void* ctx, uint32_t basis_kind,
                               const uint64_t* words, size_t n_words,
                               uint32_t n_units, uint32_t weight,
                               double coeff_magnitude,
                               uint32_t layer_index, uint32_t n_layers);
                                                             // required; nonzero = keep

int32_t  propaq_truncator_keep_batch(void* ctx, uint32_t basis_kind,
                                     const uint64_t* words, size_t n_words_per_term,
                                     uint32_t n_units, const uint32_t* weights,
                                     const double* coeff_magnitudes,
                                     uint32_t layer_index, uint32_t n_layers,
                                     uint8_t* out_keep, size_t n_terms);  // optional, faster
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

## What a plugin receives

`basis_kind` is `0` for Pauli and `1` for Majorana. `words` is the term's
key as raw storage words, two bits per unit, interleaved:

- Pauli: bit `2q` is qubit `q`'s X component, bit `2q + 1` its Z component.
- Majorana: bit `2k` is mode `gamma_{2k}`, bit `2k + 1` is `gamma_{2k + 1}`.

`n_units` is the register size in qubits (Pauli) or fermionic *sites*
(Majorana, so half the mode count). `weight` is the term's weight, passed
because the engine has computed it already. In the batch forms the terms
are contiguous, `n_words_per_term` words each.

`layer_index` is the zero-based circuit layer and `n_layers` the circuit's
layer count, so a model can normalize depth to `layer_index / n_layers`
without being told the circuit size separately.

## The batch entry points

`propaq_noise_factor_batch` / `propaq_truncator_keep_batch` are an optional
performance path. These apply the plugin to a contiguous array of terms in one
call in an effort to amortize the overhead of FFI boundary crossing.

`propaq_truncator_keep_batch` is reachable only from the reclaim sweep
that follows a noise layer, which visits whole rows. An emit-time decision
is taken inside the branching loop one child at a time, so there is
nothing to batch there and `propaq_truncator_keep` carries it.

## Safety

Loading a plugin runs unsandboxed native code in-process. Only load
libraries you trust, the same way you'd trust any other compiled
dependency you link against. Plugin code must never panic, unwind, or
`longjmp` across the call boundary.