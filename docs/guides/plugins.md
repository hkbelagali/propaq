# Native plugins

Noise models and truncation policies sit in the propagator's innermost loop. A
Python implementation is fine for prototyping, but every call pays GIL
acquisition and interpreter dispatch, which will dominate the runtime of any
large propagation.

The plugin ABI lets you write the same model in **C, Rust, or AOT-compiled
Julia**, ship it as a shared library, and have propaq call it with no interpreter
involvement.

## Loading a plugin

```python
from propaq.noise import NativeNoiseModel
from propaq.propagators import PauliPropagator
from propaq.truncation import NativeTruncator

prop = PauliPropagator(
    noise=NativeNoiseModel(
        path="./thermal_decay_noise.so",
        config='{"gamma": 0.02, "beta": 0.8}',
    ),
    truncation=NativeTruncator(
        path="./pareto_truncator.so",
        config='{"threshold": 1e-6, "alpha": 0.5}',
    ),
)
```

`config` is an optional JSON string handed once to the plugin's
`propaq_*_create` entry point, if it exports one.

!!! danger "Plugins run unsandboxed"

    Loading a plugin executes native code in-process with no isolation. Only
    load libraries you trust, exactly as you would any other compiled dependency
    you link against. Plugin code must never panic, unwind, or `longjmp` across
    the call boundary.

## Example implementations

The repository ships working plugins in all three languages under
[`examples/plugins/`](https://github.com/hkbelagali/propaq/tree/main/examples/plugins),
including re-implementations of the built-in `UniformNoiseModel` and
`WeightTruncator` that you can use to check your toolchain end to end.

The four plugin notebooks build and run them:

- [C plugins](../examples/plugins/01_c_plugins.ipynb)
- [Rust plugins](../examples/plugins/02_rust_plugins.ipynb)
- [Julia AOT plugins](../examples/plugins/03_julia_aot_plugins.ipynb)
- [Batch ABI benchmark](../examples/plugins/04_batch_abi_benchmark.ipynb)

--8<-- "examples/plugins/README.md"
