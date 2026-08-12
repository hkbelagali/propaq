# Noise models

A noise model damps term coefficients as the observable propagates. It can be used to 
model noisy devices, or drive coefficients to zero to make a simulation tractable.

```python
from propaq.noise import UniformNoiseModel
from propaq.propagators import PauliPropagator

prop = PauliPropagator(noise=UniformNoiseModel(damping=0.001))
```

The model can also be swapped after construction with `set_noise`, and read back
off the propagator's `noise` property.

## Uniform depolarizing noise

[`UniformNoiseModel`][propaq.noise.UniformNoiseModel] is the built-in
depolarizing-style model: a term of weight \(w\) is scaled by

\[
\exp(-\gamma w)
\]

with \(\gamma\) the `damping` rate. It runs natively, with no Python callback in
the propagator's inner loop.

## Python-defined models

[`GateNoiseModel`][propaq.noise.GateNoiseModel] delegates to an arbitrary Python
object. Implement the two methods of
[`NoiseModel`][propaq.noise.base.NoiseModel]:

```python
import math

from propaq.noise import GateNoiseModel


class StretchedExponentialNoise:
    def __init__(self, gamma: float, beta: float) -> None:
        self.gamma = gamma
        self.beta = beta

    def damping_factor(self, term_weight: float, active_modes: int) -> float:
        return math.exp(-((self.gamma * term_weight) ** self.beta))

    def apply_noise(self, term_sum):
        term_sum.apply_damping(self.damping_factor)


prop = PauliPropagator(noise=GateNoiseModel(inner=StretchedExponentialNoise(0.01, 0.8)))
```

!!! warning "Python noise is on the hot path"

    `damping_factor` is called from the propagator's inner loop, so every call
    pays GIL acquisition and Python dispatch. This is fine for prototyping and
    for models applied at coarse granularity, but it will dominate the runtime
    of a large propagation. For anything performance-sensitive, write the model
    as a [native plugin](plugins.md) instead.

## Native plugins

[`NativeNoiseModel`][propaq.noise.NativeNoiseModel] loads a noise model from a
C, Rust or AOT-compiled Julia shared library:

```python
from propaq.noise import NativeNoiseModel

noise = NativeNoiseModel(
    path="./thermal_decay_noise.so",
    config='{"gamma": 0.02, "beta": 0.8}',
)
```

`config` is a JSON string handed once to the plugin's `propaq_noise_create`.
See the [plugin guide](plugins.md) for the full ABI.

## Interaction with truncation

Noise and truncation compound. As depth increases, damping pushes more and more
coefficient mass under `CoefficientTruncator`'s cutoff, so a run with noise
enabled typically has a smaller live term count than the same run without it.
A noise model with a moderate damping rate can be used to make a simulation tractable.
If you want the *noiseless* answer, run a sweep of damping rates and
extrapolate to \(\gamma \to 0\). See [extrapolation](extrapolation.md).

## Worked examples

- [Notebook 08 - Zero noise extrapolation](../examples/usage/08_zero_noise_extrapolation.ipynb)
- [Notebook 01 - C plugins](../examples/plugins/01_c_plugins.ipynb)
- [Notebook 02 - Rust plugins](../examples/plugins/02_rust_plugins.ipynb)
- [Notebook 03 - Julia plugins](../examples/plugins/03_julia_aot_plugins.ipynb)
- [Notebook 04 - Batch ABI benchmark](../examples/plugins/04_batch_abi_benchmark.ipynb)
