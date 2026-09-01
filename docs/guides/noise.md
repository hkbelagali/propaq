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

with \(\gamma\) the `damping` rate.

## Python-defined models

Subclass [`GateNoiseModel`][propaq.noise.GateNoiseModel] and define the appropriate method for your model.
### Weight-only noise

Implement `damping_factor(term_weight, active_modes) -> float` for a model
whose damping depends only on a term's Pauli/Majorana weight (like
[`UniformNoiseModel`][propaq.noise.UniformNoiseModel]):

```python
import math

from propaq.noise import GateNoiseModel


class StretchedExponentialNoise(GateNoiseModel):
    def __init__(self, gamma: float, beta: float) -> None:
        self.gamma = gamma
        self.beta = beta

    def damping_factor(self, term_weight: float, active_modes: int) -> float:
        return math.exp(-((self.gamma * term_weight) ** self.beta))


prop = PauliPropagator(noise=StretchedExponentialNoise(0.01, 0.8))
```

We precompute the values for each weight and cache them, so these models 
are effectively zero overhead.

### Key-aware noise
In general, a noise model will need the actual operator's string representation, 
not just its weight. For these cases, implement `damping_factor_term(basis_kind, words, n_units, weight) -> float`.

```python
class BoundaryQubitNoise(GateNoiseModel):
    """Damp terms acting nontrivially on qubit 0 more than the rest."""

    def __init__(self) -> None:
        pass

    def damping_factor_term(self, basis_kind, words, n_units, weight):
        touches_qubit_0 = bool(words[0] & 0b11)
        gamma = 0.3 if touches_qubit_0 else 0.05
        return math.exp(-gamma * weight)


prop = PauliPropagator(noise=BoundaryQubitNoise())
```

!!! warning "Key-aware noise is on the hot path"

    `damping_factor_term` is called once per live term at every noise
    application boundary, with the GIL held, so every call pays GIL
    acquisition and Python dispatch. We have observed that this can 
    cost up to 50% added runtime for a circuit.
    We recommend prototyping your model in Python first, then porting it to a native plugin for production runs.

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
