# Extrapolation

propaq's noise and truncation models are naturally compatible with error mitigation techniques such as 
zero noise extrapolation. One can also sweep different cutoffs and extrapolate to zero cutoff, i.e. zero-cutoff extrapolation.


## Zero-noise extrapolation

[`ZeroNoiseExtrapolator`][propaq.extrapolators.ZeroNoiseExtrapolator] sweeps the
damping rate of a [`NoiseModel`][propaq.noise.base.NoiseModel] and
extrapolates to \(\gamma \to 0\).

```python
import numpy as np

from propaq.extrapolators import ZeroNoiseExtrapolator
from propaq.propagators import PauliPropagator


def linear(gamma, a, b):
    return a + b * gamma


zne = ZeroNoiseExtrapolator(
    fitting_fn=linear,
    noise_values=[0.01, 0.02, 0.03, 0.04, 0.05],
)

result = zne.run(PauliPropagator(truncation=..., noise=...), observable, circuit, initial_state=0)

print("zero-noise estimate:", result.zero_noise_value)
print("sweep:", list(zip(result.noise_values, result.expectation_values)))
print("fit params:", result.fit_params)
```

`fitting_fn` is passed straight to `scipy.optimize.curve_fit`, so its first
argument is the noise value and the rest are fit parameters. Extra keyword
arguments to `run` (such as `p0=`) are forwarded to `curve_fit`.

The propagator's existing noise model is replaced for the duration of the sweep
and **restored afterwards**, including if the sweep raises.

The result is a [`ZNEResult`][propaq.extrapolators.ZNEResult], which carries the
extrapolated value along with the raw sweep, the fitted parameters and their
covariance.

## Zero-cutoff extrapolation

The truncation analogue sweeps a cutoff already present in the propagator's
truncation pipeline, and extrapolates to zero cutoff. There are two concrete
extrapolators, one per truncator kind:

| Extrapolator | Sweeps |
| --- | --- |
| [`WeightCutoffExtrapolator`][propaq.extrapolators.WeightCutoffExtrapolator] | the `weight` of a [`WeightTruncator`][propaq.truncation.WeightTruncator] |
| [`CoefficientCutoffExtrapolator`][propaq.extrapolators.CoefficientCutoffExtrapolator] | the `coefficient` of a [`CoefficientTruncator`][propaq.truncation.CoefficientTruncator] |

```python
from propaq.extrapolators import CoefficientCutoffExtrapolator
from propaq.propagators import PauliPropagator
from propaq.truncation import CoefficientTruncator, WeightTruncator

prop = PauliPropagator(
    truncation=[WeightTruncator(weight=12), CoefficientTruncator(coefficient=1e-4)]
)

zce = CoefficientCutoffExtrapolator(
    fitting_fn=linear,
    cutoff_values=[1e-3, 5e-4, 2e-4, 1e-4],
)

result = zce.run(prop, observable, circuit, initial_state=0)
print("zero-cutoff estimate:", result.zero_cutoff_value)
```
A user can also specify custom truncators to sweep by subclassing
[`ZeroCutoffExtrapolator`][propaq.extrapolators.ZeroCutoffExtrapolator] and implementing its three
abstract methods: [`_rust_cls`][propaq.extrapolators.ZeroCutoffExtrapolator._rust_cls] (the Rust
truncator class to sweep), [`_read`][propaq.extrapolators.ZeroCutoffExtrapolator._read] (pull the
cutoff out of a matching truncator) and
[`_build`][propaq.extrapolators.ZeroCutoffExtrapolator._build] (construct one carrying a given
cutoff).

The result is a [`ZCEResult`][propaq.extrapolators.ZCEResult], with the same
fields as `ZNEResult` but keyed on `cutoff_values` / `zero_cutoff_value`.

## Worked examples

- [Notebook 07 - Zero cutoff extrapolation](../examples/usage/07_zero_cutoff_extrapolation.ipynb)
- [Notebook 08 - Zero noise extrapolation](../examples/usage/08_zero_noise_extrapolation.ipynb)
