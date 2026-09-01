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
and restored afterwards, including if the sweep raises.

A user can also sweep a custom noise model by subclassing
[`ZeroNoiseExtrapolator`][propaq.extrapolators.ZeroNoiseExtrapolator] and overriding
[`build_noise`][propaq.extrapolators.ZeroNoiseExtrapolator.build_noise], which
builds the model instance for a given sweep value (default:
`UniformNoiseModel(value)`):

```python
import math

from propaq.noise import GateNoiseModel


class DephasingNoise(GateNoiseModel):
    """Per-qubit dephasing noise"""

    _X_MASK = 0x5555555555555555  # the low bit of every interleaved (x, z) pair

    def __init__(self, gamma: float) -> None:
        self.gamma = gamma

    def damping_factor_term(self, basis_kind, words, n_units, weight):
        x_count = sum(bin(w & self._X_MASK).count("1") for w in words)
        return math.exp(-self.gamma * x_count)


class DephasingExtrapolator(ZeroNoiseExtrapolator):
    def build_noise(self, value):
        return DephasingNoise(gamma=value)


zne_dephasing = DephasingExtrapolator(
    fitting_fn=linear,
    noise_values=[0.01, 0.02, 0.03, 0.04, 0.05],
)
result = zne_dephasing.run(PauliPropagator(), observable, circuit, initial_state=0)
```

Note that only one parameter can be swept at a time, so if the model depends on multiple parameters, the others must be fixed.

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
A user can also specify a custom truncator to sweep by subclassing
[`ZeroCutoffExtrapolator`][propaq.extrapolators.ZeroCutoffExtrapolator] and implementing its two
abstract methods: [`truncator_cls`][propaq.extrapolators.ZeroCutoffExtrapolator.truncator_cls] (the
truncator class to match against, used to locate the target truncator in the propagator's
pipeline) and [`build_truncator`][propaq.extrapolators.ZeroCutoffExtrapolator.build_truncator]
(construct a fresh truncator carrying a given cutoff), mirroring
[`build_noise`][propaq.extrapolators.ZeroNoiseExtrapolator.build_noise] on the noise side:

```python
from propaq.extrapolators import ZeroCutoffExtrapolator
from propaq.truncation import WeightTruncator


class LightConeWeightCutoffExtrapolator(ZeroCutoffExtrapolator):
    """Light-cone-like truncation, where we only keep terms with weight <= 2 * depth + 1"""

    def truncator_cls(self):
        return WeightTruncator

    def build_truncator(self, depth):
        return WeightTruncator(2 * int(depth) + 1 if depth is not None else None)


extrapolator = LightConeWeightCutoffExtrapolator(
    fitting_fn=linear,
    cutoff_values=[1, 2, 3, 4],
)
result = extrapolator.run(
    PauliPropagator(truncation=WeightTruncator(weight=16)), observable, circuit, initial_state=0
)
```
Note the extra method [`truncator_cls`][propaq.extrapolators.ZeroCutoffExtrapolator.truncator_cls]. Since we have a composable truncation pipeline,
we need to identify which truncator to sweep. This allows one to hold other truncators fixed while sweeping a single one.

The result is a [`ZCEResult`][propaq.extrapolators.ZCEResult], with the same
fields as `ZNEResult` but keyed on `cutoff_values` / `zero_cutoff_value`.

## Worked examples

- [Notebook 07 - Zero cutoff extrapolation](../examples/usage/07_zero_cutoff_extrapolation.ipynb)
- [Notebook 08 - Zero noise extrapolation](../examples/usage/08_zero_noise_extrapolation.ipynb)
