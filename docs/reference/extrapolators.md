# `propaq.extrapolators`

Extrapolation to the zero-noise and zero-cutoff limits, by sweeping the
controlling parameter and fitting a curve.

See the [extrapolation guide](../guides/extrapolation.md).

## Zero-noise extrapolation

::: propaq.extrapolators.ZeroNoiseExtrapolator
    options:
      heading_level: 3

::: propaq.extrapolators.ZNEResult
    options:
      heading_level: 3

## Zero-cutoff extrapolation

::: propaq.extrapolators.ZeroCutoffExtrapolator
    options:
      heading_level: 3
      members:
        - fitting_fn
        - cutoff_values
        - run
        - _rust_cls
        - _read
        - _build

::: propaq.extrapolators.WeightCutoffExtrapolator
    options:
      heading_level: 3

::: propaq.extrapolators.CoefficientCutoffExtrapolator
    options:
      heading_level: 3

::: propaq.extrapolators.ZCEResult
    options:
      heading_level: 3
