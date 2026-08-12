# `propaq.truncation`

Composable truncation operators, applied together at every flush by both the
numerical and the surrogate propagators.

See the [truncation guide](../guides/truncation.md).

!!! note "Budget arguments are keyword-only"

    [`TermBudget`][propaq.truncation.TermBudget] and
    [`MonomialBudget`][propaq.truncation.MonomialBudget] take their `min_*`
    parameter first and their `max_*` parameter second, and both reject
    positional arguments with a `TypeError`.

::: propaq.truncation
    options:
      members:
        - WeightTruncator
        - CoefficientTruncator
        - TermBudget
        - FrequencyTruncator
        - MonomialBudget
        - Simplify
        - NativeTruncator
        - Truncator
