# `propaq.truncation`

Composable truncation operators, applied together after every gate by both the
numerical and the surrogate propagators.

See the [truncation guide](../guides/truncation.md).

::: propaq.truncation
    options:
      members:
        - WeightTruncator
        - CoefficientTruncator
        - TermBudget
        - FrequencyTruncator
        - Simplify
        - NativeTruncator
        - Truncator
        - TruncationPolicy
        - FrequencyTruncationPolicy
        - resolve_truncation
        - ResolvedTruncation
