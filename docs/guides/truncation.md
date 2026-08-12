# Truncation

Every non-Clifford rotation can split a term in two, so the term count of a
back-propagated observable grows exponentially with circuit depth. Two
mechanisms hold that in check:

- **Merging** - distinct branches that arrive at the same basis operator have
  their coefficients summed.
- **Truncation** - terms judged not to matter are discarded. 

Truncation runs at a **flush**, a point at which the live term store is
deduplicated and the truncator pipeline is applied. Flushes happen periodically,
whenever a budget's threshold is reached, and always at the end of the circuit.

## The truncator pipeline

A propagator takes a list of truncators, all of which are applied at each
flush:

```python
from propaq.propagators import PauliPropagator
from propaq.truncation import CoefficientTruncator, TermBudget, WeightTruncator

prop = PauliPropagator(
    truncation=[
        WeightTruncator(weight=12),
        CoefficientTruncator(coefficient=1e-6),
        TermBudget(max_terms=5_000_000, min_terms=1_000_000),
    ]
)
```

You can also pass a single truncator, or `None` for no truncation. The currently
installed pipeline is readable back off the propagator via its `truncators`
property, and replaceable with `set_truncation`.

## Available truncators

| Truncator | Discards | Applies to |
| --- | --- | --- |
| [`WeightTruncator`][propaq.truncation.WeightTruncator] | terms with operator weight above `weight` | both |
| [`CoefficientTruncator`][propaq.truncation.CoefficientTruncator] | contributions with coefficient magnitude below `coefficient` | both |
| [`TermBudget`][propaq.truncation.TermBudget] | nothing directly - triggers a truncation pass at `max_terms` | both |
| [`FrequencyTruncator`][propaq.truncation.FrequencyTruncator] | monomials whose trig-factor count exceeds `frequency` | surrogate only |
| [`MonomialBudget`][propaq.truncation.MonomialBudget] | nothing directly - triggers a flush at `max_monomials` | surrogate only |
| [`Simplify`][propaq.truncation.Simplify] | nothing - lossless collapse of monomials sharing a canonical trig-factor run | surrogate only |
| [`NativeTruncator`][propaq.truncation.NativeTruncator] | whatever your plugin decides | both |

### Weight

```python
WeightTruncator(weight=12)
```
A term of weight \(w\) is exponentially unlikely
in \(w\) to contribute to an expectation value against a computational-basis
state, so cutting the tail of high-weight terms removes a large fraction of the
work for a small error.

### Coefficient

```python
CoefficientTruncator(coefficient=1e-6)
```

Drops contributions whose magnitude has fallen below the cutoff. Pairs naturally
with a noise model, which drives coefficients down as depth increases, and so
lets this truncator do progressively more work deeper into the circuit.

### Budgets

```python
TermBudget(min_terms=1_000_000, max_terms=5_000_000)
```

A budget does not discard anything by itself. `max_terms` is a trigger, and when
the live term count reaches it, a truncation pass fires. `min_terms` is a
**floor**: below that count, the lossy truncators in the pipeline are suppressed,
so a small term sum is not eroded by cutoffs that were meant for a large one.
Either field may be `None` to disable that bound.

!!! note "Budget arguments are keyword-only"

    Both [`TermBudget`][propaq.truncation.TermBudget] and
    [`MonomialBudget`][propaq.truncation.MonomialBudget] take their floor first
    and their ceiling second, and both reject positional arguments. The pair is
    easy to transpose, and a transposed pair used to fail *silently* - the
    budget simply never fired - so a positional call now raises `TypeError`.

## Choosing a pipeline

A reasonable starting point for a numerical run:

```python
truncation = [
    WeightTruncator(weight=10),
    CoefficientTruncator(coefficient=1e-5),
    TermBudget(max_terms=10_000_000),
]
```

Truncations are heuristic, and the right cutoffs must be tailored to your problem. The following strategies help:

1. Enable a [`Logger`](logging.md) and look at `terms_before`/`terms_after` and
   `discarded_coeff_l1` for each truncation event. A large discarded \(L^1\)
    norm indicates that the truncation is generally aggressive, however note that 
    the \(L^!\) norm is often a pessimistic estimate of the actual error in the expectation value.
2. If you need to *quantify* the truncation error rather than just bound it,
   sweep the cutoff and extrapolate to zero. See
   [extrapolation](extrapolation.md).

## Legacy `TruncationPolicy`

Older code configures truncation with a single
[`TruncationPolicy`][propaq.noise.TruncationPolicy] object:

```python
from propaq.noise import TruncationPolicy

TruncationPolicy(weight_cutoff=10, coeff_cutoff=1e-5, truncation_range=(None, 10_000_000))
```

This is still accepted, and is decomposed internally into the equivalent
truncator list. New code should use the pipeline form, which composes and is
easier to inspect.

## Worked examples

- [Notebook 03 - Truncation pipelines](../examples/usage/03_truncation_pipelines.ipynb)
- [Notebook 07 - Zero cutoff extrapolation](../examples/usage/07_zero_cutoff_extrapolation.ipynb)
