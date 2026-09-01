# Truncation

Every non-Clifford rotation can split a term in two, so the term count of a
back-propagated observable grows exponentially with circuit depth. Two
mechanisms hold that in check:

- **Merging** - distinct branches that arrive at the same basis operator have
  their coefficients summed.
- **Truncation** - terms judged not to matter are discarded.

The truncator pipeline is applied after every gate, unless the live term count is 
below a floor set by the user. 
## The truncator pipeline

A propagator takes a list of truncators, all of which are applied after each
gate:

```python
from propaq.propagators import PauliPropagator
from propaq.truncation import CoefficientTruncator, TermBudget, WeightTruncator

prop = PauliPropagator(
    truncation=[
        WeightTruncator(weight=12),
        CoefficientTruncator(coefficient=1e-6),
        TermBudget(min_terms=1_000_000),
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
| [`TermBudget`][propaq.truncation.TermBudget] | nothing directly - suppresses the other truncators below `min_terms` | both |
| [`FrequencyTruncator`][propaq.truncation.FrequencyTruncator] | monomials whose trig-factor count exceeds `frequency` | surrogate only |
| [`Simplify`][propaq.truncation.Simplify] | nothing - lossless collapse of monomials sharing a canonical trig-factor run | surrogate only |
| [`NativeTruncator`][propaq.truncation.NativeTruncator] | whatever your plugin decides | numerical only |

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

### Budget

```python
TermBudget(min_terms=1_000_000)
```

A budget does not discard anything by itself. `min_terms` defines a floor 
for the live term count. Below this floor, the other truncators are suppressed, 
so that we don't truncate too aggressively when it's unnecessary.

## Choosing a pipeline

A reasonable starting point for a numerical run:

```python
truncation = [
    WeightTruncator(weight=10),
    CoefficientTruncator(coefficient=1e-5),
    TermBudget(min_terms=1_000_000),
]
```

Truncations are heuristic, and the right cutoffs must be tailored to your problem. The following strategies help:

1. Enable a [`Logger`](logging.md) and look at `terms_before`/`terms_after` and
   `discarded_coeff_l1` for each truncation event. A large discarded \(L^1\)
    norm indicates that the truncation is generally aggressive, however note that 
    the \(L^1\) norm is often a pessimistic estimate of the actual error in the expectation value.
2. If you need to *quantify* the truncation error rather than just bound it,
   sweep the cutoff and extrapolate to zero. See
   [extrapolation](extrapolation.md).

## Legacy `TruncationPolicy`

Older code configures truncation with a single
[`TruncationPolicy`][propaq.noise.TruncationPolicy] object:

```python
from propaq.noise import TruncationPolicy

TruncationPolicy(weight_cutoff=10, coeff_cutoff=1e-5, min_terms=1_000_000)
```

This is still accepted, and is decomposed internally into the equivalent
truncator list. New code should use the pipeline form, which composes and is
easier to inspect.

## Worked examples

- [Notebook 03 - Truncation pipelines](../examples/usage/03_truncation_pipelines.ipynb)
- [Notebook 07 - Zero cutoff extrapolation](../examples/usage/07_zero_cutoff_extrapolation.ipynb)
