# Logging and profiling

Propagation is opaque from the outside: you get a number back, but not whether
the run spent its time in a sensible place or whether truncation quietly ate the
signal. propaq's [`Logger`][propaq._rust_core.Logger] writes a structured **JSON Lines**
event log that answers both questions.

## Enabling logging

```python
from propaq import Logger
from propaq.propagators import PauliPropagator

prop = PauliPropagator(
    truncation=...,
    logger=Logger("run.jsonl", log_every=10),
)

result = prop.expectation_value(observable, circuit)
```

`log_every` emits a gate record every *N* gate applications. Leave it at `1`
for short runs; raise it for deep circuits, where one record per gate is more
data than you want.

## Reading the log back

[`LogParser`][propaq.log_parser.LogParser] reads a log file into typed event lists:

```python
from propaq import LogParser

log = LogParser("run.jsonl")

print(len(log.gate_events), "gate events")
print(len(log.truncation_events), "truncation events")
```

It also exposes each field as a flat list, ready to plot without a
comprehension:

```python
import matplotlib.pyplot as plt

plt.plot(log.gate_indices, log.map_terms)
plt.xlabel("gate")
plt.ylabel("live terms")
```

## The events

| Event | Emitted by | Tells you |
| --- | --- | --- |
| [`GateEvent`][propaq.log_parser.GateEvent] | all propagators | live term count (and, for surrogates, monomial count) as the circuit is applied, plus average wall time per gate |
| [`TruncationEvent`][propaq.log_parser.TruncationEvent] | numerical propagators | what each truncation pass discarded, and what triggered it |
| [`SurrogateFlushEvent`][propaq.log_parser.SurrogateFlushEvent] | surrogate propagators | the surrogate's equivalent, keyed on monomials as well as terms |
| [`SurrogateFlushDeferredEvent`][propaq.log_parser.SurrogateFlushDeferredEvent] | surrogate propagators | flush triggers that fired but were deferred |
| [`EnginePhasesEvent`][propaq.log_parser.EnginePhasesEvent] | all propagators | one closing summary per run: phase timings, worker occupancy, store statistics |

## What to look at

### Is the truncation too aggressive? 

Look at [`TruncationEvent`][propaq.log_parser.TruncationEvent]:

- `terms_before` / `terms_after` / `terms_discarded` - how much each pass removed.
- `discarded_coeff_l1` - the total |coefficient| mass thrown away. 
- `discarded_coeff_max` - the single largest discarded coefficient. If this is
  close to the coefficients you are keeping, the cutoff is too aggressive.

### Where is the time going?

Look at [`GateEvent.avg_ms_per_gate`][propaq.log_parser.GateEvent] to find the point in the
circuit where cost takes off.

Then look at the closing [`EnginePhasesEvent`][propaq.log_parser.EnginePhasesEvent]:

- `scan_s` / `absorb_s` / `claims_s` - the split between emitting branches,
  absorbing the routing exchange, and the pair rule's rescue round.
- `scan_occupancy` / `absorb_occupancy` / `claims_occupancy` - the fraction of workers 
  that were busy in each phase. If this is low, certain workers are waiting on others to finish
  often.
- `overflow_share` - the fraction of rows whose keys spilled past the store's
  inline capacity, each costing an extra lookup per read.
- `emitted_share` / `declined_share` - how many of the branches the scan visited
  it actually formed, versus refused up front.

!!! note "Release builds inline the phase split"

    In an optimized build, the scan/absorb/claims split is inlined into 
    the gate loop, and therefore not profiled by tools like `perf`. 
    The [`EnginePhasesEvent`][propaq.log_parser.EnginePhasesEvent] 
    reports this split by measuring the time spent in each phase.

## Progress bars

For interactive work, a `tqdm` bar over the gate loop is often enough:

```python
PauliPropagator(progress_bar=True, progress_every=10)
```

`progress_every` sets the number of gates between ticks. This is independent of
the logger and cheaper.
