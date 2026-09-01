# Logging and profiling

Generally, the propagators either return an expectation value, or a backpropagated observable. However, this obscures what happened during the propagation, such as 
how many terms were removed, how long each gate took, and how many threads were busy at each point. 
propaq's [`Logger`][propaq._rust_core.Logger] records this information to a JSONL file, which is automatically parsed using [`LogParser`][propaq.log_parser.LogParser] for easy access to the events and their fields. The details of the API and events are described below.
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
for short runs.

!!! note "Logger overwrites files"

    The [`Logger`][propaq._rust_core.Logger] overwrites the contents of the specified file.  

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

plt.plot(log.gate_indices, log.terms)
plt.xlabel("gate")
plt.ylabel("live terms")
```

## Execution order

The logger emits events in the order they occur. This means that any structural indices, such as layer/gate indices will be reversed in the logs, since we operate in the 
Heisenberg picture. The events are listed below. 

## The events

| Event | Emitted by | Tells you |
| --- | --- | --- |
| [`GateEvent`][propaq.log_parser.GateEvent] | all propagators | live term count (and, for surrogates, monomial count) as the circuit is applied, plus wall time for that gate |
| [`TruncationEvent`][propaq.log_parser.TruncationEvent] | numerical propagators | what each per-gate truncation pass discarded |
| [`SurrogateMergeEvent`][propaq.log_parser.SurrogateMergeEvent] | surrogate propagators | the surrogate's equivalent, keyed on monomials as well as terms |
| [`EnginePhasesEvent`][propaq.log_parser.EnginePhasesEvent] | all propagators | runtimes of each propagation phase/load balance statistics |

## What to look at

### Is the truncation too aggressive?

Look at [`TruncationEvent`][propaq.log_parser.TruncationEvent]:

- `terms_before` / `terms_after` / `terms_gained` - the live term count either
  side of the gate, and how many of those are new keys created by branching
- `terms_discarded` - branches the emit cutoff declined to form this gate.
  Independent of the three counts above: a declined branch never reaches the
  store, so it's not a "loss" subtracted from `terms_gained`, it's a branch
  that was never attempted.
- `discarded_coeff_l1` - the total |coefficient| mass thrown away this gate.
- `discarded_coeff_max` - the single largest discarded coefficient this gate. If this is
  close to the coefficients you are keeping, the cutoff is too aggressive.

!!! note "Term count semantics" 

    From the naming, one might expect that `terms_after = terms_before + terms_gained - terms_discarded`. However, this is not the case. The propagator declines 
    branches that fall below the cutoff, and those branches never reach the store, and are not registered in `terms_after`.
    The surrogate build's equivalent, [`SurrogateMergeEvent`][propaq.log_parser.SurrogateMergeEvent],
    runs its truncation pipeline as a separate pass after each gate rather than
    declining branches during emission, so it has no per-branch decline count or
    coefficient magnitudes to report.

### Where is the time going?

Look at [`GateEvent.ms_per_gate`][propaq.log_parser.GateEvent] to find the point in the
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

!!! note Changing the store's inline capacity

    If the engine detects that overflows are frequent, it will automatically double the store's inline capacity for the next gate, as long as
    a weight truncation is not set. If one is set, the store's inline capacity is fixed to twice the truncator's cutoff, ensure that the store never overflows.
    A hard cap of inline capacity 32 is enforced to avoid unnecessary memory growth.
    
!!! note "Release builds inline the phase split"

    In an optimized build, the scan/absorb/claims split is inlined into 
    the gate loop, and therefore not profiled by tools like `perf`. 
    The [`EnginePhasesEvent`][propaq.log_parser.EnginePhasesEvent] 
    reports this split by measuring the time spent in each phase.

## Progress bars

For interactive work, a `tqdm` bar over the gate loop reports the number of live terms and progress:

```python
PauliPropagator(progress_bar=True, progress_every=10)
```

`progress_every` sets the number of gates between ticks.