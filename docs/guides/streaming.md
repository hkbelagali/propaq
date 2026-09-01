# Streaming and I/O

A propagated term sum can be very large in memory. One might want to conduct long propagation runs 
in a cluster environmentm, and then run post-processing on workstations. In order to support such workflows, 
propaq's term sums can be written to disk as gzip-compressed binary files and read back either
eagerly or **lazily, one term at a time**.

## Saving

Both `propagate` and `expectation_value` take a `filename=` argument that writes
the final terms as they are produced:

```python
from propaq.circuits import PauliCircuit
from propaq.datatypes import PauliTermSum
from propaq.propagators import PauliPropagator

prop = PauliPropagator(n_threads=4, progress_bar=True)
propagated = prop.propagate(observable, circuit, filename="propagated_terms.gz")
```

An existing term sum can be written directly too:

```python
propagated.save("propagated_terms.gz")
```

## Loading eagerly

[`from_file`][propaq.datatypes.PauliTermSum.from_file] reads the whole file back
into a term sum:

```python
reloaded = PauliTermSum.from_file("propagated_terms.gz")

print(len(reloaded.items()), reloaded.norm_squared())
```

## Streaming lazily

When the file is too large to load at once, use a **streamer**. It is an
iterator over `(term, coefficient)` pairs that never materialises the whole sum:

```python
from propaq.datatypes import PauliTermStreamer

streamer = PauliTermStreamer.from_file("propagated_terms.gz")

for term, coeff in streamer:
    if term.weight <= 4:
        ...  # process one term at a time
```

The Majorana counterpart is
[`MajoranaTermStreamer`][propaq.datatypes.MajoranaTermStreamer]. Both accept
files written by the corresponding term sum's `save()`.

## Merging from a stream

To accumulate a file's terms into an existing term sum without loading it
separately first, use
[`merge_from_file`][propaq.datatypes.PauliTermSum.merge_from_file]:

```python
accumulator = PauliTermSum()
accumulator.merge_from_file(PauliTermStreamer.from_file("propagated_terms.gz"))
```

This enables distributed propagation runs, where different nodes write their own term sums
to disk, and the results are merged later on a single node for post-processing and analysis.

## Worked examples

- [Notebook 06 - Term streaming](../examples/usage/06_term_streaming.ipynb)