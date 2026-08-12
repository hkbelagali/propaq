# `propaq.datatypes`

Operator representations: individual basis terms, the weighted sums that collect
them, and the lazy streamers that read a saved sum back one term at a time.

See [Core concepts](../getting-started/concepts.md) for how these relate to the
Pauli and Majorana bases, and [Streaming and I/O](../guides/streaming.md) for
the file format and streamers.

::: propaq.datatypes
    options:
      members:
        - PauliString
        - PauliTermSum
        - MajoranaMonomial
        - MajoranaTermSum
        - PauliTermStreamer
        - MajoranaTermStreamer
        - AbstractTerm
        - AbstractTermSum
