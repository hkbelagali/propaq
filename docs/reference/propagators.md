# `propaq.propagators`

The propagators. A numerical propagator back-propagates an observable through a
concrete circuit; a surrogate propagator compiles a *parameterized* circuit into
a reusable symbolic model.

See the [quickstart](../getting-started/quickstart.md) and the
[surrogate guide](../guides/surrogate.md).

::: propaq.propagators
    options:
      members:
        - PauliPropagator
        - MajoranaPropagator
        - PauliSurrogatePropagator
        - MajoranaSurrogatePropagator
        - AbstractPropagator
        - CircuitLike
