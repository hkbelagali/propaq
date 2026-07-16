"""Evaluate a compiled surrogate model directly against Qiskit Parameter values."""

from collections.abc import Mapping, Sequence

from qiskit.circuit import Parameter

from .._rust_core import MajoranaSurrogateModel, PauliSurrogateModel
from ..circuits._qiskit_symbolic import ParamSource


class VariationalSurrogateModel:
    """
    Wraps a compiled `PauliSurrogateModel`/`MajoranaSurrogateModel` together with
    the qiskit Parameter bindings recorded by `SurrogatePauliCircuit.from_qiskit`/
    `SurrogateMajoranaCircuit.from_qiskit`, so it can be evaluated directly against
    Qiskit `Parameter` values instead of raw propaq parameter-index vectors.

    Attributes:
        parameters: Distinct Qiskit Parameters this model can be evaluated against,
            in the order used for positional `evaluate` calls.
    """

    def __init__(
        self,
        model: "PauliSurrogateModel | MajoranaSurrogateModel",
        parameter_sources: list[ParamSource],
        qiskit_parameters: tuple[Parameter, ...],
    ):
        """Construct a VariationalSurrogateModel from a compiled surrogate model and its Qiskit parameter sources."""
        self._model = model
        self._parameter_sources = parameter_sources
        self.parameters = qiskit_parameters

    @property
    def n_params(self) -> int:
        """Number of underlying propaq parameter slots (may exceed len(parameters))."""
        return len(self._parameter_sources)

    def _bind(self, values: "Mapping[Parameter, float] | Sequence[float]") -> list[float]:
        """Resolve Qiskit Parameter values into the raw propaq parameter vector."""
        if isinstance(values, Mapping):
            binding = values
        else:
            if len(values) != len(self.parameters):
                raise ValueError(
                    f"values has {len(values)} elements but model has "
                    f"{len(self.parameters)} distinct parameters"
                )
            binding = dict(zip(self.parameters, values))

        return [
            source.scale * binding[source.parameter] if source.parameter is not None else source.scale
            for source in self._parameter_sources
        ]

    def evaluate(self, values: "Mapping[Parameter, float] | Sequence[float]") -> float:
        """
        Evaluate the expectation value for the given Qiskit Parameter values.

        Arguments:
            values: Either a mapping from Parameter to its value, or a plain
                sequence of values aligned with `self.parameters`.
        """
        return self._model.evaluate(self._bind(values))

    def evaluate_batch(
        self, values_batch: "Sequence[Mapping[Parameter, float] | Sequence[float]]"
    ) -> list[float]:
        """
        Evaluate many Qiskit Parameter assignments at once (parallelized across
        assignments in Rust), returning one expectation value per assignment.

        Arguments:
            values_batch: A sequence of assignments, each accepted in either
                form `evaluate` takes.
        """
        return self._model.evaluate_batch([self._bind(values) for values in values_batch])
