"""Abstract propagator interface, and the reusable Heisenberg engine behind it."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Generic, Protocol, TypeAlias, TypeVar

from propaq._rust_core import PropagationResult
from propaq.datatypes.abstract import AbstractTerm, AbstractTermSum, FockState
from propaq.datatypes.term_io import save_terms as _save_terms_to_file
from propaq.truncation._apply import ResolvedTruncation, resolve_truncation

if TYPE_CHECKING:
    from collections.abc import Iterable

    from propaq.noise import GateNoiseModel, NativeNoiseModel, UniformNoiseModel
    from propaq.truncation import TruncationPolicy

TermT = TypeVar("TermT", bound=AbstractTerm)
RotationT = TypeVar("RotationT")

#: Covariant, since both structural shapes below only ever *produce* rotations.
_RotationT_co = TypeVar("_RotationT_co", covariant=True)


class _HasLayers(Protocol[_RotationT_co]):
    """Structural match for a circuit exposing ``.layers``."""

    @property
    def layers(self) -> Sequence[Sequence[_RotationT_co]]: ...


class _HasRotations(Protocol[_RotationT_co]):
    """Structural match for anything exposing a flat ``.rotations`` list."""

    @property
    def rotations(self) -> Sequence[_RotationT_co]: ...


CircuitLike: TypeAlias = (
    _HasLayers[RotationT]
    | _HasRotations[RotationT]
    | Sequence[RotationT]
    | Sequence[Sequence[RotationT]]
)


class AbstractPropagator(ABC, Generic[TermT, RotationT]):
    """Heisenberg-picture propagation over an arbitrary operator basis.

    Arguments:
        noise: Optional noise model (`UniformNoiseModel`, `GateNoiseModel`,
            `NativeNoiseModel`, or any object exposing `damping_factor` /
            `damping_factor_term`).
        truncation: A truncator, a sequence of truncators, a
            `TruncationPolicy`, or None. Surrogate-only and engine-only
            truncators are rejected
    """

    basis_kind: ClassVar[int] = -1
    """The integer a key-aware native noise plugin sees as ``basis_kind``.

    ``0`` is Pauli and ``1`` is Majorana, and a plugin is only guaranteed to
    interpret those two. The default, ``-1`` denies plugin access to the basis kind. 
    Subclass if your custom basis supports the representation the plugin expects.
    """

    term_sum_type: ClassVar[type[AbstractTermSum] | None] = None
    """Container type built for a propagated result."""

    def __init__(
        self,
        noise: UniformNoiseModel | GateNoiseModel | NativeNoiseModel | None = None,
        truncation: object | Sequence[object] | TruncationPolicy | None = None,
    ) -> None:
        """Construct a propagator with an optional noise model and truncation pipeline."""
        self.set_noise(noise)
        self.set_truncation(truncation)

    @property
    def noise(self) -> UniformNoiseModel | GateNoiseModel | NativeNoiseModel | None:
        """The current noise model, or None."""
        return getattr(self, "_noise", None)

    def set_noise(
        self, noise: UniformNoiseModel | GateNoiseModel | NativeNoiseModel | None = None
    ) -> None:
        """Replace the noise model.

        Arguments:
            noise: The new model, or None to disable noise.
        """
        self._noise = noise

    @property
    def truncators(self) -> list[object]:
        """The current truncation pipeline, in application order."""
        return list(getattr(self, "_truncators", ()))

    def set_truncation(
        self,
        truncation: object | Sequence[object] | TruncationPolicy | None = None,
    ) -> None:
        """Replace the truncation pipeline.

        Arguments:
            truncation: A truncator, a sequence of truncators, a legacy
                `TruncationPolicy`, or None.

        Raises:
            TypeError: If ``truncation`` is not one of the accepted forms, or
                names a surrogate-only (`FrequencyTruncator`/`Simplify`) or
                engine-only (`NativeTruncator`) truncator. See
                `resolve_truncation`.
        """
        self._truncators = resolve_truncation(truncation)
        self._cutoff = ResolvedTruncation.from_truncators(self._truncators)

    @abstractmethod
    def apply_gate(
        self, term: TermT, coeff: complex, rotation: RotationT
    ) -> Iterable[tuple[TermT, complex]]:
        r"""Branch one basis term under one gate

        Arguments:
            term: The basis term being conjugated.
            coeff: The term's current coefficient.
            rotation: The gate, typically an `AbstractRotation` carrying a
                ``generator`` and an ``angle``.

        Returns:
            The ``(child_term, child_coeff)`` pairs of the expansion.
        """

    @staticmethod
    def layers_of(circuit: CircuitLike) -> list[list[RotationT]]:
        """Normalize a circuit into its gate layers, in Heisenberg application order.

        Arguments:
            circuit: The circuit to normalize.

        Returns:
            One (reversed) list of rotations per (reversed) layer.

        Raises:
            TypeError: If ``circuit`` is none of the accepted forms.
        """
        layers = getattr(circuit, "layers", None)
        if layers is None:
            rotations = getattr(circuit, "rotations", None)
            if rotations is not None:
                layers = [[r] for r in rotations]
            elif isinstance(circuit, Sequence):
                layers = (
                    [list(layer) for layer in circuit]
                    if circuit and isinstance(circuit[0], list | tuple)
                    else [[r] for r in circuit]
                )
            else:
                raise TypeError(
                    "circuit must expose `layers` or `rotations`, or be a sequence "
                    "of rotations or of layers of rotations, got "
                    f"{type(circuit).__name__}"
                )
        return [list(reversed(layer)) for layer in reversed(list(layers))]

    def apply_layer(
        self, terms: dict[TermT, complex], rotation: RotationT, layer_index: int
    ) -> dict[TermT, complex]:
        """Apply one gate to every live term, folding branches back together.

        Override this to transform a whole term map at once.

        Arguments:
            terms: The live terms, as a ``{term: coefficient}`` map.
            rotation: The gate to apply.
            layer_index: Index of this gate's layer, in the reversed
                (Heisenberg) order `layers_of` produces.

        Returns:
            The evolved term map.
        """
        cutoff = self._cutoff.at_size(len(terms))
        out: dict[TermT, complex] = {}
        for term, coeff in terms.items():
            for child, child_coeff in self.apply_gate(term, coeff, rotation):
                if not cutoff.admits(child.weight, child_coeff):
                    continue
                out[child] = out.get(child, 0j) + child_coeff
        return out

    def apply_noise(
        self, terms: dict[TermT, complex], layer_index: int, n_layers: int
    ) -> dict[TermT, complex]:
        """Damp every live coefficient by this layer's noise, then reclaim.

        Arguments:
            terms: The live terms, as a ``{term: coefficient}`` map.
            layer_index: Index of the layer about to be applied, in the
                reversed (Heisenberg) order `layers_of` produces.
            n_layers: Total number of layers in the circuit.

        Raises:
            NotImplementedError: If the noise model exposes neither
                ``damping_factor_term``, ``factor_term``, nor
                ``damping_factor``.
        """
        noise = self._noise
        if noise is None or not terms:
            return terms

        hook = getattr(noise, "damping_factor_term", None)
        native = getattr(noise, "factor_term", None)
        if hook is not None:
            for term in terms:
                terms[term] *= hook(self.basis_kind, term.words, term.n_units, term.weight)
        elif native is not None:
            if getattr(noise, "depends", 0) & 1:
                for term in terms:
                    terms[term] *= native(
                        self.basis_kind,
                        term.words,
                        term.n_units,
                        term.weight,
                        layer_index,
                        n_layers,
                    )
            else:
                table: dict[int, float] = {}
                for term in list(terms):
                    weight = term.weight
                    factor = table.get(weight)
                    if factor is None:
                        factor = table[weight] = native(
                            self.basis_kind, [], term.n_units, weight, layer_index, n_layers
                        )
                    terms[term] *= factor
        elif hasattr(noise, "damping_factor"):
            table = {}
            for term in list(terms):
                weight = term.weight
                factor = table.get(weight)
                if factor is None:
                    factor = table[weight] = noise.damping_factor(weight, 0)
                terms[term] *= factor
        else:
            raise NotImplementedError(
                f"{type(noise).__name__} exposes none of damping_factor_term, "
                "factor_term, or damping_factor"
            )

        cfg = self._cutoff
        if cfg.weight_cutoff is None and cfg.coeff_cutoff is None:
            return terms
        return {t: c for t, c in terms.items() if cfg.admits(t.weight, c)}

    def _run(
        self, observable: AbstractTermSum[TermT], circuit: CircuitLike
    ) -> tuple[dict[TermT, complex], list[int]]:
        """Back-propagate ``circuit``, returning the raw term map and per-gate live counts."""
        cutoff = self._cutoff
        terms: dict[TermT, complex] = {}
        for term, coeff in observable.items():
            value = complex(coeff)
            if cutoff.admits(term.weight, value):
                terms[term] = terms.get(term, 0j) + value

        layers = self.layers_of(circuit)
        n_layers = len(layers)
        n_terms: list[int] = []
        has_noise = self._noise is not None

        for layer_index, layer in enumerate(layers):
            if has_noise:
                terms = self.apply_noise(terms, layer_index, n_layers)
            for rotation in layer:
                terms = self.apply_layer(terms, rotation, layer_index)
                n_terms.append(len(terms))
        return terms, n_terms

    def propagate(
        self, observable: AbstractTermSum[TermT], circuit: CircuitLike, filename: str | None = None
    ) -> AbstractTermSum[TermT]:
        r"""Back-propagate *circuit* through *observable* in the Heisenberg picture.

        Layers, and the gates within each layer, are applied in reverse, so the
        result is \(U^\dagger O U\).

        If *filename* is given, the final term sum is saved to a gzip-compressed
        binary file at that path.

        Arguments:
            observable: The term sum to back-propagate (matching this
                propagator's basis).
            circuit: The circuit to propagate through: anything `layers_of`
                accepts.
            filename: Optional path to save the evolved term sum to,
                gzip-compressed.

        Returns:
            The evolved term sum, in the same container class as *observable*
            unless `term_sum_type` says otherwise.
        """
        terms, _ = self._run(observable, circuit)
        out = self._wrap(terms, observable)
        if filename is not None:
            self.save_terms(out, filename)
        return out

    def expectation_value(
        self,
        observable: AbstractTermSum[TermT],
        circuit: CircuitLike,
        initial_state: FockState = 0,
        filename: str | None = None,
    ) -> PropagationResult:
        r"""Compute the expectation value of *observable* after evolving through *circuit*.

        Evaluates \(\langle f | U^\dagger O U | f \rangle\) by summing each
        evolved term's `AbstractTerm.trace_with_fock_state` against
        *initial_state*.

        Arguments:
            observable: The term sum whose expectation value is computed.
            circuit: The circuit to propagate through.
            initial_state: The reference state, passed through unchanged to
                `AbstractTerm.trace_with_fock_state`. Both `PauliString` and
                `MajoranaMonomial` read it as an integer bitmask.
            filename: Optional path to save the evolved term sum to,
                gzip-compressed.

        Returns:
            The expectation value, plus the diagnostics collected during the
            run

        Raises:
            ValueError: If the summed value has a non-negligible imaginary
                part. `PropagationResult` is real-valued, matching the two
                built-in propagators
        """
        terms, n_terms = self._run(observable, circuit)
        out = self._wrap(terms, observable)
        if filename is not None:
            self.save_terms(out, filename)

        value = sum(
            (
                coeff * complex(term.trace_with_fock_state(initial_state))
                for term, coeff in terms.items()
            ),
            0j,
        )
        if abs(value.imag) > 1e-9 * max(abs(value.real), 1.0):
            raise ValueError(
                f"expectation_value produced a non-negligible imaginary part ({value!r}); "
                "PropagationResult is real-valued, matching PauliPropagator/"
                "MajoranaPropagator. Pass a Hermitian observable (see "
                "DictTermSum.hermitian), or call propagate() and inspect the "
                "evolved term sum directly."
            )
        floor = self._cutoff.coeff_cutoff
        terms_below_cutoff = (
            0 if floor is None else sum(1 for c in terms.values() if abs(c) < floor)
        )
        return PropagationResult(
            expectation_value=value.real,
            n_terms=n_terms,
            sparse_key_bytes=sum(len(term.to_bytes()) for term in terms),
            terms_below_cutoff=terms_below_cutoff,
        )

    def save_terms(self, terms: AbstractTermSum[TermT], filename: str) -> None:
        """Write *terms* to a gzip-compressed binary file at *filename*.

        Delegates to the term sum's own ``save`` when it has one, otherwise
        uses the `from_bytes` and `to_bytes` methods to serialize terms.

        Arguments:
            terms: The term sum to write.
            filename: Destination path.
        """
        save = getattr(terms, "save", None)
        if callable(save):
            save(filename)
            return
        _save_terms_to_file(terms.items(), filename)

    def _wrap(
        self, terms: dict[TermT, complex], like: AbstractTermSum[TermT]
    ) -> AbstractTermSum[TermT]:
        """
        Put a raw term map back into a term sum container.
        """
        container = self.term_sum_type or type(like)
        demoted = {term: self._demote(coeff) for term, coeff in terms.items()}
        return container(demoted)  # type: ignore[call-arg]

    @staticmethod
    def _demote(coeff: complex) -> complex | float:
        """Collapse *coeff* to a real `float` if its imaginary part is negligible."""
        if abs(coeff.imag) <= 1e-9 * max(abs(coeff.real), 1.0):
            return coeff.real
        return coeff
