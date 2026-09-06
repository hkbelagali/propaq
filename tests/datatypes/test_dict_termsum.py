"""Tests for `DictTermSum`, the reusable dict-backed `AbstractTermSum`."""

from dataclasses import dataclass

import pytest

from propaq.datatypes import AbstractTermSum, DictTermSum
from propaq.datatypes.abstract import AbstractTerm
from propaq.truncation import CoefficientTruncator, TermBudget, TruncationPolicy, WeightTruncator


@dataclass(frozen=True, slots=True)
class ToyTerm(AbstractTerm):
    """A minimal `AbstractTerm`: an integer bitmask, weight = popcount."""

    bits: int
    size: int = 4

    @property
    def weight(self) -> int:
        return bin(self.bits).count("1")

    @property
    def n_units(self) -> int:
        return self.size

    def commutes_with(self, other: "ToyTerm") -> bool:
        return True

    def to_bytes(self) -> bytes:
        return self.bits.to_bytes(1, "little")

    @classmethod
    def from_bytes(cls, data: bytes, n_units: int) -> "ToyTerm":
        return cls(int.from_bytes(data, "little"), n_units)

    def __matmul__(self, other: "ToyTerm") -> tuple[complex, "ToyTerm"]:
        return (1 + 0j, self)

    def __hash__(self) -> int:
        return hash(self.bits)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ToyTerm) and self.bits == other.bits

    def trace_with_fock_state(self, fock_state) -> complex:
        return 1.0 + 0j


class ToyTermSum(DictTermSum[ToyTerm]):
    term_type = ToyTerm


def test_is_an_abstract_term_sum():
    assert isinstance(DictTermSum(), AbstractTermSum)


def test_add_accumulates():
    ts = DictTermSum[ToyTerm]()
    ts.add(ToyTerm(0b01), 1 + 0j)
    ts.add(ToyTerm(0b01), 2 + 0j)
    ts.add(ToyTerm(0b10), 3 + 0j)
    assert dict(ts.items()) == {ToyTerm(0b01): 3 + 0j, ToyTerm(0b10): 3 + 0j}
    assert len(ts) == 2


def test_scale_multiplies_every_coefficient():
    ts = DictTermSum({ToyTerm(0b01): 1 + 0j, ToyTerm(0b10): 2 + 0j})
    ts.scale(2j)
    assert ts[ToyTerm(0b01)] == 2j
    assert ts[ToyTerm(0b10)] == 4j


def test_merge_adds_other_terms_in():
    a = DictTermSum({ToyTerm(0b01): 1 + 0j})
    b = DictTermSum({ToyTerm(0b01): 1 + 0j, ToyTerm(0b10): 5 + 0j})
    a.merge(b)
    assert a[ToyTerm(0b01)] == 2 + 0j
    assert a[ToyTerm(0b10)] == 5 + 0j


def test_getitem_missing_is_zero():
    ts = DictTermSum[ToyTerm]()
    assert ts[ToyTerm(0b111)] == 0j


def test_setitem_replaces_rather_than_accumulates():
    ts = DictTermSum({ToyTerm(0b01): 1 + 0j})
    ts[ToyTerm(0b01)] = 9 + 0j
    assert ts[ToyTerm(0b01)] == 9 + 0j


def test_copy_is_independent():
    a = DictTermSum({ToyTerm(0b01): 1 + 0j})
    b = a.copy()
    b.add(ToyTerm(0b01), 1 + 0j)
    assert a[ToyTerm(0b01)] == 1 + 0j
    assert b[ToyTerm(0b01)] == 2 + 0j


def test_norm_squared():
    ts = DictTermSum({ToyTerm(0b01): 3 + 4j, ToyTerm(0b10): 1 + 0j})
    assert ts.norm_squared() == pytest.approx(25.0 + 1.0)


@pytest.mark.parametrize(
    "policy",
    [
        WeightTruncator(weight=1),
        [WeightTruncator(weight=1)],
        TruncationPolicy(weight_cutoff=1, coeff_cutoff=0.0),
    ],
)
def test_truncate_by_weight(policy):
    ts = DictTermSum({ToyTerm(0b0): 1 + 0j, ToyTerm(0b1): 1 + 0j, ToyTerm(0b11): 1 + 0j})
    ts.truncate(policy)
    assert set(t.bits for t, _ in ts.items()) == {0b0, 0b1}


def test_truncate_by_coefficient():
    ts = DictTermSum({ToyTerm(0b0): 1.0, ToyTerm(0b1): 1e-6})
    ts.truncate(CoefficientTruncator(coefficient=1e-3))
    assert list(ts.items()) == [(ToyTerm(0b0), 1.0)]


def test_truncate_none_is_a_no_op():
    ts = DictTermSum({ToyTerm(0b0): 1.0, ToyTerm(0b111): 1e-9})
    ts.truncate(None)
    assert len(ts) == 2


def test_truncate_ignores_term_budget_outside_a_propagator():
    ts = DictTermSum({ToyTerm(0b0): 1.0, ToyTerm(0b111): 1e-9})
    ts.truncate([CoefficientTruncator(coefficient=1e-3), TermBudget(min_terms=2)])
    assert len(ts) == 1


def test_hermitian_builds_the_symmetrized_combination():
    out = DictTermSum.hermitian(ToyTerm(0b01), coeff=2 + 0j)
    assert dict(out.items()) == {ToyTerm(0b01): 2 + 0j}


def test_from_file_needs_a_term_type():
    with pytest.raises(TypeError, match="term_type"):
        DictTermSum.from_file("does-not-matter.gz")


def test_save_and_from_file_round_trip(tmp_path):
    path = str(tmp_path / "toy.gz")
    ts = ToyTermSum({ToyTerm(0b01): 1.5, ToyTerm(0b11): -2.0})
    ts.save(path)
    reloaded = ToyTermSum.from_file(path)
    assert dict(reloaded.items()) == {ToyTerm(0b01): 1.5 + 0j, ToyTerm(0b11): -2.0 + 0j}


def test_save_rejects_complex_coefficients(tmp_path):
    path = str(tmp_path / "toy.gz")
    ts = ToyTermSum({ToyTerm(0b01): 1 + 2j})
    with pytest.raises(ValueError, match="complex"):
        ts.save(path)


def test_repr_names_the_term_count():
    ts = DictTermSum({ToyTerm(0b01): 1.0})
    assert "1" in repr(ts)
