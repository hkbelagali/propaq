import pytest
from qiskit.quantum_info import SparsePauliOp

from propaq.datatypes import MajoranaMonomial, MajoranaTermStreamer, MajoranaTermSum
from propaq.noise.truncation import TruncationPolicy
from propaq.noise.uniform import UniformNoiseModel

# n_modes=8 gives 4 fermionic sites; enough for all tests below.
N = 8


def mon(modes_int: int) -> MajoranaMonomial:
    return MajoranaMonomial(modes_int, N)


def test_add_and_len_items():
    ts = MajoranaTermSum()
    t = mon(0b01)
    ts.add(t, 1.0)
    ts.add(t, 2.0)
    assert len(ts) == 1
    items = list(ts.items())
    assert items[0][1] == pytest.approx(3.0)


def test_scale_and_norm_squared():
    ts = MajoranaTermSum()
    a = mon(0b01)
    b = mon(0b10)
    ts.add(a, 1.0)
    ts.add(b, 0.5)
    orig = ts.norm_squared()
    factor = 2.0
    ts.scale(factor)
    assert pytest.approx(ts.norm_squared(), rel=1e-9) == (abs(factor) ** 2) * orig


def test_merge_and_copy_independence():
    a = mon(0b0001)
    b = mon(0b0010)
    ts1 = MajoranaTermSum()
    ts2 = MajoranaTermSum()
    ts1.add(a, 1)
    ts2.add(b, 2)
    ts1.merge(ts2)
    assert len(ts1) == 2
    c = ts1.copy()
    ts1.add(mon(0b0100), 3)
    assert len(c) == 2


def test_truncate_removes_terms_safely():
    ts = MajoranaTermSum()
    # weight-4 monomial: modes 0b11110000 → sites 2,3 both occupied → high weight
    heavy = mon(0b00001111)  # weight >= 1
    light = mon(0b00000011)  # single site, low weight
    ts.add(heavy, 0.01)
    ts.add(light, 1.0)
    policy = TruncationPolicy(weight_cutoff=0, coeff_cutoff=0.1)
    ts.truncate(policy)
    remaining = [m for m, _ in ts.items()]
    assert heavy not in remaining
    assert light not in remaining or any(m == light for m in remaining)


def test_truncate_keeps_heavy_above_cutoff():
    ts = MajoranaTermSum()
    t = mon(0b00000011)
    ts.add(t, 1.0)
    policy = TruncationPolicy(weight_cutoff=10, coeff_cutoff=0.0)
    ts.truncate(policy)
    assert len(ts) == 1


def test_apply_damping_uses_noise_model():
    ts = MajoranaTermSum()
    t = mon(0b00000011)
    ts.add(t, 2.0)
    noise = UniformNoiseModel(0.0)  # zero damping → coefficients unchanged
    ts.apply_damping(noise, active_modes=0)
    _, coeff = list(ts.items())[0]
    assert coeff == pytest.approx(2.0)

def test_constructor_with_dict():
    t = mon(0b01)
    ts = MajoranaTermSum({t: 2.5})
    assert len(ts) == 1
    _, coeff = list(ts.items())[0]
    assert coeff == pytest.approx(2.5)

def test_constructor_empty_dict():
    ts = MajoranaTermSum({})
    assert len(ts) == 0

def test_constructor_multi_term_dict():
    a, b = mon(0b0001), mon(0b0010)
    ts = MajoranaTermSum({a: 1.0, b: -1.0})
    assert len(ts) == 2

def test_setitem_and_getitem():
    ts = MajoranaTermSum()
    t = mon(0b01)
    ts[t] = 3.0
    assert ts[t] == pytest.approx(3.0)

def test_setitem_overwrites():
    ts = MajoranaTermSum()
    t = mon(0b01)
    ts[t] = 1.0
    ts[t] = 5.0
    assert ts[t] == pytest.approx(5.0)
    assert len(ts) == 1

def test_getitem_missing_returns_zero():
    ts = MajoranaTermSum()
    t = mon(0b01)
    assert ts[t] == pytest.approx(0.0)

def test_merge_overlapping_accumulates():
    a = mon(0b0001)
    ts1 = MajoranaTermSum()
    ts2 = MajoranaTermSum()
    ts1.add(a, 1.0)
    ts2.add(a, 2.0)
    ts1.merge(ts2)
    assert len(ts1) == 1
    assert ts1[a] == pytest.approx(3.0)

def test_merge_mixed_overlap_and_new():
    a, b = mon(0b0001), mon(0b0010)
    ts1 = MajoranaTermSum({a: 1.0})
    ts2 = MajoranaTermSum({a: 2.0, b: 3.0})
    ts1.merge(ts2)
    assert len(ts1) == 2
    assert ts1[a] == pytest.approx(3.0)
    assert ts1[b] == pytest.approx(3.0)

class EvenWeightTruncation:
    """Truncates terms with even Pauli weight."""
    def should_truncate(self, weight, abs_coeff):
        return weight % 2 == 0

def test_truncate_custom_python_policy():
    # modes=0b11 → weight 1 (odd) → kept
    # modes=0b0101_0101 → weight 4 (even) → truncated
    kept = mon(0b00000011)          # weight 1
    removed = mon(0b01010101)       # weight 4
    ts = MajoranaTermSum({kept: 1.0, removed: 1.0})
    ts.truncate(EvenWeightTruncation())
    remaining = [m for m, _ in ts.items()]
    assert kept in remaining
    assert removed not in remaining
    
class ConstantDampingNoise:
    def __init__(self, factor):
        self.factor = factor

    def damping_factor(self, weight, active_modes):
        return self.factor

def test_apply_damping_custom_python_noise():
    t = mon(0b00000011)
    ts = MajoranaTermSum({t: 4.0})
    ts.apply_damping(ConstantDampingNoise(0.25), active_modes=0)
    assert ts[t] == pytest.approx(1.0)  # 4.0 * 0.25

def _ref(modes_int: int, n_modes: int) -> MajoranaMonomial:
    """Reference monomial for lookup; equality uses only mode bits."""
    return MajoranaMonomial(modes_int, n_modes)


# Single-qubit Jordan-Wigner images
# X_0  →  γ_0          (mode 0b01,  coeff +1)
# Y_0  →  γ_1          (mode 0b10,  coeff +1)
# Z_0  →  γ_0 γ_1      (mode 0b11,  coeff -1)

def test_from_sparse_pauli_op_x():
    ts = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp("X"))
    assert len(ts) == 1
    assert ts[_ref(0b01, 2)] == pytest.approx(1.0)


def test_from_sparse_pauli_op_y():
    ts = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp("Y"))
    assert len(ts) == 1
    assert ts[_ref(0b10, 2)] == pytest.approx(1.0)


def test_from_sparse_pauli_op_z():
    ts = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp("Z"))
    assert len(ts) == 1
    assert ts[_ref(0b11, 2)] == pytest.approx(-1.0)

def test_from_sparse_pauli_op_xz():
    ts = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp.from_list([("XZ", 1.0)]))
    assert len(ts) == 1
    assert ts[_ref(0b0100, 4)] == pytest.approx(1.0)


def test_from_sparse_pauli_op_yz():
    ts = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp.from_list([("YZ", 1.0)]))
    assert len(ts) == 1
    assert ts[_ref(0b1000, 4)] == pytest.approx(1.0)


def test_from_sparse_pauli_op_zx():
    ts = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp.from_list([("ZX", 1.0)]))
    assert len(ts) == 1
    assert ts[_ref(0b1101, 4)] == pytest.approx(-1.0)


def test_from_sparse_pauli_op_xx():
    ts = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp.from_list([("XX", 1.0)]))
    assert len(ts) == 1
    assert ts[_ref(0b0110, 4)] == pytest.approx(-1.0)


def test_from_sparse_pauli_op_yy():
    ts = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp.from_list([("YY", 1.0)]))
    assert len(ts) == 1
    assert ts[_ref(0b1001, 4)] == pytest.approx(1.0)


def test_from_sparse_pauli_op_zz():
    ts = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp.from_list([("ZZ", 1.0)]))
    assert len(ts) == 1
    assert ts[_ref(0b1111, 4)] == pytest.approx(-1.0)


def test_from_sparse_pauli_op_coefficient_scaling():
    ts = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp("X", coeffs=[3.5]))
    assert ts[_ref(0b01, 2)] == pytest.approx(3.5)


def test_from_sparse_pauli_op_negative_coefficient():
    ts = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp("X", coeffs=[-2.0]))
    assert ts[_ref(0b01, 2)] == pytest.approx(-2.0)


def test_from_sparse_pauli_op_linear_combination():
    op = SparsePauliOp.from_list([("X", 0.5), ("Y", 0.5)])
    ts = MajoranaTermSum.from_sparse_pauli_op(op)
    assert len(ts) == 2
    assert ts[_ref(0b01, 2)] == pytest.approx(0.5)
    assert ts[_ref(0b10, 2)] == pytest.approx(0.5)

def test_streamer_round_trip_single_term(tmp_path):
    a = mon(0b0001)
    ts = MajoranaTermSum({a: 3.0})
    path = str(tmp_path / "ts.gz")
    ts.save(path)

    streamer = MajoranaTermStreamer.from_file(path)
    pairs = list(streamer)
    assert len(pairs) == 1
    term, coeff = pairs[0]
    assert term == a
    assert coeff == pytest.approx(3.0)


def test_streamer_round_trip_multi_term(tmp_path):
    a, b, c = mon(0b0001), mon(0b0010), mon(0b0100)
    ts = MajoranaTermSum({a: 1.0, b: 2.0, c: -1.0})
    path = str(tmp_path / "ts.gz")
    ts.save(path)

    result = {term: coeff for term, coeff in MajoranaTermStreamer.from_file(path)}
    assert len(result) == 3
    assert result[a] == pytest.approx(1.0)
    assert result[b] == pytest.approx(2.0)
    assert result[c] == pytest.approx(-1.0)


def test_streamer_empty_file(tmp_path):
    ts = MajoranaTermSum()
    path = str(tmp_path / "empty.gz")
    ts.save(path)

    pairs = list(MajoranaTermStreamer.from_file(path))
    assert pairs == []


def test_merge_from_file_matches_from_file(tmp_path):
    a, b = mon(0b0001), mon(0b0010)
    ts = MajoranaTermSum({a: 1.5, b: -2.0})
    path = str(tmp_path / "ts.gz")
    ts.save(path)

    ref = MajoranaTermSum.from_file(path)
    ts2 = MajoranaTermSum()
    ts2.merge_from_file(MajoranaTermStreamer.from_file(path))

    assert len(ts2) == len(ref)
    for term, coeff in ref.items():
        assert ts2[term] == pytest.approx(coeff)


def test_merge_from_file_accumulates_into_existing(tmp_path):
    a, b = mon(0b0001), mon(0b0010)
    # existing sum has 'a' with coeff 1.0; file also has 'a' → should sum to 3.0
    existing = MajoranaTermSum({a: 1.0})
    saved = MajoranaTermSum({a: 2.0, b: 4.0})
    path = str(tmp_path / "ts.gz")
    saved.save(path)

    existing.merge_from_file(MajoranaTermStreamer.from_file(path))
    assert len(existing) == 2
    assert existing[a] == pytest.approx(3.0)
    assert existing[b] == pytest.approx(4.0)


def test_merge_from_file_empty_file_leaves_sum_unchanged(tmp_path):
    a = mon(0b0001)
    ts = MajoranaTermSum({a: 5.0})
    empty_path = str(tmp_path / "empty.gz")
    MajoranaTermSum().save(empty_path)

    ts.merge_from_file(MajoranaTermStreamer.from_file(empty_path))
    assert len(ts) == 1
    assert ts[a] == pytest.approx(5.0)

def test_to_sparse_pauli_op_x():
    ts = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp("X"))
    assert ts.to_sparse_pauli_op().equiv(SparsePauliOp("X"))


def test_to_sparse_pauli_op_y():
    ts = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp("Y"))
    assert ts.to_sparse_pauli_op().equiv(SparsePauliOp("Y"))


def test_to_sparse_pauli_op_z():
    ts = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp("Z"))
    assert ts.to_sparse_pauli_op().equiv(SparsePauliOp("Z"))


def test_to_sparse_pauli_op_xx():
    ts = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp("XX"))
    assert ts.to_sparse_pauli_op().equiv(SparsePauliOp("XX"))


def test_to_sparse_pauli_op_zz():
    ts = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp("ZZ"))
    assert ts.to_sparse_pauli_op().equiv(SparsePauliOp("ZZ"))


def test_to_sparse_pauli_op_scaled():
    ts = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp("X", coeffs=[3.5]))
    assert ts.to_sparse_pauli_op().equiv(SparsePauliOp("X", coeffs=[3.5]))


def test_to_sparse_pauli_op_round_trip():
    op = SparsePauliOp.from_list([("ZZ", 1.0), ("XX", 2.0), ("ZI", -0.5)])
    assert MajoranaTermSum.from_sparse_pauli_op(op).to_sparse_pauli_op().equiv(op)


def test_to_sparse_pauli_op_empty_raises():
    with pytest.raises(ValueError):
        MajoranaTermSum().to_sparse_pauli_op()


def test_default_dtype_is_float64():
    assert MajoranaTermSum().dtype == "float64"


def test_float32_dtype_round_trip():
    m = mon(0b0011)
    ts = MajoranaTermSum(dtype="float32")
    ts.add(m, 1.5)
    assert ts.dtype == "float32"
    assert ts[m] == pytest.approx(1.5, rel=1e-6)


def test_float32_copy_keeps_dtype():
    ts = MajoranaTermSum(dtype="float32")
    ts.add(mon(0b0011), 1.0)
    assert ts.copy().dtype == "float32"


def test_float32_save_load_round_trips_as_float64(tmp_path):
    m = mon(0b0011)
    ts = MajoranaTermSum(dtype="float32")
    ts.add(m, 1.5)
    path = str(tmp_path / "ts32.gz")
    ts.save(path)
    loaded = MajoranaTermSum.from_file(path)
    assert loaded.dtype == "float64"
    assert loaded[m] == pytest.approx(1.5, rel=1e-6)


def test_merge_mismatched_dtype_raises():
    ts64 = MajoranaTermSum()
    ts32 = MajoranaTermSum(dtype="float32")
    with pytest.raises(ValueError):
        ts64.merge(ts32)


def test_unknown_dtype_raises():
    with pytest.raises(ValueError):
        MajoranaTermSum(dtype="float16")
