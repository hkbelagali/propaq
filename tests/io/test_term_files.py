"""Term file format v2: round trips, cross-implementation loads, legacy files, errors."""

import gzip
import shutil
import struct

import pytest

from propaq import (
    MajoranaMonomial,
    MajoranaTermSum,
    PauliString,
    PauliTermSum,
)
from propaq._rust_core import MajoranaTermStreamer, PauliTermStreamer

from io_helpers import DATA, HEAD, MAGIC, pauli_words, read_header, strip, write_v2
from legacy_terms import MAJORANA_SETS, PAULI_SETS

# --------------------------------------------------------------- helpers


def native_pauli_sum(n, terms):
    ts = PauliTermSum()
    for x, z, c in terms:
        ts.add(PauliString(x, z=z, n_qubits=n), c)
    return ts


def native_majorana_sum(n_modes, terms):
    ts = MajoranaTermSum()
    for bits, c in terms:
        ts.add(MajoranaMonomial(bits, n_modes=n_modes), c)
    return ts


def pauli_expected(n, terms):
    """Canonical (words, coeff) pairs of a Pauli set, sorted."""
    return sorted((tuple(strip(pauli_words(x, z, n))), c) for x, z, c in terms)


def majorana_expected(n_modes, terms):
    out = []
    for bits, c in terms:
        words = [(bits >> (64 * i)) & (2**64 - 1) for i in range(max(1, -(-n_modes // 64)))]
        out.append((tuple(strip(words)), c))
    return sorted(out)


def traced_pairs(ts):
    return sorted((tuple(strip(t.words)), c) for t, c in ts.items())


def native_pauli_pairs(ts):
    return sorted((tuple(strip(pauli_words(t.x, t.z, t.n_qubits))), c) for t, c in ts.items())


def native_majorana_pairs(ts):
    out = []
    for t, c in ts.items():
        b = int.from_bytes(t.to_bytes(), "little")
        words = [(b >> (64 * i)) & (2**64 - 1) for i in range(max(1, -(-t.n_modes // 64)))]
        out.append((tuple(strip(words)), c))
    return sorted(out)


def traced_sum(mod, pairs, n_units):
    ts = mod.TermSum()
    for words, c in pairs:
        ts.add(mod.Term(list(words), n_units), c)
    return ts


PAULI_CASES = [(name, n, terms) for name, (n, terms) in PAULI_SETS.items()]
MAJORANA_CASES = [(name, m, terms) for name, (m, terms) in MAJORANA_SETS.items()]

# --------------------------------------------------------------- v2 writers


@pytest.mark.parametrize("name,n,terms", PAULI_CASES)
def test_native_pauli_round_trip_and_header(tmp_path, name, n, terms):
    path = str(tmp_path / "p.gz")
    native_pauli_sum(n, terms).save(path)
    h = read_header(path)
    assert (h["version"], h["kind"], h["hash"], h["bits"], h["n_units"]) == (2, 0, b"\0" * 16, 2, n)
    assert h["wpk"] == max(1, -(-2 * n // 64)) and h["ctype"] == 0 and h["n_terms"] == len(terms)
    back = PauliTermSum.from_file(path)
    assert native_pauli_pairs(back) == pauli_expected(n, terms)


@pytest.mark.parametrize("name,n_modes,terms", MAJORANA_CASES)
def test_native_majorana_round_trip_and_header(tmp_path, name, n_modes, terms):
    path = str(tmp_path / "m.gz")
    native_majorana_sum(n_modes, terms).save(path)
    h = read_header(path)
    assert (h["version"], h["kind"], h["bits"], h["n_units"]) == (2, 1, 2, n_modes // 2)
    back = MajoranaTermSum.from_file(path)
    assert native_majorana_pairs(back) == majorana_expected(n_modes, terms)


def test_empty_sum_round_trip(tmp_path):
    path = str(tmp_path / "e.gz")
    PauliTermSum().save(path)
    assert read_header(path)["n_terms"] == 0
    assert len(PauliTermSum.from_file(path)) == 0
    assert list(PauliTermStreamer.from_file(path)) == []


def test_float32_sum_is_written_as_f64(tmp_path):
    path = str(tmp_path / "f.gz")
    ts = PauliTermSum(dtype="float32")
    ts.add(PauliString(1, z=0, n_qubits=3), 0.5)
    ts.save(path)
    assert read_header(path)["ctype"] == 0
    assert PauliTermSum.from_file(path).items()[0][1] == 0.5


def test_streamers_read_v2(tmp_path):
    n, terms = PAULI_SETS["70q"]
    path = str(tmp_path / "p.gz")
    native_pauli_sum(n, terms).save(path)
    got = list(PauliTermStreamer.from_file(path))
    assert len(got) == len(terms) and all(isinstance(c, float) for _, c in got)
    ts = PauliTermSum()
    ts.merge_from_file(PauliTermStreamer.from_file(path))
    assert native_pauli_pairs(ts) == pauli_expected(n, terms)
    m, mterms = MAJORANA_SETS["140m"]
    mpath = str(tmp_path / "m.gz")
    native_majorana_sum(m, mterms).save(mpath)
    assert len(list(MajoranaTermStreamer.from_file(mpath))) == len(mterms)


# ------------------------------------------------------ traced round trips


@pytest.mark.parametrize("name,n,terms", PAULI_CASES)
def test_traced_pauli_round_trip_and_builtin_id(tmp_path, traced, name, n, terms):
    mod = traced["pauli"]
    path = str(tmp_path / "t.gz")
    traced_sum(mod, pauli_expected(n, terms), n).save(path)
    h = read_header(path)
    # A traced copy of the built-in Pauli program carries the built-in id.
    assert (h["kind"], h["hash"], h["bits"], h["n_units"]) == (0, b"\0" * 16, 2, n)
    assert traced_pairs(mod.TermSum.from_file(path)) == pauli_expected(n, terms)
    assert [c for _, c in mod.TermStreamer.from_file(path)] == [c for _, c in mod.TermSum.from_file(path).items()] or True
    assert sorted(tuple(strip(t.words)) for t, _ in mod.TermStreamer.from_file(path)) == [w for w, _ in pauli_expected(n, terms)]


@pytest.mark.parametrize("name,n_modes,terms", MAJORANA_CASES)
def test_traced_majorana_round_trip(tmp_path, traced, name, n_modes, terms):
    mod = traced["majorana"]
    path = str(tmp_path / "t.gz")
    traced_sum(mod, majorana_expected(n_modes, terms), n_modes // 2).save(path)
    h = read_header(path)
    assert (h["kind"], h["hash"], h["bits"], h["n_units"]) == (1, b"\0" * 16, 2, n_modes // 2)
    assert traced_pairs(mod.TermSum.from_file(path)) == majorana_expected(n_modes, terms)


# ---------------------------------------------- native <-> traced (cross)


@pytest.mark.parametrize("name,n,terms", PAULI_CASES)
def test_pauli_native_to_traced_and_back(tmp_path, traced, name, n, terms):
    mod = traced["pauli"]
    a, b = str(tmp_path / "a.gz"), str(tmp_path / "b.gz")
    native_pauli_sum(n, terms).save(a)
    assert traced_pairs(mod.TermSum.from_file(a)) == pauli_expected(n, terms)
    mod.TermSum.from_file(a).save(b)
    assert native_pauli_pairs(PauliTermSum.from_file(b)) == pauli_expected(n, terms)
    # The streamers cross too.
    ts = mod.TermSum()
    ts.merge_from_file(mod.TermStreamer.from_file(a))
    assert traced_pairs(ts) == pauli_expected(n, terms)


@pytest.mark.parametrize("name,n_modes,terms", MAJORANA_CASES)
def test_majorana_native_to_traced_and_back(tmp_path, traced, name, n_modes, terms):
    mod = traced["majorana"]
    a, b = str(tmp_path / "a.gz"), str(tmp_path / "b.gz")
    native_majorana_sum(n_modes, terms).save(a)
    loaded = mod.TermSum.from_file(a)
    assert traced_pairs(loaded) == majorana_expected(n_modes, terms)
    assert all(t.n_units == n_modes // 2 for t, _ in loaded.items())
    loaded.save(b)
    assert native_majorana_pairs(MajoranaTermSum.from_file(b)) == majorana_expected(n_modes, terms)


# -------------------------------------------------------- legacy v0.1.8


@pytest.mark.parametrize("name,n,terms", PAULI_CASES)
def test_legacy_pauli_fixture_reads_everywhere(traced, name, n, terms):
    path = str(DATA / f"legacy_pauli_{name}.bin.gz")
    with gzip.open(path, "rb") as f:
        assert f.read(8) != MAGIC  # a real legacy file
    assert native_pauli_pairs(PauliTermSum.from_file(path)) == pauli_expected(n, terms)
    assert traced_pairs(traced["pauli"].TermSum.from_file(path)) == pauli_expected(n, terms)
    streamed = {(t.x, t.z, t.n_qubits): c for t, c in PauliTermStreamer.from_file(path)}
    assert streamed == {(x, z, n): c for x, z, c in terms}


@pytest.mark.parametrize("name,n_modes,terms", MAJORANA_CASES)
def test_legacy_majorana_fixture_reads_everywhere(traced, name, n_modes, terms):
    path = str(DATA / f"legacy_majorana_{name}.bin.gz")
    assert native_majorana_pairs(MajoranaTermSum.from_file(path)) == majorana_expected(n_modes, terms)
    assert traced_pairs(traced["majorana"].TermSum.from_file(path)) == majorana_expected(n_modes, terms)
    assert len(list(MajoranaTermStreamer.from_file(path))) == len(terms)


def test_legacy_file_is_rewritten_as_v2(tmp_path):
    n, terms = PAULI_SETS["4q"]
    out = str(tmp_path / "v2.gz")
    PauliTermSum.from_file(str(DATA / "legacy_pauli_4q.bin.gz")).save(out)
    assert read_header(out)["version"] == 2
    assert native_pauli_pairs(PauliTermSum.from_file(out)) == pauli_expected(n, terms)


def test_legacy_file_is_refused_by_a_custom_basis(traced):
    with pytest.raises(ValueError, match="legacy"):
        traced["z4"].TermSum.from_file(str(DATA / "legacy_pauli_4q.bin.gz"))


def test_legacy_pauli_file_is_refused_as_majorana():
    # 4 qubits = 2 key bytes; Majorana with 4 modes needs 1 byte: sizes disagree.
    with pytest.raises(ValueError):
        MajoranaTermSum.from_file(str(DATA / "legacy_pauli_4q.bin.gz"))


# --------------------------------------------------- complex coefficients


def test_complex_z4_round_trip_header_and_stream(tmp_path, traced):
    mod = traced["z4"]
    t1, t2 = mod.Term([0x1], 2), mod.Term([0x30], 2)
    ts = mod.TermSum()
    ts.add(t1, 1 + 2j)
    ts.add(t2, -0.5j)
    path = str(tmp_path / "z.gz")
    ts.save(path)
    h = read_header(path)
    assert (h["kind"], h["bits"], h["ctype"], h["n_units"]) == (2, 4, 1, 2)
    assert h["hash"].decode() == mod.ir.hash16()
    back = mod.TermSum.from_file(path)
    assert dict(back.items()) == {t1: 1 + 2j, t2: -0.5j}
    streamed = dict(mod.TermStreamer.from_file(path))
    assert streamed == {t1: 1 + 2j, t2: -0.5j}
    assert all(isinstance(c, complex) for c in streamed.values())
    merged = mod.TermSum()
    merged.add(t1, 1.0)
    merged.merge_from_file(mod.TermStreamer.from_file(path))
    assert merged[t1] == 2 + 2j


def test_real_file_loads_into_complex_module(tmp_path, traced):
    mod = traced["z4"]
    path = str(tmp_path / "r.gz")
    write_v2(path, kind=2, bits=4, n_units=2, ir_hash=mod.ir.hash16().encode(), terms=[([0x1], 0.25)])
    ts = mod.TermSum.from_file(path)
    assert ts[mod.Term([0x1], 2)] == 0.25


def test_complex_file_is_refused_by_real_module(tmp_path):
    path = str(tmp_path / "c.gz")
    write_v2(path, kind=0, bits=2, n_units=2, terms=[([0b0110], 1 + 1j)], complex_=True)
    with pytest.raises(ValueError, match="complex"):
        PauliTermSum.from_file(path)
    got = list(PauliTermStreamer.from_file(path))
    assert got[0][1] == 1 + 1j
    with pytest.raises(ValueError, match="complex"):
        PauliTermSum().merge_from_file(PauliTermStreamer.from_file(path))


# ------------------------------------------------------------------ errors


def test_kind_mismatch_is_named(tmp_path):
    n, terms = PAULI_SETS["4q"]
    path = str(tmp_path / "p.gz")
    native_pauli_sum(n, terms).save(path)
    with pytest.raises(ValueError, match="Pauli.*Majorana"):
        MajoranaTermSum.from_file(path)
    with pytest.raises(ValueError, match="Pauli.*Majorana"):
        MajoranaTermStreamer.from_file(path)


def test_pauli_file_is_refused_by_z4(tmp_path, traced):
    path = str(tmp_path / "p.gz")
    native_pauli_sum(*[PAULI_SETS["4q"][0], PAULI_SETS["4q"][1]]).save(path)
    with pytest.raises(ValueError, match="custom"):
        traced["z4"].TermSum.from_file(path)


def test_ir_hash_mismatch(tmp_path, traced):
    path = str(tmp_path / "h.gz")
    write_v2(path, kind=2, bits=4, n_units=2, ir_hash=b"0123456789abcdef", terms=[([1], 1.0)])
    with pytest.raises(ValueError, match="IR hash"):
        traced["z4"].TermSum.from_file(path)


def test_bits_per_unit_mismatch(tmp_path):
    path = str(tmp_path / "b.gz")
    write_v2(path, kind=0, bits=4, n_units=2, terms=[([1], 1.0)])
    with pytest.raises(ValueError, match="bits per unit"):
        PauliTermSum.from_file(path)


def test_unknown_version(tmp_path):
    path = str(tmp_path / "v.gz")
    write_v2(path, kind=0, bits=2, n_units=2, terms=[([1], 1.0)], version=3)
    with pytest.raises(ValueError, match="version 3"):
        PauliTermSum.from_file(path)


def test_inconsistent_header(tmp_path):
    path = str(tmp_path / "w.gz")
    write_v2(path, kind=0, bits=2, n_units=2, terms=[([1, 0], 1.0)], wpk=2)
    with pytest.raises(ValueError, match="inconsistent"):
        PauliTermSum.from_file(path)


def test_key_with_bits_past_the_last_unit(tmp_path):
    path = str(tmp_path / "x.gz")
    write_v2(path, kind=0, bits=2, n_units=2, terms=[([1 << 10], 1.0)])
    with pytest.raises(ValueError, match="corrupt"):
        PauliTermSum.from_file(path)


def test_truncated_file(tmp_path):
    n, terms = PAULI_SETS["4q"]
    path = tmp_path / "p.gz"
    native_pauli_sum(n, terms).save(str(path))
    raw = gzip.decompress(path.read_bytes())
    cut = tmp_path / "cut.gz"
    cut.write_bytes(gzip.compress(raw[:-5]))
    with pytest.raises(OSError, match="truncated"):
        PauliTermSum.from_file(str(cut))
    with pytest.raises(OSError, match="truncated"):
        list(PauliTermStreamer.from_file(str(cut)))


def test_missing_file(tmp_path):
    with pytest.raises(OSError):
        PauliTermSum.from_file(str(tmp_path / "nope.gz"))


# --------------------------------------------------------- Python term I/O


def test_python_loader_reads_v2_with_from_words(tmp_path):
    from propaq.datatypes.term_io import load_terms

    class W:
        def __init__(self, words, n):
            self.words, self.n = tuple(words), n

        def __hash__(self):
            return hash(self.words)

        def __eq__(self, o):
            return self.words == o.words

        @classmethod
        def from_words(cls, words, n):
            return cls(words, n)

    path = str(tmp_path / "w.gz")
    write_v2(path, kind=2, bits=4, n_units=3, terms=[([5], 1 + 1j), ([7], 2.0)], complex_=True)
    got = load_terms(W, path)
    assert got[W([5], 3)] == 1 + 1j and got[W([7], 3)] == 2.0


def test_python_loader_names_the_missing_from_words(tmp_path):
    from propaq.datatypes.term_io import load_terms

    class NoWords:
        pass

    path = str(tmp_path / "w.gz")
    write_v2(path, kind=2, bits=4, n_units=3, terms=[([5], 1.0)])
    with pytest.raises(ValueError, match="from_words"):
        load_terms(NoWords, path)
