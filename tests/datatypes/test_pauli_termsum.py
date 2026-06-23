import pytest

from propaq.datatypes import PauliString, PauliTermStreamer, PauliTermSum


def ps(x: int, z: int, n_qubits: int = 4) -> PauliString:
    return PauliString(x, z=z, n_qubits=n_qubits)


def test_streamer_round_trip_single_term(tmp_path):
    a = ps(0b0001, 0)
    ts = PauliTermSum({a: 2.0 + 1j})
    path = str(tmp_path / "ts.gz")
    ts.save(path)

    pairs = list(PauliTermStreamer.from_file(path))
    assert len(pairs) == 1
    term, coeff = pairs[0]
    assert term == a
    assert coeff == pytest.approx(2.0 + 1j)


def test_streamer_round_trip_multi_term(tmp_path):
    a, b, c = ps(0b0001, 0), ps(0, 0b0010), ps(0b0001, 0b0001)
    ts = PauliTermSum({a: 1.0, b: -0.5j, c: 3.0 + 2j})
    path = str(tmp_path / "ts.gz")
    ts.save(path)

    result = {term: coeff for term, coeff in PauliTermStreamer.from_file(path)}
    assert len(result) == 3
    assert result[a] == pytest.approx(1.0)
    assert result[b] == pytest.approx(-0.5j)
    assert result[c] == pytest.approx(3.0 + 2j)


def test_streamer_empty_file(tmp_path):
    path = str(tmp_path / "empty.gz")
    PauliTermSum().save(path)

    assert list(PauliTermStreamer.from_file(path)) == []


def test_merge_from_file_matches_from_file(tmp_path):
    a, b = ps(0b0001, 0), ps(0, 0b0010)
    ts = PauliTermSum({a: 1.5, b: -2.0 + 1j})
    path = str(tmp_path / "ts.gz")
    ts.save(path)

    ref = PauliTermSum.from_file(path)
    ts2 = PauliTermSum()
    ts2.merge_from_file(PauliTermStreamer.from_file(path))

    assert len(ts2) == len(ref)
    for term, coeff in ref.items():
        assert ts2[term] == pytest.approx(coeff)


def test_merge_from_file_accumulates_into_existing(tmp_path):
    a, b = ps(0b0001, 0), ps(0, 0b0010)
    existing = PauliTermSum({a: 1.0})
    saved = PauliTermSum({a: 2.0, b: 4.0})
    path = str(tmp_path / "ts.gz")
    saved.save(path)

    existing.merge_from_file(PauliTermStreamer.from_file(path))
    assert len(existing) == 2
    assert existing[a] == pytest.approx(3.0)
    assert existing[b] == pytest.approx(4.0)


def test_merge_from_file_empty_file_leaves_sum_unchanged(tmp_path):
    a = ps(0b0001, 0)
    ts = PauliTermSum({a: 5.0})
    empty_path = str(tmp_path / "empty.gz")
    PauliTermSum().save(empty_path)

    ts.merge_from_file(PauliTermStreamer.from_file(empty_path))
    assert len(ts) == 1
    assert ts[a] == pytest.approx(5.0)
