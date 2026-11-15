"""IR serialization, canonical JSON and hash stability."""

import json
import random

import pytest

from propaq.tracer.analysis import add_fold
from propaq.tracer.bases import ALL
from propaq.tracer.evaluate import evaluate
from propaq.tracer.ir import IRError, BasisIR, Node, Program

NAMES = sorted(ALL)


def full_ir(name):
    return add_fold(ALL[name]().trace_ir())


@pytest.mark.parametrize("name", NAMES)
def test_json_round_trip(name):
    ir = full_ir(name)
    back = BasisIR.from_json(ir.to_json())
    assert back.to_json() == ir.to_json()
    assert back.hash() == ir.hash()
    rng = random.Random(1)
    for pname, p in ir.programs.items():
        q = back.programs[pname]
        ins = [[rng.getrandbits(64) for _ in range(2)] for _ in p.inputs]
        assert evaluate(p, ins, n_units=20) == evaluate(q, ins, n_units=20)


@pytest.mark.parametrize("name", NAMES)
def test_hash_is_stable_across_traces(name):
    assert full_ir(name).hash() == full_ir(name).hash()
    assert len(full_ir(name).hash16()) == 16


def test_hashes_differ_between_bases():
    assert len({full_ir(n).hash() for n in NAMES}) == len(NAMES)


def test_canonical_json_form():
    text = full_ir("pauli").to_json()
    obj = json.loads(text)
    assert obj["ir_version"] == 1
    assert text == json.dumps(obj, sort_keys=True, separators=(",", ":"))
    prog = obj["programs"]["character"]
    assert set(prog) == {"inputs", "nodes", "outputs"}
    assert all(len(n) == 4 for n in prog["nodes"])  # [op, kind, args, imm]
    assert "meta" in obj and obj["meta"]["name"] == "pauli"


def test_hash_changes_with_a_node():
    ir = full_ir("pauli")
    h = ir.hash()
    ir.programs["weight"].nodes.append(Node("const", "int", (), 1))
    assert ir.hash() != h


def test_hash_changes_with_meta():
    ir = full_ir("pauli")
    h = ir.hash()
    ir.meta["phase_order"] = 8
    assert ir.hash() != h


def test_hash_is_sha256_of_canonical_json():
    import hashlib

    ir = full_ir("majorana")
    assert ir.hash() == hashlib.sha256(ir.to_json().encode()).hexdigest()


def test_unknown_version_rejected():
    obj = json.loads(full_ir("pauli").to_json())
    obj["ir_version"] = 2
    with pytest.raises(IRError):
        BasisIR.from_json(json.dumps(obj))


@pytest.mark.parametrize("node", [
    Node("and", "bits", (0,), None),        # wrong arity
    Node("and", "bits", (0, 5), None),      # forward reference
    Node("popcount", "int", (0,), 3),       # stray immediate
    Node("shl_w", "bits", (0,), 64),        # bad shift
    Node("expand_units", "bits", (0,), (3, 0)),  # b does not divide 64
    Node("rem", "int", (1,), 0),            # bad modulus
    Node("nope", "bits", (), None),         # unknown op
])
def test_validation_rejects_bad_nodes(node):
    p = Program("x", ["bits", "int"], [Node("input", "bits", (), 0), Node("input", "int", (), 1), node], [2])
    with pytest.raises(IRError):
        p.validate()


def test_validation_rejects_kind_mismatch():
    nodes = [Node("input", "bits", (), 0), Node("add", "int", (0, 0), None)]
    with pytest.raises(IRError):
        Program("x", ["bits"], nodes, [1]).validate()
