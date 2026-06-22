import pytest

from propaq.datatypes import PauliString


def X(qubit=0, n_qubits=1):
    return PauliString(1 << qubit, z=0, n_qubits=n_qubits)


def Z(qubit=0, n_qubits=1):
    return PauliString(0, z=(1 << qubit), n_qubits=n_qubits)


def Y(qubit=0, n_qubits=1):
    return PauliString(1 << qubit, z=(1 << qubit), n_qubits=n_qubits)


def I(qubit=0, n_qubits=1):
    return PauliString(0, z=0, n_qubits=n_qubits)

def test_xz_anticommute():
    x = X(n_qubits=1)
    z = Z(n_qubits=1)
    assert not x.commutes_with(z)


def test_xx_commute():
    x1 = X(n_qubits=1)
    x2 = X(n_qubits=1)
    assert x1.commutes_with(x2)


def test_xz_multiply():
    x = X(n_qubits=1)
    z = Z(n_qubits=1)
    phase, result = x @ z  # X @ Z = -iY
    assert phase == -1j
    assert result == Y(n_qubits=1)


@pytest.mark.parametrize(
    "a,b,expected_phase,expected_term",
    [
        (X(), X(), 1, I()),
        (Y(), Y(), 1, I()),
        (Z(), Z(), 1, I()),
        (X(), Y(), 1j, Z()),
        (Y(), X(), -1j, Z()),
        (Y(), Z(), 1j, X()),
        (Z(), Y(), -1j, X()),
        (Z(), X(), 1j, Y()),
        (X(), Z(), -1j, Y()),
    ],
)
def test_single_qubit_multiplication_table(a, b, expected_phase, expected_term):
    phase, result = a @ b
    assert phase == expected_phase
    assert result == expected_term


def test_associativity_up_to_phase():
    # (A @ B) @ C and A @ (B @ C) should give the same Pauli term ignoring phase
    ops = [X(), Y(), Z()]
    for A in ops:
        for B in ops:
            for C in ops:
                _, r1 = (A @ B)
                _, r1 = r1 @ C
                _, r2 = (B @ C)
                _, r2 = A @ r2
                assert r1 == r2


def single_qubit_char_from_masks(x_mask, z_mask, bit):
    xb = (x_mask >> bit) & 1
    zb = (z_mask >> bit) & 1
    if xb == 0 and zb == 0:
        return "I"
    if xb == 1 and zb == 0:
        return "X"
    if xb == 0 and zb == 1:
        return "Z"
    return "Y"


SINGLE_QUBIT_MUL = {
    ("I", "I"): (1, "I"),
    ("X", "X"): (1, "I"),
    ("Y", "Y"): (1, "I"),
    ("Z", "Z"): (1, "I"),
    ("X", "Y"): (1j, "Z"),
    ("Y", "X"): (-1j, "Z"),
    ("Y", "Z"): (1j, "X"),
    ("Z", "Y"): (-1j, "X"),
    ("Z", "X"): (1j, "Y"),
    ("X", "Z"): (-1j, "Y"),
}


def test_multi_qubit_multiplication_masks_and_phase():
    p1 = PauliString(x=0b01, z=0b10, n_qubits=2)  # X on q0, Z on q1
    p2 = PauliString(x=0b10, z=0b01, n_qubits=2)  # X on q1, Z on q0

    phase, result = p1 @ p2

    # masks should XOR
    assert result.x == (p1.x ^ p2.x)
    assert result.z == (p1.z ^ p2.z)

    # compute expected phase by per-qubit multiplication
    expected_phase = 1
    for bit in range(2):
        a = single_qubit_char_from_masks(p1.x, p1.z, bit)
        b = single_qubit_char_from_masks(p2.x, p2.z, bit)
        ph, _ = SINGLE_QUBIT_MUL[(a, b)]
        expected_phase *= ph

    assert phase == expected_phase


def test_disjoint_qubits_commute():
    a = X(qubit=0, n_qubits=2)  # acts on q0
    b = Z(qubit=1, n_qubits=2)  # acts on q1
    assert a.commutes_with(b)


def test_overlapping_odd_parity_anticommute():
    a = X(qubit=0, n_qubits=1)
    b = Z(qubit=0, n_qubits=1)
    assert not a.commutes_with(b)


def test_weight_and_identity():
    identity = I(n_qubits=3)
    assert identity.weight == 0
    p = PauliString(x=0b101, z=0b010, n_qubits=3)  # nontrivial on 2 qubits
    assert p.weight == 3


def test_to_bytes_consistent_for_equal_terms():
    p1 = PauliString(x=0b11, z=0b00, n_qubits=2)
    p2 = PauliString(x=0b11, z=0b00, n_qubits=2)
    assert p1.to_bytes() == p2.to_bytes()


def test_multiply_result_equals_expected_term_ignoring_phase():
    # X @ Z -> -iY but term should equal Y
    a = X(n_qubits=1)
    b = Z(n_qubits=1)
    _, result = a @ b
    assert result == Y(n_qubits=1)
