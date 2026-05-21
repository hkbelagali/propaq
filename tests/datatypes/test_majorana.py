import pytest 
from propaq.datatypes import MajoranaMonomial 

def monomial(*indices, n_modes=8): 
    modes = 0
    for i in indices:
        modes |= 1 << (i-1) 
    return MajoranaMonomial(modes, n_modes=n_modes)

I = MajoranaMonomial(0, n_modes=8) 

def test_identity_weight(): 
    assert I.weight == 0

def test_single_mode():
    m = monomial(1)
    assert m.weight == 1

def test_weight_4(): 
    m = monomial(1, 3, 5, 7)
    assert m.weight == 4

@pytest.mark.parametrize("w, expected", [
    (0, 0),   # w(w-1)/2 = 0, even  → Hermitian
    (1, 0),   # w(w-1)/2 = 0, even  → Hermitian
    (2, 1),   # w(w-1)/2 = 1, odd   → anti-Hermitian, needs i
    (3, 1),   # w(w-1)/2 = 3, odd   → anti-Hermitian, needs i
    (4, 0),   # w(w-1)/2 = 6, even  → Hermitian
    (5, 0),   # w(w-1)/2 = 10, even → Hermitian
    (6, 1),   # w(w-1)/2 = 15, odd  → anti-Hermitian, needs i
    (7, 1),   # w(w-1)/2 = 21, odd  → anti-Hermitian, needs i
    (8, 0),   # w(w-1)/2 = 28, even → Hermitian
])

def test_hermiticity_exp(w, expected):
    from propaq.datatypes.majorana import _hermiticity_exp
    assert _hermiticity_exp(w) == expected

def test_same_mode_commutes(): 
    m1 = monomial(1) 
    assert m1.commutes_with(m1)

def test_even_length_disjoint_commute(): 
    m1 = monomial(1, 2) 
    m2 = monomial(3, 4) 
    assert m1.commutes_with(m2)

def test_even_length_overlap_1_anticommute(): 
    m1 = monomial(1, 2) 
    m2 = monomial(2, 3) 
    assert not m1.commutes_with(m2)

def test_identity_times():
    m = monomial(1, 2) 
    phase, result = I @ m
    assert abs(phase - 1) < 1e-6
    assert result == m

def test_self_product_is_identity(): 
    m = monomial(1, 2, 3) 
    phase, result = m @ m
    assert abs(phase - 1) < 1e-6
    assert result == I

def test_disjoint_product(): 
    m1 = monomial(1, 2) 
    m2 = monomial(3, 4)
    phase, result = m1 @ m2
    assert result == monomial(1, 2, 3, 4)
    assert phase == -1

def test_overlap_product():
    m1 = monomial(1, 2) 
    m2 = monomial(2, 3)
    phase, result = m1 @ m2
    assert result == monomial(1, 3)
    assert result.weight == 2

def test_result_weight(): 
    m1 = monomial(1, 2) 
    m2 = monomial(1, 3) 
    assert m1.resulting_weight(m2) == (m1 @ m2)[1].weight

def test_product_not_commutative(): 
    m1 = monomial(1, 2) 
    m2 = monomial(2, 3) 
    phase_ab, result_ab = m1 @ m2
    phase_ba, result_ba = m2 @ m1

    assert result_ab == result_ba
    assert abs(phase_ab - (-phase_ba)) < 1e-6

def test_weight2_preserves_weight(): 
    gate = monomial(1, 2) 
    obs = monomial(2, 3) 
    assert not gate.commutes_with(obs)
    _, result = gate @ obs
    assert result.weight == obs.weight

def test_to_bytes_and_from_bytes():
    m = monomial(1, 3, 5) 
    b = m.to_bytes()
    reconstructed = MajoranaMonomial(int.from_bytes(b, byteorder='little'), n_modes=m.n_modes) 
    assert m == reconstructed

def test_hash_and_equality():
    m1 = monomial(1, 2) 
    m2 = MajoranaMonomial(modes=0b11, n_modes=8) 
    assert m1 == m2
    assert hash(m1) == hash(m2)

def test_different_monomials_not_equal(): 
    m1 = monomial(1, 2) 
    m2 = monomial(2, 3) 
    assert m1 != m2

