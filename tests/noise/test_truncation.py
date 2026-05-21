import pytest

from propaq.noise.truncation import TruncationPolicy


def test_should_truncate_logic():
    p = TruncationPolicy(weight_cutoff=2, coeff_cutoff=0.5)
    assert p.should_truncate(3, 1.0) is True
    assert p.should_truncate(2, 0.4) is True
    assert p.should_truncate(2, 0.6) is False
    assert p.should_truncate(1, 0.4) is True


def test_error_bound_placeholder():
    p = TruncationPolicy(1, 0.1)
    # current implementation is a stub; ensure it doesn't raise
    assert p.error_bound(0.1, 10) is None
