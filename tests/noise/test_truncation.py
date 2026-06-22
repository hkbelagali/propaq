
from propaq.noise.truncation import TruncationPolicy


def test_should_truncate_logic():
    p = TruncationPolicy(weight_cutoff=2, coeff_cutoff=0.5)
    assert p.should_truncate(3, 1.0) is True
    assert p.should_truncate(2, 0.4) is True
    assert p.should_truncate(2, 0.6) is False
    assert p.should_truncate(1, 0.4) is True


def test_weight_cutoff_none_never_truncates_on_weight():
    p = TruncationPolicy(weight_cutoff=None, coeff_cutoff=0.0)
    assert p.should_truncate(100, 1.0) is False
    assert p.should_truncate(1000, 0.5) is False


def test_weight_cutoff_none_still_truncates_on_coeff():
    p = TruncationPolicy(weight_cutoff=None, coeff_cutoff=0.5)
    assert p.should_truncate(100, 0.1) is True
    assert p.should_truncate(100, 0.6) is False


def test_error_bound_placeholder():
    p = TruncationPolicy(1, 0.1)
    # current implementation is a stub; ensure it doesn't raise
    assert p.error_bound(0.1, 10) is None
