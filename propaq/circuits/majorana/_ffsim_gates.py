"""
ffsim gate primitives to Majorana rotation generators.
"""

import numpy as np
import scipy.linalg

from ...datatypes.majorana.majorana import MajoranaMonomial
from ...datatypes.majorana.termsum import MajoranaTermSum, _expand_ladder_term, _flush_ladder_terms


def orbital_rotation_generators(
    mat: np.ndarray, n_modes: int, mode_offset: int = 0, tol: float = 1e-12
) -> list[tuple[MajoranaMonomial, float]]:
    """
    (generator, angle) pairs for an orbital rotation on one spin sector.

    Arguments:
        mat: The unitary orbital rotation matrix for this spin sector.
        n_modes: The total number of Majorana modes in the system.
        mode_offset: Qubit/orbital index of orbital 0 in this spin sector (0
            for alpha, norb for beta).
        tol: Matrix entries smaller than this value are treated as zero.

    Returns:
        Ordered (generator, angle) pairs, to be applied in sequence.
    """
    import ffsim

    givens_rotations, phase_shifts = ffsim.linalg.givens_decomposition(mat, tol=tol)
    terms: list[tuple[MajoranaMonomial, float]] = []

    def _flush_piece(actions_coeffs: list[tuple[list[tuple[int, bool]], complex]]) -> None:
        acc: dict[int, complex] = {}
        for actions, coeff in actions_coeffs:
            for bitmask, value in _expand_ladder_term(actions, coeff, n_modes).items():
                acc[bitmask] = acc.get(bitmask, 0) + value
        piece_sum = MajoranaTermSum()
        _flush_ladder_terms(piece_sum, acc, n_modes)
        terms.extend((gen, 2 * coeff) for gen, coeff in piece_sum.items())

    for c, s, i, j in givens_rotations:

        block = np.array([[c, np.conj(s)], [-s, c]], dtype=complex)
        k_block = scipy.linalg.logm(block)

        p, q = mode_offset + i, mode_offset + j
        k01 = k_block[0, 1]
        r, phi = abs(k01), np.angle(k01)
        if r <= tol:
            continue

        _flush_piece([([(p, True), (p, False)], phi)])
        _flush_piece([([(p, True), (q, False)], 1j * r), ([(q, True), (p, False)], -1j * r)])
        _flush_piece([([(p, True), (p, False)], -phi)])

    for i, phase_shift in enumerate(phase_shifts):
        k = np.log(phase_shift)
        if abs(k) <= tol:
            continue
        mode = mode_offset + i
        _flush_piece([([(mode, True), (mode, False)], 1j * k)])

    return terms


def diag_coulomb_generators(
    mat_aa: "np.ndarray | None",
    mat_ab: "np.ndarray | None",
    mat_bb: "np.ndarray | None",
    time: float,
    norb: int,
    n_modes: int,
) -> list[tuple[MajoranaMonomial, float]]:
    """
    (generator, angle) pairs for a diagonal Coulomb evolution.

    Arguments:
        mat_aa: The alpha-alpha diagonal Coulomb matrix, or None to skip it.
        mat_ab: The alpha-beta diagonal Coulomb matrix, or None to skip it.
        mat_bb: The beta-beta diagonal Coulomb matrix, or None to skip it.
        time: The evolution time.
        norb: The number of spatial orbitals.
        n_modes: The total number of Majorana modes in the system.

    Returns:
        (generator, angle) pairs; order does not matter since all terms commute.
    """
    acc: dict[int, complex] = {}

    def _add_number_term(mode_p: int, mode_q: int, h_coeff: complex) -> None:
        actions = (
            [(mode_p, True), (mode_p, False)]
            if mode_p == mode_q
            else [(mode_p, True), (mode_p, False), (mode_q, True), (mode_q, False)]
        )
        term_acc = _expand_ladder_term(actions, h_coeff, n_modes)
        for bitmask, value in term_acc.items():
            acc[bitmask] = acc.get(bitmask, 0) + value

    for sigma, this_mat in enumerate([mat_aa, mat_bb]):
        if this_mat is None:
            continue
        offset = sigma * norb
        for i in range(norb):
            if this_mat[i, i]:
                _add_number_term(offset + i, offset + i, 0.5 * this_mat[i, i])
        for i in range(norb):
            for j in range(i + 1, norb):
                if this_mat[i, j]:
                    _add_number_term(offset + i, offset + j, this_mat[i, j])

    if mat_ab is not None:
        for i in range(norb):
            for j in range(norb):
                if mat_ab[i, j]:
                    _add_number_term(i, norb + j, mat_ab[i, j])

    term_sum = MajoranaTermSum()
    _flush_ladder_terms(term_sum, acc, n_modes)
    return [(gen, 2 * time * coeff) for gen, coeff in term_sum.items()]


def uccsd_generator_terms(
    t1: np.ndarray, t2: np.ndarray, n_modes: int
) -> list[tuple[MajoranaMonomial, float]]:
    """
    (generator, angle) pairs for a single first-order Trotter step of a
    restricted UCCSD generator T - T^dagger.

    Arguments:
        t1: The t1 amplitudes, shape (nocc, nvrt).
        t2: The t2 amplitudes, shape (nocc, nocc, nvrt, nvrt).
        n_modes: The total number of Majorana modes in the system (4 * norb).

    Returns:
        (generator, angle) pairs, to be applied as one Trotter sweep.
    """
    nocc, nvrt = t1.shape
    norb = nocc + nvrt

    one_body = np.zeros((norb, norb), dtype=complex)
    two_body = np.zeros((norb, norb, norb, norb), dtype=complex)
    one_body[:nocc, nocc:] = -t1.conj()
    one_body[nocc:, :nocc] = t1.T
    two_body[nocc:, :nocc, nocc:, :nocc] = t2.transpose(2, 0, 3, 1)
    two_body[:nocc, nocc:, :nocc, nocc:] = -t2.transpose(0, 2, 1, 3).conj()

    acc: dict[int, complex] = {}

    def _add(actions: list[tuple[int, bool]], coeff: complex) -> None:
        for bitmask, value in _expand_ladder_term(actions, coeff, n_modes).items():
            acc[bitmask] = acc.get(bitmask, 0) + value

    for p in range(norb):
        for q in range(norb):
            if one_body[p, q] == 0:
                continue
            for sigma in (0, 1):
                _add([(sigma * norb + p, True), (sigma * norb + q, False)], 1j * one_body[p, q])

    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    if two_body[p, q, r, s] == 0:
                        continue
                    coeff = 0.5j * two_body[p, q, r, s]
                    for sigma in (0, 1):
                        for tau in (0, 1):
                            _add(
                                [
                                    (sigma * norb + p, True),
                                    (tau * norb + r, True),
                                    (tau * norb + s, False),
                                    (sigma * norb + q, False),
                                ],
                                coeff,
                            )

    term_sum = MajoranaTermSum()
    _flush_ladder_terms(term_sum, acc, n_modes)
    return [(gen, 2 * coeff) for gen, coeff in term_sum.items()]
