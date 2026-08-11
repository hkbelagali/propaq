//!
//! Defines the core algebra of Majorana monomials, products of Majorana operators
//!

use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};

use propaq_core::bitset::Bitset;
use propaq_core::helpers::{bitset_to_pyint, pyint_to_bitset};
use propaq_core::sparse::{intersection_count, symmetric_difference_into};
use propaq_core::store::{hash_positions, split_planes, Position, TermBasis};
use propaq_core::traits::AbstractTerm;

/// A Majorana monomial, a product of Majorana operators encoded as a mode bitmask.
///
/// Bit 2k is set if $\gamma_{2k}$ (even mode) is active on site k.
/// Bit 2k+1 is set if $\gamma_{2k+1}$ (odd mode) is active on site k.
///
/// Arguments:
///     modes: Integer bitmask encoding occupied Majorana modes.
///     n_modes: Total number of Majorana modes (must be even, equal to 2 * n_qubits).
///     is_number_preserving: Whether the monomial preserves particle number (default True).
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(module = "propaq._rust_core")]
#[derive(Clone)]
pub struct MajoranaMonomial {
    pub modes: Bitset,
    #[pyo3(get)]
    pub n_modes: usize,
    #[pyo3(get)]
    pub is_number_preserving: bool,
    pub weight: u32,
    pub p: Bitset,
}

impl MajoranaMonomial {
    fn commutes_with_impl(&self, other: &MajoranaMonomial) -> bool {
        if self.modes == other.modes {
            return true;
        }
        let a = self.modes.as_words();
        let b = other.modes.as_words();
        let overlap: u32 = (0..a.len().min(b.len()))
            .map(|i| (a[i] & b[i]).count_ones())
            .sum();
        (self.modes.count_ones() as usize * other.modes.count_ones() as usize + overlap as usize)
            .is_multiple_of(2)
    }

    fn trace_fock_state_impl(&self, fock_state: &Bitset) -> f64 {
        let n_fermionic = self.n_modes / 2;
        let mut p = 0i32;
        let mut product = 1i32;

        for k in 0..n_fermionic {
            let low = self.modes.bit(2 * k) as i32;
            let high = self.modes.bit(2 * k + 1) as i32;

            if low != high {
                return 0.0;
            }
            if low == 1 {
                let n_k = fock_state.bit(k) as i32;
                product *= 2 * n_k - 1;
                p += 1;
            }
        }

        let phase = if (p / 2) % 2 == 0 { 1 } else { -1 };
        (phase * product) as f64
    }

    fn compress_single_occupied(modes: &Bitset, n_qubits: usize) -> (Bitset, Bitset) {
        let x_bits = compress_to_qubits(modes, n_qubits, 0);
        let y_bits = compress_to_qubits(modes, n_qubits, 1);
        let occupied = &x_bits | &y_bits;
        let single = &x_bits ^ &y_bits;
        (single, occupied)
    }

    fn scan_p(single: &Bitset, n_qubits: usize, qubit_mask: &Bitset) -> Bitset {
        let mut p = single.clone();
        let mut shift = 1usize;
        while shift < n_qubits {
            p = &p ^ &(&p.shl(shift) & qubit_mask);
            shift <<= 1;
        }
        p
    }

    /// Final weight from the compressed parts: complements `p` into the
    /// Jordan-Wigner Z-string parity, then counts non-identity qubits.
    fn weight_from_parts(
        single: &Bitset,
        occupied: &Bitset,
        p: &Bitset,
        qubit_mask: &Bitset,
    ) -> u32 {
        let string = if single.count_ones() & 1 == 1 {
            p ^ qubit_mask
        } else {
            p.clone()
        };
        (single | &(occupied ^ &string)).count_ones()
    }

    /// Builds a monomial from a mode set, deriving everything else.
    pub fn from_modes(modes: Bitset, n_modes: usize) -> Self {
        let (weight, p) = Self::weight_and_p_for(&modes, n_modes);
        let is_number_preserving =
            (0..n_modes / 2).all(|k| modes.bit(2 * k) == modes.bit(2 * k + 1));
        MajoranaMonomial {
            modes,
            n_modes,
            is_number_preserving,
            weight,
            p,
        }
    }

    /// Computes the Jordan-Wigner qubit weight of a Majorana mode set
    pub fn compute_weight_for(modes: &Bitset, n_modes: usize) -> u32 {
        let n_qubits = n_modes / 2;
        if n_qubits == 0 {
            return 0;
        }
        let qubit_mask = Bitset::all_ones_upto(n_qubits);
        let (single, occupied) = Self::compress_single_occupied(modes, n_qubits);
        let p = Self::scan_p(&single, n_qubits, &qubit_mask);
        Self::weight_from_parts(&single, &occupied, &p, &qubit_mask)
    }

    /// Computes both the Jordan-Wigner qubit weight and the `p` (Z-string parity) plane for a
    /// Majorana mode set.
    pub fn weight_and_p_for(modes: &Bitset, n_modes: usize) -> (u32, Bitset) {
        let n_qubits = n_modes / 2;
        if n_qubits == 0 {
            return (0, Bitset::zero());
        }
        let qubit_mask = Bitset::all_ones_upto(n_qubits);
        let (single, occupied) = Self::compress_single_occupied(modes, n_qubits);
        let p = Self::scan_p(&single, n_qubits, &qubit_mask);
        let weight = Self::weight_from_parts(&single, &occupied, &p, &qubit_mask);
        (weight, p)
    }

    pub(crate) fn weight_and_p_from_product(
        result_modes: &Bitset,
        n_modes: usize,
        self_p: &Bitset,
        other_p: &Bitset,
    ) -> (u32, Bitset) {
        let n_qubits = n_modes / 2;
        if n_qubits == 0 {
            return (0, Bitset::zero());
        }
        let qubit_mask = Bitset::all_ones_upto(n_qubits);
        let (single, occupied) = Self::compress_single_occupied(result_modes, n_qubits);
        let p = self_p ^ other_p;
        let weight = Self::weight_from_parts(&single, &occupied, &p, &qubit_mask);
        (weight, p)
    }

    pub(crate) fn matmul_internal(
        &self,
        other: &MajoranaMonomial,
    ) -> (Complex64, MajoranaMonomial) {
        let result_modes = &self.modes ^ &other.modes;
        let (weight, p) =
            Self::weight_and_p_from_product(&result_modes, self.n_modes, &self.p, &other.p);
        let n_fermionic = self.n_modes / 2;
        let is_np =
            (0..n_fermionic).all(|k| result_modes.bit(2 * k) == result_modes.bit(2 * k + 1));
        let result = MajoranaMonomial {
            modes: result_modes,
            n_modes: self.n_modes,
            is_number_preserving: is_np,
            weight,
            p,
        };

        let r_a = hermiticity_exp(self.length());
        let r_b = hermiticity_exp(other.length());
        let r_c = hermiticity_exp(result.length());
        let total_parity = resorting_parity(self.modes.as_words(), other.modes.as_words());
        let phase_exp = (r_a + r_b - r_c + 2 * (total_parity as i32)).rem_euclid(4);

        let phase = match phase_exp {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, -1.0),
            _ => unreachable!(),
        };

        (phase, result)
    }
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl MajoranaMonomial {
    /// Construct a Majorana monomial from a mode bitmask.
    ///
    /// Arguments:
    ///     modes: Integer bitmask where bit 2k (2k+1) is set if Majorana mode gamma_{2k} (gamma_{2k+1}) is active.
    ///     n_modes: Total number of Majorana modes (must be even).
    ///     is_number_preserving: Whether the monomial preserves particle number.
    #[new]
    #[pyo3(signature = (modes, n_modes, is_number_preserving = true))]
    fn new(modes: &Bound<'_, PyAny>, n_modes: usize, is_number_preserving: bool) -> PyResult<Self> {
        let bitset = pyint_to_bitset(modes, n_modes)?;
        let (weight, p) = Self::weight_and_p_for(&bitset, n_modes);
        Ok(MajoranaMonomial {
            modes: bitset,
            n_modes,
            is_number_preserving,
            weight,
            p,
        })
    }

    /// The active mode indices as a Python integer bitmask.
    #[getter]
    fn modes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        bitset_to_pyint(py, &self.modes)
    }

    /// The number of Majorana modes in the system
    #[getter]
    fn n_modes(&self) -> usize {
        self.n_modes
    }

    /// Whether or not the monomial preserves particle number (i.e. fully paired).
    #[getter]
    fn is_number_preserving(&self) -> bool {
        self.is_number_preserving
    }

    /// Number of active Majorana modes in the monomial (popcount of the mode bitmask).
    #[getter]
    fn length(&self) -> usize {
        self.modes.count_ones() as usize
    }

    /// Pauli weight of this monomial under the Jordan-Wigner mapping.
    #[getter]
    fn weight(&self) -> u32 {
        self.weight
    }

    /// Number of Majorana modes shared with *other* (popcount of modes & other.modes).
    /// Arguments:
    ///     other: Another MajoranaMonomial to compare with.
    /// Returns:
    ///     The number of Majorana modes that are active in both self and other.
    fn overlap(&self, other: &MajoranaMonomial) -> u32 {
        (&self.modes & &other.modes).count_ones()
    }

    /// Return True if this monomial commutes with *other*.
    /// Arguments:
    ///     other: Another MajoranaMonomial to check commutation with.
    /// Returns:
    ///     True if self and other commute, False otherwise.
    pub fn commutes_with(&self, other: &MajoranaMonomial) -> bool {
        self.commutes_with_impl(other)
    }

    /// Pauli weight of the product monomial self @ other, without computing the full product.
    /// Arguments:
    ///     other: Another MajoranaMonomial to multiply with.
    /// Returns:
    ///     The Pauli weight of the resulting monomial from multiplying self and other.
    fn resulting_weight(&self, other: &MajoranaMonomial) -> u32 {
        let result_modes = &self.modes ^ &other.modes;
        Self::compute_weight_for(&result_modes, self.n_modes)
    }

    /// Multiply two Majorana monomials, returning (phase, product).
    ///
    /// The phase factor accounts for the anticommutation relations of Majorana operators.
    fn __matmul__(&self, other: &MajoranaMonomial) -> PyResult<(Complex64, MajoranaMonomial)> {
        Ok(self.matmul_internal(other))
    }

    /// Compute $\langle \psi |M| \psi \rangle$ for this Majorana monomial M.
    ///
    /// Returns 0.0 if M has any unpaired modes.
    /// For paired modes, returns the product of $(2n_k - 1)$ values for each occupied pair.
    ///
    /// Arguments:
    ///     fock_state: Computational basis state as a bitstring integer.
    /// Returns:
    ///     Expectation value of the Majorana monomial in the given Fock state.
    pub fn trace_with_fock_state(&self, fock_state: &Bound<'_, PyAny>) -> PyResult<f64> {
        let bs = pyint_to_bitset(fock_state, self.n_modes)?;
        Ok(self.trace_fock_state_impl(&bs))
    }

    /// Serialize the mode bitmask as a little-endian byte string.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let byte_length = self.n_modes.div_ceil(8);
        let mut bytes = self.modes.to_le_bytes();
        bytes.resize(byte_length, 0);
        PyBytes::new(py, &bytes)
    }

    fn __hash__(&self) -> u64 {
        let mut h = FxHasher::default();
        self.modes.hash(&mut h);
        h.finish()
    }

    fn __eq__(&self, other: &MajoranaMonomial) -> bool {
        self.modes == other.modes
    }
}

impl AbstractTerm for MajoranaMonomial {
    fn weight(&self) -> u32 {
        self.weight
    }
    fn commutes_with(&self, other: &Self) -> bool {
        self.commutes_with_impl(other)
    }
    fn matmul_internal(&self, other: &Self) -> (Complex64, Self) {
        MajoranaMonomial::matmul_internal(self, other)
    }
    fn trace_with_fock_state(&self, fock_state: &Bitset) -> f64 {
        self.trace_fock_state_impl(fock_state)
    }
    fn to_bytes_vec(&self) -> Vec<u8> {
        let byte_length = self.n_modes.div_ceil(8);
        let mut bytes = self.modes.to_le_bytes();
        bytes.resize(byte_length, 0);
        bytes
    }
    fn partition_key(&self) -> u64 {
        let mut h = FxHasher::default();
        self.modes.hash(&mut h);
        h.finish()
    }
    fn is_number_preserving(&self) -> bool {
        self.is_number_preserving
    }
    fn system_size(&self) -> u64 {
        self.n_modes as u64
    }
    fn from_bytes_vec(bytes: &[u8], system_size: u64) -> Self {
        let n_modes = system_size as usize;
        let modes = Bitset::from_le_bytes(bytes);
        let (weight, p) = Self::weight_and_p_for(&modes, n_modes);
        let n_fermionic = n_modes / 2;
        let is_np = (0..n_fermionic).all(|k| modes.bit(2 * k) == modes.bit(2 * k + 1));
        MajoranaMonomial {
            modes,
            n_modes,
            is_number_preserving: is_np,
            weight,
            p,
        }
    }
}

impl PartialEq for MajoranaMonomial {
    fn eq(&self, other: &Self) -> bool {
        self.modes == other.modes
    }
}

impl Eq for MajoranaMonomial {}

impl Hash for MajoranaMonomial {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.modes.hash(state);
    }
}

/// The `TermBasis` implementation for Majorana fermion strings, encoded as a Jordan-Wigner
/// mode-occupation bitmask (`MajoranaMonomial`).
pub struct MajoranaBasis;

impl TermBasis for MajoranaBasis {
    type Term = MajoranaMonomial;

    fn commutes(term: [&[u64]; 2], gen: [&[u64]; 2]) -> bool {
        // Mirrors `commutes_with_impl(self=term, other=gen)`.
        if term[0] == gen[0] {
            return true;
        }
        let overlap: u32 = term[0]
            .iter()
            .zip(gen[0])
            .map(|(a, b)| (a & b).count_ones())
            .sum();
        let term_len: usize = term[0].iter().map(|w| w.count_ones()).sum::<u32>() as usize;
        let gen_len: usize = gen[0].iter().map(|w| w.count_ones()).sum::<u32>() as usize;
        (term_len * gen_len + overlap as usize).is_multiple_of(2)
    }

    fn product(term: [&[u64]; 2], gen: [&[u64]; 2], out: [&mut [u64]; 2]) -> Complex64 {
        for i in 0..out[0].len() {
            out[0][i] = gen[0][i] ^ term[0][i];
            out[1][i] = gen[1][i] ^ term[1][i];
        }
        let gen_len = gen[0].iter().map(|w| w.count_ones()).sum::<u32>() as usize;
        let term_len = term[0].iter().map(|w| w.count_ones()).sum::<u32>() as usize;
        let result_len = out[0].iter().map(|w| w.count_ones()).sum::<u32>() as usize;
        let r_a = hermiticity_exp(gen_len);
        let r_b = hermiticity_exp(term_len);
        let r_c = hermiticity_exp(result_len);
        let total_parity = resorting_parity(gen[0], term[0]);
        let phase_exp = (r_a + r_b - r_c + 2 * (total_parity as i32)).rem_euclid(4);
        match phase_exp {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, -1.0),
            _ => unreachable!(),
        }
    }

    fn weight(term: [&[u64]; 2], n_units: usize) -> u32 {
        let n_qubits = n_units / 2;
        if n_qubits == 0 {
            return 0;
        }
        let modes = Bitset::from_slice(term[0]);
        let p = Bitset::from_slice(term[1]);
        let qubit_mask = Bitset::all_ones_upto(n_qubits);
        let (single, occupied) = MajoranaMonomial::compress_single_occupied(&modes, n_qubits);
        MajoranaMonomial::weight_from_parts(&single, &occupied, &p, &qubit_mask)
    }

    fn trace(term: [&[u64]; 2], n_units: usize, fock: &[u64]) -> f64 {
        let modes = Bitset::from_slice(term[0]);
        let m = MajoranaMonomial {
            modes,
            n_modes: n_units,
            is_number_preserving: false,
            weight: 0,
            p: Bitset::zero(),
        };
        let fock_bs = Bitset::from_slice(fock);
        m.trace_fock_state_impl(&fock_bs)
    }

    fn key_hash(term: [&[u64]; 2]) -> u64 {
        let mut h = FxHasher::default();
        term[0].hash(&mut h);
        h.finish()
    }

    fn key_eq(a: [&[u64]; 2], b: [&[u64]; 2]) -> bool {
        a[0] == b[0]
    }

    fn term_from_planes(term: [&[u64]; 2], n_units: usize) -> MajoranaMonomial {
        let modes = Bitset::from_slice(term[0]);
        let p = Bitset::from_slice(term[1]);
        let n_qubits = n_units / 2;
        let weight = if n_qubits == 0 {
            0
        } else {
            let qubit_mask = Bitset::all_ones_upto(n_qubits);
            let (single, occupied) = MajoranaMonomial::compress_single_occupied(&modes, n_qubits);
            MajoranaMonomial::weight_from_parts(&single, &occupied, &p, &qubit_mask)
        };
        let is_np = (0..n_qubits).all(|k| modes.bit(2 * k) == modes.bit(2 * k + 1));
        MajoranaMonomial {
            modes,
            n_modes: n_units,
            is_number_preserving: is_np,
            weight,
            p,
        }
    }

    fn term_into_planes(term: &MajoranaMonomial, _n_units: usize, out: [&mut [u64]; 2]) {
        let mw = term.modes.as_words();
        out[0].fill(0);
        out[0][..mw.len()].copy_from_slice(mw);
        let pw = term.p.as_words();
        out[1].fill(0);
        out[1][..pw.len()].copy_from_slice(pw);
    }

    fn key_hash_sparse(row: &[Position], plane_span: usize) -> u64 {
        hash_positions(split_planes(row, plane_span).0)
    }

    fn key_eq_sparse(a: &[Position], b: &[Position], plane_span: usize) -> bool {
        split_planes(a, plane_span).0 == split_planes(b, plane_span).0
    }

    /// The Jordan-Wigner qubit weight, as set algebra over the qubits the mode
    /// positions and the `p` positions touch.
    fn weight_sparse(row: &[Position], plane_span: usize, n_units: usize) -> u32 {
        let n_qubits = n_units / 2;
        if n_qubits == 0 {
            return 0;
        }
        let parts = QubitSets::of(row, plane_span, n_qubits);
        let (occ, single, p) = (parts.occupied as i64, parts.single as i64, parts.p as i64);
        let (occ_p, single_p) = (parts.occupied_and_p as i64, parts.single_and_p as i64);

        let weight = if parts.single.is_multiple_of(2) {
            occ + p - 2 * occ_p + single_p
        } else {
            n_qubits as i64 + 2 * occ_p - occ - p + single - single_p
        };
        debug_assert!(weight >= 0 && weight <= n_qubits as i64);
        weight as u32
    }

    /// A Majorana monomial is diagonal only where every occupied site carries
    /// both of its modes, so the trace walks the mode positions in pairs.
    fn trace_sparse(row: &[Position], plane_span: usize, n_units: usize, fock: &[u64]) -> f64 {
        let n_fermionic = n_units / 2;
        let (modes, _) = split_planes(row, plane_span);
        let limit = (2 * n_fermionic) as Position;
        let modes = &modes[..modes.partition_point(|&m| m < limit)];
        let (mut occupied_pairs, mut product) = (0i32, 1i32);
        let mut i = 0usize;
        while i < modes.len() {
            let m = modes[i];

            if m % 2 != 0 || i + 1 >= modes.len() || modes[i + 1] != m + 1 {
                return 0.0;
            }
            let k = (m / 2) as usize;
            let n_k = ((fock.get(k >> 6).copied().unwrap_or(0) >> (k & 63)) & 1) as i32;
            product *= 2 * n_k - 1;
            occupied_pairs += 1;
            i += 2;
        }
        let phase = if (occupied_pairs / 2) % 2 == 0 { 1 } else { -1 };
        (phase * product) as f64
    }

    fn commutes_sparse(term: &[Position], gen: &[Position], plane_span: usize) -> bool {
        let (tm, _) = split_planes(term, plane_span);
        let (gm, _) = split_planes(gen, plane_span);
        if tm == gm {
            return true;
        }
        let overlap = intersection_count(tm, gm) as usize;
        (tm.len() * gm.len() + overlap).is_multiple_of(2)
    }

    fn product_sparse(
        term: &[Position],
        gen: &[Position],
        plane_span: usize,
        out: &mut Vec<Position>,
    ) -> Complex64 {
        let start = out.len();
        symmetric_difference_into(term, gen, out);
        let (tm, _) = split_planes(term, plane_span);
        let (gm, _) = split_planes(gen, plane_span);
        let (nm, _) = split_planes(&out[start..], plane_span);
        let r_a = hermiticity_exp(gm.len());
        let r_b = hermiticity_exp(tm.len());
        let r_c = hermiticity_exp(nm.len());
        let total_parity = resorting_parity_sparse(gm, tm);
        let phase_exp = (r_a + r_b - r_c + 2 * (total_parity as i32)).rem_euclid(4);
        match phase_exp {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, -1.0),
            _ => unreachable!(),
        }
    }
}


struct QubitSets {
    /// Qubits with at least one Majorana mode set.
    occupied: u32,
    /// Qubits with exactly one of their two modes set.
    single: u32,
    /// Qubits set in the `p` plane.
    p: u32,
    occupied_and_p: u32,
    single_and_p: u32,
}

impl QubitSets {
    fn of(row: &[Position], plane_span: usize, n_qubits: usize) -> Self {
        let (modes, p_positions) = split_planes(row, plane_span);
        let mode_limit = (2 * n_qubits) as Position;
        let modes = &modes[..modes.partition_point(|&m| m < mode_limit)];
        let p_limit = (plane_span + n_qubits) as Position;
        let p_positions = &p_positions[..p_positions.partition_point(|&p| p < p_limit)];

        let mut sets = QubitSets {
            occupied: 0,
            single: 0,
            p: p_positions.len() as u32,
            occupied_and_p: 0,
            single_and_p: 0,
        };
        let (mut i, mut j) = (0usize, 0usize);
        while i < modes.len() {
            let k = (modes[i] >> 1) as usize;
            let paired = i + 1 < modes.len() && (modes[i + 1] >> 1) as usize == k;
            i += if paired { 2 } else { 1 };
            sets.occupied += 1;
            if !paired {
                sets.single += 1;
            }

            while j < p_positions.len() && (p_positions[j] as usize - plane_span) < k {
                j += 1;
            }
            if j < p_positions.len() && (p_positions[j] as usize - plane_span) == k {
                sets.occupied_and_p += 1;
                if !paired {
                    sets.single_and_p += 1;
                }
            }
        }
        sets
    }
}

fn resorting_parity_sparse(a: &[Position], b: &[Position]) -> bool {
    let mut count = 0u64;
    let mut i = 0usize;
    for &bv in b {
        while i < a.len() && a[i] <= bv {
            i += 1;
        }
        count += (a.len() - i) as u64;
    }
    (count & 1) != 0
}

fn compress_to_qubits(modes: &Bitset, n_qubits: usize, offset: usize) -> Bitset {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("bmi2") {
        return unsafe { compress_to_qubits_bmi2(modes, n_qubits, offset) };
    }
    compress_to_qubits_scalar(modes, n_qubits, offset)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn compress_to_qubits_bmi2(modes: &Bitset, n_qubits: usize, offset: usize) -> Bitset {
    use std::arch::x86_64::_pext_u64;
    let mask: u64 = if offset == 0 {
        0x5555_5555_5555_5555
    } else {
        0xAAAA_AAAA_AAAA_AAAA
    };
    let n_qubit_words = n_qubits.div_ceil(64);
    let mut words = vec![0u64; n_qubit_words];
    let mode_words = modes.as_words();
    for (qw, word) in words.iter_mut().enumerate().take(n_qubit_words) {
        let lo = mode_words.get(2 * qw).copied().unwrap_or(0);
        let hi = mode_words.get(2 * qw + 1).copied().unwrap_or(0);
        *word = _pext_u64(lo, mask) | (_pext_u64(hi, mask) << 32);
    }
    Bitset::from_words(words)
}

fn compress_to_qubits_scalar(modes: &Bitset, n_qubits: usize, offset: usize) -> Bitset {
    let n_words = n_qubits.div_ceil(64);
    let mut words = vec![0u64; n_words];
    for k in 0..n_qubits {
        if modes.bit(2 * k + offset) != 0 {
            words[k / 64] |= 1u64 << (k % 64);
        }
    }
    Bitset::from_words(words)
}

pub(crate) fn hermiticity_exp(length: usize) -> i32 {
    if matches!(length % 4, 0 | 1) {
        0
    } else {
        1
    }
}

pub(crate) fn resorting_parity(a_words: &[u64], b_words: &[u64]) -> bool {
    let total: u64 = a_words.iter().map(|w| w.count_ones() as u64).sum();
    let mut running = 0u64;
    let mut count = 0u64;

    for (wi, &bw) in b_words.iter().enumerate() {
        let a_word = a_words.get(wi).copied().unwrap_or(0);
        running += a_word.count_ones() as u64;
        let above_higher = total - running;
        let mut bword = bw;
        while bword != 0 {
            let bi = bword.trailing_zeros() as usize;
            let above_same = if bi < 63 {
                (a_word >> (bi + 1)).count_ones() as u64
            } else {
                0
            };
            count += above_same + above_higher;
            bword &= bword - 1;
        }
    }
    (count & 1) != 0
}

#[cfg(test)]
#[path = "../tests/unit/monomial.rs"]
mod tests;
