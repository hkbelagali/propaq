//!
//! Defines the core algebra of Pauli strings.
//!

use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};

use propaq_core::bitset::Bitset;
use propaq_core::helpers::{bitset_to_pyint, pyint_to_bitset};
use propaq_core::sparse::{shifted_intersection_count, symmetric_difference_into};
use propaq_core::store::{split_planes, Position, TermBasis};
use propaq_core::traits::AbstractTerm;

/// An n-qubit Pauli operator encoded as two integer bitmasks.
///
/// `x` and `z` together encode the single-qubit Pauli on each qubit:
///
/// 00 -> I, 01 -> X, 10 -> Z, 11 -> Y
///
/// Arguments:
///     x: Integer bitmask where bit k is set if qubit k has an X or Y component.
///     z: Integer bitmask where bit k is set if qubit k has a Z or Y component.
///     n_qubits: Total number of qubits in the system.
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(module = "propaq._rust_core")]
#[derive(Clone)]
pub struct PauliString {
    pub x: Bitset,
    pub z: Bitset,
    #[pyo3(get)]
    pub n_qubits: usize,
    pub weight: u32,
}

impl PauliString {
    fn commutes_with_impl(&self, other: &PauliString) -> bool {
        // Anticommutator parity = popcount(x1 & z2) + popcount(z1 & x2) mod 2.
        // Compute word-by-word to avoid allocating intermediate Bitsets.
        let xz: u32 = self
            .x
            .as_words()
            .iter()
            .zip(other.z.as_words())
            .map(|(a, b)| (a & b).count_ones())
            .sum();
        let zx: u32 = self
            .z
            .as_words()
            .iter()
            .zip(other.x.as_words())
            .map(|(a, b)| (a & b).count_ones())
            .sum();
        (xz + zx).is_multiple_of(2)
    }

    pub(crate) fn matmul_impl(&self, other: &PauliString) -> (Complex64, PauliString) {
        let new_x = &self.x ^ &other.x;
        let new_z = &self.z ^ &other.z;
        let new_weight = (&new_x | &new_z).count_ones();

        let p = ((&self.x & &self.z).count_ones() as i32
            + (&other.x & &other.z).count_ones() as i32
            - (&new_x & &new_z).count_ones() as i32
            + 2 * (&self.z & &other.x).count_ones() as i32)
            .rem_euclid(4);

        let phase = match p {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, -1.0),
            _ => unreachable!(),
        };

        let result = PauliString {
            x: new_x,
            z: new_z,
            n_qubits: self.n_qubits,
            weight: new_weight,
        };
        (phase, result)
    }

    fn trace_fock_state_impl(&self, fock_state: &Bitset) -> f64 {
        if !self.x.is_zero() {
            return 0.0;
        }
        let parity = (&self.z & fock_state).count_ones();
        if parity.is_multiple_of(2) {
            1.0
        } else {
            -1.0
        }
    }
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl PauliString {
    /// Construct a Pauli monomial from X and Z bitmasks.
    ///
    /// Arguments:
    ///     x: Integer bitmask where bit k is set if qubit k has an X or Y component.
    ///     z: Integer bitmask where bit k is set if qubit k has a Z or Y component.
    ///     n_qubits: Total number of qubits in the system.
    #[new]
    #[pyo3(signature = (x, z, n_qubits))]
    fn new(x: &Bound<'_, PyAny>, z: &Bound<'_, PyAny>, n_qubits: usize) -> PyResult<Self> {
        let x_bits = pyint_to_bitset(x, n_qubits)?;
        let z_bits = pyint_to_bitset(z, n_qubits)?;
        let weight = (&x_bits | &z_bits).count_ones();
        Ok(PauliString {
            x: x_bits,
            z: z_bits,
            n_qubits,
            weight,
        })
    }

    /// X-component bitmask as a Python int.
    #[getter]
    fn x(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        bitset_to_pyint(py, &self.x)
    }

    /// Z-component bitmask as a Python int.
    #[getter]
    fn z(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        bitset_to_pyint(py, &self.z)
    }

    /// @private
    #[getter]
    fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Number of non-identity single-qubit Pauli operators (popcount of x | z).
    #[getter]
    fn weight(&self) -> u32 {
        self.weight
    }

    /// Return True if this Pauli string commutes with *other*.
    ///
    /// Two Pauli strings commute iff the number of positions where they
    /// anticommute is even.
    ///
    /// Arguments:
    ///     other: Another PauliString to check commutation with.
    ///
    /// Returns:
    ///    True if self and other commute, False otherwise.
    fn commutes_with(&self, other: &PauliString) -> bool {
        self.commutes_with_impl(other)
    }

    /// Multiply two Pauli strings, returning (phase, product).
    ///
    /// The phase factor is in {1, i, -1, -i}. Phase and monomial are returned
    /// separately so that equal monomials (modulo phase) hash identically.
    fn __matmul__(&self, other: &PauliString) -> PyResult<(Complex64, PauliString)> {
        Ok(self.matmul_impl(other))
    }

    /// Compute $\langle \psi | P | \psi \rangle$ for this Pauli string P.
    ///
    /// Returns 0.0 if P has any X or Y components (off-diagonal).
    /// For Z-only P, returns $(-1)^{\text{popcount}(z \text{ AND } \psi)}$.
    ///
    /// Arguments:
    ///     fock_state: Computational basis state as a bitstring integer.
    /// Returns:
    ///     Expectation value of the Pauli string in the given Fock state.
    fn trace_with_fock_state(&self, fock_state: &Bound<'_, PyAny>) -> PyResult<f64> {
        let bs = pyint_to_bitset(fock_state, self.n_qubits)?;
        Ok(self.trace_fock_state_impl(&bs))
    }

    /// Serialize the monomial as little-endian X bytes concatenated with Z bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let n_bytes = self.n_qubits.div_ceil(8);
        let mut x_bytes = self.x.to_le_bytes();
        let mut z_bytes = self.z.to_le_bytes();
        x_bytes.resize(n_bytes, 0);
        z_bytes.resize(n_bytes, 0);
        x_bytes.extend_from_slice(&z_bytes);
        PyBytes::new(py, &x_bytes)
    }

    fn __hash__(&self) -> u64 {
        let mut h = FxHasher::default();
        self.x.hash(&mut h);
        self.z.hash(&mut h);
        h.finish()
    }

    fn __eq__(&self, other: &PauliString) -> bool {
        self.x == other.x && self.z == other.z
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let x_int: u64 = bitset_to_pyint(py, &self.x)?.extract(py)?;
        let z_int: u64 = bitset_to_pyint(py, &self.z)?.extract(py)?;
        Ok(format!(
            "PauliString(x={x_int:#b}, z={z_int:#b}, n_qubits={})",
            self.n_qubits
        ))
    }
}

impl AbstractTerm for PauliString {
    fn weight(&self) -> u32 {
        self.weight
    }
    fn commutes_with(&self, other: &Self) -> bool {
        self.commutes_with_impl(other)
    }
    fn matmul_internal(&self, other: &Self) -> (Complex64, Self) {
        self.matmul_impl(other)
    }
    fn trace_with_fock_state(&self, fock_state: &Bitset) -> f64 {
        self.trace_fock_state_impl(fock_state)
    }
    fn to_bytes_vec(&self) -> Vec<u8> {
        let n_bytes = self.n_qubits.div_ceil(8);
        let mut x_bytes = self.x.to_le_bytes();
        let mut z_bytes = self.z.to_le_bytes();
        x_bytes.resize(n_bytes, 0);
        z_bytes.resize(n_bytes, 0);
        x_bytes.extend_from_slice(&z_bytes);
        x_bytes
    }
    fn partition_key(&self) -> u64 {
        let mut h = FxHasher::default();
        self.x.hash(&mut h);
        self.z.hash(&mut h);
        h.finish()
    }
    fn system_size(&self) -> u64 {
        self.n_qubits as u64
    }
    fn from_bytes_vec(bytes: &[u8], system_size: u64) -> Self {
        let n_qubits = system_size as usize;
        let n_bytes = n_qubits.div_ceil(8);
        let xb = Bitset::from_le_bytes(&bytes[..n_bytes]);
        let zb = Bitset::from_le_bytes(&bytes[n_bytes..2 * n_bytes]);
        let weight = (&xb | &zb).count_ones();
        PauliString {
            x: xb,
            z: zb,
            n_qubits,
            weight,
        }
    }
}

impl PartialEq for PauliString {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.z == other.z
    }
}

impl Eq for PauliString {}

impl Hash for PauliString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.z.hash(state);
    }
}

pub struct PauliBasis;

impl TermBasis for PauliBasis {
    type Term = PauliString;

    fn commutes(term: [&[u64]; 2], gen: [&[u64]; 2]) -> bool {
        // Anticommutator parity = popcount(term.x & gen.z) + popcount(term.z & gen.x) mod 2.
        let xz: u32 = term[0]
            .iter()
            .zip(gen[1])
            .map(|(a, b)| (a & b).count_ones())
            .sum();
        let zx: u32 = term[1]
            .iter()
            .zip(gen[0])
            .map(|(a, b)| (a & b).count_ones())
            .sum();
        (xz + zx).is_multiple_of(2)
    }

    fn product(term: [&[u64]; 2], gen: [&[u64]; 2], out: [&mut [u64]; 2]) -> Complex64 {
        // gen @ term, matching `matmul_impl(self=gen, other=term)`.
        for i in 0..out[0].len() {
            out[0][i] = gen[0][i] ^ term[0][i];
            out[1][i] = gen[1][i] ^ term[1][i];
        }
        let gxz: u32 = gen[0]
            .iter()
            .zip(gen[1])
            .map(|(a, b)| (a & b).count_ones())
            .sum();
        let txz: u32 = term[0]
            .iter()
            .zip(term[1])
            .map(|(a, b)| (a & b).count_ones())
            .sum();
        let nxz: u32 = out[0]
            .iter()
            .zip(out[1].iter())
            .map(|(a, b)| (a & b).count_ones())
            .sum();
        let gzx: u32 = gen[1]
            .iter()
            .zip(term[0])
            .map(|(a, b)| (a & b).count_ones())
            .sum();
        let p = (gxz as i32 + txz as i32 - nxz as i32 + 2 * gzx as i32).rem_euclid(4);
        match p {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, -1.0),
            _ => unreachable!(),
        }
    }

    fn local_word(gen: [&[u64]; 2]) -> Option<usize> {
        let mut found: Option<usize> = None;
        for (i, (&gx, &gz)) in gen[0].iter().zip(gen[1]).enumerate() {
            if gx != 0 || gz != 0 {
                if found.is_some() {
                    return None; // gen spans more than one word
                }
                found = Some(i);
            }
        }
        found
    }

    fn commutes_at_word(term_word: [u64; 2], gen_word: [u64; 2]) -> bool {
        let xz = (term_word[0] & gen_word[1]).count_ones();
        let zx = (term_word[1] & gen_word[0]).count_ones();
        (xz + zx).is_multiple_of(2)
    }

    fn product_at_word(term_word: [u64; 2], gen_word: [u64; 2]) -> ([u64; 2], Complex64) {
        let out_word = [gen_word[0] ^ term_word[0], gen_word[1] ^ term_word[1]];
        let gxz = (gen_word[0] & gen_word[1]).count_ones();
        let txz = (term_word[0] & term_word[1]).count_ones();
        let nxz = (out_word[0] & out_word[1]).count_ones();
        let gzx = (gen_word[1] & term_word[0]).count_ones();
        let p = (gxz as i32 + txz as i32 - nxz as i32 + 2 * gzx as i32).rem_euclid(4);
        let phase = match p {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, -1.0),
            _ => unreachable!(),
        };
        (out_word, phase)
    }

    fn weight(term: [&[u64]; 2], _n_units: usize) -> u32 {
        term[0]
            .iter()
            .zip(term[1])
            .map(|(a, b)| (a | b).count_ones())
            .sum()
    }

    fn trace(term: [&[u64]; 2], _n_units: usize, fock: &[u64]) -> f64 {
        if term[0].iter().any(|&w| w != 0) {
            return 0.0;
        }
        let parity: u32 = term[1]
            .iter()
            .enumerate()
            .map(|(i, &w)| {
                let f = fock.get(i).copied().unwrap_or(0);
                (w & f).count_ones()
            })
            .sum();
        if parity.is_multiple_of(2) {
            1.0
        } else {
            -1.0
        }
    }

    fn key_hash(term: [&[u64]; 2]) -> u64 {
        let mut h = FxHasher::default();
        term[0].hash(&mut h);
        term[1].hash(&mut h);
        h.finish()
    }

    fn key_eq(a: [&[u64]; 2], b: [&[u64]; 2]) -> bool {
        a[0] == b[0] && a[1] == b[1]
    }

    fn term_from_planes(term: [&[u64]; 2], n_units: usize) -> PauliString {
        let x = Bitset::from_slice(term[0]);
        let z = Bitset::from_slice(term[1]);
        let weight = (&x | &z).count_ones();
        PauliString {
            x,
            z,
            n_qubits: n_units,
            weight,
        }
    }

    fn term_into_planes(term: &PauliString, _n_units: usize, out: [&mut [u64]; 2]) {
        let xw = term.x.as_words();
        let zw = term.z.as_words();
        out[0].fill(0);
        out[0][..xw.len()].copy_from_slice(xw);
        out[1].fill(0);
        out[1][..zw.len()].copy_from_slice(zw);
    }

    fn trace_sparse(row: &[Position], plane_span: usize, _n_units: usize, fock: &[u64]) -> f64 {
        let (x, z) = split_planes(row, plane_span);
        if !x.is_empty() {
            return 0.0;
        }
        let parity = z
            .iter()
            .filter(|&&p| {
                let q = p as usize - plane_span;
                (fock.get(q >> 6).copied().unwrap_or(0) >> (q & 63)) & 1 == 1
            })
            .count();
        if parity.is_multiple_of(2) {
            1.0
        } else {
            -1.0
        }
    }

    fn commutes_sparse(term: &[Position], gen: &[Position], plane_span: usize) -> bool {
        let (tx, tz) = split_planes(term, plane_span);
        let (gx, gz) = split_planes(gen, plane_span);
        let xz = shifted_intersection_count(tx, gz, plane_span);
        let zx = shifted_intersection_count(gx, tz, plane_span);
        (xz + zx).is_multiple_of(2)
    }

    fn product_sparse(
        term: &[Position],
        gen: &[Position],
        plane_span: usize,
        out: &mut Vec<Position>,
    ) -> Complex64 {
        let start = out.len();
        symmetric_difference_into(term, gen, out);
        let (tx, tz) = split_planes(term, plane_span);
        let (gx, gz) = split_planes(gen, plane_span);
        let (nx, nz) = split_planes(&out[start..], plane_span);
        let gxz = shifted_intersection_count(gx, gz, plane_span);
        let txz = shifted_intersection_count(tx, tz, plane_span);
        let nxz = shifted_intersection_count(nx, nz, plane_span);
        let gzx = shifted_intersection_count(tx, gz, plane_span);
        let p = (gxz as i32 + txz as i32 - nxz as i32 + 2 * gzx as i32).rem_euclid(4);
        match p {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, -1.0),
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/string.rs"]
mod tests;
