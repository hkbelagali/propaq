///
/// Defines the core algebra of Pauli strings.
///
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use num_complex::Complex64;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use rustc_hash::FxHasher;

use propaq_core::bitset::Bitset;
use propaq_core::helpers::{pyint_to_bitset, bitset_to_pyint};
use propaq_core::traits::AbstractTerm;
use propaq_core::soa::SoaBasis;

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
        let xz: u32 = self.x.as_words().iter()
            .zip(other.z.as_words())
            .map(|(a, b)| (a & b).count_ones())
            .sum();
        let zx: u32 = self.z.as_words().iter()
            .zip(other.x.as_words())
            .map(|(a, b)| (a & b).count_ones())
            .sum();
        (xz + zx) % 2 == 0
    }

    pub(crate) fn matmul_impl(&self, other: &PauliString) -> (Complex64, PauliString) {
        let new_x = &self.x ^ &other.x;
        let new_z = &self.z ^ &other.z;
        let new_weight = (&new_x | &new_z).count_ones();

        let p = (
            (&self.x & &self.z).count_ones() as i32
            + (&other.x & &other.z).count_ones() as i32
            - (&new_x & &new_z).count_ones() as i32
            + 2 * (&self.z & &other.x).count_ones() as i32
        ).rem_euclid(4);

        let phase = match p {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, -1.0),
            _ => unreachable!(),
        };

        let result = PauliString { x: new_x, z: new_z, n_qubits: self.n_qubits, weight: new_weight };
        (phase, result)
    }

    fn trace_fock_state_impl(&self, fock_state: u64) -> f64 {
        if !self.x.is_zero() {
            return 0.0;
        }
        let fock_bits = Bitset::from_le_bytes(&fock_state.to_le_bytes());
        let parity = (&self.z & &fock_bits).count_ones();
        if parity % 2 == 0 { 1.0 } else { -1.0 }
    }
}

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
        Ok(PauliString { x: x_bits, z: z_bits, n_qubits, weight })
    }

    /// X-component bitmask as a Python int.
    #[getter]
    fn x(&self, py: Python<'_>) -> PyResult<PyObject> {
        bitset_to_pyint(py, &self.x)
    }

    /// Z-component bitmask as a Python int.
    #[getter]
    fn z(&self, py: Python<'_>) -> PyResult<PyObject> {
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
    fn trace_with_fock_state(&self, fock_state: u64) -> f64 {
        self.trace_fock_state_impl(fock_state)
    }

    /// Serialize the monomial as little-endian X bytes concatenated with Z bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let n_bytes = (self.n_qubits + 7) / 8;
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
        Ok(format!("PauliString(x={x_int:#b}, z={z_int:#b}, n_qubits={})", self.n_qubits))
    }
}

impl AbstractTerm for PauliString {
    fn weight(&self) -> u32 { self.weight }
    fn commutes_with(&self, other: &Self) -> bool { self.commutes_with_impl(other) }
    fn matmul_internal(&self, other: &Self) -> (Complex64, Self) { self.matmul_impl(other) }
    fn trace_with_fock_state(&self, fock_state: u64) -> f64 { self.trace_fock_state_impl(fock_state) }
    fn to_bytes_vec(&self) -> Vec<u8> {
        let n_bytes = (self.n_qubits + 7) / 8;
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
    fn system_size(&self) -> u64 { self.n_qubits as u64 }
    fn from_bytes_vec(bytes: &[u8], system_size: u64) -> Self {
        let n_qubits = system_size as usize;
        let n_bytes = (n_qubits + 7) / 8;
        let xb = Bitset::from_le_bytes(&bytes[..n_bytes]);
        let zb = Bitset::from_le_bytes(&bytes[n_bytes..2 * n_bytes]);
        let weight = (&xb | &zb).count_ones();
        PauliString { x: xb, z: zb, n_qubits, weight }
    }
}

impl PartialEq for PauliString {
    fn eq(&self, other: &Self) -> bool { self.x == other.x && self.z == other.z }
}

impl Eq for PauliString {}

impl Hash for PauliString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.z.hash(state);
    }
}

/// SoA engine seam for Pauli strings: the same symplectic algebra as
/// `commutes_with_impl`/`matmul_impl`/`trace_fock_state_impl` above, applied
/// directly to the `x`/`z` word planes of `SoaTermSum<C>` instead of a pair
/// of per-term `Bitset`s. Both planes are identity (a Pauli string is
/// exactly its `(x, z)` pair), so `key_cmp` compares both.
pub struct PauliBasis;

impl SoaBasis for PauliBasis {
    type Term = PauliString;

    fn commutes(term: [&[u64]; 2], gen: [&[u64]; 2]) -> bool {
        // Anticommutator parity = popcount(term.x & gen.z) + popcount(term.z & gen.x) mod 2.
        let xz: u32 = term[0].iter().zip(gen[1]).map(|(a, b)| (a & b).count_ones()).sum();
        let zx: u32 = term[1].iter().zip(gen[0]).map(|(a, b)| (a & b).count_ones()).sum();
        (xz + zx) % 2 == 0
    }

    fn product(term: [&[u64]; 2], gen: [&[u64]; 2], out: [&mut [u64]; 2]) -> Complex64 {
        // gen @ term, matching `matmul_impl(self=gen, other=term)`.
        for i in 0..out[0].len() {
            out[0][i] = gen[0][i] ^ term[0][i];
            out[1][i] = gen[1][i] ^ term[1][i];
        }
        let gxz: u32 = gen[0].iter().zip(gen[1]).map(|(a, b)| (a & b).count_ones()).sum();
        let txz: u32 = term[0].iter().zip(term[1]).map(|(a, b)| (a & b).count_ones()).sum();
        let nxz: u32 = out[0].iter().zip(out[1].iter()).map(|(a, b)| (a & b).count_ones()).sum();
        let gzx: u32 = gen[1].iter().zip(term[0]).map(|(a, b)| (a & b).count_ones()).sum();
        let p = (gxz as i32 + txz as i32 - nxz as i32 + 2 * gzx as i32).rem_euclid(4);
        match p {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, -1.0),
            _ => unreachable!(),
        }
    }

    fn weight(term: [&[u64]; 2], _n_units: usize) -> u32 {
        term[0].iter().zip(term[1]).map(|(a, b)| (a | b).count_ones()).sum()
    }

    fn trace(term: [&[u64]; 2], _n_units: usize, fock: u64) -> f64 {
        if term[0].iter().any(|&w| w != 0) {
            return 0.0;
        }
        let fock_words = fock.to_le_bytes();
        let parity: u32 = term[1]
            .iter()
            .enumerate()
            .map(|(i, &w)| {
                let f = if i == 0 { u64::from_le_bytes(fock_words) } else { 0 };
                (w & f).count_ones()
            })
            .sum();
        if parity % 2 == 0 { 1.0 } else { -1.0 }
    }

    fn key_cmp(a: [&[u64]; 2], b: [&[u64]; 2]) -> Ordering {
        a[0].cmp(b[0]).then_with(|| a[1].cmp(b[1]))
    }

    fn term_from_planes(term: [&[u64]; 2], n_units: usize) -> PauliString {
        let x = Bitset::from_words(term[0].to_vec());
        let z = Bitset::from_words(term[1].to_vec());
        let weight = (&x | &z).count_ones();
        PauliString { x, z, n_qubits: n_units, weight }
    }

    fn term_into_planes(term: &PauliString, _n_units: usize, out: [&mut [u64]; 2]) {
        let xw = term.x.as_words();
        let zw = term.z.as_words();
        out[0].fill(0);
        out[0][..xw.len()].copy_from_slice(xw);
        out[1].fill(0);
        out[1][..zw.len()].copy_from_slice(zw);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pauli(x: u64, z: u64, n: usize) -> PauliString {
        let xb = Bitset::from_le_bytes(&x.to_le_bytes());
        let zb = Bitset::from_le_bytes(&z.to_le_bytes());
        let weight = (&xb | &zb).count_ones();
        PauliString { x: xb, z: zb, n_qubits: n, weight }
    }

    #[test]
    fn identity_weight_zero() { assert_eq!(pauli(0, 0, 4).weight, 0); }

    #[test]
    fn single_x_weight_one() { assert_eq!(pauli(0b01, 0, 4).weight, 1); }

    #[test]
    fn single_z_weight_one() { assert_eq!(pauli(0, 0b01, 4).weight, 1); }

    #[test]
    fn single_y_weight_one() { assert_eq!(pauli(0b01, 0b01, 4).weight, 1); }

    #[test]
    fn identity_commutes_with_everything() {
        let id = pauli(0, 0, 4);
        let x = pauli(0b01, 0, 4);
        assert!(id.commutes_with_impl(&x));
        assert!(x.commutes_with_impl(&id));
    }

    #[test]
    fn x_commutes_with_itself() {
        let x = pauli(0b01, 0, 4);
        assert!(x.commutes_with_impl(&x));
    }

    #[test]
    fn x_anticommutes_z_same_qubit() {
        let x = pauli(0b01, 0, 4);
        let z = pauli(0, 0b01, 4);
        assert!(!x.commutes_with_impl(&z));
    }

    #[test]
    fn x0_commutes_z1_different_qubits() {
        let x0 = pauli(0b01, 0, 4);
        let z1 = pauli(0, 0b10, 4);
        assert!(x0.commutes_with_impl(&z1));
    }

    #[test]
    fn matmul_x_times_x_is_identity() {
        let x = pauli(0b01, 0, 4);
        let (phase, result) = x.matmul_impl(&x);
        assert!((phase - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert_eq!(result.weight, 0);
    }

    #[test]
    fn matmul_x_times_z_gives_y_with_phase() {
        let x = pauli(0b01, 0, 4);
        let z = pauli(0, 0b01, 4);
        let (phase, result) = x.matmul_impl(&z);
        assert!((phase - Complex64::new(0.0, -1.0)).norm() < 1e-10);
        assert_eq!(result.weight, 1);
    }

    #[test]
    fn trace_identity_is_one() { assert_eq!(pauli(0, 0, 4).trace_fock_state_impl(0), 1.0); }

    #[test]
    fn trace_x_is_zero() { assert_eq!(pauli(0b01, 0, 4).trace_fock_state_impl(0), 0.0); }

    #[test]
    fn trace_z0_empty_state() { assert_eq!(pauli(0, 0b01, 4).trace_fock_state_impl(0b00), 1.0); }

    #[test]
    fn trace_z0_occupied_state() { assert_eq!(pauli(0, 0b01, 4).trace_fock_state_impl(0b01), -1.0); }

    #[test]
    fn trace_zz_all_combinations() {
        let zz = pauli(0, 0b11, 4);
        assert_eq!(zz.trace_fock_state_impl(0b00),  1.0);
        assert_eq!(zz.trace_fock_state_impl(0b01), -1.0);
        assert_eq!(zz.trace_fock_state_impl(0b10), -1.0);
        assert_eq!(zz.trace_fock_state_impl(0b11),  1.0);
    }

    // --- `PauliBasis` (SoA word-plane kernels) vs `PauliString` (AoS,
    // already exhaustively tested above) cross-checks. Both must agree
    // exactly, since `PauliBasis` is meant to be a bit-for-bit vectorized
    // restatement of the same symplectic algebra.

    fn planes_of(p: &PauliString, stride: usize) -> (Vec<u64>, Vec<u64>) {
        let mut gx = vec![0u64; stride];
        let mut gz = vec![0u64; stride];
        PauliBasis::term_into_planes(p, p.n_qubits, [&mut gx, &mut gz]);
        (gx, gz)
    }

    fn assert_basis_matches(a: &PauliString, b: &PauliString) {
        let stride = 1;
        let (ax, az) = planes_of(a, stride);
        let (bx, bz) = planes_of(b, stride);
        let a_planes = [ax.as_slice(), az.as_slice()];
        let b_planes = [bx.as_slice(), bz.as_slice()];
        let ctx = || format!("a=(x={ax:?},z={az:?}) b=(x={bx:?},z={bz:?})");

        assert_eq!(
            PauliBasis::commutes(a_planes, b_planes),
            a.commutes_with_impl(b),
            "commutes mismatch for {}", ctx(),
        );
        assert_eq!(PauliBasis::weight(a_planes, a.n_qubits), a.weight, "weight mismatch for {}", ctx());

        let (expected_phase, expected_result) = a.matmul_impl(b);
        let mut out_x = vec![0u64; stride];
        let mut out_z = vec![0u64; stride];
        let phase = PauliBasis::product(b_planes, a_planes, [&mut out_x, &mut out_z]);
        assert!((phase - expected_phase).norm() < 1e-10, "phase mismatch for {}", ctx());
        let result = PauliBasis::term_from_planes([&out_x, &out_z], a.n_qubits);
        assert_eq!(result.x, expected_result.x, "product x mismatch for {}", ctx());
        assert_eq!(result.z, expected_result.z, "product z mismatch for {}", ctx());

        for fock in 0u64..16 {
            assert_eq!(
                PauliBasis::trace(a_planes, a.n_qubits, fock),
                a.trace_fock_state_impl(fock),
                "trace mismatch for {} fock={fock}", ctx(),
            );
        }
    }

    #[test]
    fn pauli_basis_matches_aos_exhaustive_4_qubit() {
        for xa in 0u64..16 {
            for za in 0u64..16 {
                let a = pauli(xa, za, 4);
                for xb in 0u64..16 {
                    for zb in 0u64..16 {
                        let b = pauli(xb, zb, 4);
                        assert_basis_matches(&a, &b);
                    }
                }
            }
        }
    }

    #[test]
    fn pauli_basis_key_cmp_orders_like_equality() {
        let a = pauli(0b01, 0b10, 4);
        let b = pauli(0b01, 0b10, 4);
        let c = pauli(0b11, 0b10, 4);
        let (ax, az) = planes_of(&a, 1);
        let (bx, bz) = planes_of(&b, 1);
        let (cx, cz) = planes_of(&c, 1);
        assert_eq!(
            PauliBasis::key_cmp([&ax, &az], [&bx, &bz]),
            Ordering::Equal,
            "identical strings must compare equal under key_cmp",
        );
        assert_ne!(
            PauliBasis::key_cmp([&ax, &az], [&cx, &cz]),
            Ordering::Equal,
            "distinct strings must not compare equal under key_cmp",
        );
    }
}
