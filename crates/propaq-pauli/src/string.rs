use pyo3::prelude::*;
use pyo3::types::PyBytes;
use num_complex::Complex64;
use std::hash::{Hash, Hasher};
use rustc_hash::FxHasher;

use propaq_core::bitset::Bitset;
use propaq_core::helpers::{pyint_to_bitset, bitset_to_pyint};
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
        self.x.as_words().iter().chain(self.z.as_words().iter())
            .fold(0u64, |acc, &w| acc ^ w)
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
}
