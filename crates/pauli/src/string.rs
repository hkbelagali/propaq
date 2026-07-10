///
/// Defines the core algebra of Pauli strings.
///
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use num_complex::Complex64;
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

/// One of `gen`'s (at most two) touched-qubit positions, located once per
/// `commutes`/`product` call by `classify_gen` — see `GenShape`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct GenSite {
    word: usize,
    /// Exactly one bit set: the qubit's position within `word`.
    mask: u64,
    gx: bool,
    gz: bool,
}

/// Structural classification of `gen`'s support (`popcount(gen.x | gen.z)`),
/// computed fresh at the top of `commutes`/`product`. `gen` is fixed for an
/// entire `soa::kernels::apply_rotation` call (the same generator is checked
/// against every live term), so real circuits — which emit weight-1
/// (`Rz`-shaped) or weight-2 (`XX+YY`/`CP`-shaped) generators for essentially
/// every gate (see `propaq/datatypes/pauli/termsum.py`) — hit `Weight1`/
/// `Weight2` on every call, letting `commutes_fast`/`product_fast` replace
/// the generic multi-word popcount/XOR reductions with O(1) bit tests at the
/// known site(s). Wider generators fall back to the unchanged generic path.
#[derive(Debug, PartialEq)]
enum GenShape {
    Identity,
    Weight1(GenSite),
    Weight2(GenSite, GenSite),
    Wide,
}

/// Single fused pass: accumulates `gen`'s weight and locates up to two set
/// bits at once (never a separate popcount-then-locate pass, so a narrow
/// generator touching a late word costs no more than the weight scan alone
/// already would).
#[inline]
fn classify_gen(gen: [&[u64]; 2]) -> GenShape {
    let mut weight: u32 = 0;
    let mut site0: Option<GenSite> = None;
    let mut site1: Option<GenSite> = None;
    for (word, (&gx_w, &gz_w)) in gen[0].iter().zip(gen[1]).enumerate() {
        let combined = gx_w | gz_w;
        weight += combined.count_ones();
        if weight > 2 {
            return GenShape::Wide;
        }
        let mut remaining = combined;
        while remaining != 0 {
            let mask = 1u64 << remaining.trailing_zeros();
            let site = GenSite { word, mask, gx: gx_w & mask != 0, gz: gz_w & mask != 0 };
            if site0.is_none() {
                site0 = Some(site);
            } else {
                site1 = Some(site);
            }
            remaining &= remaining - 1;
        }
    }
    match (site0, site1) {
        (None, None) => GenShape::Identity,
        (Some(s), None) => GenShape::Weight1(s),
        (Some(a), Some(b)) => GenShape::Weight2(a, b),
        (None, Some(_)) => unreachable!("a site is only ever recorded as site1 after site0"),
    }
}

/// Whether `term` anticommutes with `gen` *at this one site only* —
/// `(tx & gz) ^ (tz & gx)`, restricted to `gen`'s zero-elsewhere support this
/// is the whole anticommutator parity contribution from this site.
#[inline]
fn site_anticommutes(term: [&[u64]; 2], site: &GenSite) -> bool {
    let tx = term[0][site.word] & site.mask != 0;
    let tz = term[1][site.word] & site.mask != 0;
    (tx && site.gz) ^ (tz && site.gx)
}

/// This site's contribution to the product's mod-4 phase exponent (see the
/// derivation in the design plan: `txz`/`nxz`'s difference telescopes to a
/// sum over `gen`'s touched sites alone, and `i^(sum) = product of i^(term)`,
/// so each site's exponent can be summed independently and reduced mod 4
/// once at the end).
#[inline]
fn site_phase_exponent(term: [&[u64]; 2], site: &GenSite) -> i32 {
    let tx = (term[0][site.word] & site.mask != 0) as i32;
    let tz = (term[1][site.word] & site.mask != 0) as i32;
    let gx = site.gx as i32;
    let gz = site.gz as i32;
    let ox = gx ^ tx;
    let oz = gz ^ tz;
    gx * gz + tx * tz - ox * oz + 2 * gz * tx
}

/// Fast commute check for `Identity`/`Weight1`/`Weight2` shapes; `None` for
/// `Wide` (caller falls back to `commutes_generic`).
#[inline]
fn commutes_fast(term: [&[u64]; 2], shape: &GenShape) -> Option<bool> {
    match shape {
        GenShape::Identity => Some(true),
        GenShape::Weight1(s) => Some(!site_anticommutes(term, s)),
        GenShape::Weight2(a, b) => Some(!(site_anticommutes(term, a) ^ site_anticommutes(term, b))),
        GenShape::Wide => None,
    }
}

/// Fast product for `Identity`/`Weight1`/`Weight2` shapes: `out` is built via
/// `copy_from_slice` (not a full XOR loop) plus targeted XORs at the ≤2
/// touched words, and the phase is a sum of ≤2 independent per-site
/// exponents. `None` for `Wide` (caller falls back to `product_generic`,
/// which writes `out` itself).
#[inline]
fn product_fast(
    term: [&[u64]; 2],
    gen: [&[u64]; 2],
    shape: &GenShape,
    out: [&mut [u64]; 2],
) -> Option<Complex64> {
    let sites: [Option<GenSite>; 2] = match *shape {
        GenShape::Identity => [None, None],
        GenShape::Weight1(s) => [Some(s), None],
        GenShape::Weight2(a, b) => [Some(a), Some(b)],
        GenShape::Wide => return None,
    };

    out[0].copy_from_slice(term[0]);
    out[1].copy_from_slice(term[1]);
    let mut phase_exp = 0i32;
    for site in sites.into_iter().flatten() {
        out[0][site.word] ^= gen[0][site.word] & site.mask;
        out[1][site.word] ^= gen[1][site.word] & site.mask;
        phase_exp += site_phase_exponent(term, &site);
    }
    Some(match phase_exp.rem_euclid(4) {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        3 => Complex64::new(0.0, -1.0),
        _ => unreachable!(),
    })
}

fn commutes_generic(term: [&[u64]; 2], gen: [&[u64]; 2]) -> bool {
    // Anticommutator parity = popcount(term.x & gen.z) + popcount(term.z & gen.x) mod 2.
    let xz: u32 = term[0].iter().zip(gen[1]).map(|(a, b)| (a & b).count_ones()).sum();
    let zx: u32 = term[1].iter().zip(gen[0]).map(|(a, b)| (a & b).count_ones()).sum();
    (xz + zx) % 2 == 0
}

fn product_generic(term: [&[u64]; 2], gen: [&[u64]; 2], out: [&mut [u64]; 2]) -> Complex64 {
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

/// SoA engine seam for Pauli strings: the same symplectic algebra as
/// `commutes_with_impl`/`matmul_impl`/`trace_fock_state_impl` above, applied
/// directly to the `x`/`z` word planes of `SoaTermSum<C>` instead of a pair
/// of per-term `Bitset`s. Both planes are identity (a Pauli string is
/// exactly its `(x, z)` pair), so `key_hash`/`key_eq` cover both.
pub struct PauliBasis;

impl SoaBasis for PauliBasis {
    type Term = PauliString;

    fn commutes(term: [&[u64]; 2], gen: [&[u64]; 2]) -> bool {
        let shape = classify_gen(gen);
        if let Some(fast) = commutes_fast(term, &shape) {
            debug_assert_eq!(
                fast, commutes_generic(term, gen),
                "PauliBasis::commutes fast/generic mismatch"
            );
            return fast;
        }
        commutes_generic(term, gen)
    }

    fn product(term: [&[u64]; 2], gen: [&[u64]; 2], out: [&mut [u64]; 2]) -> Complex64 {
        let [ox, oz] = out;
        let shape = classify_gen(gen);
        if let Some(phase) = product_fast(term, gen, &shape, [&mut *ox, &mut *oz]) {
            #[cfg(debug_assertions)]
            {
                let mut ref_x = vec![0u64; ox.len()];
                let mut ref_z = vec![0u64; oz.len()];
                let ref_phase = product_generic(term, gen, [&mut ref_x, &mut ref_z]);
                debug_assert!(
                    (phase - ref_phase).norm() < 1e-9,
                    "PauliBasis::product fast/generic phase mismatch"
                );
                debug_assert_eq!(*ox, ref_x[..], "PauliBasis::product fast/generic x mismatch");
                debug_assert_eq!(*oz, ref_z[..], "PauliBasis::product fast/generic z mismatch");
            }
            return phase;
        }
        product_generic(term, gen, [ox, oz])
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
        assert_basis_matches_at(a, b, 1);
    }

    /// Generalizes `assert_basis_matches` over `stride` so the fast-path
    /// tests below can exercise multi-word (`stride > 1`) placements — the
    /// existing exhaustive test only ever needs `stride = 1`.
    fn assert_basis_matches_at(a: &PauliString, b: &PauliString, stride: usize) {
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

        assert_eq!(PauliBasis::key_eq(a_planes, b_planes), *a == *b, "key_eq mismatch for {}", ctx());
        if PauliBasis::key_eq(a_planes, b_planes) {
            assert_eq!(
                PauliBasis::key_hash(a_planes), PauliBasis::key_hash(b_planes),
                "key_eq strings must key_hash equally for {}", ctx(),
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
    fn pauli_basis_key_eq_and_hash_agree_with_equality() {
        let a = pauli(0b01, 0b10, 4);
        let b = pauli(0b01, 0b10, 4);
        let c = pauli(0b11, 0b10, 4);
        let (ax, az) = planes_of(&a, 1);
        let (bx, bz) = planes_of(&b, 1);
        let (cx, cz) = planes_of(&c, 1);
        assert!(PauliBasis::key_eq([&ax, &az], [&bx, &bz]), "identical strings must be key_eq");
        assert_eq!(
            PauliBasis::key_hash([&ax, &az]), PauliBasis::key_hash([&bx, &bz]),
            "key_eq strings must key_hash equally (merge's parallel-batch correctness depends on this)",
        );
        assert!(!PauliBasis::key_eq([&ax, &az], [&cx, &cz]), "distinct strings must not be key_eq");
    }

    // --- Narrow-generator fast path (`classify_gen`/`commutes_fast`/
    // `product_fast`): the exhaustive test above already exercises the
    // dispatch at `stride=1`; these target what it can't reach — multi-word
    // placements, including the qubit-63/64 word boundary, the
    // wide-generator fallback at `stride>1`, and a large randomized sweep.

    fn multiword_pauli(x_words: &[u64], z_words: &[u64], n: usize) -> PauliString {
        let x = Bitset::from_words(x_words.to_vec());
        let z = Bitset::from_words(z_words.to_vec());
        let weight = (&x | &z).count_ones();
        PauliString { x, z, n_qubits: n, weight }
    }

    #[test]
    fn classify_gen_locates_sites_correctly() {
        const N: usize = 192;
        let stride = PauliBasis::stride_words(N);

        let (gx, gz) = planes_of(&multiword_pauli(&[], &[], N), stride);
        assert_eq!(classify_gen([gx.as_slice(), gz.as_slice()]), GenShape::Identity);

        // Weight 1: a lone Z at qubit 130 (word 2, bit 2).
        let (gx, gz) = planes_of(&multiword_pauli(&[0, 0, 0], &[0, 0, 1 << 2], N), stride);
        assert_eq!(
            classify_gen([gx.as_slice(), gz.as_slice()]),
            GenShape::Weight1(GenSite { word: 2, mask: 1 << 2, gx: false, gz: true }),
        );

        // Weight 2, same word: X at qubit 128 (word 2, bit 0), Z at qubit 130 (word 2, bit 2).
        let (gx, gz) = planes_of(&multiword_pauli(&[0, 0, 1], &[0, 0, 1 << 2], N), stride);
        assert_eq!(
            classify_gen([gx.as_slice(), gz.as_slice()]),
            GenShape::Weight2(
                GenSite { word: 2, mask: 1, gx: true, gz: false },
                GenSite { word: 2, mask: 1 << 2, gx: false, gz: true },
            ),
        );

        // Weight 3: falls back to Wide.
        let (gx, gz) = planes_of(&multiword_pauli(&[0, 0, 0b111], &[0, 0, 0], N), stride);
        assert_eq!(classify_gen([gx.as_slice(), gz.as_slice()]), GenShape::Wide);
    }

    #[test]
    fn pauli_basis_fast_path_weight1_high_word() {
        const N: usize = 192;
        let stride = PauliBasis::stride_words(N);
        let gen = multiword_pauli(&[0, 0, 0], &[0, 0, 1 << 2], N); // Z at qubit 130
        let backgrounds = [
            multiword_pauli(&[0, 0, 0], &[0, 0, 0], N),
            multiword_pauli(&[u64::MAX, u64::MAX, u64::MAX], &[u64::MAX, u64::MAX, u64::MAX], N),
            multiword_pauli(&[0xAAAA_AAAA_AAAA_AAAA, 0, 0], &[0x5555_5555_5555_5555, 0, 0], N),
            multiword_pauli(&[0, 0, 1 << 2], &[0, 0, 1 << 2], N), // Y at the exact touched qubit
        ];
        for term in &backgrounds {
            assert_basis_matches_at(&gen, term, stride);
        }
    }

    #[test]
    fn pauli_basis_fast_path_weight2_same_late_word() {
        const N: usize = 192;
        let stride = PauliBasis::stride_words(N);
        let gen = multiword_pauli(&[0, 0, 1], &[0, 0, 1 << 5], N); // X at qubit 128, Z at qubit 133
        let term = multiword_pauli(&[0, 0, 0b1010_1010], &[0, 0, 0b0101_0101], N);
        assert_basis_matches_at(&gen, &term, stride);
    }

    #[test]
    fn pauli_basis_fast_path_weight2_split_words() {
        const N: usize = 128;
        let stride = PauliBasis::stride_words(N);
        // Qubit 63 (word 0, top bit) and qubit 64 (word 1, bottom bit) — the
        // exact word boundary, the likeliest off-by-one spot.
        let gen = multiword_pauli(&[1u64 << 63, 1], &[0, 0], N);
        let backgrounds = [
            multiword_pauli(&[0, 0], &[0, 0], N),
            multiword_pauli(&[u64::MAX, u64::MAX], &[u64::MAX, u64::MAX], N),
            multiword_pauli(&[1u64 << 63, 1], &[1u64 << 63, 1], N), // Y at both touched qubits
            multiword_pauli(&[1u64 << 63, 0], &[0, 1], N),
        ];
        for term in &backgrounds {
            assert_basis_matches_at(&gen, term, stride);
        }
    }

    #[test]
    fn pauli_basis_fast_path_wide_gen_multiword_matches_generic() {
        const N: usize = 192;
        let stride = PauliBasis::stride_words(N);
        let gen = multiword_pauli(&[0xFFFF, 0, 0], &[0x0F0F, 0, 0], N); // weight > 2
        let term = multiword_pauli(&[0x1234, 0x5678, 0x9ABC], &[0xDEF0, 0x1111, 0x2222], N);
        assert_basis_matches_at(&gen, &term, stride);
    }

    #[test]
    fn pauli_basis_fast_path_randomized_cross_word() {
        const N: usize = 256;
        let stride = PauliBasis::stride_words(N);
        let mut seed = 0x243F_6A88_85A3_08D3u64;
        let mut next_u64 = move || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };

        for _ in 0..10_000 {
            // gen: weight 0-2 at random (possibly cross-word) positions.
            let gen_weight = next_u64() % 3;
            let mut gx = vec![0u64; stride];
            let mut gz = vec![0u64; stride];
            for _ in 0..gen_weight {
                let bit = (next_u64() as usize) % (stride * 64);
                if next_u64() % 2 == 0 {
                    gx[bit / 64] |= 1u64 << (bit % 64);
                } else {
                    gz[bit / 64] |= 1u64 << (bit % 64);
                }
            }
            let gen = multiword_pauli(&gx, &gz, N);

            let tx: Vec<u64> = (0..stride).map(|_| next_u64()).collect();
            let tz: Vec<u64> = (0..stride).map(|_| next_u64()).collect();
            let term = multiword_pauli(&tx, &tz, N);

            assert_basis_matches_at(&gen, &term, stride);
        }
    }
}
