//!
//! Ingests MPS tensors from Python as zero-copy numpy array views
//! and exposes the one hybrid expectation-value entry point
//!
use num_complex::Complex64;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use propaq_pauli::PauliTermSum;

use crate::contract::{
    check_dims, hybrid_expectation_sum, hybrid_expectation_sum_raw, LocalTable,
};
use crate::mps::{build_environments, norm_squared, Mps, MpsTensor};

fn mps_from_numpy(arrays: &[PyReadonlyArray3<'_, Complex64>]) -> PyResult<Mps> {
    let mut tensors = Vec::with_capacity(arrays.len());
    for arr in arrays {
        let view = arr.as_array();
        let (chi_l, phys, chi_r) = view.dim();
        let mut data = Vec::with_capacity(chi_l * phys * chi_r);
        for l in 0..chi_l {
            for s in 0..phys {
                for r in 0..chi_r {
                    data.push(view[[l, s, r]]);
                }
            }
        }
        tensors.push(MpsTensor::new(data, chi_l, phys, chi_r));
    }
    Mps::new(tensors).map_err(PyValueError::new_err)
}

/// Computes \(\sum_i c_i \langle\Psi|P_i|\Psi\rangle\) over every term in `observable`,
/// given \(|\Psi\rangle\) as a list of MPS site tensors (each rank-3, shape
/// `(bond_left, 2, bond_right)`, with dummy size-1 bonds at the open
/// boundaries
///
/// Arguments:
///     observable: A Heisenberg-propagated `PauliTermSum` (e.g. the result
///         of `PauliPropagator.propagate(observable, circuit1)`).
///     mps_arrays: \(|\Psi\rangle\)'s site tensors in left-to-right qubit order,
///         complex128, rank-3, contiguous.
#[gen_stub_pyfunction(module = "propaq._rust_core")]
#[pyfunction]
pub fn hybrid_expectation(
    observable: &PauliTermSum,
    mps_arrays: Vec<PyReadonlyArray3<'_, Complex64>>,
) -> PyResult<f64> {
    let mps = mps_from_numpy(&mps_arrays)?;
    if mps.n_sites != observable.n_units() {
        return Err(PyValueError::new_err(format!(
            "mps_arrays has {} sites but observable is defined on {} qubits",
            mps.n_sites,
            observable.n_units()
        )));
    }
    let table = LocalTable::pauli();
    check_dims(&mps, &table).map_err(PyValueError::new_err)?;
    let env = build_environments(&mps);
    let terms = observable.as_f64();
    let sum = hybrid_expectation_sum(&mps, &env, &table, &terms);
    let norm2 = norm_squared(&env);
    let result = sum / norm2;
    debug_assert!(
        result.im.abs() < 1e-6 * result.norm().max(1.0),
        "hybrid_expectation produced a non-negligible imaginary part: {result:?} \
         (observable must be Hermitian and |Psi> a valid state)"
    );
    Ok(result.re)
}

/// Computes \(\langle\Psi|\Theta|\Psi\rangle / \langle\Psi|\Psi\rangle\) for
/// an observable of any basis, given as raw words, and returns the complex
/// value. The basis supplies one matrix for each unit label (`local_table`).
///
/// Arguments:
///     words: Term keys, `uint64` array of shape `(n_terms, stride)`. Unit `u`
///         owns the bits `u * bits_per_unit` to `(u + 1) * bits_per_unit - 1`.
///     coeffs: Complex coefficients, one per term.
///     bits_per_unit: Bits of one unit label. Divides 64 and is at most 8.
///     local_dim: Physical dimension of one unit.
///     local_table: Flat complex array of shape `(2 ** bits_per_unit * local_dim ** 2,)`;
///         matrix `[label][sigma * local_dim + sigma_p]`. Label 0 is the identity.
///     mps_arrays: Site tensors `(bond_left, local_dim, bond_right)`, complex128.
#[gen_stub_pyfunction(module = "propaq._rust_core")]
#[pyfunction]
pub fn hybrid_expectation_raw(
    words: PyReadonlyArray2<'_, u64>,
    coeffs: PyReadonlyArray1<'_, Complex64>,
    bits_per_unit: usize,
    local_dim: usize,
    local_table: PyReadonlyArray1<'_, Complex64>,
    mps_arrays: Vec<PyReadonlyArray3<'_, Complex64>>,
) -> PyResult<Complex64> {
    if bits_per_unit == 0 || bits_per_unit > 8 || !64usize.is_multiple_of(bits_per_unit) {
        return Err(PyValueError::new_err(
            "bits_per_unit must divide 64 and be at most 8",
        ));
    }
    let words = words.as_array();
    let (n_terms, stride) = words.dim();
    let coeffs = coeffs.as_array();
    if coeffs.len() != n_terms {
        return Err(PyValueError::new_err(format!(
            "{} coefficients for {n_terms} terms",
            coeffs.len()
        )));
    }
    let mps = mps_from_numpy(&mps_arrays)?;
    if stride * 64 < mps.n_sites * bits_per_unit {
        return Err(PyValueError::new_err(format!(
            "terms have {stride} words, too few for {} units of {bits_per_unit} bits",
            mps.n_sites
        )));
    }
    let table = LocalTable::new(local_dim, local_table.as_array().iter().copied().collect())
        .map_err(PyValueError::new_err)?;
    if table.n_labels() != 1usize << bits_per_unit {
        return Err(PyValueError::new_err(format!(
            "local table has {} labels, expected {}",
            table.n_labels(),
            1usize << bits_per_unit
        )));
    }
    check_dims(&mps, &table).map_err(PyValueError::new_err)?;
    let flat_words: Vec<u64> = words.iter().copied().collect();
    let flat_coeffs: Vec<Complex64> = coeffs.iter().copied().collect();
    let env = build_environments(&mps);
    let sum = hybrid_expectation_sum_raw(
        &mps,
        &env,
        &table,
        &flat_words,
        stride,
        bits_per_unit,
        &flat_coeffs,
    );
    Ok(sum / norm_squared(&env))
}
