///
/// Ingests MPS tensors from Python as zero-copy numpy array views 
/// and exposes the one hybrid expectation-value entry point
///
use num_complex::Complex64;
use numpy::PyReadonlyArray3;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use propaq_pauli::PauliTermSum;

use crate::contract::hybrid_expectation_sum;
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

/// Computes `sum_i coeff_i * <Psi|P_i|Psi>` over every term in `observable`,
/// given `|Psi>` as a list of MPS site tensors (each rank-3, shape
/// `(bond_left, 2, bond_right)`, with dummy size-1 bonds at the open
/// boundaries
///
/// Arguments:
///     observable: A Heisenberg-propagated `PauliTermSum` (e.g. the result
///         of `PauliPropagator.propagate(observable, circuit1)`).
///     mps_arrays: `|Psi>`'s site tensors in left-to-right qubit order,
///         complex128, rank-3, contiguous.
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
    let env = build_environments(&mps);
    let terms = observable.as_f64();
    let sum = hybrid_expectation_sum(&mps, &env, &terms);
    let norm2 = norm_squared(&env);
    let result = sum / norm2;
    debug_assert!(
        result.im.abs() < 1e-6 * result.norm().max(1.0),
        "hybrid_expectation produced a non-negligible imaginary part: {result:?} \
         (observable must be Hermitian and |Psi> a valid state)"
    );
    Ok(result.re)
}
