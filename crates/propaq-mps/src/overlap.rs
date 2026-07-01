use faer::linalg::matmul::matmul as fgemm;
use faer::prelude::c64 as fc64;
use faer::Parallelism;
use num_complex::Complex64;
use numpy::ndarray::{ArrayViewD, Axis};
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

/// One MPS tensor stored as its two physical-index slices.
///
/// `proj[s]` is the (D_left × D_right) matrix A[:, :, s] stored row-major.
/// The leftmost site has `rows == 1` (no left bond), the rightmost has
/// `cols == 1` (no right bond), and interior sites have `rows == D_left`,
/// `cols == D_right`.
pub struct ProjectedSite {
    pub rows: usize,
    pub cols: usize,
    pub proj: [Vec<Complex64>; 2],
}

fn project_site(
    arr: &ArrayViewD<'_, Complex64>,
    is_first: bool,
    is_last: bool,
) -> PyResult<ProjectedSite> {
    match arr.shape() {
        [d_left, d_right, 2] => {
            let (d_left, d_right) = (*d_left, *d_right);
            let proj0: Vec<Complex64> = arr.index_axis(Axis(2), 0).iter().copied().collect();
            let proj1: Vec<Complex64> = arr.index_axis(Axis(2), 1).iter().copied().collect();
            Ok(ProjectedSite { rows: d_left, cols: d_right, proj: [proj0, proj1] })
        }
        [d, 2] if is_first != is_last => {
            let d = *d;
            let proj0: Vec<Complex64> = arr.index_axis(Axis(1), 0).iter().copied().collect();
            let proj1: Vec<Complex64> = arr.index_axis(Axis(1), 1).iter().copied().collect();
            if is_first {
                Ok(ProjectedSite { rows: 1, cols: d, proj: [proj0, proj1] })
            } else {
                Ok(ProjectedSite { rows: d, cols: 1, proj: [proj0, proj1] })
            }
        }
        shape => Err(PyValueError::new_err(format!(
            "unexpected MPS tensor shape {shape:?}; expected (D_left, D_right, 2) for \
             interior sites or (D, 2) for the first/last site"
        ))),
    }
}

/// Accumulate one physical-index contribution into the left environment.
///
/// Complex64 and faer::c64 share the same #[repr(C)] layout, so the
/// pointer casts below are safe.
pub fn update_l(
    l_new: &mut [Complex64],
    d_r: usize,
    d_l: usize,
    l: &[Complex64],
    bra: &[Complex64],
    ket: &[Complex64],
    scale: Complex64,
) {
    let mut tmp = vec![fc64::new(0.0, 0.0); d_r * d_l];
    let sc = fc64::new(scale.re, scale.im);
    let one = fc64::new(1.0, 0.0);

    // Complex64 and faer::c64 share #[repr(C)] {re: f64, im: f64}; pointer cast is safe.
    let bra_fc = unsafe { std::slice::from_raw_parts(bra.as_ptr() as *const fc64, bra.len()) };
    let l_fc   = unsafe { std::slice::from_raw_parts(l.as_ptr()   as *const fc64, l.len()) };
    let ket_fc = unsafe { std::slice::from_raw_parts(ket.as_ptr() as *const fc64, ket.len()) };

    // All matrices are stored row-major.
    let bra_mat = faer::mat::from_row_major_slice::<fc64>(bra_fc, d_l, d_r);
    let l_mat   = faer::mat::from_row_major_slice::<fc64>(l_fc,   d_l, d_l);
    let ket_mat = faer::mat::from_row_major_slice::<fc64>(ket_fc, d_l, d_r);

    {
        let tmp_mat = faer::mat::from_row_major_slice_mut::<fc64>(&mut tmp, d_r, d_l);
        fgemm(tmp_mat, bra_mat.adjoint(), l_mat, None, one, Parallelism::None);
    }

    {
        let tmp_ref = faer::mat::from_row_major_slice::<fc64>(&tmp, d_r, d_l);
        let l_new_fc = unsafe {
            std::slice::from_raw_parts_mut(l_new.as_mut_ptr() as *mut fc64, l_new.len())
        };
        let l_new_mat = faer::mat::from_row_major_slice_mut::<fc64>(l_new_fc, d_r, d_r);
        fgemm(l_new_mat, tmp_ref, ket_mat, Some(one), sc, Parallelism::None);
    }
}

/// Compute `<MPS|P|MPS>` for a Pauli string P via a left-to-right transfer
/// matrix sweep.
///
/// At each site the local Pauli acts on the ket copy of the tensor:
///   I: ket unchanged
///   X: ket_0 = proj[1],      ket_1 = proj[0]
///   Y: ket_0 = -i·proj[1],   ket_1 =  i·proj[0]   (Y = [[0,-i],[i,0]])
///   Z: ket_0 = proj[0],      ket_1 = -proj[1]
///
/// `pauli_str` follows Qiskit's big-endian convention (leftmost character =
/// highest qubit index), so we pair sites in forward order with characters in
/// reverse order.
pub fn pauli_expectation(sites: &[ProjectedSite], pauli_str: &str) -> Complex64 {
    let mut l = vec![Complex64::new(1.0, 0.0)]; // 1×1 left boundary

    for (site, ch) in sites.iter().zip(pauli_str.chars().rev()) {
        let d_l = site.rows;
        let d_r = site.cols;
        let mut l_new = vec![Complex64::new(0.0, 0.0); d_r * d_r];

        let one  = Complex64::new(1.0, 0.0);
        let neg  = Complex64::new(-1.0, 0.0);
        let ic   = Complex64::new(0.0, 1.0);
        let neg_i = Complex64::new(0.0, -1.0);

        match ch {
            'X' => {
                update_l(&mut l_new, d_r, d_l, &l, &site.proj[0], &site.proj[1], one);
                update_l(&mut l_new, d_r, d_l, &l, &site.proj[1], &site.proj[0], one);
            }
            'Y' => {
                update_l(&mut l_new, d_r, d_l, &l, &site.proj[0], &site.proj[1], neg_i);
                update_l(&mut l_new, d_r, d_l, &l, &site.proj[1], &site.proj[0], ic);
            }
            'Z' => {
                update_l(&mut l_new, d_r, d_l, &l, &site.proj[0], &site.proj[0], one);
                update_l(&mut l_new, d_r, d_l, &l, &site.proj[1], &site.proj[1], neg);
            }
            _ => {
                update_l(&mut l_new, d_r, d_l, &l, &site.proj[0], &site.proj[0], one);
                update_l(&mut l_new, d_r, d_l, &l, &site.proj[1], &site.proj[1], one);
            }
        }

        l = l_new;
    }

    l[0]
}

/// Compute `sum_k coeff_k * <MPS|P_k|MPS>` for a batch of Pauli strings.
///
/// The sum over terms is parallelized with rayon.
///
/// Arguments:
///     tensors: MPS site tensors in left-to-right order (each tensor's `.data`
///         from iterating a quimb `MatrixProductState`, dtype complex128).
///         Interior tensors must have shape `(D_left, D_right, 2)`; boundary
///         tensors must have shape `(D, 2)`.
///     terms: List of `(pauli_string, coefficient)` pairs where `pauli_string`
///         uses Qiskit's big-endian convention (as from `SparsePauliOp.to_list()`)
///         and has one character per tensor.
///
/// Returns:
///     The accumulated complex sum.
#[pyfunction]
#[pyo3(signature = (tensors, terms))]
pub fn mps_pauli_overlap_sum(
    py: Python<'_>,
    tensors: Vec<PyReadonlyArrayDyn<'_, Complex64>>,
    terms: Vec<(String, Complex64)>,
) -> PyResult<Complex64> {
    let n_sites = tensors.len();
    if n_sites == 0 {
        return Err(PyValueError::new_err("tensors must be non-empty"));
    }

    let mut sites = Vec::with_capacity(n_sites);
    for (idx, t) in tensors.iter().enumerate() {
        let arr = t.as_array();
        sites.push(project_site(&arr, idx == 0, idx == n_sites - 1)?);
    }
    drop(tensors);

    for w in sites.windows(2) {
        if w[0].cols != w[1].rows {
            return Err(PyValueError::new_err(format!(
                "bond dimension mismatch between adjacent tensors: {} != {}",
                w[0].cols, w[1].rows
            )));
        }
    }

    for (pauli_str, _) in &terms {
        if pauli_str.len() != n_sites {
            return Err(PyValueError::new_err(format!(
                "Pauli string {pauli_str:?} has length {}, expected {n_sites} \
                 (one character per tensor)",
                pauli_str.len()
            )));
        }
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .build()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    const CHUNK_SIZE: usize = 10_000;
    let mut total = Complex64::new(0.0, 0.0);
    for chunk in terms.chunks(CHUNK_SIZE) {
        py.check_signals()?;
        total += py.allow_threads(|| {
            pool.install(|| {
                chunk
                    .par_iter()
                    .map(|(pauli_str, coeff)| *coeff * pauli_expectation(&sites, pauli_str))
                    .reduce(|| Complex64::new(0.0, 0.0), |a, b| a + b)
            })
        });
    }

    py.allow_threads(|| drop(pool));

    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Bell state as a 2-site MPS
    fn bell_sites() -> Vec<ProjectedSite> {
        let s = 1.0_f64 / 2.0_f64.sqrt();
        vec![
            ProjectedSite {
                rows: 1,
                cols: 2,
                proj: [vec![c(1.0), c(0.0)], vec![c(0.0), c(1.0)]],
            },
            ProjectedSite {
                rows: 2,
                cols: 1,
                proj: [vec![c(s), c(0.0)], vec![c(0.0), c(s)]],
            },
        ]
    }

    fn c(x: f64) -> Complex64 { Complex64::new(x, 0.0) }

    #[test]
    fn bell_norm_via_identity() {
        let sites = bell_sites();
        let v = pauli_expectation(&sites, "II");
        assert!((v - c(1.0)).norm() < 1e-12, "got {v}");
    }

    #[test]
    fn bell_zz_equals_one() {
        let sites = bell_sites();
        let v = pauli_expectation(&sites, "ZZ");
        assert!((v - c(1.0)).norm() < 1e-12, "got {v}");
    }

    #[test]
    fn bell_xx_equals_one() {
        let sites = bell_sites();
        let v = pauli_expectation(&sites, "XX");
        assert!((v - c(1.0)).norm() < 1e-12, "got {v}");
    }

    #[test]
    fn bell_yy_equals_minus_one() {
        let sites = bell_sites();
        let v = pauli_expectation(&sites, "YY");
        assert!((v - c(-1.0)).norm() < 1e-12, "got {v}");
    }

    #[test]
    fn bell_iz_equals_zero() {
        let sites = bell_sites();
        let v = pauli_expectation(&sites, "IZ");
        assert!(v.norm() < 1e-12, "got {v}");
    }

    #[test]
    fn bell_sum_zz_yy_cancels() {
        let sites = bell_sites();
        let zz = pauli_expectation(&sites, "ZZ");
        let yy = pauli_expectation(&sites, "YY");
        assert!((zz + yy).norm() < 1e-12, "got {}", zz + yy);
    }
}
