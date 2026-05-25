use pyo3::prelude::*;
use rayon::prelude::*;
use num_complex::Complex64;
use indexmap::IndexMap;
use std::sync::Arc;

use crate::monomial::MajoranaMonomial;
use crate::termsum::MajoranaTermSum;
use crate::noise::UniformNoiseModel;

#[pyclass]
pub struct MajoranaPropagator {
    noise: Option<PyObject>,
    truncation: Option<PyObject>,
    pool: Arc<rayon::ThreadPool>,
    progress_bar: bool,
}

#[pymethods]
impl MajoranaPropagator {
    #[new]
    #[pyo3(signature = (noise=None, truncation=None, n_threads=None, progress_bar=false))]
    fn new(
        noise: Option<PyObject>,
        truncation: Option<PyObject>,
        n_threads: Option<usize>,
        progress_bar: bool,
    ) -> PyResult<Self> {
        let mut builder = rayon::ThreadPoolBuilder::new();
        if let Some(n) = n_threads {
            builder = builder.num_threads(n);
        }
        let pool = Arc::new(
            builder
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        );
        Ok(MajoranaPropagator { noise, truncation, pool, progress_bar })
    }

    fn propagate(
        &self,
        py: Python<'_>,
        observable: &MajoranaTermSum,
        circuit: &Bound<'_, PyAny>,
    ) -> PyResult<MajoranaTermSum> {
        let rotations: Vec<PyObject> = circuit.getattr("rotations")?.extract()?;
        let mut evolved = observable.copy();

        let damping: Option<f64> = if let Some(ref noise_obj) = self.noise {
            let noise = noise_obj.bind(py);
            if let Ok(unm) = noise.extract::<PyRef<UniformNoiseModel>>() {
                Some(unm.damping)
            } else {
                None
            }
        } else {
            None
        };

        let (pbar, postfix) = if self.progress_bar {
            let tqdm = py.import("tqdm.auto")?;
            let postfix = pyo3::types::PyDict::new(py);
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("total", rotations.len())?;
            kwargs.set_item("desc", "Propagating through gates")?;
            let pbar = tqdm.call_method("tqdm", (), Some(&kwargs))?;
            (Some(pbar), Some(postfix))
        } else {
            (None, None)
        };
        

        for rotation_obj in rotations.iter().rev() {
            let rot = rotation_obj.bind(py);
            let generator: MajoranaMonomial = rot.getattr("generator")?.extract()?;
            let angle: f64 = rot.getattr("angle")?.extract()?;

            // Release GIL for the gate application
            evolved = py.allow_threads(|| {
                self.apply_gate(&evolved, &generator, angle, damping)
            });

            // Reacquire GIL for non-uniform noise, truncation, and progress bar
            if self.noise.is_some() && damping.is_none() {
                let noise = self.noise.as_ref().unwrap().bind(py);
                evolved.apply_damping(noise, generator.compute_weight())?;
            }

            if generator.is_number_preserving {
                if let Some(ref trunc_obj) = self.truncation {
                    evolved.truncate(trunc_obj.bind(py))?;
                }
            }

            if let (Some(pbar), Some(postfix)) = (&pbar, &postfix) {
                postfix.set_item("terms", evolved.terms.len())?;
                pbar.call_method("set_postfix", (), Some(postfix))?;
                pbar.call_method0("update")?;
            }
            
        }

        if let Some(pbar) = &pbar {
            pbar.call_method0("close")?;
        }

        if let Some(ref t) = self.truncation {
            evolved.truncate(t.bind(py))?;
        }

        Ok(evolved)
    }

    #[pyo3(signature = (observable, circuit, fock_state=0))]
    fn expectation_value(
        &self,
        py: Python<'_>,
        observable: &MajoranaTermSum,
        circuit: &Bound<'_, PyAny>,
        fock_state: u64,
    ) -> PyResult<f64> {
        let evolved = self.propagate(py, observable, circuit)?;
        let total: Complex64 = evolved
            .terms
            .iter()
            .map(|(term, coeff)| coeff * term.trace_with_fock_state(fock_state))
            .sum();
        Ok(total.re)
    }
}

impl MajoranaPropagator {
    fn apply_gate(
        &self,
        terms: &MajoranaTermSum,
        generator: &MajoranaMonomial,
        angle: f64,
        damping: Option<f64>,
    ) -> MajoranaTermSum {
        let cos_t = angle.cos();
        let sin_t = angle.sin();

        let pairs: Vec<(MajoranaMonomial, Complex64)> = self.pool.install(|| {
            terms
                .terms
                .par_iter()
                .flat_map(|(term, coeff)| {
                    if term.commutes_with(generator) {
                        vec![(term.clone(), *coeff)]
                    } else {
                        let (phase, new_term) = generator.matmul_internal(term);
                        vec![
                            (term.clone(), *coeff * cos_t),
                            (new_term, *coeff * Complex64::new(0.0, sin_t) * phase),
                        ]
                    }
                })
                .collect()
        });

        let mut result_map: IndexMap<MajoranaMonomial, Complex64> = IndexMap::new();
        for (term, coeff) in pairs {
            *result_map.entry(term).or_insert(Complex64::new(0.0, 0.0)) += coeff;
        }
        let mut result = MajoranaTermSum { terms: result_map };

        // Apply uniform noise if damping was pre-extracted
        if let Some(d) = damping {
            for (term, coeff) in result.terms.iter_mut() {
                *coeff *= (-d * term.compute_weight() as f64).exp();
            }
        }

        result
    }
}
