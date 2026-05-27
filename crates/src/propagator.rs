use pyo3::prelude::*;
use rayon::prelude::*;
use num_complex::Complex64;
use std::sync::Arc;

use crate::monomial::MajoranaMonomial;
use crate::termsum::MajoranaTermSum;
use crate::noise::UniformNoiseModel;

#[pyclass]
pub struct PropagationResult {
    #[pyo3(get)]
    pub n_terms: Vec<usize>,
    #[pyo3(get)]
    pub expectation_value: f64,
}

#[pymethods]
impl PropagationResult {
    fn __repr__(&self) -> String {
        format!(
            "PropagationResult(expectation_value={}, n_terms=[{} entries])",
            self.expectation_value,
            self.n_terms.len()
        )
    }
}

#[pyclass]
pub struct MajoranaPropagator {
    noise: Option<PyObject>,
    truncation: Option<PyObject>,
    pool: Arc<rayon::ThreadPool>,
    progress_bar: bool,
    #[pyo3(get)]
    truncation_interval: usize,
    flags_buf: Vec<bool>,
}

#[pymethods]
impl MajoranaPropagator {
    #[new]
    #[pyo3(signature = (noise=None, truncation=None, n_threads=None, progress_bar=false, truncation_interval=1))]
    fn new(
        noise: Option<PyObject>,
        truncation: Option<PyObject>,
        n_threads: Option<usize>,
        progress_bar: bool,
        truncation_interval: usize,
    ) -> PyResult<Self> {
        if truncation_interval == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "truncation_interval must be >= 1",
            ));
        }
        let mut builder = rayon::ThreadPoolBuilder::new();
        if let Some(n) = n_threads {
            builder = builder.num_threads(n);
        }
        let pool = Arc::new(
            builder
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        );
        Ok(MajoranaPropagator { noise, truncation, pool, progress_bar, truncation_interval, flags_buf: Vec::new() })
    }

    fn propagate(
        &mut self,
        py: Python<'_>,
        observable: &MajoranaTermSum,
        circuit: &Bound<'_, PyAny>,
    ) -> PyResult<MajoranaTermSum> {
        let layers: Vec<Vec<PyObject>> = circuit.getattr("layers")?.extract()?;
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

        let total_rotations: usize = layers.iter().map(|l| l.len()).sum();
        let (pbar, postfix) = if self.progress_bar {
            let tqdm = py.import("tqdm.auto")?;
            let postfix = pyo3::types::PyDict::new(py);
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("total", total_rotations)?;
            kwargs.set_item("desc", "Propagating through gates")?;
            let pbar = tqdm.call_method("tqdm", (), Some(&kwargs))?;
            (Some(pbar), Some(postfix))
        } else {
            (None, None)
        };

        let mut trunc_counter: usize = 0;
        for layer in layers.iter().rev() {
            for rotation_obj in layer.iter().rev() {
                let rot = rotation_obj.bind(py);
                let generator: MajoranaMonomial = rot.getattr("generator")?.extract()?;
                let angle: f64 = rot.getattr("angle")?.extract()?;
                let is_intermediate: bool = rot.getattr("is_intermediate")?.extract()?;

                py.allow_threads(|| {
                    self.apply_gate_inplace(&mut evolved, &generator, angle)
                });

                if !(is_intermediate && generator.is_number_preserving) {
                    trunc_counter += 1;
                    if trunc_counter % self.truncation_interval == 0 {
                        if let Some(ref trunc_obj) = self.truncation {
                            evolved.truncate(trunc_obj.bind(py))?;
                        }
                    }
                }

                if let (Some(pbar), Some(postfix)) = (&pbar, &postfix) {
                    postfix.set_item("terms", evolved.terms.len())?;
                    pbar.call_method("set_postfix", (), Some(postfix))?;
                    pbar.call_method0("update")?;
                }
            }

            if self.noise.is_some() {
                if let Some(d) = damping {
                    let pool = Arc::clone(&self.pool);
                    py.allow_threads(|| {
                        pool.install(|| {
                            evolved.terms.par_iter_mut().for_each(|(term, coeff)| {
                                *coeff *= (-d * term.weight as f64).exp();
                            });
                        });
                    });
                } else {
                    let noise = self.noise.as_ref().unwrap().bind(py);
                    evolved.apply_damping(noise, 0)?;
                }
            }
        }

        if let Some(pbar) = &pbar {
            pbar.call_method0("close")?;
        }

        if let Some(ref t) = self.truncation {
            evolved.truncate(t.bind(py))?;
        } else {
            py.allow_threads(|| evolved.consolidate());
        }

        Ok(evolved)
    }

    #[pyo3(signature = (observable, circuit, fock_state=0))]
    fn expectation_value(
        &mut self,
        py: Python<'_>,
        observable: &MajoranaTermSum,
        circuit: &Bound<'_, PyAny>,
        fock_state: u64,
    ) -> PyResult<PropagationResult> {
        let layers: Vec<Vec<PyObject>> = circuit.getattr("layers")?.extract()?;
        let mut evolved = observable.copy();
        let mut n_terms: Vec<usize> = Vec::new();

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

        let total_rotations: usize = layers.iter().map(|l| l.len()).sum();
        let (pbar, postfix) = if self.progress_bar {
            let tqdm = py.import("tqdm.auto")?;
            let postfix = pyo3::types::PyDict::new(py);
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("total", total_rotations)?;
            kwargs.set_item("desc", "Propagating through gates")?;
            let pbar = tqdm.call_method("tqdm", (), Some(&kwargs))?;
            (Some(pbar), Some(postfix))
        } else {
            (None, None)
        };

        let mut trunc_counter: usize = 0;
        for layer in layers.iter().rev() {
            for rotation_obj in layer.iter().rev() {
                let rot = rotation_obj.bind(py);
                let generator: MajoranaMonomial = rot.getattr("generator")?.extract()?;
                let angle: f64 = rot.getattr("angle")?.extract()?;
                let is_intermediate: bool = rot.getattr("is_intermediate")?.extract()?;

                py.allow_threads(|| {
                    self.apply_gate_inplace(&mut evolved, &generator, angle)
                });

                if !(is_intermediate && generator.is_number_preserving) {
                    trunc_counter += 1;
                    if trunc_counter % self.truncation_interval == 0 {
                        if let Some(ref trunc_obj) = self.truncation {
                            evolved.truncate(trunc_obj.bind(py))?;
                        }
                    }
                }

                n_terms.push(evolved.terms.len());

                if let (Some(pbar), Some(postfix)) = (&pbar, &postfix) {
                    postfix.set_item("terms", evolved.terms.len())?;
                    pbar.call_method("set_postfix", (), Some(postfix))?;
                    pbar.call_method0("update")?;
                }
            }

            if self.noise.is_some() {
                if let Some(d) = damping {
                    let pool = Arc::clone(&self.pool);
                    py.allow_threads(|| {
                        pool.install(|| {
                            evolved.terms.par_iter_mut().for_each(|(term, coeff)| {
                                *coeff *= (-d * term.weight as f64).exp();
                            });
                        });
                    });
                } else {
                    let noise = self.noise.as_ref().unwrap().bind(py);
                    evolved.apply_damping(noise, 0)?;
                }
            }
        }

        if let Some(pbar) = &pbar {
            pbar.call_method0("close")?;
        }

        if let Some(ref t) = self.truncation {
            evolved.truncate(t.bind(py))?;
        } else {
            py.allow_threads(|| evolved.consolidate());
        }

        let total: Complex64 = evolved
            .terms
            .iter()
            .map(|(term, coeff)| *coeff * term.trace_with_fock_state(fock_state))
            .sum();

        Ok(PropagationResult { n_terms, expectation_value: total.re })
    }
}

impl MajoranaPropagator {
    fn apply_gate_inplace(
        &mut self,
        evolved: &mut MajoranaTermSum,
        generator: &MajoranaMonomial,
        angle: f64,
    ) {
        let cos_t = angle.cos();
        let sin_t = angle.sin();
        let n = evolved.terms.len();

        // Clone the Arc once per call to avoid borrowing self.pool and self.flags_buf
        // through self simultaneously (which the borrow checker disallows).
        let pool = Arc::clone(&self.pool);

        // Classification pass: resize reuses the existing allocation after the first gate.
        self.flags_buf.resize(n, false);
        {
            let flags = &mut self.flags_buf;
            pool.install(|| {
                flags
                    .par_iter_mut()
                    .zip(evolved.terms.par_iter())
                    .for_each(|(f, (term, _))| *f = !term.commutes_with(generator));
            });
        }

        // Collect new sin-terms using original coefficients (before in-place scaling).
        // Allocates O(N_anti) rather than O(N + N_anti) as the old flat_map approach did.
        let new_terms: Vec<(MajoranaMonomial, Complex64)> = {
            let flags = &self.flags_buf;
            pool.install(|| {
                evolved
                    .terms
                    .par_iter()
                    .zip(flags.par_iter())
                    .filter_map(|((term, coeff), &is_anti)| {
                        if !is_anti {
                            return None;
                        }
                        let (phase, new_term) = generator.matmul_internal(term);
                        Some((new_term, *coeff * Complex64::new(0.0, sin_t) * phase))
                    })
                    .collect()
            })
        };

        // Scale anticommuting terms in place by cos_t.
        {
            let flags = &self.flags_buf;
            pool.install(|| {
                evolved
                    .terms
                    .par_iter_mut()
                    .zip(flags.par_iter())
                    .for_each(|((_, coeff), &is_anti)| {
                        if is_anti {
                            *coeff *= cos_t;
                        }
                    });
            });
        }

        // Append; Vec::extend grows geometrically, so amortised O(1) per term.
        evolved.terms.extend(new_terms);
    }
}
