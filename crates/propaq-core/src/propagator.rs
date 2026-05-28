use pyo3::prelude::*;
use rayon::prelude::*;
use num_complex::Complex64;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::termsum::AbstractTermSum;
use crate::noise::UniformNoiseModel;
use crate::traits::AbstractTerm;

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

const PARALLEL_THRESHOLD: usize = 512;

// Abstract propagator (not a pyclass — only concrete wrappers are exposed)
pub struct AbstractPropagator<M: AbstractTerm> {
    pub noise: Option<PyObject>,
    pub truncation: Option<PyObject>,
    pub pool: Arc<rayon::ThreadPool>,
    pub progress_bar: bool,
    pub truncation_interval: usize,
    staging: Vec<(M, Complex64)>,
    _marker: PhantomData<M>,
}

impl<M: AbstractTerm> AbstractPropagator<M> {
    pub fn new(
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
        Ok(AbstractPropagator {
            noise,
            truncation,
            pool,
            progress_bar,
            truncation_interval,
            staging: Vec::new(),
            _marker: PhantomData,
        })
    }

    fn apply_gate_inplace(
        &mut self,
        evolved: &mut AbstractTermSum<M>,
        generator: &M,
        angle: f64,
    ) {
        let cos_t = angle.cos();
        let sin_t = angle.sin();

        let mut staging = std::mem::take(&mut self.staging);
        staging.clear();
        // Reserve worst case (every term non-commuting → one new term each).
        staging.reserve(evolved.terms.len());

        if evolved.terms.len() >= PARALLEL_THRESHOLD {
            self.pool.install(|| {
                evolved.terms
                    .par_iter_mut()
                    .filter_map(|(term, coeff)| {
                        if term.commutes_with(generator) {
                            return None;
                        }
                        let (phase, new_term) = generator.matmul_internal(term);
                        let new_coeff = *coeff * Complex64::new(0.0, sin_t) * phase;
                        *coeff *= cos_t;
                        Some((new_term, new_coeff))
                    })
                    .collect_into_vec(&mut staging);
            });
        } else {
            for (term, coeff) in evolved.terms.iter_mut() {
                if term.commutes_with(generator) {
                    continue;
                }
                let (phase, new_term) = generator.matmul_internal(term);
                let new_coeff = *coeff * Complex64::new(0.0, sin_t) * phase;
                *coeff *= cos_t;
                staging.push((new_term, new_coeff));
            }
        }

        // Reserve before merging to avoid HashMap rehashing mid-loop.
        evolved.terms.reserve(staging.len());
        for (k, v) in staging.drain(..) {
            *evolved.terms.entry(k).or_insert(Complex64::new(0.0, 0.0)) += v;
        }

        self.staging = staging;
    }

    pub fn run_propagate(
        &mut self,
        py: Python<'_>,
        evolved: &mut AbstractTermSum<M>,
        circuit: &Bound<'_, PyAny>,
    ) -> PyResult<()>
    where
        M: for<'py> FromPyObject<'py>,
    {
        let layers: Vec<Vec<PyObject>> = circuit.getattr("layers")?.extract()?;

        // Pre-extract all rotation data while holding the GIL once, rather than
        // re-crossing the Python boundary on every gate inside the hot loop.
        let circuit_data: Vec<Vec<(M, f64, bool)>> = layers
            .iter()
            .map(|layer| {
                layer.iter().map(|rot_obj| -> PyResult<(M, f64, bool)> {
                    let rot = rot_obj.bind(py);
                    let generator: M = rot.getattr("generator")?.extract()?;
                    let angle: f64 = rot.getattr("angle")?.extract()?;
                    let is_intermediate: bool = rot.getattr("is_intermediate")?.extract()?;
                    Ok((generator, angle, is_intermediate))
                }).collect::<PyResult<_>>()
            })
            .collect::<PyResult<_>>()?;

        let damping = self.uniform_damping(py);
        let total_rotations: usize = circuit_data.iter().map(|l| l.len()).sum();
        let (pbar, postfix) = self.make_progress_bar(py, total_rotations)?;
        let pool = Arc::clone(&self.pool);

        let mut trunc_counter: usize = 0;
        for layer_data in circuit_data.iter().rev() {
            for (generator, angle, is_intermediate) in layer_data.iter().rev() {
                py.allow_threads(|| self.apply_gate_inplace(evolved, generator, *angle));

                if !(*is_intermediate && generator.is_number_preserving()) {
                    trunc_counter += 1;
                    if trunc_counter % self.truncation_interval == 0 {
                        if let Some(ref trunc_obj) = self.truncation {
                            evolved.truncate(trunc_obj.bind(py))?;
                        }
                    }
                }

                Self::tick_progress_bar(py, &pbar, &postfix, evolved.terms.len())?;
            }

            self.apply_layer_noise(py, &pool, evolved, damping)?;
        }

        Self::close_progress_bar(py, &pbar)?;

        if let Some(ref t) = self.truncation {
            evolved.truncate(t.bind(py))?;
        }

        Ok(())
    }

    pub fn run_expectation_value(
        &mut self,
        py: Python<'_>,
        evolved: &mut AbstractTermSum<M>,
        circuit: &Bound<'_, PyAny>,
        fock_state: u64,
    ) -> PyResult<PropagationResult>
    where
        M: for<'py> FromPyObject<'py>,
    {
        let layers: Vec<Vec<PyObject>> = circuit.getattr("layers")?.extract()?;

        let circuit_data: Vec<Vec<(M, f64, bool)>> = layers
            .iter()
            .map(|layer| {
                layer.iter().map(|rot_obj| -> PyResult<(M, f64, bool)> {
                    let rot = rot_obj.bind(py);
                    let generator: M = rot.getattr("generator")?.extract()?;
                    let angle: f64 = rot.getattr("angle")?.extract()?;
                    let is_intermediate: bool = rot.getattr("is_intermediate")?.extract()?;
                    Ok((generator, angle, is_intermediate))
                }).collect::<PyResult<_>>()
            })
            .collect::<PyResult<_>>()?;

        let mut n_terms: Vec<usize> = Vec::new();
        let damping = self.uniform_damping(py);
        let total_rotations: usize = circuit_data.iter().map(|l| l.len()).sum();
        let (pbar, postfix) = self.make_progress_bar(py, total_rotations)?;
        let pool = Arc::clone(&self.pool);

        let mut trunc_counter: usize = 0;
        for layer_data in circuit_data.iter().rev() {
            for (generator, angle, is_intermediate) in layer_data.iter().rev() {
                py.allow_threads(|| self.apply_gate_inplace(evolved, generator, *angle));

                if !(*is_intermediate && generator.is_number_preserving()) {
                    trunc_counter += 1;
                    if trunc_counter % self.truncation_interval == 0 {
                        if let Some(ref trunc_obj) = self.truncation {
                            evolved.truncate(trunc_obj.bind(py))?;
                        }
                    }
                }

                n_terms.push(evolved.terms.len());
                Self::tick_progress_bar(py, &pbar, &postfix, evolved.terms.len())?;
            }

            self.apply_layer_noise(py, &pool, evolved, damping)?;
        }

        Self::close_progress_bar(py, &pbar)?;

        if let Some(ref t) = self.truncation {
            evolved.truncate(t.bind(py))?;
        }

        let total: Complex64 = evolved
            .terms
            .iter()
            .map(|(term, coeff)| *coeff * term.trace_with_fock_state(fock_state))
            .sum();

        Ok(PropagationResult { n_terms, expectation_value: total.re })
    }

    fn uniform_damping(&self, py: Python<'_>) -> Option<f64> {
        if let Some(ref noise_obj) = self.noise {
            let noise = noise_obj.bind(py);
            if let Ok(unm) = noise.extract::<PyRef<UniformNoiseModel>>() {
                return Some(unm.damping);
            }
        }
        None
    }

    fn make_progress_bar(
        &self,
        py: Python<'_>,
        total: usize,
    ) -> PyResult<(Option<Py<PyAny>>, Option<Py<PyAny>>)> {
        if !self.progress_bar {
            return Ok((None, None));
        }
        let tqdm = py.import("tqdm.auto")?;
        let postfix = pyo3::types::PyDict::new(py);
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("total", total)?;
        kwargs.set_item("desc", "Propagating through gates")?;
        let pbar = tqdm.call_method("tqdm", (), Some(&kwargs))?;
        Ok((Some(pbar.into()), Some(postfix.into())))
    }

    fn tick_progress_bar(
        py: Python<'_>,
        pbar: &Option<Py<PyAny>>,
        postfix: &Option<Py<PyAny>>,
        n_terms: usize,
    ) -> PyResult<()> {
        if let (Some(pbar), Some(postfix)) = (pbar, postfix) {
            let pbar = pbar.bind(py);
            let postfix = postfix.bind(py);
            postfix.set_item("terms", n_terms)?;
            pbar.call_method("set_postfix", (), Some(postfix.downcast()?))?;
            pbar.call_method0("update")?;
        }
        Ok(())
    }

    fn close_progress_bar(py: Python<'_>, pbar: &Option<Py<PyAny>>) -> PyResult<()> {
        if let Some(pbar) = pbar {
            pbar.bind(py).call_method0("close")?;
        }
        Ok(())
    }

    fn apply_layer_noise(
        &self,
        py: Python<'_>,
        pool: &rayon::ThreadPool,
        evolved: &mut AbstractTermSum<M>,
        damping: Option<f64>,
    ) -> PyResult<()> {
        if self.noise.is_none() {
            return Ok(());
        }
        if let Some(d) = damping {
            py.allow_threads(|| {
                pool.install(|| {
                    evolved.terms.par_iter_mut().for_each(|(term, coeff)| {
                        *coeff *= (-d * term.weight() as f64).exp();
                    });
                });
            });
        } else {
            let noise = self.noise.as_ref().unwrap().bind(py);
            evolved.apply_damping(noise, 0)?;
        }
        Ok(())
    }
}
