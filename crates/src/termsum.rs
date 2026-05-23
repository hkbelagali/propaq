use pyo3::prelude::*;
use pyo3::types::PyDict;
use num_complex::Complex64;
use indexmap::IndexMap;

use crate::monomial::MajoranaMonomial;
use crate::truncation::TruncationPolicy;
use crate::noise::UniformNoiseModel;

#[pyclass(subclass)]
pub struct MajoranaTermSum {
    pub terms: IndexMap<MajoranaMonomial, Complex64>,
}

#[pymethods]
impl MajoranaTermSum {
    #[new]
    #[pyo3(signature = (terms=None))]
    fn new(terms: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut map = IndexMap::new();
        if let Some(dict) = terms {
            for (k, v) in dict.iter() {
                let key: MajoranaMonomial = k.extract()?;
                let val: Complex64 = v.extract()?;
                map.insert(key, val);
            }
        }
        Ok(MajoranaTermSum { terms: map })
    }

    fn add(&mut self, term: MajoranaMonomial, coeff: Complex64) {
        *self.terms.entry(term).or_insert(Complex64::new(0.0, 0.0)) += coeff;
    }

    fn scale(&mut self, factor: Complex64) {
        for val in self.terms.values_mut() {
            *val *= factor;
        }
    }

    fn merge(&mut self, other: &MajoranaTermSum) {
        for (term, coeff) in other.terms.iter() {
            self.add(term.clone(), *coeff);
        }
    }

    pub fn truncate(&mut self, policy: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(tp) = policy.extract::<PyRef<TruncationPolicy>>() {
            let wc = tp.weight_cutoff;
            let cc = tp.coeff_cutoff;
            self.terms.retain(|term, coeff| {
                !(term.compute_weight() > wc || coeff.norm() < cc)
            });
            return Ok(());
        }
        let mut to_remove = Vec::new();
        for (term, coeff) in self.terms.iter() {
            let weight = term.compute_weight();
            let abs_coeff = coeff.norm();
            let should: bool = policy
                .call_method1("should_truncate", (weight, abs_coeff))?
                .extract()?;
            if should {
                to_remove.push(term.clone());
            }
        }
        for key in to_remove {
            self.terms.swap_remove(&key);
        }
        Ok(())
    }

    pub fn apply_damping(&mut self, noise: &Bound<'_, PyAny>, active_modes: u32) -> PyResult<()> {
        if let Ok(unm) = noise.extract::<PyRef<UniformNoiseModel>>() {
            let d = unm.damping;
            for (term, coeff) in self.terms.iter_mut() {
                *coeff *= (-d * term.compute_weight() as f64).exp();
            }
            return Ok(());
        }
        for (term, coeff) in self.terms.iter_mut() {
            let weight = term.compute_weight();
            let damping: f64 = noise
                .call_method1("damping_factor", (weight, active_modes))?
                .extract()?;
            *coeff *= damping;
        }
        Ok(())
    }

    fn norm_squared(&self) -> f64 {
        self.terms.values().map(|c| c.norm_sqr()).sum()
    }

    fn items(&self) -> Vec<(MajoranaMonomial, Complex64)> {
        self.terms.iter().map(|(k, v)| (k.clone(), *v)).collect()
    }

    fn __len__(&self) -> usize {
        self.terms.len()
    }

    fn __setitem__(&mut self, term: MajoranaMonomial, coeff: Complex64) {
        self.terms.insert(term, coeff);
    }

    fn __getitem__(&self, term: &MajoranaMonomial) -> Complex64 {
        self.terms.get(term).copied().unwrap_or(Complex64::new(0.0, 0.0))
    }

    pub fn copy(&self) -> MajoranaTermSum {
        MajoranaTermSum { terms: self.terms.clone() }
    }
}
