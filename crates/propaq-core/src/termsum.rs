use pyo3::prelude::*;
use num_complex::Complex64;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::truncation::TruncationPolicy;
use crate::noise::UniformNoiseModel;
use crate::traits::AbstractTerm;

// Abstract term sum (not a pyclass — only concrete wrappers are exposed)
pub struct AbstractTermSum<M: AbstractTerm> {
    pub terms: FxHashMap<M, Complex64>,
}

impl<M: AbstractTerm> AbstractTermSum<M> {
    pub fn new() -> Self {
        AbstractTermSum { terms: FxHashMap::default() }
    }

    pub fn copy(&self) -> Self {
        AbstractTermSum { terms: self.terms.clone() }
    }

    pub fn add(&mut self, term: M, coeff: Complex64) {
        *self.terms.entry(term).or_insert(Complex64::new(0.0, 0.0)) += coeff;
    }

    pub fn scale(&mut self, factor: Complex64) {
        for coeff in self.terms.values_mut() {
            *coeff *= factor;
        }
    }

    pub fn merge(&mut self, other: &AbstractTermSum<M>) {
        for (term, coeff) in other.terms.iter() {
            self.add(term.clone(), *coeff);
        }
    }

    pub fn truncate(&mut self, policy: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(tp) = policy.extract::<PyRef<TruncationPolicy>>() {
            let wc = tp.weight_cutoff;
            let cc = tp.coeff_cutoff;
            self.terms.retain(|term, coeff| !(term.weight() > wc || coeff.norm() < cc));
            return Ok(());
        }

        let mut to_remove: Vec<M> = Vec::new();
        for (term, coeff) in self.terms.iter() {
            let should_remove: bool = policy
                .call_method1("should_truncate", (term.weight(), coeff.norm()))?
                .extract()?;
            if should_remove {
                to_remove.push(term.clone());
            }
        }
        for term in &to_remove {
            self.terms.remove(term);
        }
        Ok(())
    }

    pub fn apply_damping(&mut self, noise: &Bound<'_, PyAny>, active_modes: u32) -> PyResult<()> {
        if let Ok(unm) = noise.extract::<PyRef<UniformNoiseModel>>() {
            let d = unm.damping;
            self.terms.par_iter_mut().for_each(|(term, coeff)| {
                *coeff *= (-d * term.weight() as f64).exp();
            });
            return Ok(());
        }
        for (term, coeff) in self.terms.iter_mut() {
            let damping: f64 = noise
                .call_method1("damping_factor", (term.weight(), active_modes))?
                .extract()?;
            *coeff *= damping;
        }
        Ok(())
    }

    pub fn norm_squared(&self) -> f64 {
        self.terms.values().map(|c| c.norm_sqr()).sum()
    }
}
