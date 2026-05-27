use pyo3::prelude::*;
use num_complex::Complex64;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::truncation::TruncationPolicy;
use crate::noise::UniformNoiseModel;
use crate::traits::AbstractTerm;

// Abstract term sum (not a pyclass — only concrete wrappers are exposed)
pub struct AbstractTermSum<M: AbstractTerm> {
    pub terms: Vec<(M, Complex64)>,
}

impl<M: AbstractTerm> AbstractTermSum<M> {
    pub fn new() -> Self {
        AbstractTermSum { terms: Vec::new() }
    }

    pub fn copy(&self) -> Self {
        AbstractTermSum { terms: self.terms.clone() }
    }

    pub fn add(&mut self, term: M, coeff: Complex64) {
        if let Some((_, c)) = self.terms.iter_mut().find(|(t, _)| t == &term) {
            *c += coeff;
        } else {
            self.terms.push((term, coeff));
        }
    }

    pub fn scale(&mut self, factor: Complex64) {
        for (_, coeff) in self.terms.iter_mut() {
            *coeff *= factor;
        }
    }

    pub fn merge(&mut self, other: &AbstractTermSum<M>) {
        for (term, coeff) in other.terms.iter() {
            self.add(term.clone(), *coeff);
        }
    }

    /// Deduplicate in-place using a parallel fold into FxHashMap.
    /// Term order is not preserved; callers must not rely on ordering after consolidation.
    pub fn consolidate(&mut self) {
        if self.terms.len() <= 1 {
            return;
        }

        let map = std::mem::take(&mut self.terms)
            .into_par_iter()
            .fold(
                || FxHashMap::<M, Complex64>::default(),
                |mut m, (term, coeff)| {
                    *m.entry(term).or_insert(Complex64::new(0.0, 0.0)) += coeff;
                    m
                },
            )
            .reduce(FxHashMap::default, |mut a, b| {
                for (k, v) in b {
                    *a.entry(k).or_insert(Complex64::new(0.0, 0.0)) += v;
                }
                a
            });

        self.terms = map.into_iter().collect();
    }

    pub fn truncate(&mut self, policy: &Bound<'_, PyAny>) -> PyResult<()> {
        self.consolidate();

        if let Ok(tp) = policy.extract::<PyRef<TruncationPolicy>>() {
            let wc = tp.weight_cutoff;
            let cc = tp.coeff_cutoff;
            self.terms = std::mem::take(&mut self.terms)
                .into_par_iter()
                .filter(|(term, coeff)| !(term.weight() > wc || coeff.norm() < cc))
                .collect();
            return Ok(());
        }

        let mut kept = Vec::with_capacity(self.terms.len());
        for (term, coeff) in self.terms.drain(..) {
            let should_remove: bool = policy
                .call_method1("should_truncate", (term.weight(), coeff.norm()))?
                .extract()?;
            if !should_remove {
                kept.push((term, coeff));
            }
        }
        self.terms = kept;
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
        self.terms.iter().map(|(_, c)| c.norm_sqr()).sum()
    }
}
