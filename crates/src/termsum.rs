use pyo3::prelude::*;
use pyo3::types::PyDict;
use num_complex::Complex64;
use rayon::prelude::*;

use crate::monomial::MajoranaMonomial;
use crate::truncation::TruncationPolicy;
use crate::noise::UniformNoiseModel;

#[pyclass(subclass)]
pub struct MajoranaTermSum {
    pub terms: Vec<(MajoranaMonomial, Complex64)>,
}

impl MajoranaTermSum {
    /// Deduplicate in-place: parallel sort by monomial then sequential coefficient accumulation.
    pub fn consolidate(&mut self) {
        if self.terms.len() <= 1 {
            return;
        }
        self.terms.par_sort_unstable_by(|(a, _), (b, _)| a.modes.cmp(&b.modes));
        // Merge adjacent equal monomials.
        let mut write = 0usize;
        for read in 1..self.terms.len() {
            if self.terms[read].0 == self.terms[write].0 {
                let coeff = self.terms[read].1;
                self.terms[write].1 += coeff;
            } else {
                write += 1;
                self.terms.swap(write, read);
            }
        }
        self.terms.truncate(write + 1);
    }
}

#[pymethods]
impl MajoranaTermSum {
    #[new]
    #[pyo3(signature = (terms=None))]
    fn new(terms: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut vec = Vec::new();
        if let Some(dict) = terms {
            vec.reserve(dict.len());
            for (k, v) in dict.iter() {
                let key: MajoranaMonomial = k.extract()?;
                let val: Complex64 = v.extract()?;
                vec.push((key, val));
            }
        }
        Ok(MajoranaTermSum { terms: vec })
    }

    fn add(&mut self, term: MajoranaMonomial, coeff: Complex64) {
        if let Some((_, c)) = self.terms.iter_mut().find(|(t, _)| t == &term) {
            *c += coeff;
        } else {
            self.terms.push((term, coeff));
        }
    }

    fn scale(&mut self, factor: Complex64) {
        for (_, coeff) in self.terms.iter_mut() {
            *coeff *= factor;
        }
    }

    fn merge(&mut self, other: &MajoranaTermSum) {
        for (term, coeff) in other.terms.iter() {
            self.add(term.clone(), *coeff);
        }
    }

    /// Consolidate duplicates then remove terms that exceed the truncation policy.
    pub fn truncate(&mut self, policy: &Bound<'_, PyAny>) -> PyResult<()> {
        self.consolidate();

        if let Ok(tp) = policy.extract::<PyRef<TruncationPolicy>>() {
            let wc = tp.weight_cutoff;
            let cc = tp.coeff_cutoff;
            self.terms = std::mem::take(&mut self.terms)
                .into_par_iter()
                .filter(|(term, coeff)| !(term.weight > wc || coeff.norm() < cc))
                .collect();
            return Ok(());
        }

        let mut kept = Vec::with_capacity(self.terms.len());
        for (term, coeff) in self.terms.drain(..) {
            let should_remove: bool = policy
                .call_method1("should_truncate", (term.weight, coeff.norm()))?
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
                *coeff *= (-d * term.weight as f64).exp();
            });
            return Ok(());
        }
        for (term, coeff) in self.terms.iter_mut() {
            let damping: f64 = noise
                .call_method1("damping_factor", (term.weight, active_modes))?
                .extract()?;
            *coeff *= damping;
        }
        Ok(())
    }

    fn norm_squared(&self) -> f64 {
        self.terms.iter().map(|(_, c)| c.norm_sqr()).sum()
    }

    fn items(&self) -> Vec<(MajoranaMonomial, Complex64)> {
        self.terms.clone()
    }

    fn __len__(&self) -> usize {
        self.terms.len()
    }

    fn __setitem__(&mut self, term: MajoranaMonomial, coeff: Complex64) {
        self.terms.retain(|(t, _)| t != &term);
        self.terms.push((term, coeff));
    }

    fn __getitem__(&self, term: &MajoranaMonomial) -> Complex64 {
        self.terms
            .iter()
            .filter(|(t, _)| t == term)
            .map(|(_, c)| *c)
            .sum()
    }

    pub fn copy(&self) -> MajoranaTermSum {
        MajoranaTermSum { terms: self.terms.clone() }
    }

    fn consolidate_py(&mut self) {
        self.consolidate();
    }
}
