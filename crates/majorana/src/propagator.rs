///
/// impl for the Majorana propagator, which works with observables 
/// represented in the Majorana operator basis. The propagator is 
/// just a wrapper around the generic `AbstractPropagator`, incorporating 
/// the Majorana algebra and the Majorana monomial representation.
///
use pyo3::prelude::*;

use propaq_core::propagator::{AbstractPropagator, PropagationResult};
use propaq_core::truncators::{reject_surrogate_only, resolve_truncation, FlushSchedule};

use crate::monomial::MajoranaMonomial;
use crate::termsum::MajoranaTermSum;

/// Back-propagates Majorana observables through quantum circuits in the Heisenberg picture.
///
/// Arguments:
///     noise: Optional noise model (UniformNoiseModel, GateNoiseModel, or custom).
///     truncation: A list of truncators
///         (WeightTruncator, CoefficientTruncator, TermBudget), a single such
///         truncator, a legacy TruncationPolicy (decomposed), or None. The
///         symbolic-only FrequencyTruncator/MonomialBudget are rejected.
///     schedule: Optional FlushSchedule controlling the lossless merge cadence.
///     n_threads: Number of worker threads. Defaults to the system thread count.
///     progress_bar: Display a tqdm progress bar during propagation.
///     logger: Optional Logger for verbose JSON Lines event logging.
#[pyclass(module = "propaq._rust_core")]
pub struct MajoranaPropagator {
    inner: AbstractPropagator<MajoranaMonomial, f64>,
}

#[pymethods]
impl MajoranaPropagator {
    /// Initialize the Majorana propagator. See the class docstring for arguments.
    #[new]
    #[pyo3(signature = (noise=None, truncation=None, schedule=None, n_threads=None, progress_bar=false, logger=None))]
    fn new(
        noise: Option<PyObject>,
        truncation: Option<Bound<'_, PyAny>>,
        schedule: Option<FlushSchedule>,
        n_threads: Option<usize>,
        progress_bar: bool,
        logger: Option<PyObject>,
    ) -> PyResult<Self> {
        let (schedule, truncators) = resolve_truncation(truncation.as_ref(), schedule)?;
        reject_surrogate_only(&truncators)?;
        Ok(MajoranaPropagator {
            inner: AbstractPropagator::new(noise, schedule, truncators, n_threads, progress_bar, logger)?,
        })
    }

    /// Back-propagate *circuit* through *observable*, returning the evolved term sum.
    ///
    /// Arguments:
    ///     observable: The Majorana observable to back-propagate.
    ///     circuit: A MajoranaCircuit whose rotations are applied in reverse.
    ///     filename: If given, save the final terms to a gzip-compressed binary file at this path.
    #[pyo3(signature = (observable, circuit, filename=None))]
    fn propagate(
        &mut self,
        py: Python<'_>,
        observable: &MajoranaTermSum,
        circuit: &Bound<'_, PyAny>,
        filename: Option<String>,
    ) -> PyResult<MajoranaTermSum> {
        let mut evolved = observable.inner.copy();
        self.inner.run_propagate(py, &mut evolved, circuit, filename.as_deref())?;
        Ok(MajoranaTermSum { inner: evolved })
    }

    /// Compute the expectation value of *observable* in the state prepared by *circuit*.
    ///
    /// Arguments:
    ///     observable: The Majorana observable.
    ///     circuit: A MajoranaCircuit applied to the reference state.
    ///     initial_state: Fock state as a bitstring integer.
    ///     filename: If given, save the final terms to a gzip-compressed binary file at this path.
    #[pyo3(signature = (observable, circuit, initial_state=0, filename=None))]
    fn expectation_value(
        &mut self,
        py: Python<'_>,
        observable: &MajoranaTermSum,
        circuit: &Bound<'_, PyAny>,
        initial_state: u64,
        filename: Option<String>,
    ) -> PyResult<PropagationResult> {
        let mut evolved = observable.inner.copy();
        self.inner.run_expectation_value(py, &mut evolved, circuit, initial_state, filename.as_deref())
    }

    #[getter]
    fn noise(&self, py: Python<'_>) -> Option<PyObject> {
        self.inner.noise.as_ref().map(|n| n.clone_ref(py))
    }

    #[pyo3(signature = (noise=None))]
    fn set_noise(&mut self, noise: Option<PyObject>) {
        self.inner.noise = noise;
    }

    /// The active truncation pipeline as a list of truncator objects.
    #[getter]
    fn truncators(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.inner.truncators.iter().map(|t| t.to_object(py)).collect()
    }

    /// The flush/merge schedule.
    #[getter]
    fn schedule(&self) -> FlushSchedule {
        self.inner.schedule.clone()
    }

    #[setter]
    fn set_schedule(&mut self, schedule: FlushSchedule) {
        self.inner.schedule = schedule;
    }

    /// Replace the truncation pipeline (accepts the same forms as the
    /// constructor's `truncation`); the current schedule is preserved.
    #[pyo3(signature = (truncation=None))]
    fn set_truncation(&mut self, truncation: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let (schedule, truncators) =
            resolve_truncation(truncation.as_ref(), Some(self.inner.schedule.clone()))?;
        reject_surrogate_only(&truncators)?;
        self.inner.schedule = schedule;
        self.inner.truncators = truncators;
        Ok(())
    }
}

#[cfg(test)]
mod gate_batch_integration_tests {
    //! Exercises the *real* `apply_gate_inplace` (real rayon parallelism —
    //! `par_iter_mut`/`fold`/`with_min_len`, real partition hashing, real
    //! outbox flush) with `M = MajoranaMonomial`, whose `SUPPORTS_BATCHING`
    //! is unconditionally `true` — so every call here goes through the
    //! `GateBatch` path, not just the per-item scalar path unit-tested in
    //! `monomial.rs`. The oracle is a small hand-computed reference built
    //! directly from the validated scalar primitives (`commutes_with`,
    //! `matmul_internal`, `f64::apply_rotation`'s documented formula) —
    //! independent of both `apply_gate_inplace` and `matmul_batch`.
    use super::*;
    use propaq_core::bitset::Bitset;
    use propaq_core::termsum::AbstractTermSum;
    use std::collections::HashMap;

    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn next_f64_signed(&mut self) -> f64 {
            let unit = (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64); // [0, 1)
            unit * 2.0 - 1.0 // [-1, 1)
        }
    }

    fn random_mon(rng: &mut Rng, n_modes: usize) -> MajoranaMonomial {
        let n_words = (n_modes + 63) / 64;
        let mut words: Vec<u64> = (0..n_words).map(|_| rng.next_u64()).collect();
        let rem = n_modes % 64;
        if rem != 0 {
            *words.last_mut().unwrap() &= (1u64 << rem) - 1;
        }
        let modes = Bitset::from_words(words);
        let (weight, p, is_np) = MajoranaMonomial::weight_and_p_for(&modes, n_modes);
        MajoranaMonomial { modes, n_modes, is_number_preserving: is_np, weight, p }
    }

    /// N_MODES=32 (16 qubits) keeps `modes` a single word for this test, so
    /// "first word of `modes`" is a safe, unique dedup/comparison key —
    /// multi-word correctness is already covered exhaustively at the
    /// `matmul_batch` unit level in `monomial.rs`; this test's job is to
    /// validate the *propagator plumbing* (real parallelism, real GateBatch
    /// dispatch, real flush/merge), not re-cover multi-word arithmetic.
    fn key_of(m: &MajoranaMonomial) -> u64 {
        m.modes.as_words().first().copied().unwrap_or(0)
    }

    #[test]
    fn apply_gate_inplace_batched_matches_hand_computed_reference() {
        const N_MODES: usize = 32;
        // Comfortably straddles both `gate_par_min_len()` (default 256) and
        // `GATE_BATCH_SIZE` (64) across `n_threads=4` partitions (avg. ~500
        // terms/partition), so this exercises the big-parallel-partition
        // branch of `apply_gate_inplace`'s batched path, not just the small
        // serial-partition fallback.
        const N_TERMS: usize = 2000;
        let mut rng = Rng(0xA11CE_B0B_1234_5678);

        let mut seed_map: HashMap<u64, (MajoranaMonomial, f64)> = HashMap::new();
        while seed_map.len() < N_TERMS {
            let term = random_mon(&mut rng, N_MODES);
            let coeff = rng.next_f64_signed();
            seed_map.entry(key_of(&term)).or_insert((term, coeff));
        }

        let generator = random_mon(&mut rng, N_MODES);
        let angle: f64 = 0.4123;

        // --- Reference: hand-computed directly from the validated scalar
        // primitives, independent of `apply_gate_inplace`/`matmul_batch`. ---
        let (sin_t, cos_t) = angle.sin_cos();
        let mut reference: HashMap<u64, f64> = HashMap::new();
        for (term, coeff) in seed_map.values() {
            if term.commutes_with(&generator) {
                *reference.entry(key_of(term)).or_insert(0.0) += coeff;
                continue;
            }
            let (phase, new_term) = generator.matmul_internal(term);
            let cos_branch = coeff * cos_t;
            let sin_branch = coeff * sin_t * (-phase.im);
            *reference.entry(key_of(term)).or_insert(0.0) += cos_branch;
            *reference.entry(key_of(&new_term)).or_insert(0.0) += sin_branch;
        }

        // --- Real propagator: real rayon parallelism, real GateBatch dispatch. ---
        let mut prop: AbstractPropagator<MajoranaMonomial, f64> =
            AbstractPropagator::new(None, FlushSchedule::none(), Vec::new(), Some(4), false, None)
                .expect("propagator construction");
        let mut seed = AbstractTermSum::new();
        for (term, coeff) in seed_map.values() {
            seed.add(term.clone(), *coeff);
        }
        prop.initialize_from(&seed);
        prop.apply_gate_inplace(&generator, angle);
        prop.flush_outboxes_to_maps();
        let results: Vec<(MajoranaMonomial, f64)> = prop.drain_collect_terms(|t, c| Some((t, c)));

        let mut actual: HashMap<u64, f64> = HashMap::new();
        for (term, coeff) in &results {
            *actual.entry(key_of(term)).or_insert(0.0) += coeff;
        }

        assert_eq!(
            actual.len(),
            reference.len(),
            "live term count mismatch: propagator={}, reference={}",
            actual.len(),
            reference.len()
        );
        for (key, &ref_coeff) in &reference {
            let actual_coeff = actual
                .get(key)
                .unwrap_or_else(|| panic!("propagator is missing term key {key:#x} present in reference"));
            assert!(
                (actual_coeff - ref_coeff).abs() < 1e-9,
                "coefficient mismatch for term {key:#x}: propagator={actual_coeff}, reference={ref_coeff}"
            );
        }
    }
}
