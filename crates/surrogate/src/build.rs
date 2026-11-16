//!
//! Surrogate build on the core run loop.
//!
//! The loop is `propaq_core::run`. This file adds a hook for it. The hook
//! truncates the symbolic coefficients after each gate. It compiles them
//! after the last gate. The build works for any basis `A: Basis<W>` with
//! `BRANCHES == 2`.
//!
use std::sync::Mutex;

use pyo3::prelude::*;

use propaq_core::basis::Basis;
use propaq_core::operator_index::Pos;
use propaq_core::partitioned_termsum::{PartitionedTermSum, PhaseStats};
use propaq_core::progress::Progress;
use propaq_core::run::{hooked_runner_for, BasisAll, GateCtx, GateHook, RunGate, RunInputs};
use propaq_core::strings::BasisString;
use propaq_core::termsum::EmitCutoff;
use propaq_core::truncators::ResolvedConfig;
use propaq_core::CoeffRepr;

use crate::model::{SurrogateModel, SurrogateTerm};
use crate::symcoeff::{simplify_sharded, CompiledCoeff, GateParam, SymbolicCoeff};

/// Shards per worker for the coefficient passes.
const SHARD_OVERSUBSCRIPTION: usize = 1;

/// One truncation pass, recorded for the verbose log.
pub struct MergeRecord {
    pub gate_idx: usize,
    pub layer_idx: usize,
    pub qiskit_gate_idx: Option<usize>,
    pub trigger: &'static str,
    pub terms_before: usize,
    pub terms_after: usize,
    pub monomials_before: u128,
    pub monomials_after: u128,
    pub frequency: Option<usize>,
    pub weight: Option<u32>,
    pub coefficient: Option<f64>,
    pub elapsed_ms: f64,
}

/// One gate application, recorded for the verbose log.
pub struct GateRecord {
    pub gate_idx: usize,
    pub layer_idx: usize,
    pub qiskit_gate_idx: Option<usize>,
    pub terms: usize,
    pub monomials: u128,
    pub ms_per_gate: f64,
}

/// Applies the surrogate's truncation pipeline to the whole operator.
/// Returns the monomial count after the pass.
pub fn apply_truncation<A, P, const W: usize>(
    op: &mut PartitionedTermSum<SymbolicCoeff, P, W>,
    cfg: &ResolvedConfig,
    n_units: usize,
) -> u128
where
    A: Basis<W>,
    P: Pos,
{
    let total_before = op.len();
    let apply_lossy = total_before >= cfg.min_terms.unwrap_or(0);

    if cfg.simplify {
        let n_shards = (rayon::current_num_threads() * SHARD_OVERSUBSCRIPTION).max(1);
        op.with_coeffs_mut(|coeffs| simplify_sharded(coeffs, n_shards));
    }

    if apply_lossy {
        let max_frequency = cfg.frequency.map(|f| f.min(u32::MAX as usize) as u32);
        let coefficient = cfg.coefficient;
        op.with_coeffs_mut(|coeffs| {
            for c in coeffs.iter_mut() {
                c.prune(max_frequency, coefficient);
            }
        });
        let weight = cfg.weight;
        let _ = op.retain::<A>(|key: &BasisString<W>, c: &SymbolicCoeff| {
            let weight_ok = weight.is_none_or(|w| A::weight(key, n_units) <= w);
            weight_ok && !c.is_empty()
        });
    }

    op.sum_coeffs(|c| c.monomial_count())
}

/// Compiles the surviving terms into one shared tape.
pub fn compile<A, P, const W: usize>(
    op: &mut PartitionedTermSum<SymbolicCoeff, P, W>,
    n_units: usize,
    initial_state: &[u64],
) -> (CompiledCoeff, Vec<SurrogateTerm>)
where
    A: Basis<W>,
    P: Pos,
{
    let shards: Vec<(CompiledCoeff, Vec<(f64, usize)>)> = op.map_partitions(|part, frame| {
        let mut overlaps: Vec<f64> = Vec::new();
        let mut survivors: Vec<SymbolicCoeff> = Vec::new();
        part.for_each_term_mut(|key, c| {
            let (image, sign) = frame.conjugate::<A>(&key);
            let overlap = sign * A::trace(&image, n_units, initial_state);

            let coeff = std::mem::take(c);
            if overlap.abs() > 1e-15 {
                overlaps.push(overlap);
                survivors.push(coeff);
            }
        });
        let (tape, roots) = SymbolicCoeff::compile_batch(survivors);
        (tape, overlaps.into_iter().zip(roots).collect())
    });

    let (shard_tapes, shard_terms): (Vec<_>, Vec<_>) = shards.into_iter().unzip();
    let (tape, offsets) = CompiledCoeff::merge_shards(shard_tapes);
    let mut raw = Vec::with_capacity(shard_terms.iter().map(|s| s.len()).sum());
    for (terms, offset) in shard_terms.into_iter().zip(offsets) {
        for (overlap, local_root) in terms {
            let root = if local_root == usize::MAX {
                usize::MAX
            } else {
                local_root + offset
            };
            raw.push(SurrogateTerm { overlap, root });
        }
    }
    (tape, raw)
}

/// What the hook collects during a run.
#[derive(Default)]
struct HookState {
    monomials: u128,
    gates: Vec<GateRecord>,
    merges: Vec<MergeRecord>,
    compiled: Option<(CompiledCoeff, Vec<SurrogateTerm>)>,
}

/// The surrogate's hook for the core run loop.
pub struct SurrogateHook<'a> {
    cfg: &'a ResolvedConfig,
    n_units: usize,
    initial_state: &'a [u64],
    state: Mutex<HookState>,
}

impl<'a> SurrogateHook<'a> {
    /// `initial_monomials` is the monomial count of the starting operator.
    pub fn new(
        cfg: &'a ResolvedConfig,
        n_units: usize,
        initial_state: &'a [u64],
        initial_monomials: u128,
    ) -> Self {
        SurrogateHook {
            cfg,
            n_units,
            initial_state,
            state: Mutex::new(HookState {
                monomials: initial_monomials,
                ..HookState::default()
            }),
        }
    }
}

impl GateHook<SymbolicCoeff> for SurrogateHook<'_> {
    fn after_gate<A, P, const W: usize>(
        &self,
        op: &mut PartitionedTermSum<SymbolicCoeff, P, W>,
        ctx: &GateCtx,
    ) where
        A: Basis<W>,
        P: Pos,
    {
        let mut st = self.state.lock().expect("surrogate hook state");
        let (before, mono_before) = (op.len(), st.monomials);
        let t1 = std::time::Instant::now();
        let monomials = apply_truncation::<A, P, W>(op, self.cfg, self.n_units);
        let merge_elapsed_ms = t1.elapsed().as_secs_f64() * 1e3;
        st.monomials = monomials;
        st.gates.push(GateRecord {
            gate_idx: ctx.gate_idx,
            layer_idx: ctx.layer_idx,
            qiskit_gate_idx: ctx.qiskit_gate_idx,
            terms: op.len(),
            monomials,
            ms_per_gate: ctx.rotate_ms,
        });
        st.merges.push(MergeRecord {
            gate_idx: ctx.gate_idx,
            layer_idx: ctx.layer_idx,
            qiskit_gate_idx: ctx.qiskit_gate_idx,
            trigger: "emit",
            terms_before: before,
            terms_after: op.len(),
            monomials_before: mono_before,
            monomials_after: monomials,
            frequency: self.cfg.frequency,
            weight: self.cfg.weight,
            coefficient: self.cfg.coefficient,
            elapsed_ms: merge_elapsed_ms,
        });
    }

    fn finish<A, P, const W: usize>(
        &self,
        op: &mut PartitionedTermSum<SymbolicCoeff, P, W>,
        n_units: usize,
    ) where
        A: Basis<W>,
        P: Pos,
    {
        let compiled = compile::<A, P, W>(op, n_units, self.initial_state);
        self.state.lock().expect("surrogate hook state").compiled = Some(compiled);
    }

    fn tick(&self, progress: &Progress, advance: usize, n_terms: usize) {
        let monomials = self.state.lock().expect("surrogate hook state").monomials;
        progress.tick_surrogate(advance, n_terms, monomials);
    }
}

/// The result of one surrogate build.
pub struct BuildOutput {
    pub model: SurrogateModel,
    pub gates: Vec<GateRecord>,
    pub merges: Vec<MergeRecord>,
    pub phases: PhaseStats,
}

/// Why a basis cannot build a surrogate, if it cannot.
pub fn check_supported<A: Basis<1>>() -> PyResult<()> {
    if <A as Basis<1>>::BRANCHES > 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "propaq: surrogate propagation supports bases with 2 branches per rotation; \
             this basis has {}. A symbolic coefficient holds real sums of cos and sin factors. \
             A basis with more branches needs complex trigonometric polynomials with \
             several frequencies, and the basis would have to give that table symbolically.",
            <A as Basis<1>>::BRANCHES
        )));
    }
    Ok(())
}

/// Builds a surrogate model by running `observable` through `layers` on the
/// core loop. `bits` is the key size in bits. It selects the storage width.
/// Returns `Ok(None)` when the key is wider than the dispatch.
#[allow(clippy::too_many_arguments)]
pub fn build<A: BasisAll>(
    observable: &[(Vec<u64>, f64)],
    layers: &[Vec<RunGate<SymbolicCoeff>>],
    n_units: usize,
    bits: usize,
    partitions: usize,
    cfg: &ResolvedConfig,
    initial_state: &[u64],
    n_params: usize,
    progress: Option<&Progress>,
) -> PyResult<Option<BuildOutput>> {
    check_supported::<A>()?;
    let Some(run) = hooked_runner_for::<A, SymbolicCoeff, SurrogateHook<'_>>(bits) else {
        return Ok(None);
    };
    let obs: Vec<(Vec<u64>, SymbolicCoeff)> = observable
        .iter()
        .map(|(w, c)| (w.clone(), <SymbolicCoeff as CoeffRepr>::from_real(*c)))
        .collect();
    let hook = SurrogateHook::new(cfg, n_units, initial_state, obs.len() as u128);
    let cutoff = EmitCutoff::none();
    let out = run(
        &RunInputs {
            observable: &obs,
            layers,
            n_units,
            partitions,
            cutoff: &cutoff,
            noise: None,
            collect_counts: false,
            diag_state: None,
            want_terms: false,
            log_gates: false,
            progress,
        },
        &hook,
    )?;
    let HookState {
        gates,
        merges,
        compiled,
        ..
    } = hook.state.into_inner().expect("surrogate hook state");
    let (tape, raw) = compiled.expect("the run loop calls finish once");
    Ok(Some(BuildOutput {
        model: SurrogateModel::new(raw, tape, n_params),
        gates,
        merges,
        phases: out.phases,
    }))
}

/// A gate parameter, read from a Python rotation object.
pub fn extract_param(rot: &Bound<'_, PyAny>) -> PyResult<GateParam> {
    <SymbolicCoeff as CoeffRepr>::extract_gate_param(rot)
}
