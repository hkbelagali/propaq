///
/// Surrogate build on the partitioned engine.
///
/// The surrogate propagates the same operator through the same rotations as a
/// numerical run; what differs is the coefficient. `SymbolicCoeff` is a heap DAG
/// rather than a number, which changes the arithmetic per term but nothing about
/// the keys, so the store, the exchange and the Clifford tableau are all reused
/// unchanged. Only the truncation differs, and it differs in kind: a symbolic
/// coefficient is dropped when its *structure* empties out, not when its
/// magnitude falls, so the emit gate cannot express it and this keeps the
/// post-accumulation pass the previous surrogate had.
///
/// That is why `EmitCutoff` is left empty here. Frequency, monomial budgets and
/// simplification are all properties of an accumulated coefficient, so gating a
/// branch before it is formed would be gating on the wrong thing.
///
use pyo3::prelude::*;

use propaq_core::algebra::Algebra;
use propaq_core::monomial::Monomial;
use propaq_core::operator::EmitCutoff;
use propaq_core::operator_index::Pos;
use propaq_core::partitioned::PartitionedOperator;
use propaq_core::coeff::CoeffRepr;
use propaq_core::truncators::ResolvedConfig;

use crate::model::{SurrogateModel, SurrogateTerm};
use crate::symcoeff::{simplify_sharded, CompiledCoeff, GateParam, SymbolicCoeff};

/// Shards per worker for the coefficient passes.
///
/// A partition is already one shard, so this is only used where the work is
/// split inside a partition.
const SHARD_OVERSUBSCRIPTION: usize = 1;

/// Applies the surrogate's truncation pipeline to the whole operator.
///
/// Order matters and is the previous pipeline's: simplify first, so that collapsing
/// a coefficient's structure can bring it under the cutoffs that follow, then
/// prune, then drop whatever emptied out. Returns the monomial count after.
pub fn apply_truncation<A, P, const W: usize>(
    op: &mut PartitionedOperator<SymbolicCoeff, P, W>,
    cfg: &ResolvedConfig,
    n_units: usize,
    monomials_before: u128,
) -> u128
where
    A: Algebra<W>,
    P: Pos,
{
    let total_before = op.len();
    // Both floors must clear, as in the previous pipeline: each is its own "not
    // warmed up yet" veto, so they are ANDed rather than ORed.
    let apply_lossy = total_before >= cfg.min_terms.unwrap_or(0)
        && monomials_before >= cfg.min_monomials.unwrap_or(0);

    if cfg.simplify {
        let n_shards = (rayon::current_num_threads() * SHARD_OVERSUBSCRIPTION).max(1);
        op.with_coeffs_mut(|coeffs| simplify_sharded(coeffs, n_shards));
    }

    if apply_lossy {
        // Saturating cast: a cap beyond `u32::MAX` is indistinguishable from no
        // cap, but should clamp rather than wrap.
        let max_frequency = cfg.frequency.map(|f| f.min(u32::MAX as usize) as u32);
        let coefficient = cfg.coefficient;
        op.with_coeffs_mut(|coeffs| {
            for c in coeffs.iter_mut() {
                c.prune(max_frequency, coefficient);
            }
        });
        let weight = cfg.weight;
        let _ = op.retain::<A>(|key: &Monomial<W>, c: &SymbolicCoeff| {
            let weight_ok = weight.map_or(true, |w| A::weight(key, n_units) <= w);
            weight_ok && !c.is_empty()
        });
    }

    op.sum_coeffs(|c| c.monomial_count())
}

/// Compiles the surviving terms into one shared tape.
///
/// One shard per partition. Sharding matters for memory rather than speed: a
/// subtree shared across many terms is emitted once per shard instead of once
/// per term, which is what stopped a reported multi-hundred-gigabyte blowup
/// under heavy parameter reuse.
pub fn compile<A, P, const W: usize>(
    op: &mut PartitionedOperator<SymbolicCoeff, P, W>,
    n_units: usize,
    initial_state: &[u64],
) -> (CompiledCoeff, Vec<SurrogateTerm>)
where
    A: Algebra<W>,
    P: Pos,
{
    let shards: Vec<(CompiledCoeff, Vec<(f64, usize)>)> = op.map_partitions(|part, frame| {
        let mut overlaps: Vec<f64> = Vec::new();
        let mut survivors: Vec<SymbolicCoeff> = Vec::new();
        part.for_each_term_mut(|key, c| {
            // Stored keys are pre-conjugation: the engine defers Cliffords into
            // a tableau rather than rewriting rows, so the operator this term
            // actually represents is its image under the frame, with a sign.
            let (image, sign) = frame.conjugate::<A>(&key);
            let overlap = sign * A::trace(&image, n_units, initial_state);
            // Taken even when the row will not survive: a term with no overlap
            // has a coefficient DAG that is pure waste, so this releases it (or
            // the part of it not shared with a survivor) now rather than
            // holding it until the build unwinds.
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
            let root = if local_root == usize::MAX { usize::MAX } else { local_root + offset };
            raw.push(SurrogateTerm { overlap, root });
        }
    }
    (tape, raw)
}

/// One truncation pass, recorded for the verbose log.
///
/// Collected during the build rather than written there: the build runs with
/// the GIL released, and writing needs it back. The events are replayed in
/// order once it returns.
pub struct FlushRecord {
    pub gate_idx: usize,
    pub layer_idx: usize,
    pub trigger: &'static str,
    pub terms_before: usize,
    pub terms_after: usize,
    pub monomials_before: u128,
    pub monomials_after: u128,
}

/// One gate, already converted out of Python.
pub struct Gate<T> {
    pub generator: T,
    pub param: GateParam,
}

/// Builds a surrogate model by propagating `observable` through `layers`.
///
/// The caller supplies the monomial conversion, so this stays basis-agnostic
/// while the two bases keep their own key types at the Python boundary.
#[allow(clippy::too_many_arguments)]
pub fn build<A, P, T, const W: usize>(
    observable: &[(T, f64)],
    layers: &[Vec<Gate<T>>],
    to_mono: impl Fn(&T) -> Monomial<W>,
    n_units: usize,
    partitions: usize,
    inline_positions: usize,
    cfg: &ResolvedConfig,
    initial_state: &[u64],
    n_params: usize,
) -> PyResult<(SurrogateModel, Vec<FlushRecord>)>
where
    A: Algebra<W>,
    P: Pos,
{
    let mut op: PartitionedOperator<SymbolicCoeff, P, W> =
        PartitionedOperator::with_inline_positions(n_units, partitions, inline_positions);
    for (term, coeff) in observable {
        op.add(&to_mono(term), <SymbolicCoeff as CoeffRepr>::from_real(*coeff))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }

    // Empty on purpose: every surrogate cutoff is a property of an accumulated
    // symbolic coefficient, so none of them can decide a branch before it is
    // formed. Truncation happens between gates instead.
    let no_emit_gate = EmitCutoff::none();
    let mut monomials = op.sum_coeffs(|c| c.monomial_count());

    let mut flushes: Vec<FlushRecord> = Vec::new();
    let mut gate_idx = 0usize;
    for (layer_idx, layer) in layers.iter().enumerate() {
        for gate in layer {
            let gen = to_mono(&gate.generator);
            op.apply_rotation::<A>(&gen, &gate.param, &no_emit_gate)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            gate_idx += 1;

            // A budget is a ceiling, so it has to be checked as the operator
            // grows rather than once a layer has finished. Term count wins the
            // logged name when both cross on the same gate; a flush fires
            // either way.
            let terms_trigger = cfg.max_terms.is_some_and(|max| op.len() >= max);
            let monomials_trigger = !terms_trigger
                && cfg.max_monomials.is_some_and(|max| {
                    op.sum_coeffs(|c| c.monomial_count()) >= max
                });
            if terms_trigger || monomials_trigger {
                let trigger = if terms_trigger { "threshold" } else { "monomial_threshold" };
                let (before, mono_before) = (op.len(), monomials);
                monomials = apply_truncation::<A, P, W>(&mut op, cfg, n_units, monomials);
                flushes.push(FlushRecord {
                    gate_idx,
                    layer_idx,
                    trigger,
                    terms_before: before,
                    terms_after: op.len(),
                    monomials_before: mono_before,
                    monomials_after: monomials,
                });
            }
        }
        let (before, mono_before) = (op.len(), monomials);
        monomials = apply_truncation::<A, P, W>(&mut op, cfg, n_units, monomials);
        flushes.push(FlushRecord {
            gate_idx,
            layer_idx,
            trigger: "merge",
            terms_before: before,
            terms_after: op.len(),
            monomials_before: mono_before,
            monomials_after: monomials,
        });
    }

    let (tape, raw) = compile::<A, P, W>(&mut op, n_units, initial_state);
    Ok((SurrogateModel::new(raw, tape, n_params), flushes))
}

/// Reads a circuit's layers into gates, in application order.
///
/// Heisenberg propagation consumes layers in reverse and each layer's rotations
/// in reverse, matching every other engine here.
pub fn extract_layers<T>(py: Python<'_>, circuit: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<Gate<T>>>>
where
    T: for<'py> FromPyObject<'py>,
{
    let layers: Vec<Vec<PyObject>> = circuit.getattr("layers")?.extract()?;
    let mut out = Vec::with_capacity(layers.len());
    for layer in layers.iter().rev() {
        let mut gates = Vec::with_capacity(layer.len());
        for rot_obj in layer.iter().rev() {
            let rot = rot_obj.bind(py);
            let generator: T = rot.getattr("generator")?.extract()?;
            let param = <SymbolicCoeff as CoeffRepr>::extract_gate_param(rot)?;
            gates.push(Gate { generator, param });
        }
        out.push(gates);
    }
    Ok(out)
}

/// Inline row width when nothing bounds a term's support. See the numerical
/// engines for the sweep behind it.
pub const INITIAL_INLINE_POSITIONS: usize = 24;
