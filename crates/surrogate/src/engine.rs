///
/// Surrogate build on the partitioned engine.
///
use pyo3::prelude::*;

use propaq_core::basis::Basis;
use propaq_core::coeff::CoeffRepr;
use propaq_core::operator_index::Pos;
use propaq_core::partitioned_termsum::{PartitionedTermSum, PhaseStats};
use propaq_core::progress::Progress;
use propaq_core::strings::BasisString;
use propaq_core::termsum::EmitCutoff;
use propaq_core::truncators::ResolvedConfig;

use crate::model::{SurrogateModel, SurrogateTerm};
use crate::symcoeff::{simplify_sharded, CompiledCoeff, GateParam, SymbolicCoeff};

/// Shards per worker for the coefficient passes.
const SHARD_OVERSUBSCRIPTION: usize = 1;

/// Applies the surrogate's truncation pipeline to the whole operator.
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

/// One gate, already converted out of Python.
pub struct Gate<T> {
    /// Position of the originating qiskit instruction, when there was one.
    pub qiskit_gate_idx: Option<usize>,
    pub generator: T,
    pub param: GateParam,
}

/// Builds a surrogate model by propagating `observable` through `layers`.
#[allow(clippy::too_many_arguments)]
pub fn build<A, P, T, const W: usize>(
    observable: &[(T, f64)],
    layers: &[Vec<Gate<T>>],
    to_basis: impl Fn(&T) -> BasisString<W>,
    n_units: usize,
    partitions: usize,
    inline_positions: usize,
    cfg: &ResolvedConfig,
    initial_state: &[u64],
    n_params: usize,
    progress: Option<&Progress>,
) -> PyResult<(
    SurrogateModel,
    Vec<GateRecord>,
    Vec<MergeRecord>,
    PhaseStats,
)>
where
    A: Basis<W>,
    P: Pos,
{
    let mut op: PartitionedTermSum<SymbolicCoeff, P, W> =
        PartitionedTermSum::with_inline_positions(n_units, partitions, inline_positions);
    for (term, coeff) in observable {
        op.add(
            &to_basis(term),
            <SymbolicCoeff as CoeffRepr>::from_real(*coeff),
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }

    let no_emit_gate = EmitCutoff::none();
    let mut monomials = op.sum_coeffs(|c| c.monomial_count());

    let mut gates_log: Vec<GateRecord> = Vec::new();
    let mut merges: Vec<MergeRecord> = Vec::new();
    let mut gate_idx = 0usize;
    for (layer_idx, layer) in layers.iter().enumerate() {
        for gate in layer {
            let gen = to_basis(&gate.generator);
            let t0 = std::time::Instant::now();
            op.apply_rotation::<A>(&gen, &gate.param, &no_emit_gate)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let ms_per_gate = t0.elapsed().as_secs_f64() * 1e3;
            gate_idx += 1;

            let (before, mono_before) = (op.len(), monomials);
            let t1 = std::time::Instant::now();
            monomials = apply_truncation::<A, P, W>(&mut op, cfg, n_units);
            let merge_elapsed_ms = t1.elapsed().as_secs_f64() * 1e3;
            gates_log.push(GateRecord {
                gate_idx,
                layer_idx,
                qiskit_gate_idx: gate.qiskit_gate_idx,
                terms: op.len(),
                monomials,
                ms_per_gate,
            });
            merges.push(MergeRecord {
                gate_idx,
                layer_idx,
                qiskit_gate_idx: gate.qiskit_gate_idx,
                trigger: "emit",
                terms_before: before,
                terms_after: op.len(),
                monomials_before: mono_before,
                monomials_after: monomials,
                frequency: cfg.frequency,
                weight: cfg.weight,
                coefficient: cfg.coefficient,
                elapsed_ms: merge_elapsed_ms,
            });
            if let Some(p) = progress {
                if gate_idx.is_multiple_of(p.every()) {
                    p.tick_surrogate(p.every(), op.len(), monomials);
                }
            }
        }
    }

    // Gates left over when the circuit does not divide by the tick interval,
    // so the bar still lands on its total.
    if let Some(p) = progress {
        let remainder = gate_idx % p.every();
        if remainder != 0 {
            p.tick_surrogate(remainder, op.len(), monomials);
        }
    }

    let phases = op.phase_stats(inline_positions);
    let (tape, raw) = compile::<A, P, W>(&mut op, n_units, initial_state);
    Ok((
        SurrogateModel::new(raw, tape, n_params),
        gates_log,
        merges,
        phases,
    ))
}

/// Reads a circuit's layers into gates, in application order.
pub fn extract_layers<T>(py: Python<'_>, circuit: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<Gate<T>>>>
where
    T: for<'a, 'py> FromPyObject<'a, 'py>,
{
    let layers: Vec<Vec<Py<PyAny>>> = circuit.getattr("layers")?.extract()?;
    let mut out = Vec::with_capacity(layers.len());
    for layer in layers.iter().rev() {
        let mut gates = Vec::with_capacity(layer.len());
        for rot_obj in layer.iter().rev() {
            let rot = rot_obj.bind(py);
            let generator: T = rot.getattr("generator")?.extract().map_err(Into::into)?;
            let param = <SymbolicCoeff as CoeffRepr>::extract_gate_param(rot)?;
            let qiskit_gate_idx = rot
                .getattr("qiskit_gate_idx")
                .ok()
                .and_then(|v| v.extract::<Option<usize>>().ok())
                .flatten();
            gates.push(Gate {
                qiskit_gate_idx,
                generator,
                param,
            });
        }
        out.push(gates);
    }
    Ok(out)
}

pub const INITIAL_INLINE_POSITIONS: usize = 24;
