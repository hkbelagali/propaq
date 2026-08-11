//!
//! Implement the propagator for the Majorana algebra, taking the generic
//! architecture over the hash-partitioned term storage and injecting the
//! Majorana-specific algebraic operations.
//!

use pyo3::prelude::*;

use propaq_core::basis::BasisKind;
use propaq_core::coeff::CoeffRepr;
use propaq_core::noise_resolver::{
    resolve_noise, retabulate, NoiseTable, ResolvedNoise, PYTHON_TERM_HOOK,
};
use propaq_core::operator_index::OperatorIndex;
use propaq_core::partitioned_termsum::{PartitionedTermSum, PhaseStats};
use propaq_core::progress::Progress;
use propaq_core::results::PropagationResult;
use propaq_core::strings::BasisString;
use propaq_core::term_kernel::LayerContext;
use propaq_core::termsum::EmitCutoff;
use propaq_core::truncators::ResolvedConfig;

use crate::algebra::{from_basis_string, to_basis_string, MajoranaAlgebra};
use crate::monomial::MajoranaMonomial;

/// Widest site count the dispatch covers, so twice this many Majorana modes.
pub const MAX_DISPATCH_SITES: usize = 2048;

/// Inline row width the store starts at when nothing bounds a term's support.
const INITIAL_INLINE_POSITIONS: usize = 24;

/// Share of overflowed rows that triggers a repack to a wider row.
const OVERFLOW_REPACK_THRESHOLD: f64 = 0.20;

/// One gate of a circuit, already converted out of Python.
struct Gate<C: CoeffRepr> {
    /// Position of the originating qiskit instruction, when there was one.
    qiskit_gate_idx: Option<usize>,
    generator: MajoranaMonomial,
    angle: C::GateParam,
}

/// Reads a circuit's layers into a flat gate list in application order.
fn extract_layers<C: CoeffRepr>(
    py: Python<'_>,
    circuit: &Bound<'_, PyAny>,
) -> PyResult<Vec<Vec<Gate<C>>>> {
    let layers: Vec<Vec<Py<PyAny>>> = circuit.getattr("layers")?.extract()?;
    let mut out = Vec::with_capacity(layers.len());
    for layer in layers.iter().rev() {
        let mut gates = Vec::with_capacity(layer.len());
        for rot_obj in layer.iter().rev() {
            let rot = rot_obj.bind(py);
            let generator: MajoranaMonomial = rot.getattr("generator")?.extract()?;
            let angle = C::extract_gate_param(rot)?;
            let qiskit_gate_idx = rot
                .getattr("qiskit_gate_idx")
                .ok()
                .and_then(|v| v.extract::<Option<usize>>().ok())
                .flatten();
            gates.push(Gate {
                generator,
                angle,
                qiskit_gate_idx,
            });
        }
        out.push(gates);
    }
    Ok(out)
}

/// Runs one circuit at a fixed storage width. `n_sites` is modes / 2.
#[allow(clippy::too_many_arguments)]
fn run_at_width<C, const W: usize, P>(
    observable: &[(MajoranaMonomial, f64)],
    layers: &[Vec<Gate<C>>],
    n_sites: usize,
    partitions: usize,
    cutoff: &EmitCutoff,
    noise: Option<&ResolvedNoise>,
    collect_counts: bool,
    fock: Option<&[u64]>,
    want_terms: bool,
    log_gates: bool,
    progress: Option<&Progress>,
) -> PyResult<RunOutput<C>>
where
    C: CoeffRepr,
    P: propaq_core::operator_index::Pos,
{

    let mut cutoff = cutoff.clone();
    let n_layers = layers.len() as u32;
    let inline_positions = match cutoff.max_weight {
        Some(w) => OperatorIndex::<P, W>::inline_width_for_support_cutoff(w as usize),
        None => INITIAL_INLINE_POSITIONS.min(2 * n_sites),
    };
    let adaptive_width = cutoff.max_weight.is_none();
    let mut op: PartitionedTermSum<C, P, W> =
        PartitionedTermSum::with_inline_positions(n_sites, partitions, inline_positions);
    if cutoff.depends_on_key() || noise.is_some_and(ResolvedNoise::depends_on_key) {
        op.set_defer_cliffords(false);
    }
    for (term, coeff) in observable {
        let (key, c) = (to_basis_string::<W>(term), C::from_real(*coeff));
        if !cutoff.admits_initial::<MajoranaAlgebra, C, W>(&key, &c, n_sites) {
            continue;
        }
        op.add(&key, c)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }

    let mut n_terms = Vec::new();
    let mut gate_records: Vec<GateRecord> = Vec::new();
    let mut gate_idx = 0usize;
    let mut layer_table: NoiseTable = Vec::new();
    for (layer_idx, layer) in layers.iter().enumerate() {
        cutoff.layer = LayerContext::new(layer_idx as u32, n_layers);
        if let Some(noise) = noise {
            match noise {
                ResolvedNoise::WeightTable(table) => {
                    op.scale_by_weight::<MajoranaAlgebra>(|w| {
                        table[(w as usize).min(table.len() - 1)]
                    });
                }
                ResolvedNoise::LayeredWeightTable(kernel) => {

                    retabulate(
                        kernel.as_ref(),
                        BasisKind::Majorana,
                        n_sites,
                        cutoff.layer,
                        &mut layer_table,
                    );
                    op.scale_by_weight::<MajoranaAlgebra>(|w| {
                        layer_table[(w as usize).min(layer_table.len() - 1)]
                    });
                }
                ResolvedNoise::TermKernel(kernel) => {
                    op.scale_by_key::<MajoranaAlgebra>(kernel.as_ref(), cutoff.layer);
                }

                ResolvedNoise::PythonTerm(model) => Python::attach(|py| -> PyResult<()> {
                    let model = model.bind(py);
                    op.try_scale_by_key::<MajoranaAlgebra, PyErr>(|key, weight| {
                        model
                            .call_method1(
                                PYTHON_TERM_HOOK,
                                (
                                    BasisKind::Majorana.as_u32(),
                                    key.words().to_vec(),
                                    n_sites,
                                    weight,
                                ),
                            )?
                            .extract()
                    })
                })?,
            }
            op.reclaim::<MajoranaAlgebra>(&cutoff)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }
        if adaptive_width {
            op.repack_if_overflowing(OVERFLOW_REPACK_THRESHOLD);
        }
        for gate in layer {
            let gen: BasisString<W> = to_basis_string(&gate.generator);
            let (before, declined_before, t0) = if log_gates {
                (
                    op.len(),
                    op.scan_counts().1,
                    Some(std::time::Instant::now()),
                )
            } else {
                (0, 0, None)
            };
            op.apply_rotation::<MajoranaAlgebra>(&gen, &gate.angle, &cutoff)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            if let Some(t0) = t0 {
                gate_records.push(GateRecord {
                    gate_idx,
                    layer_idx,
                    qiskit_gate_idx: gate.qiskit_gate_idx,
                    terms_before: before,
                    terms_after: op.len(),
                    declined: op.scan_counts().1 - declined_before,
                    elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
                });
            }
            gate_idx += 1;
            if collect_counts {
                n_terms.push(op.len());
            }
            if let Some(p) = progress {
                if gate_idx.is_multiple_of(p.every()) {
                    p.tick(p.every(), op.len());
                }
            }
        }
    }

    if let Some(p) = progress {
        let remainder = gate_idx % p.every();
        if remainder != 0 {
            p.tick(remainder, op.len());
        }
    }

    let phases = op.phase_stats(inline_positions);

    let terms = want_terms.then(|| {
        op.iter::<MajoranaAlgebra>()
            .map(|(key, sign, c)| {
                let mut coeff = c.clone();
                coeff.scale_real(sign);
                (from_basis_string::<W>(&key, 2 * n_sites), coeff)
            })
            .collect::<Vec<_>>()
    });
    let expectation_value = fock.map_or(0.0, |f| op.expectation::<MajoranaAlgebra>(f));
    let terms_below_cutoff = cutoff.min_coeff.map_or(0, |c| op.terms_below(c));
    Ok(RunOutput {
        gates: gate_records,
        phases,
        result: PropagationResult {
            n_terms,
            expectation_value,
            sparse_key_bytes: op.key_bytes(),
            terms_below_cutoff,
        },
        terms,
    })
}

/// Dispatches on site count to a monomorphized width, then runs the circuit.
macro_rules! dispatch_width {
    ($c:ty, $n_sites:expr, $($limit:expr => ($w:expr, $pos:ty)),+ $(,)?) => {{
        let n = $n_sites;
        $(if n <= $limit {
            return run_at_width::<$c, $w, $pos>;
        })+
        unreachable!("caller must check MAX_DISPATCH_SITES first")
    }};
}

/// One gate, as the verbose log sees it.
pub struct GateRecord {
    pub gate_idx: usize,
    pub layer_idx: usize,
    pub qiskit_gate_idx: Option<usize>,
    pub terms_before: usize,
    pub terms_after: usize,
    pub declined: u64,
    pub elapsed_ms: f64,
}

pub struct RunOutput<C> {
    pub result: PropagationResult,
    pub gates: Vec<GateRecord>,
    pub terms: Option<Vec<(MajoranaMonomial, C)>>,
    /// Phase timings and kernel counters for the run's log.
    pub phases: PhaseStats,
}

type Runner<C> = fn(
    &[(MajoranaMonomial, f64)],
    &[Vec<Gate<C>>],
    usize,
    usize,
    &EmitCutoff,
    Option<&ResolvedNoise>,
    bool,
    Option<&[u64]>,
    bool,
    bool,
    Option<&Progress>,
) -> PyResult<RunOutput<C>>;

fn runner_for<C: CoeffRepr>(n_sites: usize) -> Runner<C> {
    dispatch_width!(
        C,
        n_sites,
        32 => (1, u8),
        64 => (2, u8),
        128 => (4, u16),
        256 => (8, u16),
        512 => (16, u16),
        1024 => (32, u16),
        2048 => (64, u16),
    )
}

/// Runs `observable` through `circuit` on the partitioned engine.
#[allow(clippy::too_many_arguments)]
pub fn run<C: CoeffRepr>(
    py: Python<'_>,
    observable: &[(MajoranaMonomial, f64)],
    circuit: &Bound<'_, PyAny>,
    fock: Option<&[u64]>,
    n_modes: usize,
    cfg: &ResolvedConfig,
    pool: &rayon::ThreadPool,
    n_threads: Option<usize>,
    noise: Option<&Bound<'_, PyAny>>,
    collect_counts: bool,
    want_terms: bool,
    log_gates: bool,
    progress_bar: bool,
    progress_every: usize,
) -> PyResult<Option<RunOutput<C>>> {
    if !n_modes.is_multiple_of(2) {
        return Ok(None);
    }
    let n_sites = n_modes / 2;
    if n_sites > MAX_DISPATCH_SITES {
        return Ok(None);
    }
    let layers = extract_layers::<C>(py, circuit)?;
    let cutoff = EmitCutoff::from(cfg);
    let partitions = n_threads.unwrap_or_else(rayon::current_num_threads).max(1);
    let noise = resolve_noise(noise, n_sites, BasisKind::Majorana)?;
    let run = runner_for::<C>(n_sites);

    let total_gates = layers.iter().map(Vec::len).sum();
    let progress = Progress::new(py, progress_bar, total_gates, progress_every)?;

    let out = py.detach(|| {
        pool.install(|| {
            run(
                observable,
                &layers,
                n_sites,
                partitions,
                &cutoff,
                noise.as_ref(),
                collect_counts,
                fock,
                want_terms,
                log_gates,
                progress.as_ref(),
            )
        })
    });

    if let Some(p) = progress.as_ref() {
        p.close();
    }
    Ok(Some(out?))
}
