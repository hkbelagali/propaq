//!
//! The one generic run loop.
//!
//! The loop is generic over the basis `A` and the coefficient type `C`.
//! A basis crate only lowers its circuit to `RunGate` values with raw
//! words and calls `run_circuit`. Width dispatch lives here too.
//!
//! The loop shape (layers, noise, reclaim, adaptive width) follows the
//! per-basis loops it replaces.
//!

use pyo3::prelude::*;

use crate::basis::Basis;
use crate::coeff::CoeffRepr;
use crate::noise_resolver::{resolve_noise, retabulate, NoiseTable, ResolvedNoise, PYTHON_TERM_HOOK};
use crate::operator_index::{OperatorIndex, Pos};
use crate::partitioned_termsum::{PartitionedTermSum, PhaseStats};
use crate::progress::Progress;
use crate::results::PropagationResult;
use crate::strings::BasisString;
use crate::term_kernel::LayerContext;
use crate::termsum::EmitCutoff;
use crate::truncators::ResolvedConfig;

/// Inline row width the store starts at when nothing bounds a term's support.
pub const INITIAL_INLINE_POSITIONS: usize = 24;

/// Share of overflowed rows that triggers a repack to a wider row.
pub const OVERFLOW_REPACK_THRESHOLD: f64 = 0.20;

/// Widest key the dispatch covers, in bits.
pub const MAX_DISPATCH_BITS: usize = 4096;

/// A basis that has an implementation at every dispatched width.
pub trait BasisAll:
    Basis<1> + Basis<2> + Basis<4> + Basis<8> + Basis<16> + Basis<32> + Basis<64>
{
}

impl<T> BasisAll for T where
    T: Basis<1> + Basis<2> + Basis<4> + Basis<8> + Basis<16> + Basis<32> + Basis<64>
{
}

/// One gate, lowered by a basis crate. The generator is raw words.
pub struct RunGate<C: CoeffRepr> {
    /// Position of the originating qiskit instruction, when there was one.
    pub qiskit_gate_idx: Option<usize>,
    pub generator: Vec<u64>,
    pub angle: C::GateParam,
}

/// One gate, as the verbose log sees it.
pub struct GateRecord {
    pub gate_idx: usize,
    pub layer_idx: usize,
    pub qiskit_gate_idx: Option<usize>,
    pub terms_before: usize,
    pub terms_after: usize,
    /// New live-term keys created by branching this gate
    pub terms_gained: usize,

    pub declined: u64,
    pub discarded_coeff_l1: f64,
    pub discarded_coeff_max: f64,
    pub elapsed_ms: f64,
}

/// The result of one run. Terms are raw words, `W` words each at the
/// width the run used. The caller trims them to its own length.
pub struct RunOutput<C> {
    pub result: PropagationResult,
    pub gates: Vec<GateRecord>,
    pub terms: Option<Vec<(Vec<u64>, C)>>,
    /// Phase timings and kernel counters for the run's log.
    pub phases: PhaseStats,
}

/// Everything one run needs, borrowed.
pub struct RunInputs<'a, C: CoeffRepr> {
    pub observable: &'a [(Vec<u64>, C)],
    pub layers: &'a [Vec<RunGate<C>>],
    pub n_units: usize,
    pub partitions: usize,
    pub cutoff: &'a EmitCutoff,
    pub noise: Option<&'a ResolvedNoise>,
    pub collect_counts: bool,
    pub diag_state: Option<&'a [u64]>,
    pub want_terms: bool,
    pub log_gates: bool,
    pub progress: Option<&'a Progress>,
}

/// Copies raw words into a key. Extra words are dropped, missing ones are zero.
#[inline]
pub fn key_from_words<const W: usize>(words: &[u64]) -> BasisString<W> {
    let mut w = [0u64; W];
    let n = words.len().min(W);
    w[..n].copy_from_slice(&words[..n]);
    BasisString::from_words(w)
}

fn runtime_err(e: impl ToString) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

/// Runs one circuit at a fixed storage width.
pub fn run_at_width<A, C, const W: usize, P>(inp: &RunInputs<'_, C>) -> PyResult<RunOutput<C>>
where
    A: Basis<W>,
    C: CoeffRepr,
    P: Pos,
{
    let n_units = inp.n_units;
    let noise = inp.noise;
    let mut cutoff = inp.cutoff.clone();
    let n_layers = inp.layers.len() as u32;
    let inline_positions = match cutoff.max_weight {
        Some(w) => OperatorIndex::<P, W>::inline_width_for_support_cutoff(w as usize),
        None => INITIAL_INLINE_POSITIONS.min(2 * n_units),
    };
    let adaptive_width = cutoff.max_weight.is_none();
    let mut op: PartitionedTermSum<C, P, W> =
        PartitionedTermSum::with_inline_positions(n_units, inp.partitions, inline_positions);

    // Clifford frames only exist for the +-1 class.
    if <A as Basis<W>>::BRANCHES > 2
        || cutoff.depends_on_key()
        || noise.is_some_and(ResolvedNoise::depends_on_key)
    {
        op.set_defer_cliffords(false);
    }
    for (words, coeff) in inp.observable {
        let (key, c) = (key_from_words::<W>(words), coeff.clone());
        if !cutoff.admits_initial::<A, C, W>(&key, &c, n_units) {
            continue;
        }
        op.add(&key, c).map_err(runtime_err)?;
    }

    let mut n_terms = Vec::new();
    let mut gate_records: Vec<GateRecord> = Vec::new();
    let mut gate_idx = 0usize;
    let mut layer_table: NoiseTable = Vec::new();
    for (layer_idx, layer) in inp.layers.iter().enumerate() {
        cutoff.layer = LayerContext::new(layer_idx as u32, n_layers);
        if let Some(noise) = noise {
            match noise {
                ResolvedNoise::WeightTable(table) => {
                    op.scale_by_weight::<A>(|w| table[(w as usize).min(table.len() - 1)]);
                }
                ResolvedNoise::LayeredWeightTable(kernel) => {
                    retabulate(kernel.as_ref(), A::KIND, n_units, cutoff.layer, &mut layer_table);
                    op.scale_by_weight::<A>(|w| layer_table[(w as usize).min(layer_table.len() - 1)]);
                }
                ResolvedNoise::TermKernel(kernel) => {
                    op.scale_by_key::<A>(kernel.as_ref(), cutoff.layer);
                }
                ResolvedNoise::PythonTerm(model) => Python::attach(|py| -> PyResult<()> {
                    let model = model.bind(py);
                    op.try_scale_by_key::<A, PyErr>(|key, weight| {
                        model
                            .call_method1(
                                PYTHON_TERM_HOOK,
                                (A::KIND.as_u32(), key.words().to_vec(), n_units, weight),
                            )?
                            .extract()
                    })
                })?,
            }
            op.reclaim::<A>(&cutoff).map_err(runtime_err)?;
        }
        if adaptive_width {
            op.repack_if_overflowing(OVERFLOW_REPACK_THRESHOLD);
        }
        for gate in layer {
            let gen: BasisString<W> = key_from_words::<W>(&gate.generator);
            let (before, declined_before, l1_before, t0) = if inp.log_gates {
                (
                    op.len(),
                    op.scan_counts().1,
                    op.discard_stats().0,
                    Some(std::time::Instant::now()),
                )
            } else {
                (0, 0, 0.0, None)
            };
            let gained = op
                .apply_rotation::<A>(&gen, &gate.angle, &cutoff)
                .map_err(runtime_err)?;
            if let Some(t0) = t0 {
                let (l1_after, discarded_coeff_max) = op.discard_stats();
                gate_records.push(GateRecord {
                    gate_idx,
                    layer_idx,
                    qiskit_gate_idx: gate.qiskit_gate_idx,
                    terms_before: before,
                    terms_after: op.len(),
                    terms_gained: gained,
                    declined: op.scan_counts().1 - declined_before,
                    discarded_coeff_l1: l1_after - l1_before,
                    discarded_coeff_max,
                    elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
                });
            }
            gate_idx += 1;
            if inp.collect_counts {
                n_terms.push(op.len());
            }
            if let Some(p) = inp.progress {
                if gate_idx.is_multiple_of(p.every()) {
                    p.tick(p.every(), op.len());
                }
            }
        }
    }

    if let Some(p) = inp.progress {
        let remainder = gate_idx % p.every();
        if remainder != 0 {
            p.tick(remainder, op.len());
        }
    }

    let phases = op.phase_stats(inline_positions);

    let terms = inp.want_terms.then(|| {
        op.iter::<A>()
            .map(|(key, sign, c)| {
                let mut coeff = c.clone();
                coeff.scale_real(sign);
                (key.words().to_vec(), coeff)
            })
            .collect::<Vec<_>>()
    });
    let expectation_value = inp.diag_state.map_or(0.0, |f| op.expectation::<A>(f));
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

/// A run at one fixed width.
pub type Runner<C> = for<'a> fn(&RunInputs<'a, C>) -> PyResult<RunOutput<C>>;

/// Picks the monomorphized runner for a key of `bits` bits.
/// Returns `None` when the key is wider than `MAX_DISPATCH_BITS`.
pub fn runner_for<A: BasisAll, C: CoeffRepr>(bits: usize) -> Option<Runner<C>> {
    macro_rules! ladder {
        ($($limit:expr => ($w:expr, $pos:ty)),+ $(,)?) => {{
            $(if bits <= $limit {
                return Some(run_at_width::<A, C, $w, $pos>);
            })+
            None
        }};
    }
    ladder!(
        64 => (1, u8),
        128 => (2, u8),
        256 => (4, u16),
        512 => (8, u16),
        1024 => (16, u16),
        2048 => (32, u16),
        4096 => (64, u16),
    )
}

/// Runs a lowered circuit. `bits_per_unit` sets the key width used for
/// dispatch (`n_units * bits_per_unit` bits). Returns `Ok(None)` when the
/// key is too wide for the dispatch.
#[allow(clippy::too_many_arguments)]
pub fn run_circuit<A: BasisAll, C: CoeffRepr>(
    py: Python<'_>,
    observable: &[(Vec<u64>, C)],
    layers: &[Vec<RunGate<C>>],
    diag_state: Option<&[u64]>,
    n_units: usize,
    bits_per_unit: usize,
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
    let Some(run) = runner_for::<A, C>(n_units * bits_per_unit) else {
        return Ok(None);
    };
    let cutoff = EmitCutoff::from(cfg);
    let partitions = n_threads.unwrap_or_else(rayon::current_num_threads).max(1);
    let noise = resolve_noise(noise, n_units, <A as Basis<1>>::KIND)?;
    let total_gates = layers.iter().map(Vec::len).sum();
    let progress = Progress::new(py, progress_bar, total_gates, progress_every)?;

    let out = py.detach(|| {
        pool.install(|| {
            run(&RunInputs {
                observable,
                layers,
                n_units,
                partitions,
                cutoff: &cutoff,
                noise: noise.as_ref(),
                collect_counts,
                diag_state,
                want_terms,
                log_gates,
                progress: progress.as_ref(),
            })
        })
    });

    if let Some(p) = progress.as_ref() {
        p.close();
    }
    Ok(Some(out?))
}
