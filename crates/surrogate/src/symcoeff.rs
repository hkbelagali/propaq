///
/// The symbolic coefficient representation.
///
/// A symbolic coefficient is a persistent DAG of nodes:
///
///     Scalar(c)            -- a numeric leaf
///     Add(a, b)             -- a + b
///     Scale(k, a)           -- k * a, k a real constant
///     Cos(p, a)             -- cos(theta_p) * a
///     Sin(p, a)             -- sin(theta_p) * a
///
/// built via `Arc` so that wrapping an existing coefficient (every gate
/// application, every merge) is O(1) regardless of how large its prior
/// history already is -- no monomial list is ever touched, copied, or
/// resorted on the hot path. Structural sharing across coefficients is
/// automatic: two coefficients built by extending a common ancestor already
/// share that ancestor's allocation via the `Arc`, with no explicit
/// interning/hash-consing needed (unlike the CSR/trie design this replaces).
///
/// This is a deliberately narrow node set, not a general expression tree --
/// propaq's symbolic gates only ever scale a coefficient's whole history by a
/// real constant or by cos/sin of one parameter, never a general product of
/// two symbolic subexpressions, so five node kinds are enough.
///
/// Monomial-level truncation (`FrequencyTruncator`/`CoefficientTruncator`)
/// does *not* require expanding a coefficient into an explicit monomial
/// list, despite first appearances. Both cutoffs are decidable structurally:
/// frequency (a count of `Cos`/`Sin` wraps) is purely structural, and
/// coefficient magnitude has a sound upper bound (`|cos|,|sin| <= 1` always).
/// Caching those bounds per node (`min_freq`/`max_freq`/`upper_scale`, see
/// `Node`'s doc) lets `SymbolicCoeff::prune` drop whole doomed subtrees in
/// O(1) without ever visiting their insides -- see `prune`'s doc comment.
///
/// Besides `prune` (run only at truncation flushes), a coefficient's DAG is
/// only ever walked once per term, at build end, via `compile()` -- a
/// memoized flatten into a flat evaluable tape (see `CompiledCoeff`), so
/// repeated `evaluate`/`evaluate_batch` calls (a VQE optimizer's inner loop)
/// are cheap linear scans instead of per-call tree walks.
///
use std::sync::{Arc, OnceLock};

use num_complex::Complex64;
use pyo3::prelude::*;
use rayon::prelude::*;
use propaq_core::coeff::CoeffRepr;

/// One node of a coefficient's history. `count` is a cached, cheap-to-combine
/// **pre-dedup** monomial-instance upper bound: `Scalar` contributes 1;
/// `Add` sums its children's counts; `Scale`/`Cos`/`Sin` don't change how
/// many terms their inner sum has, since they multiply the *whole* sum by one
/// factor rather than distributing over it. This is the same pre-dedup
/// semantics `total_monomials`/`pending_monomials` already use elsewhere in
/// `propagator.rs`, so `monomial_count()`/`size_hint()` stay O(1) with no
/// caller-visible behavior change there.
///
/// `min_freq`/`max_freq` and `upper_scale` are cached the same way, and are
/// what let `SymbolicCoeff::prune` decide whether a whole subtree survives a
/// `FrequencyTruncator`/`CoefficientTruncator` cutoff without ever expanding
/// it into individual monomials -- see `prune`'s doc comment.
struct Node {
    kind: NodeKind,
    // `u128`, not `u64`: a deep/heavily-merged real circuit's pre-dedup
    // count can exceed `u64::MAX` (see `Node::add`'s doc comment) -- `u128`
    // pushes the saturation ceiling out to ~3.4e38.
    count: u128,
    /// Minimum/maximum number of `Cos`/`Sin` wraps between this node and any
    /// `Scalar` leaf reachable below it. Both are **exact** (frequency is a
    /// purely structural quantity with no runtime-unknown component, unlike
    /// `upper_scale` below), which is what lets `prune` prove a subtree is
    /// *either* fully doomed (`min_freq` too high) *or* fully safe (`max_freq`
    /// still within the cap) without visiting its insides.
    min_freq: u32,
    max_freq: u32,
    /// A safe (never-underestimating) upper bound on `|Scale/Scalar product|`
    /// reachable below this node, ignoring any `Scale` factors above it
    /// (those get folded in externally as a traversal descends). `Cos`/`Sin`
    /// don't change this bound: `|cos|,|sin| <= 1` always, so treating an
    /// unknown-until-parameters-are-bound trig factor as "no worse than x1"
    /// can never underestimate the true magnitude. Unlike frequency, there is
    /// no symmetric "definitely survives" bound computable here -- a trig
    /// factor can shrink a monomial arbitrarily close to zero even when its
    /// structural prefactor is large -- so `upper_scale` only ever proves a
    /// subtree "definitely prunable," never "definitely kept."
    upper_scale: f64,
}

enum NodeKind {
    Scalar(f64),
    Add(Arc<Node>, Arc<Node>),
    Scale(f64, Arc<Node>),
    Cos(u32, Arc<Node>),
    Sin(u32, Arc<Node>),
}

/// A trivial, shared leaf used only as a `mem::replace` placeholder in
/// `Node`'s custom `Drop` below -- cloning it is a cheap refcount bump (never
/// an allocation) since one canonical instance lives for the process's whole
/// lifetime, so it's never the clone that brings a *real* subtree's count to
/// zero.
fn drop_placeholder() -> Arc<Node> {
    static PLACEHOLDER: OnceLock<Arc<Node>> = OnceLock::new();
    Arc::clone(PLACEHOLDER.get_or_init(|| {
        Arc::new(Node { kind: NodeKind::Scalar(0.0), count: 1, min_freq: 0, max_freq: 0, upper_scale: 0.0 })
    }))
}

/// Without this, dropping a `Node` recurses into dropping its `Arc<Node>`
/// children, which recurses into theirs, and so on -- for a coefficient
/// whose history is as deep as the gate count that built it (the same
/// "thousands deep" scenario `SymbolicCoeff::compile` is already iterative
/// to avoid), the default derived drop stack-overflows. This walks the
/// structure with an explicit stack instead: each node's real children are
/// swapped out for a cheap shared placeholder (so the node's own recursive
/// drop, triggered when it falls out of scope below, has nothing further to
/// recurse into) and pushed onto the worklist; a child is only unwrapped
/// (and thus queued for further unlinking) if this call is the one bringing
/// its `Arc` refcount to zero, otherwise something else still holds it and
/// touching only the `Arc`'s refcount is correct.
impl Drop for Node {
    fn drop(&mut self) {
        let mut stack: Vec<Arc<Node>> = Vec::new();
        match &mut self.kind {
            NodeKind::Scalar(_) => {}
            NodeKind::Add(a, b) => {
                stack.push(std::mem::replace(a, drop_placeholder()));
                stack.push(std::mem::replace(b, drop_placeholder()));
            }
            NodeKind::Scale(_, inner) | NodeKind::Cos(_, inner) | NodeKind::Sin(_, inner) => {
                stack.push(std::mem::replace(inner, drop_placeholder()));
            }
        }
        while let Some(arc) = stack.pop() {
            if let Ok(mut node) = Arc::try_unwrap(arc) {
                match &mut node.kind {
                    NodeKind::Scalar(_) => {}
                    NodeKind::Add(a, b) => {
                        stack.push(std::mem::replace(a, drop_placeholder()));
                        stack.push(std::mem::replace(b, drop_placeholder()));
                    }
                    NodeKind::Scale(_, inner) | NodeKind::Cos(_, inner) | NodeKind::Sin(_, inner) => {
                        stack.push(std::mem::replace(inner, drop_placeholder()));
                    }
                }
                // `node` falls out of scope here with only the placeholder
                // left in its fields, so its own (recursive) drop call does
                // O(1) work, not another full subtree walk.
            }
        }
    }
}

impl Node {
    fn scalar(c: f64) -> Arc<Node> {
        Arc::new(Node { kind: NodeKind::Scalar(c), count: 1, min_freq: 0, max_freq: 0, upper_scale: c.abs() })
    }

    fn add(a: Arc<Node>, b: Arc<Node>) -> Arc<Node> {
        // `saturating_add`, not plain `+`: a deep/heavily-merged circuit's
        // true monomial count can exceed even `u128::MAX`, and this project
        // always builds in release mode, where unchecked overflow wraps
        // silently rather than panicking -- a wrapped count can land anywhere
        // in `[0, u128::MAX)`, including values smaller than a configured
        // `MonomialBudget` ceiling, silently defeating the one truncation
        // mechanism meant to bound exactly this growth. Saturating means a
        // count that's overflowed reads as "enormous" forever after (matching
        // `min_freq`/`max_freq`'s existing `saturating_add` convention below,
        // which this field should have matched from the start), so
        // `MonomialBudget`'s `>= max` check stays correct instead of going
        // silently inert once truly saturated. `count` remains a pre-dedup
        // upper bound, never used for evaluate/compile correctness -- this
        // only affects `n_monomials`/`MonomialBudget`'s accuracy at the point
        // of saturation, not the computed expectation value.
        let count = a.count.saturating_add(b.count);
        let min_freq = a.min_freq.min(b.min_freq);
        let max_freq = a.max_freq.max(b.max_freq);
        let upper_scale = a.upper_scale.max(b.upper_scale);
        Arc::new(Node { kind: NodeKind::Add(a, b), count, min_freq, max_freq, upper_scale })
    }

    fn scale(factor: f64, inner: Arc<Node>) -> Arc<Node> {
        let count = inner.count;
        let (min_freq, max_freq) = (inner.min_freq, inner.max_freq);
        let upper_scale = factor.abs() * inner.upper_scale;
        Arc::new(Node { kind: NodeKind::Scale(factor, inner), count, min_freq, max_freq, upper_scale })
    }

    fn cos(param: u32, inner: Arc<Node>) -> Arc<Node> {
        let count = inner.count;
        let min_freq = inner.min_freq.saturating_add(1);
        let max_freq = inner.max_freq.saturating_add(1);
        let upper_scale = inner.upper_scale;
        Arc::new(Node { kind: NodeKind::Cos(param, inner), count, min_freq, max_freq, upper_scale })
    }

    fn sin(param: u32, inner: Arc<Node>) -> Arc<Node> {
        let count = inner.count;
        let min_freq = inner.min_freq.saturating_add(1);
        let max_freq = inner.max_freq.saturating_add(1);
        let upper_scale = inner.upper_scale;
        Arc::new(Node { kind: NodeKind::Sin(param, inner), count, min_freq, max_freq, upper_scale })
    }
}

/// Wrap `node` in `Scale(sign, node)` unless `sign` is exactly `1.0`, in which
/// case `node` is returned unchanged (avoiding a wasted no-op node on the
/// overwhelmingly common case, since a rotation's branch phase is always
/// exactly `1.0` or `-1.0` -- see `apply_rotation_symbolic`/`_numeric`).
#[inline]
fn signed(sign: f64, node: Arc<Node>) -> Arc<Node> {
    if sign == 1.0 { node } else { Node::scale(sign, node) }
}

/// A symbolic coefficient: `None` is the additive identity (zero), matching
/// `Option::default()` for the `CoeffRepr: Default` bound.
#[derive(Clone, Default)]
pub struct SymbolicCoeff(Option<Arc<Node>>);

impl SymbolicCoeff {
    /// Single scalar monomial; used to seed from the observable.
    pub fn from_scalar(c: f64) -> Self {
        SymbolicCoeff(Some(Node::scalar(c)))
    }

    /// Pre-dedup monomial-instance upper bound (see `Node::count`'s doc), O(1).
    pub fn monomial_count(&self) -> u128 {
        self.0.as_ref().map_or(0, |n| n.count)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    /// Flatten this coefficient's DAG into a `CompiledCoeff`: a linear,
    /// topologically-ordered op tape where every node is emitted exactly
    /// once (memoized by `Arc` pointer identity) regardless of how many
    /// parents reference it, and every op only references earlier indices.
    /// `CompiledCoeff::evaluate` is then a single linear scan.
    ///
    /// This is a thin single-root wrapper around `compile_batch` -- see that
    /// function's doc for why a *batch* of roots sharing one memo/tape is
    /// what actually needs to run at build end (once per term here would
    /// redundantly re-flatten any subtree shared across terms).
    pub fn compile(&self) -> CompiledCoeff {
        let (tape, _roots) = SymbolicCoeff::compile_batch(std::iter::once(self.clone()));
        tape
    }

    /// Flatten MANY coefficients' DAGs into ONE shared `CompiledCoeff` tape,
    /// memoized (by `Arc` pointer identity) across all of them at once --
    /// unlike calling `compile()` once per coefficient, a node reachable
    /// from more than one of the given roots is emitted into the tape only
    /// once here, not once per referencing root. Returns `(tape, roots)`
    /// where `roots[i]` is the tape index of the `i`-th input coefficient's
    /// root (`usize::MAX` for an empty/zero coefficient).
    ///
    /// This is the primitive `run_build` uses (per shard of surviving terms)
    /// to avoid the multiplicative blowup of compiling every term's DAG
    /// independently when many terms share large common ancestor subtrees
    /// via `Arc` -- see `propaq.MD`'s "Evaluate & persistence" section.
    ///
    /// Iterative (explicit stack), not recursive: a coefficient's DAG can be
    /// as deep as the number of gates that touched it, which for a real
    /// circuit can be in the thousands -- recursion would risk a stack
    /// overflow that an explicit stack doesn't.
    pub fn compile_batch(
        coeffs: impl IntoIterator<Item = SymbolicCoeff>,
    ) -> (CompiledCoeff, Vec<usize>) {
        // Keeps every input's `Arc<Node>` (if any) alive for the whole
        // traversal below -- `Frame`'s borrows point into these Arcs.
        let owned: Vec<SymbolicCoeff> = coeffs.into_iter().collect();

        let mut ops: Vec<CompiledOp> = Vec::new();
        // `FxHashMap`/`FxHashSet`, not the default `std::collections::HashMap`'s
        // SipHash-based hasher -- the same fix already applied to `prune_node`'s
        // memo below, after profiling a real workload found SipHash's
        // hashing/rehashing dominating that function's runtime at millions-of-keys
        // scale. This traversal is the same shape (pointer-keyed memoization over
        // every surviving term's DAG at build end) and reaches the same key counts.
        let mut memo: rustc_hash::FxHashMap<*const Node, usize> = rustc_hash::FxHashMap::default();
        // Tracks every node that has already been pushed onto the work stack
        // (whether or not its `Exit` has run yet), so a node referenced by
        // more than one parent -- or by more than one of this batch's roots
        // -- is only ever traversed/compiled once. Without this, two `Enter`
        // frames for the same shared node could both land on the stack
        // before either's subtree finishes, each redundantly re-walking
        // (though not incorrectly -- just wastefully) that whole subtree.
        let mut scheduled: rustc_hash::FxHashSet<*const Node> = rustc_hash::FxHashSet::default();

        enum Frame<'a> {
            Enter(&'a Arc<Node>),
            Exit(&'a Arc<Node>),
        }

        let mut stack: Vec<Frame> = Vec::new();
        for c in &owned {
            if let Some(root) = &c.0 {
                if scheduled.insert(Arc::as_ptr(root)) {
                    stack.push(Frame::Enter(root));
                }
            }
        }

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(node) => {
                    stack.push(Frame::Exit(node));
                    match &node.kind {
                        NodeKind::Scalar(_) => {}
                        NodeKind::Add(a, b) => {
                            if scheduled.insert(Arc::as_ptr(b)) {
                                stack.push(Frame::Enter(b));
                            }
                            if scheduled.insert(Arc::as_ptr(a)) {
                                stack.push(Frame::Enter(a));
                            }
                        }
                        NodeKind::Scale(_, inner) | NodeKind::Cos(_, inner) | NodeKind::Sin(_, inner) => {
                            if scheduled.insert(Arc::as_ptr(inner)) {
                                stack.push(Frame::Enter(inner));
                            }
                        }
                    }
                }
                Frame::Exit(node) => {
                    let op = match &node.kind {
                        NodeKind::Scalar(c) => CompiledOp::Scalar(*c),
                        NodeKind::Add(a, b) => {
                            CompiledOp::Add(memo[&Arc::as_ptr(a)], memo[&Arc::as_ptr(b)])
                        }
                        NodeKind::Scale(f, inner) => CompiledOp::Scale(*f, memo[&Arc::as_ptr(inner)]),
                        NodeKind::Cos(p, inner) => CompiledOp::Cos(*p, memo[&Arc::as_ptr(inner)]),
                        NodeKind::Sin(p, inner) => CompiledOp::Sin(*p, memo[&Arc::as_ptr(inner)]),
                    };
                    ops.push(op);
                    memo.insert(Arc::as_ptr(node), ops.len() - 1);
                }
            }
        }

        // Read each input's root index via a *final* lookup pass, never
        // captured inline at push time: two inputs in this batch can hold
        // the literal same `Arc` root, in which case the second one is
        // deduped away by `scheduled` and never gets its own `Exit` frame --
        // the memo's final entry (written by whichever root's traversal
        // actually ran) is the only correct place to read its index from.
        let roots: Vec<usize> = owned
            .iter()
            .map(|c| {
                c.0.as_ref()
                    .and_then(|root| memo.get(&Arc::as_ptr(root)).copied())
                    .unwrap_or(usize::MAX)
            })
            .collect();

        (CompiledCoeff { ops }, roots)
    }

    /// Structurally drop monomials violating `max_frequency` and/or
    /// `coeff_cutoff` -- **without ever expanding into an explicit monomial
    /// list**. A no-op if both are `None`.
    ///
    /// Both cutoffs are decided via the cached `min_freq`/`max_freq`/
    /// `upper_scale` bounds (see `Node`'s doc): walking the DAG top-down
    /// while carrying an accumulated `(depth, scale)` context from the root,
    /// a whole subtree is dropped in O(1) the instant it's *provably*
    /// doomed (`depth + node.min_freq > max_frequency`, or
    /// `scale * node.upper_scale < coeff_cutoff`) -- no need to visit
    /// anything below it. When only `max_frequency` is active, a subtree can
    /// also be proven *fully safe* (`depth + node.max_freq <= max_frequency`)
    /// and kept unchanged (original `Arc`, no rebuild) -- frequency is exact,
    /// so both directions are provable; magnitude only ever proves
    /// "doomed," never "safe" (a trig factor can shrink a monomial toward
    /// zero regardless of how large its structural prefactor is), so
    /// `coeff_cutoff`-active subtrees always need an exact leaf-level
    /// decision once they're not already provably doomed.
    ///
    /// Iterative (explicit stack, same `Enter`/`Exit` discipline as
    /// `compile`), since a coefficient's DAG can be thousands of nodes deep.
    /// Unlike `compile`, this isn't a pure post-order walk: the
    /// doomed/safe/ambiguous decision is made top-down (at `Enter`, using
    /// context inherited from the parent), and the rebuilt subtree is
    /// assembled bottom-up (at `Exit`) -- so each memoized entry is keyed by
    /// `(node pointer, context)`, not pointer alone, since the same shared
    /// node can be reached with different context from different parents.
    ///
    /// Memoization: `depth` is exact and cheap to key on directly. `scale` is
    /// a continuous float, so it's tracked internally as `scale_exp: i32`, a
    /// rounded-**up** log2 exponent maintaining the invariant
    /// `true accumulated |scale| <= 2^scale_exp` -- a cached decision can
    /// therefore only ever *keep* slightly more than an exact cutoff would,
    /// never wrongly prune. Without this, a shared subtree reached with
    /// different exact `scale` values from different parents (routine: every
    /// `apply_rotation` creates a 2-parent diamond, and `merge` folds
    /// derivation paths back together by default after every gate) would be
    /// re-walked once per distinct context -- reopening exactly the
    /// unbounded-revisit blowup this DAG design exists to avoid. `depth`'s
    /// dimension of the memo key collapses to a constant when
    /// `max_frequency` is `None`; `scale_exp`'s collapses when `coeff_cutoff`
    /// is `None`.
    pub fn prune(&mut self, max_frequency: Option<u32>, coeff_cutoff: Option<f64>) {
        if max_frequency.is_none() && coeff_cutoff.is_none() {
            return;
        }
        let Some(root) = self.0.take() else { return };
        self.0 = prune_node(&root, max_frequency, coeff_cutoff);
    }

    /// Records that every live term's history branched on `param`: `cos`
    /// stays on `self` (mutated in place, O(1)), `sin` is returned as the new
    /// anticommuted term's coefficient. Both just wrap the existing history
    /// in one new node -- no monomial touched, regardless of how large that
    /// history already is.
    fn apply_rotation_symbolic(&mut self, param: u32, phase: Complex64) -> Self {
        let branch_phase = Complex64::new(0.0, 1.0) * phase;
        debug_assert!(branch_phase.im.abs() < 1e-9, "expected real branch phase: {branch_phase:?}");
        let branch_phase = branch_phase.re;

        let old = self.0.take();
        self.0 = old.clone().map(|n| Node::cos(param, n));
        let sin = old.map(|n| signed(branch_phase, Node::sin(param, n)));
        SymbolicCoeff(sin)
    }

    /// Numeric-angle rotation: `cos`/`sin` of `angle` are computed
    /// immediately and fold into a `Scale` node (mirrors
    /// `Complex64::apply_rotation` exactly). Also O(1) -- this subsumes what
    /// used to be a dedicated copy-on-write `Arc<Inner>` scheme for numeric
    /// gates specifically; `Scale` *is* that scheme, generalized uniformly
    /// to every gate.
    fn apply_rotation_numeric(&mut self, angle: f64, phase: Complex64) -> Self {
        let cos_t = angle.cos();
        let sin_t = angle.sin();
        let branch_phase = Complex64::new(0.0, sin_t) * phase;
        debug_assert!(branch_phase.im.abs() < 1e-9, "expected real branch phase: {branch_phase:?}");
        let branch_phase = branch_phase.re;

        let old = self.0.take();
        self.0 = old.clone().map(|n| Node::scale(cos_t, n));
        let sin = old.map(|n| Node::scale(branch_phase, n));
        SymbolicCoeff(sin)
    }

    /// Real algebraic simplification: collapse every group of monomials
    /// that share the same canonical trig-factor run into one, summing
    /// their scalars. Lossless -- never discards a legitimate contribution
    /// (an exact-zero-sum cancellation is dropped, which changes no
    /// evaluated value). This is what actually bounds `monomial_count()`,
    /// unlike `prune` (which only ever *removes* monomials failing a
    /// cutoff, never *merges* two surviving ones) -- see `simplify_batch`'s
    /// doc for the algorithm and why it must run only at explicit,
    /// infrequent points (e.g. a truncation flush), never on the per-gate
    /// `apply_rotation_*`/`add_assign` hot path.
    ///
    /// Thin single-coefficient wrapper around `simplify_batch`, mirroring
    /// `compile`'s relationship to `compile_batch` -- see that function's
    /// doc for why a *batch* of coefficients sharing one dedup memo is what
    /// actually needs to run at a flush (calling this once per coefficient
    /// independently would redundantly re-dedup any subtree shared across
    /// them, exactly the `compile()`-per-term OOM class this codebase
    /// already found and fixed once via `compile_batch`).
    pub fn simplify(&mut self) {
        simplify_batch(std::slice::from_mut(self));
    }
}

/// Combine an accumulated `scale_exp` (see `prune_node`'s doc) with one more
/// `Scale` factor `k`, rounding **up** so the invariant `true accumulated
/// |scale| <= 2^scale_exp` is preserved. `k.abs().log2()` is `-inf` when
/// `k == 0.0`; `(-inf).ceil() as i32` saturates to `i32::MIN` in Rust (no
/// special-casing needed), which correctly represents "this branch's true
/// scale is exactly zero" for the doomed check in `is_doomed_by_coeff`.
#[inline]
fn combine_scale_exp(exp: i32, k: f64) -> i32 {
    let k_exp = k.abs().log2().ceil() as i32;
    exp.saturating_add(k_exp)
}

/// Whether a subtree with accumulated `scale_exp` and cached `upper_scale` is
/// *provably* below `cutoff` -- i.e. `2^scale_exp * upper_scale < cutoff`,
/// computed in log-space to stay well-behaved at extreme `scale_exp`
/// magnitudes (a real circuit can chain thousands of `Scale` factors).
/// `cutoff <= 0.0` never dooms anything (no real magnitude is `< 0`).
#[inline]
fn is_doomed_by_coeff(scale_exp: i32, upper_scale: f64, cutoff: f64) -> bool {
    if cutoff <= 0.0 {
        return false;
    }
    (scale_exp as f64) + upper_scale.log2() < cutoff.log2()
}

/// The memo/scheduled key for `prune_node`'s traversal: a node pointer plus
/// whichever context dimensions are actually in use (the unused dimension
/// collapses to a constant, since a cutoff that isn't configured can never
/// distinguish two contexts).
type PruneKey = (*const Node, u32, i32);

#[inline]
fn prune_key(ptr: *const Node, depth: u32, scale_exp: i32, has_freq: bool, has_coeff: bool) -> PruneKey {
    (ptr, if has_freq { depth } else { 0 }, if has_coeff { scale_exp } else { 0 })
}

/// `SymbolicCoeff::prune`'s iterative top-down-decide/bottom-up-rebuild walk
/// -- see `prune`'s doc comment for the algorithm. Returns `None` if the
/// whole coefficient was pruned away.
fn prune_node(root: &Arc<Node>, max_frequency: Option<u32>, coeff_cutoff: Option<f64>) -> Option<Arc<Node>> {
    let has_freq = max_frequency.is_some();
    let has_coeff = coeff_cutoff.is_some();
    let max_freq_cap = max_frequency.unwrap_or(u32::MAX);
    let cutoff = coeff_cutoff.unwrap_or(0.0);

    // `FxHashMap`/`FxHashSet` (already a dependency, `rustc-hash`, and
    // already used for exactly this kind of hot-path memoization in
    // `soa::kernels::merge`), not the default `std::collections::HashMap`'s
    // SipHash-based hasher -- profiling a real workload found this memo
    // table's hashing/rehashing dominating (~90%+) `prune`'s total runtime,
    // since SipHash is deliberately not optimized for speed (it's designed
    // for DoS resistance), which matters enormously once the key count
    // reaches the millions this real-workload traversal produces.
    let mut memo: rustc_hash::FxHashMap<PruneKey, Option<Arc<Node>>> = rustc_hash::FxHashMap::default();
    let mut scheduled: rustc_hash::FxHashSet<PruneKey> = rustc_hash::FxHashSet::default();

    enum Frame<'a> {
        Enter { node: &'a Arc<Node>, depth: u32, scale_exp: i32 },
        Exit { node: &'a Arc<Node>, depth: u32, scale_exp: i32 },
    }

    let mut stack: Vec<Frame> = Vec::new();
    let root_key = prune_key(Arc::as_ptr(root), 0, 0, has_freq, has_coeff);
    scheduled.insert(root_key);
    stack.push(Frame::Enter { node: root, depth: 0, scale_exp: 0 });

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Enter { node, depth, scale_exp } => {
                let key = prune_key(Arc::as_ptr(node), depth, scale_exp, has_freq, has_coeff);

                let doomed_by_freq = has_freq && depth.saturating_add(node.min_freq) > max_freq_cap;
                let doomed_by_coeff = has_coeff && is_doomed_by_coeff(scale_exp, node.upper_scale, cutoff);
                if doomed_by_freq || doomed_by_coeff {
                    memo.insert(key, None);
                    continue;
                }

                // Only provable when coefficient truncation isn't also
                // active -- magnitude has no "definitely survives" bound.
                let provably_safe = has_freq && !has_coeff && depth.saturating_add(node.max_freq) <= max_freq_cap;
                if provably_safe || (!has_freq && !has_coeff) {
                    memo.insert(key, Some(Arc::clone(node)));
                    continue;
                }

                // Ambiguous: recurse. A `Scalar` leaf has no children, so it
                // falls straight through to `Exit` and is trivially kept
                // (it already survived the doomed check above).
                stack.push(Frame::Exit { node, depth, scale_exp });
                match &node.kind {
                    NodeKind::Scalar(_) => {}
                    NodeKind::Add(a, b) => {
                        let kb = prune_key(Arc::as_ptr(b), depth, scale_exp, has_freq, has_coeff);
                        let ka = prune_key(Arc::as_ptr(a), depth, scale_exp, has_freq, has_coeff);
                        if scheduled.insert(kb) {
                            stack.push(Frame::Enter { node: b, depth, scale_exp });
                        }
                        if scheduled.insert(ka) {
                            stack.push(Frame::Enter { node: a, depth, scale_exp });
                        }
                    }
                    NodeKind::Scale(k, inner) => {
                        let new_scale_exp = if has_coeff { combine_scale_exp(scale_exp, *k) } else { scale_exp };
                        let ki = prune_key(Arc::as_ptr(inner), depth, new_scale_exp, has_freq, has_coeff);
                        if scheduled.insert(ki) {
                            stack.push(Frame::Enter { node: inner, depth, scale_exp: new_scale_exp });
                        }
                    }
                    NodeKind::Cos(_, inner) | NodeKind::Sin(_, inner) => {
                        let new_depth = depth.saturating_add(1);
                        let ki = prune_key(Arc::as_ptr(inner), new_depth, scale_exp, has_freq, has_coeff);
                        if scheduled.insert(ki) {
                            stack.push(Frame::Enter { node: inner, depth: new_depth, scale_exp });
                        }
                    }
                }
            }
            Frame::Exit { node, depth, scale_exp } => {
                let key = prune_key(Arc::as_ptr(node), depth, scale_exp, has_freq, has_coeff);
                let result = match &node.kind {
                    NodeKind::Scalar(_) => Some(Arc::clone(node)),
                    NodeKind::Add(a, b) => {
                        let ka = prune_key(Arc::as_ptr(a), depth, scale_exp, has_freq, has_coeff);
                        let kb = prune_key(Arc::as_ptr(b), depth, scale_exp, has_freq, has_coeff);
                        match (memo[&ka].clone(), memo[&kb].clone()) {
                            (None, None) => None,
                            (Some(x), None) => Some(x),
                            (None, Some(y)) => Some(y),
                            (Some(x), Some(y)) => {
                                if Arc::ptr_eq(&x, a) && Arc::ptr_eq(&y, b) {
                                    Some(Arc::clone(node))
                                } else {
                                    Some(Node::add(x, y))
                                }
                            }
                        }
                    }
                    NodeKind::Scale(k, inner) => {
                        let new_scale_exp = if has_coeff { combine_scale_exp(scale_exp, *k) } else { scale_exp };
                        let ki = prune_key(Arc::as_ptr(inner), depth, new_scale_exp, has_freq, has_coeff);
                        memo[&ki].clone().map(|x| {
                            if Arc::ptr_eq(&x, inner) { Arc::clone(node) } else { Node::scale(*k, x) }
                        })
                    }
                    NodeKind::Cos(p, inner) => {
                        let new_depth = depth.saturating_add(1);
                        let ki = prune_key(Arc::as_ptr(inner), new_depth, scale_exp, has_freq, has_coeff);
                        memo[&ki].clone().map(|x| {
                            if Arc::ptr_eq(&x, inner) { Arc::clone(node) } else { Node::cos(*p, x) }
                        })
                    }
                    NodeKind::Sin(p, inner) => {
                        let new_depth = depth.saturating_add(1);
                        let ki = prune_key(Arc::as_ptr(inner), new_depth, scale_exp, has_freq, has_coeff);
                        memo[&ki].clone().map(|x| {
                            if Arc::ptr_eq(&x, inner) { Arc::clone(node) } else { Node::sin(*p, x) }
                        })
                    }
                };
                memo.insert(key, result);
            }
        }
    }

    memo.remove(&root_key).unwrap()
}

/// A monomial's canonical trig-factor run: `cos^cos_pow * sin^sin_pow` of
/// `theta_param`, for every distinct `param` the monomial's derivation
/// touched. `BTreeMap`, not a sorted `Vec` (unlike the old, removed CSR
/// design's bit-packed `make_factor` runs): a `Vec`-based sorted-insert
/// would cost `O(len)` per insertion (element shifting), which for a long
/// unbranched chain of distinct-parameter wraps (one insertion per gate)
/// sums to `O(n^2)` over the whole chain -- `BTreeMap`'s `O(log len)`
/// insertion keeps that `O(n log n)`. `BTreeMap<K: Hash, V: Hash>` derives
/// `Hash`/`Eq` via its canonical (sorted-by-key) iteration order, which is
/// exactly the property that makes two independently-derived `FactorRun`s
/// compare/hash equal iff they represent the same monomial -- no separate
/// canonicalization step needed.
#[derive(Clone, PartialEq, Eq, Hash, Default)]
struct FactorRun(std::collections::BTreeMap<u32, (u32, u32)>);

impl FactorRun {
    /// Fold in one more `Cos`/`Sin` wrap on `param`, mutating in place.
    /// `O(log len)`. Always called on an *owned* `FactorRun` freshly taken
    /// out of a `Dedup` map via `into_iter()` (see `simplify_batch`'s
    /// `Cos`/`Sin` exit handling) -- there is deliberately no
    /// clone-and-return variant of this method: every call site already
    /// holds a uniquely-owned value by the time it needs to increment one,
    /// so a clone-based alternative would only exist to be misused (as an
    /// earlier draft of this function did, reintroducing an `O(n^2)`
    /// blowup over a long unbranched chain that this method's whole reason
    /// for existing is to avoid).
    fn increment_in_place(&mut self, param: u32, is_sin: bool) {
        let entry = self.0.entry(param).or_insert((0, 0));
        if is_sin { entry.1 += 1; } else { entry.0 += 1; }
    }
}

/// A coefficient's monomial list *in flight* during `simplify_batch`, kept
/// as a plain `Vec`, not a `FxHashMap<FactorRun, f64>` -- deliberately.
/// `Scale`/`Cos`/`Sin` are injective per-entry transforms (see
/// `FactorRun::increment_in_place`'s doc): applied uniformly to every entry
/// of an already-duplicate-free list, they can never *introduce* a
/// duplicate, so they never need hashmap machinery at all. Using a
/// `FxHashMap` for them anyway would mean *hashing* every (growing)
/// `FactorRun` key on every single-entry transform -- an early version of
/// this function did exactly that, and hashing a size-`k` `FactorRun`
/// costs `O(k)`, so doing it at every level of a `k`-deep unbranched chain
/// summed to `O(n^2)` overall, *independent of* the clone-vs-move fix
/// `remaining`/`take_or_clone` below already provide (that fix only
/// avoided the *clone*, not the *hash*). Only `Add` can introduce a
/// duplicate (two independently-derived monomials happening to canonicalize
/// to the same run), so only `Add`'s combine step (`group`, below) pays
/// hashing cost -- exactly where merging is actually happening, not a cost
/// smeared across every intermediate node. The invariant "no duplicate
/// `FactorRun`s within one `Terms`" holds by construction: `Scalar` seeds
/// one entry, `Scale`/`Cos`/`Sin` preserve it (injective), and `Add`
/// explicitly restores it via `group`.
type Terms = Vec<(FactorRun, f64)>;

/// Group `terms` by canonical `FactorRun`, summing scalars on collision --
/// the actual like-term collection, and the only place in `simplify_batch`
/// that hashes a `FactorRun` at all. `O(n)` amortized (one hash + insert per
/// input entry).
fn group(terms: Terms) -> Terms {
    let mut map: rustc_hash::FxHashMap<FactorRun, f64> =
        rustc_hash::FxHashMap::with_capacity_and_hasher(terms.len(), Default::default());
    for (run, scalar) in terms {
        *map.entry(run).or_insert(0.0) += scalar;
    }
    map.into_iter().collect()
}

/// Real algebraic simplification, batched across `coeffs`: collapse every
/// monomial group sharing the same canonical `FactorRun` into one, summing
/// scalars, then rebuild each coefficient's DAG from the deduped result --
/// see `SymbolicCoeff::simplify`'s doc for why this must be batched (one
/// shared memo across every root at once, not one call per coefficient) and
/// why it must never run on the per-gate hot path.
///
/// Two passes, both iterative (explicit stack/worklist, matching every
/// other `Node` traversal in this file -- a coefficient's DAG can be
/// thousands of nodes deep):
///
/// **Pass 0** computes `remaining: FxHashMap<*const Node, u32>`, the number
/// of times each reachable node will still be consumed (its in-DAG fan-in,
/// plus one per `coeffs` entry that references it as a root). This is what
/// lets Pass 1 tell whether a child's `Terms` list is uniquely owned at the
/// moment it's consumed (`remaining` about to hit 0 -> move it out of the
/// memo, mutate in place, no clone) or still needed by another parent later
/// (`remaining` still positive -> clone). Without this, every consumption
/// would have to clone defensively, which for a long unbranched chain (no
/// sharing at all) reintroduces an `O(n)`-per-level cost (cloning every
/// `FactorRun` in the list) that `remaining` avoids entirely for the common
/// (unshared) case.
///
/// **Pass 1** is the actual dedup, a post-order `Enter`/`Exit` walk (same
/// shape as `compile_batch`'s `Frame`) memoized by `*const Node` alone --
/// deliberately simpler than `prune_node`'s `(pointer, depth, scale_exp)`
/// context key: a node's fully-expanded canonical monomial list is a pure
/// function of that subtree's own content, unlike `prune`'s doomed/safe
/// decision, which genuinely depends on ancestor-accumulated depth/scale.
/// Wrapping the same shared child in `Cos(5, child)` from one parent and
/// `Sin(3, child)` from another never changes `child`'s own deduped list,
/// only how each parent goes on to combine it -- so pointer-only
/// memoization is correct here, not just simpler.
///
/// `Scalar(c)` seeds a one-entry list; `Scale`/`Cos`/`Sin` transform every
/// entry of the (owned-or-cloned) child list in place, no hashing (see
/// `Terms`'s doc); `Add(a, b)` concatenates two owned lists and calls
/// `group` (the only hashing in this whole function) -- the actual like-term
/// collection. Pass 2 rebuilds each coefficient's root from its deduped
/// list via `rebuild_balanced`, deduping the rebuild itself by root pointer
/// (two coefficients can share the literal same `Arc` root).
fn simplify_batch(coeffs: &mut [SymbolicCoeff]) {
    use rustc_hash::{FxHashMap, FxHashSet};

    // Pass 0: fan-in / remaining-consumer counts.
    let mut remaining: FxHashMap<*const Node, u32> = FxHashMap::default();
    let mut seen0: FxHashSet<*const Node> = FxHashSet::default();
    let mut worklist: Vec<&Arc<Node>> = Vec::new();
    for c in coeffs.iter() {
        if let Some(root) = &c.0 {
            *remaining.entry(Arc::as_ptr(root)).or_insert(0) += 1;
            if seen0.insert(Arc::as_ptr(root)) {
                worklist.push(root);
            }
        }
    }
    while let Some(node) = worklist.pop() {
        match &node.kind {
            NodeKind::Scalar(_) => {}
            NodeKind::Add(a, b) => {
                *remaining.entry(Arc::as_ptr(a)).or_insert(0) += 1;
                *remaining.entry(Arc::as_ptr(b)).or_insert(0) += 1;
                if seen0.insert(Arc::as_ptr(a)) {
                    worklist.push(a);
                }
                if seen0.insert(Arc::as_ptr(b)) {
                    worklist.push(b);
                }
            }
            NodeKind::Scale(_, inner) | NodeKind::Cos(_, inner) | NodeKind::Sin(_, inner) => {
                *remaining.entry(Arc::as_ptr(inner)).or_insert(0) += 1;
                if seen0.insert(Arc::as_ptr(inner)) {
                    worklist.push(inner);
                }
            }
        }
    }

    // Pass 1: memoized post-order dedup.
    let mut memo: FxHashMap<*const Node, Terms> = FxHashMap::default();
    let mut scheduled: FxHashSet<*const Node> = FxHashSet::default();

    enum Frame<'a> {
        Enter(&'a Arc<Node>),
        Exit(&'a Arc<Node>),
    }

    /// Consume a child's `Terms` list: moved out (no clone) if this is its
    /// last remaining consumer, cloned otherwise. `#[inline]` -- called at
    /// every `Add`/`Scale`/`Cos`/`Sin` exit.
    #[inline]
    fn take_or_clone(memo: &mut FxHashMap<*const Node, Terms>, remaining: &mut FxHashMap<*const Node, u32>, ptr: *const Node) -> Terms {
        let r = remaining.get_mut(&ptr).expect("remaining must already have an entry from pass 0");
        *r -= 1;
        if *r == 0 {
            memo.remove(&ptr).expect("child must already be deduped by post-order exit ordering")
        } else {
            memo[&ptr].clone()
        }
    }

    let mut stack: Vec<Frame> = Vec::new();
    for c in coeffs.iter() {
        if let Some(root) = &c.0 {
            if scheduled.insert(Arc::as_ptr(root)) {
                stack.push(Frame::Enter(root));
            }
        }
    }
    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Enter(node) => {
                stack.push(Frame::Exit(node));
                match &node.kind {
                    NodeKind::Scalar(_) => {}
                    NodeKind::Add(a, b) => {
                        if scheduled.insert(Arc::as_ptr(b)) {
                            stack.push(Frame::Enter(b));
                        }
                        if scheduled.insert(Arc::as_ptr(a)) {
                            stack.push(Frame::Enter(a));
                        }
                    }
                    NodeKind::Scale(_, inner) | NodeKind::Cos(_, inner) | NodeKind::Sin(_, inner) => {
                        if scheduled.insert(Arc::as_ptr(inner)) {
                            stack.push(Frame::Enter(inner));
                        }
                    }
                }
            }
            Frame::Exit(node) => {
                let ptr = Arc::as_ptr(node);
                let result: Terms = match &node.kind {
                    NodeKind::Scalar(c) => vec![(FactorRun::default(), *c)],
                    NodeKind::Add(a, b) => {
                        // Order doesn't matter for correctness when `a`/`b`
                        // are the same pointer (self-addition): each call
                        // independently decrements `remaining`, so the
                        // second call always sees whatever the first left
                        // behind.
                        let mut da = take_or_clone(&mut memo, &mut remaining, Arc::as_ptr(a));
                        let db = take_or_clone(&mut memo, &mut remaining, Arc::as_ptr(b));
                        da.extend(db);
                        group(da)
                    }
                    NodeKind::Scale(k, inner) => {
                        let mut m = take_or_clone(&mut memo, &mut remaining, Arc::as_ptr(inner));
                        for (_, scalar) in m.iter_mut() {
                            *scalar *= k;
                        }
                        m
                    }
                    NodeKind::Cos(p, inner) => {
                        let mut m = take_or_clone(&mut memo, &mut remaining, Arc::as_ptr(inner));
                        for (run, _) in m.iter_mut() {
                            run.increment_in_place(*p, false);
                        }
                        m
                    }
                    NodeKind::Sin(p, inner) => {
                        let mut m = take_or_clone(&mut memo, &mut remaining, Arc::as_ptr(inner));
                        for (run, _) in m.iter_mut() {
                            run.increment_in_place(*p, true);
                        }
                        m
                    }
                };
                memo.insert(ptr, result);
            }
        }
    }

    // Pass 2: rebuild each coefficient's root, deduped by root pointer.
    let mut rebuilt: FxHashMap<*const Node, Option<Arc<Node>>> = FxHashMap::default();
    for c in coeffs.iter_mut() {
        let ptr = match &c.0 {
            Some(root) => Arc::as_ptr(root),
            None => continue,
        };
        let new_root = rebuilt.entry(ptr).or_insert_with(|| rebuild_balanced(&memo[&ptr])).clone();
        c.0 = new_root;
    }
}

/// Iterative pairwise "tournament" reduction of a deduped monomial list into
/// a balanced `Add`-tree: `O(log N)` resulting depth (not a degenerate
/// `O(N)` left-fold chain -- a long unbranched `Add`-chain would add real
/// depth back for every downstream `compile`/`prune`/`Drop` traversal,
/// undermining the point of deduping), `O(N)` total `Node::add` calls, no
/// recursion. Filters exact-zero-sum monomials -- real cancellation
/// `prune`'s upper-bound-based check can never see, since it never merges
/// derivation paths in the first place. `terms` is already duplicate-free by
/// construction (see `Terms`'s doc), so no grouping happens here.
fn rebuild_balanced(terms: &Terms) -> Option<Arc<Node>> {
    let mut leaves: Vec<Arc<Node>> = terms
        .iter()
        .filter(|(_, scalar)| *scalar != 0.0)
        .map(|(run, scalar)| build_leaf(run, *scalar))
        .collect();
    if leaves.is_empty() {
        return None;
    }
    while leaves.len() > 1 {
        let mut next = Vec::with_capacity(leaves.len().div_ceil(2));
        let mut it = leaves.into_iter();
        while let Some(a) = it.next() {
            next.push(match it.next() {
                Some(b) => Node::add(a, b),
                None => a,
            });
        }
        leaves = next;
    }
    leaves.pop()
}

/// Build one monomial's DAG leaf: `Node::scalar(scalar)` wrapped in nested
/// `Node::cos`/`Node::sin` calls per `FactorRun` entry, ascending by
/// `param`. Reuses the existing constructors only -- no new `NodeKind`
/// variant, exactly mirroring how `same_parameter_at_two_gates_collapses_
/// to_a_power` already encodes a repeated wrap as nested single wraps.
fn build_leaf(run: &FactorRun, scalar: f64) -> Arc<Node> {
    let mut node = Node::scalar(scalar);
    for (&param, &(cos_pow, sin_pow)) in &run.0 {
        for _ in 0..cos_pow {
            node = Node::cos(param, node);
        }
        for _ in 0..sin_pow {
            node = Node::sin(param, node);
        }
    }
    node
}

/// Sharded parallel entry point for `simplify`, mirroring
/// `propagator.rs::compile_surviving_terms`'s sharding exactly (same
/// `SHARD_OVERSUBSCRIPTION` rationale: a subtree shared *across* shard
/// boundaries is deduped once per shard instead of once globally --
/// the accepted tradeoff for parallelism, not a new one).
pub fn simplify_sharded(coeffs: &mut [SymbolicCoeff], n_shards: usize) {
    let chunk = coeffs.len().div_ceil(n_shards.max(1)).max(1);
    coeffs.par_chunks_mut(chunk).for_each(|shard| simplify_batch(shard));
}

/// Gate parameter for a symbolic rotation: either a symbolic parameter (a
/// slot in the parameter vector, resolved later by `CompiledCoeff::evaluate`
/// against the LUT) or a concrete numeric angle baked in immediately (mirrors
/// `Complex64::apply_rotation`'s math and never touches the DAG's structure,
/// only its `Scale` factors).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GateParam {
    Symbolic { param: u32 },
    Numeric { angle: f64 },
}

impl GateParam {
    /// A symbolic gate on parameter `x`. Convenience for tests/benchmarks.
    #[inline]
    pub fn symbolic(x: u32) -> Self {
        GateParam::Symbolic { param: x }
    }
}

impl CoeffRepr for SymbolicCoeff {
    type GateParam = GateParam;

    #[inline]
    fn from_real(c: f64) -> Self {
        // Seed observables are Hermitian, so their Pauli/Majorana-basis
        // coefficients are real.
        SymbolicCoeff::from_scalar(c)
    }

    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.0 = match (self.0.take(), other.0) {
            (None, b) => b,
            (a, None) => a,
            (Some(a), Some(b)) => Some(Node::add(a, b)),
        };
    }

    /// Dispatches to `apply_rotation_symbolic` (a `Cos`/`Sin` node recorded)
    /// for a symbolic gate or `apply_rotation_numeric` (`Scale` node) for a
    /// concrete angle.
    fn apply_rotation(&mut self, param: &GateParam, phase: Complex64) -> Self {
        match param {
            GateParam::Symbolic { param } => self.apply_rotation_symbolic(*param, phase),
            GateParam::Numeric { angle } => self.apply_rotation_numeric(*angle, phase),
        }
    }

    #[inline]
    fn scale_real(&mut self, factor: f64) {
        self.0 = self.0.take().map(|n| Node::scale(factor, n));
    }

    /// Only `GateParam::Numeric` carries a concrete angle to test (mirrors
    /// `f64`'s own `is_clifford_param` exactly); `GateParam::Symbolic` never
    /// qualifies, since its angle is unknown until `evaluate` time and could
    /// turn out to be anything -- unconditionally taking the in-place branch
    /// there would silently discard a cos-branch that isn't actually zero.
    /// Without this override the base trait default (`false` always) applies
    /// to the `Numeric` case too, which is what let every Clifford-angle
    /// (`pi/2`) numeric gate in a real circuit needlessly append a new row
    /// (and, once later remerged, double that row's monomial `count`)
    /// instead of using the cheap overwrite-in-place path the numerical
    /// propagator already gets for the same angle.
    #[inline]
    fn is_clifford_param(param: &GateParam, eps: f64) -> bool {
        match param {
            GateParam::Symbolic { .. } => false,
            GateParam::Numeric { angle } => angle.cos().abs() < eps,
        }
    }

    /// Monomial count is what actually drives memory/CPU cost for symbolic
    /// coefficients, unlike raw term count.
    #[inline]
    fn size_hint(&self) -> u128 {
        self.monomial_count()
    }

    /// A rotation's `param_index` (`Optional[int]`) takes precedence: if
    /// present, the gate is symbolic. Otherwise falls back to `angle`
    /// (`float`), a concrete numeric angle baked in at build time.
    fn extract_gate_param(obj: &Bound<'_, PyAny>) -> PyResult<GateParam> {
        let param_index: Option<u32> = obj.getattr("param_index")?.extract()?;
        if let Some(param) = param_index {
            return Ok(GateParam::Symbolic { param });
        }
        let angle: f64 = obj.getattr("angle")?.extract()?;
        Ok(GateParam::Numeric { angle })
    }
}

/// One flattened operation in a `CompiledCoeff`'s tape. Tape-referencing
/// operand indices are `usize`, not `u32`: a real large workload's merged
/// tape (many shards' worth of largely-unshared per-term history, summed
/// across millions of surviving terms) has been observed to exceed
/// `u32::MAX` total ops -- `u32` looked like plenty for one term's own
/// tape, but doesn't bound the *sum* across a whole model. `Cos`/`Sin`'s
/// first field is a parameter *index* (bounded by parameter count, not tape
/// size), so it stays `u32`. Indices always refer to earlier positions in
/// the same tape (the tape is topologically ordered by construction -- see
/// `SymbolicCoeff::compile`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum CompiledOp {
    Scalar(f64),
    Add(usize, usize),
    Scale(f64, usize),
    Cos(u32, usize),
    Sin(u32, usize),
}

/// Shift one op's tape-referencing operand index/indices by `offset`
/// (`CompiledCoeff::merge_shards`'s per-shard concatenation step). Split out
/// as its own function so the arithmetic can be unit-tested directly against
/// large synthetic offsets (beyond `u32::MAX`) without needing to actually
/// allocate a multi-billion-entry `Vec<CompiledOp>` to exercise it.
/// `checked_add` is defense-in-depth: since operand indices are `usize`,
/// this can only overflow by exhausting the whole address space.
fn shift_op(op: CompiledOp, offset: usize) -> CompiledOp {
    let shifted = |i: usize| {
        i.checked_add(offset).unwrap_or_else(|| {
            panic!(
                "compiled tape operand index overflowed usize while merging shards \
                 (index {i} + shard offset {offset})"
            )
        })
    };
    match op {
        CompiledOp::Scalar(c) => CompiledOp::Scalar(c),
        CompiledOp::Add(a, b) => CompiledOp::Add(shifted(a), shifted(b)),
        CompiledOp::Scale(f, i) => CompiledOp::Scale(f, shifted(i)),
        CompiledOp::Cos(p, i) => CompiledOp::Cos(p, shifted(i)),
        CompiledOp::Sin(p, i) => CompiledOp::Sin(p, shifted(i)),
    }
}

/// A frozen, flat evaluation tape produced once by `SymbolicCoeff::compile`.
/// `evaluate` is a single linear scan filling a register array -- no
/// recursion, no tree walk, no allocation beyond that one `Vec<f64>`.
#[derive(Clone, Debug, Default)]
pub struct CompiledCoeff {
    ops: Vec<CompiledOp>,
}

impl CompiledCoeff {
    /// Evaluate against a flat LUT indexed by `2 * param` (`cos`) /
    /// `2 * param + 1` (`sin`). Assumes a *single-root* tape (as produced by
    /// `SymbolicCoeff::compile`) where the last op is the root -- do not
    /// call this on a batched/merged tape from `compile_batch`/
    /// `merge_shards`, which has no single implicit root; use
    /// `evaluate_all` + a term's own root index instead.
    pub fn evaluate(&self, lut: &[f64]) -> f64 {
        if self.ops.is_empty() {
            return 0.0;
        }
        let results = self.evaluate_all(lut);
        results[self.ops.len() - 1]
    }

    /// Evaluate every op in the tape against a flat LUT indexed by
    /// `2 * param` (`cos`) / `2 * param + 1` (`sin`), returning the full
    /// register array. Unlike `evaluate`, this makes no assumption about
    /// which (if any) op is "the" root -- it's the primitive a batched
    /// tape's callers use, reading out whichever indices they need
    /// (`SurrogateTerm::root` per term) after one shared scan.
    ///
    /// Allocates a fresh `Vec` every call -- fine for a single evaluation,
    /// but see `evaluate_into` for a repeated-call site (e.g. one parameter
    /// set per VQE optimizer iteration) that wants to reuse one buffer
    /// across many calls instead.
    pub fn evaluate_all(&self, lut: &[f64]) -> Vec<f64> {
        let mut results = Vec::new();
        self.evaluate_into(lut, &mut results);
        results
    }

    /// Same computation as `evaluate_all`, writing into a caller-provided
    /// buffer instead of allocating a new one every call. `out` is resized
    /// to `self.ops.len()` (growing or truncating as needed); every index is
    /// unconditionally overwritten before any later op can read it (an
    /// invariant of the topologically-ordered tape: an op only ever
    /// references an earlier index), so whatever `out` previously held is
    /// irrelevant to correctness -- this exists purely to let a caller that
    /// evaluates the same tape many times (`SurrogateModel::evaluate_batch`)
    /// reuse one allocation instead of paying a fresh `ops.len()`-sized
    /// allocation on every parameter set.
    pub fn evaluate_into(&self, lut: &[f64], out: &mut Vec<f64>) {
        out.resize(self.ops.len(), 0.0);
        for (i, op) in self.ops.iter().enumerate() {
            out[i] = match *op {
                CompiledOp::Scalar(c) => c,
                CompiledOp::Add(a, b) => out[a] + out[b],
                CompiledOp::Scale(f, inner) => f * out[inner],
                CompiledOp::Cos(p, inner) => lut[2 * p as usize] * out[inner],
                CompiledOp::Sin(p, inner) => lut[2 * p as usize + 1] * out[inner],
            };
        }
    }

    /// Per-op pre-dedup monomial-instance count, mirroring `Node::count`'s
    /// own combine rules (`Add` sums both sides, `Scale`/`Cos`/`Sin` pass
    /// their inner count through unchanged, `Scalar` is a single monomial) --
    /// an upper bound on how many monomials each op's subtree represents,
    /// not a deduplicated tally. Recomputed from the flat tape rather than
    /// carried over from `Node::count` directly, since `compile`/
    /// `compile_batch` deliberately discard the original DAG's cached
    /// fields once flattened. `saturating_add`, matching `Node::add`'s own
    /// fix (a real workload was found pegging this at the `u64` ceiling and
    /// wrapping, corrupting `n_monomials`'s reported value past that point --
    /// see `Node::add`'s doc comment for the full story) -- this is a purely
    /// informational upper bound (never used for evaluate/compile
    /// correctness), but `n_monomials` itself is user-facing and should read
    /// "enormous" rather than an arbitrary wrapped remainder once saturated.
    /// `u128`, matching `Node::count`, so this ceiling is ~3.4e38, not ~1.8e19.
    pub fn monomial_counts(&self) -> Vec<u128> {
        let mut counts = vec![0u128; self.ops.len()];
        for (i, op) in self.ops.iter().enumerate() {
            counts[i] = match *op {
                CompiledOp::Scalar(_) => 1,
                CompiledOp::Add(a, b) => counts[a].saturating_add(counts[b]),
                CompiledOp::Scale(_, inner) => counts[inner],
                CompiledOp::Cos(_, inner) => counts[inner],
                CompiledOp::Sin(_, inner) => counts[inner],
            };
        }
        counts
    }

    /// Number of ops in the compiled tape (test/diagnostic use).
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Concatenate shard-local tapes (as produced by independent
    /// `compile_batch` calls on disjoint shards of terms) into ONE global
    /// tape, rewriting each shard's internal operand indices by that
    /// shard's cumulative offset. Purely arithmetic -- never re-walks any
    /// `Node`/`Arc`, since `CompiledOp` values are already fully resolved.
    /// Returns `(global tape, per-shard base offset)`; a shard's own local
    /// root indices become global via `offset[shard] + local_root` (with
    /// `usize::MAX` passed through unshifted for empty coefficients).
    pub fn merge_shards(shards: Vec<CompiledCoeff>) -> (CompiledCoeff, Vec<usize>) {
        let mut ops: Vec<CompiledOp> = Vec::with_capacity(shards.iter().map(|s| s.ops.len()).sum());
        let mut offsets: Vec<usize> = Vec::with_capacity(shards.len());

        for shard in shards {
            let offset = ops.len();
            offsets.push(offset);
            ops.extend(shard.ops.into_iter().map(|op| shift_op(op, offset)));
        }

        (CompiledCoeff { ops }, offsets)
    }

    /// Serialize the tape (little-endian): op count, then one tagged record
    /// per op. Operand (tape-referencing) indices are written as 8-byte
    /// (`u64`) fields on the wire regardless of the in-memory `usize` width,
    /// for portability; `Cos`/`Sin`'s parameter index stays 4 bytes (`u32`).
    pub fn serialize(&self, buf: &mut Vec<u8>) {
        serialize_ops(&self.ops, buf);
    }

    /// Deserialize a tape written by `serialize`, advancing `pos`.
    pub fn deserialize(b: &[u8], pos: &mut usize) -> Self {
        CompiledCoeff { ops: deserialize_ops(b, pos) }
    }

    /// Split this tape's ops into `n_shards` contiguous slices and run `f`
    /// (e.g. gzip-compression) on each shard's serialized raw bytes, in
    /// parallel via rayon -- the tape-side counterpart to
    /// `SurrogateModel::save`'s existing per-term sharding, for the same
    /// reason: a real large model's tape is big enough that serializing (and
    /// compressing) it as one single-threaded block was the last serial step
    /// in an otherwise fully parallel save pipeline. `f` is applied to each
    /// shard's raw bytes **before** returning, rather than collecting all
    /// shards' raw bytes into a `Vec<Vec<u8>>` first and mapping over that
    /// afterward -- the latter would hold every shard's full uncompressed
    /// bytes simultaneously (on top of `self`'s own already-resident `ops`),
    /// roughly doubling peak memory during `save` for no reason. Splitting a
    /// flat, already-globally-indexed tape needs no reindexing at all --
    /// every operand index already refers to an absolute tape position (set
    /// once, by `merge_shards`), unlike `merge_shards`' own reindexing, which
    /// exists because *that* step combines several *independently* compiled
    /// (locally-indexed) tapes.
    pub fn serialize_shards_with<T: Send>(
        &self,
        n_shards: usize,
        f: impl Fn(&[u8]) -> T + Sync,
    ) -> Vec<T> {
        let chunk = self.ops.len().div_ceil(n_shards.max(1)).max(1);
        self.ops
            .par_chunks(chunk)
            .map(|slice| {
                let mut buf = Vec::new();
                serialize_ops(slice, &mut buf);
                f(&buf)
            })
            .collect()
    }

    /// Concatenate ops from several already-globally-indexed `CompiledCoeff`
    /// pieces back into one tape, in order (e.g. shards each produced by
    /// decompressing + `deserialize`-ing one of `serialize_shards_with`'s
    /// blobs). Unlike `merge_shards`, this does **not** reindex operand
    /// indices -- these pieces are contiguous slices of one already-fully-
    /// indexed tape, not independently-compiled tapes that each started
    /// their own local indexing from zero.
    pub fn concat(shards: Vec<CompiledCoeff>) -> CompiledCoeff {
        let mut ops = Vec::with_capacity(shards.iter().map(|s| s.ops.len()).sum());
        for shard in shards {
            ops.extend(shard.ops);
        }
        CompiledCoeff { ops }
    }
}

/// Write `ops` (little-endian): op count, then one tagged record per op.
/// Shared codec body for `CompiledCoeff::serialize`/`serialize_shards_with`.
fn serialize_ops(ops: &[CompiledOp], buf: &mut Vec<u8>) {
    buf.extend_from_slice(&(ops.len() as u64).to_le_bytes());
    for op in ops {
        match *op {
            CompiledOp::Scalar(c) => {
                buf.push(0);
                buf.extend_from_slice(&c.to_le_bytes());
            }
            CompiledOp::Add(a, b) => {
                buf.push(1);
                buf.extend_from_slice(&(a as u64).to_le_bytes());
                buf.extend_from_slice(&(b as u64).to_le_bytes());
            }
            CompiledOp::Scale(f, i) => {
                buf.push(2);
                buf.extend_from_slice(&f.to_le_bytes());
                buf.extend_from_slice(&(i as u64).to_le_bytes());
            }
            CompiledOp::Cos(p, i) => {
                buf.push(3);
                buf.extend_from_slice(&p.to_le_bytes());
                buf.extend_from_slice(&(i as u64).to_le_bytes());
            }
            CompiledOp::Sin(p, i) => {
                buf.push(4);
                buf.extend_from_slice(&p.to_le_bytes());
                buf.extend_from_slice(&(i as u64).to_le_bytes());
            }
        }
    }
}

/// Read a `Vec<CompiledOp>` written by `serialize_ops`, advancing `pos`.
/// Shared codec body for `CompiledCoeff::deserialize` (per-shard callers
/// decompress a blob then call `deserialize` directly, one shard at a time).
fn deserialize_ops(b: &[u8], pos: &mut usize) -> Vec<CompiledOp> {
    #[inline]
    fn rd_u64(b: &[u8], pos: &mut usize) -> u64 {
        let v = u64::from_le_bytes(b[*pos..*pos + 8].try_into().unwrap());
        *pos += 8;
        v
    }
    #[inline]
    fn rd_idx(b: &[u8], pos: &mut usize) -> usize {
        rd_u64(b, pos) as usize
    }
    #[inline]
    fn rd_u32(b: &[u8], pos: &mut usize) -> u32 {
        let v = u32::from_le_bytes(b[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        v
    }
    #[inline]
    fn rd_f64(b: &[u8], pos: &mut usize) -> f64 {
        let v = f64::from_le_bytes(b[*pos..*pos + 8].try_into().unwrap());
        *pos += 8;
        v
    }

    let n = rd_u64(b, pos) as usize;
    let mut ops = Vec::with_capacity(n);
    for _ in 0..n {
        let tag = b[*pos];
        *pos += 1;
        let op = match tag {
            0 => CompiledOp::Scalar(rd_f64(b, pos)),
            1 => CompiledOp::Add(rd_idx(b, pos), rd_idx(b, pos)),
            2 => CompiledOp::Scale(rd_f64(b, pos), rd_idx(b, pos)),
            3 => CompiledOp::Cos(rd_u32(b, pos), rd_idx(b, pos)),
            4 => CompiledOp::Sin(rd_u32(b, pos), rd_idx(b, pos)),
            _ => panic!("corrupt CompiledCoeff tape: unknown op tag {tag}"),
        };
        ops.push(op);
    }
    ops
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_lut(n_params: usize) -> Vec<f64> {
        (0..n_params)
            .flat_map(|i| {
                let t = 0.37 * (i as f64 + 1.0);
                [t.cos(), t.sin()]
            })
            .collect()
    }

    fn eval(c: &SymbolicCoeff, lut: &[f64]) -> f64 {
        c.compile().evaluate(lut)
    }

    #[test]
    fn from_scalar_compiles_and_evaluates_to_itself() {
        let c = SymbolicCoeff::from_scalar(2.5);
        assert_eq!(c.monomial_count(), 1);
        assert!((eval(&c, &[]) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn count_saturates_instead_of_wrapping_past_u128_max() {
        // Doubling `count` via repeated self-add reaches past `u128::MAX` in
        // ~128 steps -- cheap enough to actually cross the ceiling in a test,
        // unlike reproducing a real multi-billion-monomial workload.
        let mut c = SymbolicCoeff::from_scalar(1.0);
        for _ in 0..135 {
            let other = c.clone();
            c.add_assign(other);
        }
        assert_eq!(
            c.monomial_count(),
            u128::MAX,
            "count must saturate at the ceiling, not wrap around past it"
        );
    }

    #[test]
    fn is_clifford_param_only_flags_a_cos_zero_numeric_angle() {
        use std::f64::consts::{FRAC_PI_2, PI};
        const EPS: f64 = 1e-9;
        // pi/2 (and pi/2 + k*pi): cos is exactly/near zero -> Clifford.
        assert!(SymbolicCoeff::is_clifford_param(&GateParam::Numeric { angle: FRAC_PI_2 }, EPS));
        assert!(SymbolicCoeff::is_clifford_param(&GateParam::Numeric { angle: FRAC_PI_2 + PI }, EPS));
        // A generic numeric angle: cos is nowhere near zero -> not Clifford.
        assert!(!SymbolicCoeff::is_clifford_param(&GateParam::Numeric { angle: 0.3 }, EPS));
        // Symbolic gates never qualify -- the angle is unknown until `evaluate`
        // time, so unconditionally discarding the cos branch would be unsound.
        assert!(!SymbolicCoeff::is_clifford_param(&GateParam::Symbolic { param: 0 }, EPS));
    }

    #[test]
    fn default_is_empty_and_evaluates_to_zero() {
        let c = SymbolicCoeff::default();
        assert!(c.is_empty());
        assert_eq!(c.monomial_count(), 0);
        assert_eq!(eval(&c, &[]), 0.0);
    }

    #[test]
    fn apply_rotation_matches_trig_identity() {
        let lut = make_lut(8);
        let mut c = SymbolicCoeff::from_scalar(0.75);
        for param in [0u32, 1, 2, 5, 7] {
            let before = eval(&c, &lut);
            let sin_branch = c.apply_rotation(&GateParam::symbolic(param), Complex64::new(0.0, -1.0));
            let (cos_t, sin_t) = (lut[(param << 1) as usize], lut[((param << 1) | 1) as usize]);
            assert!((eval(&c, &lut) - cos_t * before).abs() < 1e-12);
            assert!((eval(&sin_branch, &lut) - sin_t * before).abs() < 1e-12);
        }
    }

    #[test]
    fn same_parameter_at_two_gates_collapses_to_a_power() {
        // Two `Cos` wraps of the same parameter evaluate to cos(theta)^2, and
        // `monomial_count` (an upper bound unaffected by non-`Add` wraps)
        // correctly reports 1 throughout, matching the old CSR design's
        // "same param at two gates must merge into one monomial" property --
        // preserved here even though the new representation never explicitly
        // collapses two factors into one "power" the way the old packed
        // factor run did; it's just nested multiplication that evaluates
        // identically.
        let lut = make_lut(1);
        let mut c = SymbolicCoeff::from_scalar(1.0);
        let _ = c.apply_rotation(&GateParam::symbolic(0), Complex64::new(0.0, -1.0));
        let _ = c.apply_rotation(&GateParam::symbolic(0), Complex64::new(0.0, -1.0));
        assert_eq!(c.monomial_count(), 1);
        let expected = lut[0] * lut[0];
        assert!((eval(&c, &lut) - expected).abs() < 1e-12);
    }

    #[test]
    fn two_derivation_paths_through_the_same_parameter_sum_correctly() {
        // Two different derivation orders through the same single parameter
        // both evaluate to cos(theta_0)*sin(theta_0), via genuinely
        // different node shapes (`Cos(Sin(..))` vs `Sin(Cos(..))`) -- unlike
        // the old CSR design, they are NOT automatically collapsed into one
        // shared node here (that cross-path structural collapsing is
        // exactly the eager-dedup work Phase A defers to Phase B's
        // flush-time rebuild); this test only asserts the *value* is
        // correct, both individually and once summed.
        let lut = make_lut(1);
        let phase = Complex64::new(0.0, -1.0);

        // Path 1: take the sin branch first, then the *cos* branch of a
        // second gate on the same parameter.
        let mut a = SymbolicCoeff::from_scalar(1.0);
        let mut path1 = a.apply_rotation(&GateParam::symbolic(0), phase); // = sin(theta_0)
        let _ = path1.apply_rotation(&GateParam::symbolic(0), phase); // self -> cos(theta_0)*sin(theta_0)

        // Path 2: take the cos branch first, then the *sin* branch of a
        // second gate on the same parameter.
        let mut b = SymbolicCoeff::from_scalar(1.0);
        let _ = b.apply_rotation(&GateParam::symbolic(0), phase); // self -> cos(theta_0)
        let path2 = b.apply_rotation(&GateParam::symbolic(0), phase); // returned -> sin(theta_0)*cos(theta_0)

        let single = lut[0] * lut[1]; // cos(theta_0) * sin(theta_0)
        assert!((eval(&path1, &lut) - single).abs() < 1e-12);
        assert!((eval(&path2, &lut) - single).abs() < 1e-12);

        let mut total = SymbolicCoeff::default();
        total.add_assign(path1);
        total.add_assign(path2);
        assert!((eval(&total, &lut) - 2.0 * single).abs() < 1e-12);
    }

    #[test]
    fn simplify_collapses_two_derivation_paths_into_one_monomial() {
        // Same construction as `two_derivation_paths_through_the_same_
        // parameter_sum_correctly` above -- `Cos(Sin(x))` and `Sin(Cos(x))`
        // are genuinely different node shapes evaluating to the same
        // monomial. `simplify` is the mechanism that actually collapses
        // them into one, unlike plain `add_assign`.
        let lut = make_lut(1);
        let phase = Complex64::new(0.0, -1.0);

        let mut a = SymbolicCoeff::from_scalar(1.0);
        let mut path1 = a.apply_rotation(&GateParam::symbolic(0), phase);
        let _ = path1.apply_rotation(&GateParam::symbolic(0), phase);

        let mut b = SymbolicCoeff::from_scalar(1.0);
        let _ = b.apply_rotation(&GateParam::symbolic(0), phase);
        let path2 = b.apply_rotation(&GateParam::symbolic(0), phase);

        let mut total = SymbolicCoeff::default();
        total.add_assign(path1);
        total.add_assign(path2);
        assert_eq!(total.monomial_count(), 2, "pre-simplify: still two separate derivation paths");

        let single = lut[0] * lut[1];
        total.simplify();
        assert_eq!(total.monomial_count(), 1, "simplify must collapse the two paths into one monomial");
        assert!((eval(&total, &lut) - 2.0 * single).abs() < 1e-12, "value must be unchanged by simplify");
    }

    #[test]
    fn simplify_drops_exact_cancellation_to_empty() {
        // Two paths to the identical canonical monomial with opposite-sign
        // equal scalars: real cancellation `prune`'s upper-bound-based
        // check can never see (it never merges derivation paths), but
        // `simplify`'s exact like-term collection does.
        let phase = Complex64::new(0.0, -1.0);
        let mut a = SymbolicCoeff::from_scalar(3.0);
        let _ = a.apply_rotation(&GateParam::symbolic(0), phase); // cos(theta_0) branch, scalar 3.0

        let mut b = SymbolicCoeff::from_scalar(-3.0);
        let _ = b.apply_rotation(&GateParam::symbolic(0), phase); // cos(theta_0) branch, scalar -3.0

        let mut total = SymbolicCoeff::default();
        total.add_assign(a);
        total.add_assign(b);

        let lut = make_lut(1);
        assert!((eval(&total, &lut) - 0.0).abs() < 1e-12, "pre-simplify value should already be zero");

        total.simplify();
        assert!(total.is_empty(), "an exact cancellation must simplify away to nothing");
        assert!((eval(&total, &lut) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn simplify_is_idempotent() {
        let lut = make_lut(1);
        let phase = Complex64::new(0.0, -1.0);
        let mut a = SymbolicCoeff::from_scalar(1.0);
        let mut path1 = a.apply_rotation(&GateParam::symbolic(0), phase);
        let _ = path1.apply_rotation(&GateParam::symbolic(0), phase);
        let mut b = SymbolicCoeff::from_scalar(1.0);
        let _ = b.apply_rotation(&GateParam::symbolic(0), phase);
        let path2 = b.apply_rotation(&GateParam::symbolic(0), phase);

        let mut total = SymbolicCoeff::default();
        total.add_assign(path1);
        total.add_assign(path2);

        total.simplify();
        let v1 = eval(&total, &lut);
        let n1 = total.monomial_count();
        total.simplify();
        let v2 = eval(&total, &lut);
        let n2 = total.monomial_count();

        assert_eq!(n1, n2, "a second simplify pass on an already-simplified DAG must be a true no-op on count");
        assert!((v1 - v2).abs() < 1e-15, "and on value");
    }

    #[test]
    fn simplify_preserves_value_on_a_large_organic_dag() {
        // Same construction as `compile_is_deterministic_and_evaluates_at_
        // scale`: many merged branches mirroring what real propagation's
        // repeated `add_assign` calls produce.
        let n_params = 32usize;
        let lut = make_lut(n_params);
        let mut total = SymbolicCoeff::default();
        let mut expected = 0.0f64;
        for i in 0..500u32 {
            let mut term = SymbolicCoeff::from_scalar(0.1 * (i as f64 + 1.0));
            let p1 = i % n_params as u32;
            let branch = if i % 2 == 0 {
                term.apply_rotation(&GateParam::symbolic(p1), Complex64::new(0.0, -1.0))
            } else {
                term.apply_rotation(&GateParam::Numeric { angle: 0.05 * i as f64 }, Complex64::new(0.0, -1.0))
            };
            let _ = branch;
            expected += eval(&term, &lut);
            total.add_assign(term);
        }
        total.simplify();
        assert!((eval(&total, &lut) - expected).abs() < 1e-8 * expected.abs().max(1.0));
    }

    #[test]
    fn simplify_bounds_monomial_count_under_heavy_parameter_reuse() {
        // Mirrors `propagator.rs`'s `shared_parameter_history_matches_f64_
        // engine_under_fixed_angles` setup in spirit: a handful of distinct
        // parameters, reused across many rounds of branching+merging. The
        // pre-dedup `monomial_count()` grows with round count (real
        // pre-dedup path-instance blowup); the true distinct-monomial count
        // is bounded by (per-param power range), which `simplify` must
        // actually reach.
        const N_PARAMS: u32 = 3;
        const ROUNDS: u32 = 14;
        let phase = Complex64::new(0.0, -1.0);

        let mut base = SymbolicCoeff::from_scalar(1.0);
        for round in 0..ROUNDS {
            let param = round % N_PARAMS;
            let mut branch_a = base.clone();
            let branch_b = branch_a.apply_rotation(&GateParam::symbolic(param), phase);
            let mut merged = SymbolicCoeff::default();
            merged.add_assign(branch_a);
            merged.add_assign(branch_b);
            base = merged;
        }

        let pre = base.monomial_count();
        assert!(pre >= 1 << ROUNDS.min(20), "test setup should exercise real pre-dedup growth: {pre}");

        let lut = make_lut(N_PARAMS as usize);
        let before_val = eval(&base, &lut);

        base.simplify();
        let post = base.monomial_count();

        // True distinct-monomial bound: each of the N_PARAMS parameters can
        // independently reach at most ROUNDS total wraps (cos+sin powers
        // summing to at most the number of rounds that touched it), so the
        // count of distinct (cos_pow, sin_pow) pairs per param is bounded by
        // (ROUNDS/N_PARAMS + 2) (+1 for the 0..=k range, +1 slack), and the
        // monomial count is bounded by the product across params.
        let per_param_bound = (ROUNDS / N_PARAMS + 2) as u128;
        let bound = per_param_bound.pow(N_PARAMS);
        assert!(
            post <= bound,
            "post-simplify count {post} should be polynomially bounded (<= {bound}), not exponential",
        );
        assert!(post * 10 < pre, "simplify should be a large real reduction, not a marginal one: pre={pre} post={post}");

        let after_val = eval(&base, &lut);
        assert!((after_val - before_val).abs() < 1e-6 * before_val.abs().max(1.0));
    }

    #[test]
    fn simplify_sharded_matches_unsharded_on_shared_roots() {
        const N_PARAMS: u32 = 3;
        const ROUNDS: u32 = 10;
        let phase = Complex64::new(0.0, -1.0);

        let build = || {
            let mut base = SymbolicCoeff::from_scalar(1.0);
            for round in 0..ROUNDS {
                let param = round % N_PARAMS;
                let mut branch_a = base.clone();
                let branch_b = branch_a.apply_rotation(&GateParam::symbolic(param), phase);
                let mut merged = SymbolicCoeff::default();
                merged.add_assign(branch_a);
                merged.add_assign(branch_b);
                base = merged;
            }
            base
        };

        let lut = make_lut(N_PARAMS as usize);

        let mut unsharded = vec![build(); 8];
        for c in &mut unsharded {
            c.simplify();
        }

        let mut sharded = vec![build(); 8];
        simplify_sharded(&mut sharded, 8);

        for (a, b) in unsharded.iter().zip(sharded.iter()) {
            assert!((eval(a, &lut) - eval(b, &lut)).abs() < 1e-9, "shard count must not change the evaluated value");
        }
    }

    #[test]
    fn simplify_deep_chain_does_not_overflow_the_stack() {
        // Mirrors `dropping_a_deep_chain_does_not_overflow_the_stack`/
        // `prune_deep_chain_does_not_overflow_the_stack`: a 200,000-deep
        // unbranched chain of distinct-parameter wraps -- the worst case
        // for `FactorRun` growth (one monomial whose factor run grows by
        // one entry per level). This is exactly why `FactorRun` uses a
        // `BTreeMap` (O(log n) insertion), why `Terms` is a plain `Vec`
        // rather than a hashmap (see `Terms`'s doc -- hashing the growing
        // key at every level would itself be O(n^2), independent of any
        // cloning concern), and why `simplify_batch`'s `remaining`-tracked
        // move-not-clone path matters: without all three, this is an
        // `O(n^2)` blowup, not a stack overflow, but just as impractical to
        // complete (measured ~336ms at this size with the fix; the
        // clone-based and hash-based regressions this test guards against
        // were both multi-second-to-minutes at 5,000-10,000 already).
        let mut c = SymbolicCoeff::from_scalar(1.0);
        for p in 0..200_000u32 {
            let _ = c.apply_rotation(&GateParam::symbolic(p), Complex64::new(0.0, -1.0));
        }
        let lut = make_lut(200_000);
        let before = eval(&c, &lut);
        let start = std::time::Instant::now();
        c.simplify();
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(10),
            "simplify() took {elapsed:?} on a 200,000-deep unbranched chain -- \
             suggests FactorRun growth, the move-vs-clone path, or Terms's \
             Vec-not-hashmap design has regressed to quadratic",
        );
        let after = eval(&c, &lut);
        assert!((after - before).abs() < 1e-6 * before.abs().max(1.0));
        assert_eq!(c.monomial_count(), 1, "an unbranched chain is already exactly one monomial");
    }

    #[test]
    fn simplify_batch_shares_memo_across_roots_not_per_row() {
        // A moderately expensive-to-simplify shared structure (3-way
        // branch-then-merge over several rounds -- the same shared-diamond
        // shape `prune_memoizes_shared_subtrees_under_coefficient_cutoff`
        // uses), cloned into many batch entries that all share the
        // *literal same* Arc root. If `simplify_batch` redundantly
        // recomputed each entry's dedup independently instead of sharing
        // one memo across the whole batch, this would take roughly (batch
        // size) times as long as simplifying one instance.
        let mut base = SymbolicCoeff::from_scalar(1.0);
        let mut next_param = 0u32;
        for _round in 0..10 {
            let mut merged = SymbolicCoeff::default();
            for _branch in 0..3u32 {
                let mut b = base.clone();
                let _ = b.apply_rotation(&GateParam::symbolic(next_param), Complex64::new(0.0, -1.0));
                next_param += 1;
                merged.add_assign(b);
            }
            base = merged;
        }

        let start_one = std::time::Instant::now();
        let mut one = vec![base.clone()];
        simplify_batch(&mut one);
        let one_elapsed = start_one.elapsed();

        let start_many = std::time::Instant::now();
        let mut many = vec![base.clone(); 50];
        simplify_batch(&mut many);
        let many_elapsed = start_many.elapsed();

        assert!(
            many_elapsed < one_elapsed * 5 + std::time::Duration::from_millis(50),
            "simplifying 50 batch entries sharing one root took {many_elapsed:?} vs {one_elapsed:?} \
             for one -- suggests the memo isn't actually shared across the batch",
        );

        let lut = make_lut(next_param as usize);
        let expected = eval(&base, &lut);
        for c in &many {
            assert!((eval(c, &lut) - expected).abs() < 1e-6 * expected.abs().max(1.0));
        }
    }

    #[test]
    fn apply_rotation_numeric_matches_trig_identity() {
        let c0 = 0.75;
        let angle = 0.4;
        let phase = Complex64::new(0.0, -1.0);

        let mut c = SymbolicCoeff::from_scalar(c0);
        let sin_branch = c.apply_rotation(&GateParam::Numeric { angle }, phase);

        assert!((eval(&c, &[]) - c0 * angle.cos()).abs() < 1e-12);
        assert!((eval(&sin_branch, &[]) - c0 * angle.sin()).abs() < 1e-12);
    }

    #[test]
    fn apply_rotation_mixed_numeric_then_symbolic_composes_correctly() {
        let c0: f64 = 1.0;
        let angle: f64 = 0.6;
        let param = 3u32;
        let phase = Complex64::new(0.0, -1.0);
        let lut = make_lut(8);
        let (cos_t_sym, sin_t_sym) = (lut[(2 * param) as usize], lut[(2 * param + 1) as usize]);
        let (cos_num, sin_num) = (angle.cos(), angle.sin());

        // Numeric first, then symbolic on both resulting branches.
        let mut cos_branch = SymbolicCoeff::from_scalar(c0);
        let mut sin_branch = cos_branch.apply_rotation(&GateParam::Numeric { angle }, phase);
        let cos_cos = cos_branch.apply_rotation(&GateParam::symbolic(param), phase);
        let sin_cos = sin_branch.apply_rotation(&GateParam::symbolic(param), phase);

        assert!((eval(&cos_branch, &lut) - c0 * cos_num * cos_t_sym).abs() < 1e-12);
        assert!((eval(&cos_cos, &lut) - c0 * cos_num * sin_t_sym).abs() < 1e-12);
        assert!((eval(&sin_branch, &lut) - c0 * sin_num * cos_t_sym).abs() < 1e-12);
        assert!((eval(&sin_cos, &lut) - c0 * sin_num * sin_t_sym).abs() < 1e-12);

        // Symbolic first, then numeric on both resulting branches -- order
        // must not matter.
        let mut cos_branch2 = SymbolicCoeff::from_scalar(c0);
        let mut sin_branch2 = cos_branch2.apply_rotation(&GateParam::symbolic(param), phase);
        let cos_num2 = cos_branch2.apply_rotation(&GateParam::Numeric { angle }, phase);
        let sin_num2 = sin_branch2.apply_rotation(&GateParam::Numeric { angle }, phase);

        assert!((eval(&cos_branch2, &lut) - c0 * cos_t_sym * cos_num).abs() < 1e-12);
        assert!((eval(&cos_num2, &lut) - c0 * cos_t_sym * sin_num).abs() < 1e-12);
        assert!((eval(&sin_branch2, &lut) - c0 * sin_t_sym * cos_num).abs() < 1e-12);
        assert!((eval(&sin_num2, &lut) - c0 * sin_t_sym * sin_num).abs() < 1e-12);
    }

    #[test]
    fn apply_rotation_numeric_scalar_matches_f64_apply_rotation() {
        let c0 = 0.42;
        let angle = 1.1;
        let phase = Complex64::new(0.0, -1.0);

        let mut symbolic = SymbolicCoeff::from_scalar(c0);
        let symbolic_sin = symbolic.apply_rotation(&GateParam::Numeric { angle }, phase);

        // The numeric `CoeffRepr` is `f64`; the symbolic path must agree with it.
        let mut real = c0;
        let real_sin = real.apply_rotation(&angle, phase);

        assert!((eval(&symbolic, &[]) - real).abs() < 1e-12);
        assert!((eval(&symbolic_sin, &[]) - real_sin).abs() < 1e-12);
    }

    #[test]
    fn numeric_and_symbolic_branches_share_the_same_prior_history() {
        // Both branches of a rotation wrap the *same* prior `Arc<Node>` --
        // verified by checking a shared prior history's node is referenced
        // by both compiled tapes without being duplicated (each tape's
        // shared-prefix behavior is exercised more directly by the
        // memoization test below); here we just confirm evaluate agrees
        // regardless of which branch is taken first, i.e. no branch
        // accidentally mutates the shared prior state.
        let mut c = SymbolicCoeff::from_scalar(1.0);
        let _ = c.apply_rotation(&GateParam::symbolic(0), Complex64::new(0.0, -1.0));
        let before = c.clone();
        let _sin = c.apply_rotation(&GateParam::Numeric { angle: 0.7 }, Complex64::new(0.0, -1.0));
        let lut = make_lut(4);
        // `before`'s value must be unaffected by having since produced a
        // sin-branch derivative of `c`.
        assert!((eval(&before, &lut) - lut[0]).abs() < 1e-12);
    }

    #[test]
    fn add_assign_into_default_moves_without_copy() {
        let mut src = SymbolicCoeff::from_scalar(1.0);
        let _ = src.apply_rotation(&GateParam::symbolic(0), Complex64::new(0.0, -1.0));
        let expected = eval(&src, &make_lut(4));

        let mut dst = SymbolicCoeff::default();
        dst.add_assign(src);
        assert!((eval(&dst, &make_lut(4)) - expected).abs() < 1e-15);
    }

    #[test]
    fn add_assign_sums_values() {
        let lut = make_lut(4);
        let mut a = SymbolicCoeff::from_scalar(1.0);
        let _ = a.apply_rotation(&GateParam::symbolic(0), Complex64::new(0.0, -1.0));
        let mut b = SymbolicCoeff::from_scalar(2.0);
        let _ = b.apply_rotation(&GateParam::symbolic(1), Complex64::new(0.0, -1.0));

        let expected = eval(&a, &lut) + eval(&b, &lut);
        a.add_assign(b);
        assert!((eval(&a, &lut) - expected).abs() < 1e-12);
    }

    #[test]
    fn post_merge_default_is_a_harmless_no_op() {
        // Phase A has no eager dedup; `post_merge` falls back to the
        // `CoeffRepr` trait's default no-op. Confirm it's callable and
        // doesn't change the evaluated value.
        let mut c = SymbolicCoeff::from_scalar(3.0);
        let before = eval(&c, &[]);
        c.post_merge();
        assert_eq!(eval(&c, &[]), before);
    }

    #[test]
    fn compile_is_deterministic_and_evaluates_at_scale() {
        // Build a coefficient via many merged branches (mirrors what a real
        // propagation's repeated `add_assign` calls produce) and check
        // `compile()`/`evaluate()` reproduces a hand-computed reference sum.
        let n_params = 32usize;
        let lut = make_lut(n_params);
        let mut total = SymbolicCoeff::default();
        let mut expected = 0.0f64;
        for i in 0..500u32 {
            let mut term = SymbolicCoeff::from_scalar(0.1 * (i as f64 + 1.0));
            let p1 = i % n_params as u32;
            let p2 = (i * 7 + 3) % n_params as u32;
            let branch = if i % 2 == 0 {
                term.apply_rotation(&GateParam::symbolic(p1), Complex64::new(0.0, -1.0))
            } else {
                term.apply_rotation(&GateParam::Numeric { angle: 0.05 * i as f64 }, Complex64::new(0.0, -1.0))
            };
            let _ = branch.compile(); // exercise compiling an intermediate value too
            let _ = p2;
            expected += eval(&term, &lut);
            total.add_assign(term);
        }
        assert!((eval(&total, &lut) - expected).abs() < 1e-8 * expected.abs().max(1.0));
    }

    #[test]
    fn compile_memoizes_shared_subtrees_polynomial_not_exponential() {
        // Build a long shared prefix, then branch several times off the
        // *same* prefix and add the branches together. Without memoization
        // by shared subtree, compiling the sum would recompile the whole
        // prefix once per branch; with it, the prefix appears once in the
        // tape and every branch's tail just references it by index.
        let mut base = SymbolicCoeff::from_scalar(1.0);
        for p in 0..50u32 {
            let _ = base.apply_rotation(&GateParam::symbolic(p), Complex64::new(0.0, -1.0));
        }

        let mut total = SymbolicCoeff::default();
        for p in 50..55u32 {
            let mut b = base.clone();
            let _ = b.apply_rotation(&GateParam::symbolic(p), Complex64::new(0.0, -1.0));
            total.add_assign(b);
        }

        let compiled = total.compile();
        assert!(
            compiled.len() < 5 * 52,
            "compile() should reuse the shared 50-node prefix once, not per branch: {} ops",
            compiled.len(),
        );

        // And the value must still be correct.
        let lut = make_lut(60);
        let base_val = eval(&base, &lut);
        let expected: f64 = (50..55u32).map(|p| base_val * lut[(2 * p) as usize]).sum();
        assert!((compiled.evaluate(&lut) - expected).abs() < 1e-9);
    }

    #[test]
    fn compiled_coeff_serialize_round_trips() {
        let mut c = SymbolicCoeff::from_scalar(1.5);
        let _ = c.apply_rotation(&GateParam::symbolic(2), Complex64::new(0.0, -1.0));
        let sin = c.apply_rotation(&GateParam::Numeric { angle: 0.3 }, Complex64::new(0.0, -1.0));
        c.add_assign(sin);

        let compiled = c.compile();
        let mut buf = Vec::new();
        compiled.serialize(&mut buf);
        let mut pos = 0usize;
        let restored = CompiledCoeff::deserialize(&buf, &mut pos);
        assert_eq!(pos, buf.len());

        let lut = make_lut(8);
        assert!((restored.evaluate(&lut) - compiled.evaluate(&lut)).abs() < 1e-15);
    }

    #[test]
    fn serialize_shards_with_round_trips_and_matches_single_block_serialize() {
        // A batch tape (many roots, deliberate shared prefix) split into
        // several shards for serialization must reassemble (via per-shard
        // `deserialize` + `concat`) into a value-identical tape to what a
        // single-block `serialize`/`deserialize` round trip produces --
        // proving the "no index adjustment needed" claim in
        // `serialize_shards_with`'s doc comment, not just asserting it.
        let mut base = SymbolicCoeff::from_scalar(1.0);
        for p in 0..20u32 {
            let _ = base.apply_rotation(&GateParam::symbolic(p), Complex64::new(0.0, -1.0));
        }
        let branches: Vec<SymbolicCoeff> = (0..10u32)
            .map(|i| {
                let mut b = base.clone();
                let _ = b.apply_rotation(&GateParam::symbolic(20 + i), Complex64::new(0.0, -1.0));
                b
            })
            .collect();
        let (tape, roots) = SymbolicCoeff::compile_batch(branches.clone());

        let mut single_buf = Vec::new();
        tape.serialize(&mut single_buf);
        let mut pos = 0usize;
        let single_restored = CompiledCoeff::deserialize(&single_buf, &mut pos);

        // `f` here just clones the raw bytes (standing in for the caller's
        // real work, e.g. gzip compression in `save`), so the shard pieces
        // can be deserialized directly afterward.
        let shard_bufs: Vec<Vec<u8>> = tape.serialize_shards_with(4, |raw| raw.to_vec());
        assert!(shard_bufs.len() > 1, "test should actually exercise multiple shards");
        let shard_pieces: Vec<CompiledCoeff> = shard_bufs
            .iter()
            .map(|buf| {
                let mut pos = 0usize;
                CompiledCoeff::deserialize(buf, &mut pos)
            })
            .collect();
        let sharded_restored = CompiledCoeff::concat(shard_pieces);

        assert_eq!(single_restored.len(), sharded_restored.len());
        let lut = make_lut(30);
        let single_results = single_restored.evaluate_all(&lut);
        let sharded_results = sharded_restored.evaluate_all(&lut);
        for (branch, &root) in branches.iter().zip(&roots) {
            let expected = eval(branch, &lut);
            assert!((single_results[root] - expected).abs() < 1e-9);
            assert!((sharded_results[root] - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn dropping_a_deep_chain_does_not_overflow_the_stack() {
        // A real circuit's gate-chain can be thousands deep (see `compile`'s
        // own doc comment); the default derived `Drop` would recurse to
        // match, stack-overflowing. Regression guard for that (found via a
        // real, non-toy-sized workload -- a moderate qubit count and gate
        // count segfaulted before `Node` got a custom iterative `Drop`).
        let mut c = SymbolicCoeff::from_scalar(1.0);
        for p in 0..200_000u32 {
            let _ = c.apply_rotation(&GateParam::symbolic(p), Complex64::new(0.0, -1.0));
        }
        drop(c);
    }

    fn root_ptr(c: &SymbolicCoeff) -> *const Node {
        Arc::as_ptr(c.0.as_ref().unwrap())
    }

    #[test]
    fn prune_max_frequency_zero_drops_non_constant_keeps_constant() {
        // A constant (frequency-0) monomial and a frequency-1 monomial
        // summed together; capping frequency at 0 must drop only the latter.
        let mut total = SymbolicCoeff::from_scalar(5.0);
        let mut b = SymbolicCoeff::from_scalar(3.0);
        let _ = b.apply_rotation(&GateParam::symbolic(0), Complex64::new(0.0, -1.0));
        total.add_assign(b);

        let lut = make_lut(1);
        assert!((eval(&total, &lut) - (5.0 + 3.0 * lut[0])).abs() < 1e-12);

        total.prune(Some(0), None);
        assert!((eval(&total, &lut) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn prune_max_frequency_at_true_depth_is_exact_no_op() {
        // Capping frequency at exactly the coefficient's true max depth must
        // change nothing -- verified both by `evaluate` and by confirming
        // the "provably safe" fast path actually fired (same root `Arc`,
        // no rebuild), not just that the rebuilt structure happens to
        // evaluate the same.
        let mut c = SymbolicCoeff::from_scalar(2.0);
        let _ = c.apply_rotation(&GateParam::symbolic(0), Complex64::new(0.0, -1.0));
        let _ = c.apply_rotation(&GateParam::symbolic(1), Complex64::new(0.0, -1.0));

        let lut = make_lut(2);
        let before_val = eval(&c, &lut);
        let before_ptr = root_ptr(&c);

        c.prune(Some(2), None);

        assert!((eval(&c, &lut) - before_val).abs() < 1e-12);
        assert_eq!(root_ptr(&c), before_ptr, "provably-safe fast path should return the original Arc unchanged");
    }

    #[test]
    fn prune_with_no_cutoffs_is_a_true_no_op() {
        let mut c = SymbolicCoeff::from_scalar(1.0);
        let _ = c.apply_rotation(&GateParam::symbolic(0), Complex64::new(0.0, -1.0));
        let before_ptr = root_ptr(&c);
        c.prune(None, None);
        assert_eq!(root_ptr(&c), before_ptr);
    }

    #[test]
    fn prune_hand_built_cross_check_frequency_and_coefficient() {
        // Two hand-built monomials with known frequency and scale magnitude:
        //   m1 = cos(theta_0) * 2.0 * 3.0   -- frequency 1, upper_scale 6.0
        //   m2 = sin(theta_1) * 0.5         -- frequency 1, upper_scale 0.5
        let lut = make_lut(2);
        let (theta0, theta1) = (0.37f64, 0.74f64); // matches `make_lut`'s `0.37*(i+1)`

        let m1 = Node::cos(0, Node::scale(2.0, Node::scalar(3.0)));
        let m2 = Node::sin(1, Node::scalar(0.5));
        let total = SymbolicCoeff(Some(Node::add(m1, m2)));

        let expected_total = 6.0 * theta0.cos() + 0.5 * theta1.sin();
        assert!((eval(&total, &lut) - expected_total).abs() < 1e-12);

        // max_frequency = 0: both monomials are frequency 1, both doomed.
        let mut c = total.clone();
        c.prune(Some(0), None);
        assert!(c.is_empty());

        // max_frequency = 1: exact boundary, no-op.
        let mut c = total.clone();
        c.prune(Some(1), None);
        assert!((eval(&c, &lut) - expected_total).abs() < 1e-12);

        // coeff_cutoff = 1.0: only m2 (upper_scale 0.5 < 1.0) is doomed.
        let mut c = total.clone();
        c.prune(None, Some(1.0));
        assert!((eval(&c, &lut) - 6.0 * theta0.cos()).abs() < 1e-12);

        // coeff_cutoff = 10.0: both doomed (6.0 < 10.0, 0.5 < 10.0).
        let mut c = total.clone();
        c.prune(None, Some(10.0));
        assert!(c.is_empty());
    }

    #[test]
    fn prune_memoizes_shared_subtrees_under_coefficient_cutoff() {
        // A nested-sharing shape: each round branches the *previous* round's
        // (already-merged) result several ways and merges again -- the same
        // 2-parent-diamond-then-merge pattern real propagation produces
        // every gate, repeated across rounds so an unmemoized per-context
        // revisit cost would compound multiplicatively *per round*
        // (genuinely exponential in round count), not just linearly in
        // branch count. All factors here have magnitude exactly 1, so a
        // cutoff far below 1 can never structurally prune anything --
        // forcing every visit to fall through to "ambiguous, recurse" the
        // way a real coefficient-cutoff-active prune commonly would.
        let mut base = SymbolicCoeff::from_scalar(1.0);
        let mut next_param = 0u32;
        for _round in 0..12 {
            let mut merged = SymbolicCoeff::default();
            for _branch in 0..3u32 {
                let mut b = base.clone();
                let _ = b.apply_rotation(&GateParam::symbolic(next_param), Complex64::new(0.0, -1.0));
                next_param += 1;
                merged.add_assign(b);
            }
            base = merged;
        }

        let lut = make_lut(next_param as usize);
        let before = eval(&base, &lut);

        let start = std::time::Instant::now();
        base.prune(None, Some(1e-9));
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "prune() took {elapsed:?} for a 3-way/12-round shared structure -- \
             suggests memoization by (node, scale bucket) isn't sharing repeated visits",
        );

        let after = eval(&base, &lut);
        assert!((after - before).abs() < 1e-6 * before.abs().max(1.0));
    }

    #[test]
    fn prune_deep_chain_does_not_overflow_the_stack() {
        // Mirrors `dropping_a_deep_chain_does_not_overflow_the_stack`, but
        // for `prune`'s own iterative stack discipline (a different frame
        // shape from `Drop`'s and `compile`'s, so it needs its own
        // depth-safety proof). A coefficient-cutoff-only prune has no
        // "provably safe, stop early" fast path (see `prune`'s doc comment),
        // so a cutoff that can never trigger (every factor here has
        // magnitude exactly 1) forces a genuine top-to-bottom walk through
        // all 200,000 levels rather than resolving instantly at the root.
        let mut c = SymbolicCoeff::from_scalar(1.0);
        for p in 0..200_000u32 {
            let _ = c.apply_rotation(&GateParam::symbolic(p), Complex64::new(0.0, -1.0));
        }
        let lut = make_lut(200_000);
        let before = eval(&c, &lut);
        c.prune(None, Some(1e-300));
        let after = eval(&c, &lut);
        assert!((after - before).abs() < 1e-8 * before.abs().max(1.0));
    }

    #[test]
    fn compile_batch_two_rows_sharing_the_same_root_resolve_to_the_same_index() {
        // Two term-rows can legitimately hold the literal same `Arc<Node>`
        // root (e.g. both cloned from one surviving term before diverging
        // elsewhere). The second one is deduped away by `scheduled` and
        // never gets its own `Exit` frame -- root indices must therefore be
        // read from the memo in a final pass, not captured at push time.
        let mut base = SymbolicCoeff::from_scalar(2.0);
        let _ = base.apply_rotation(&GateParam::symbolic(0), Complex64::new(0.0, -1.0));
        let a = base.clone();
        let b = base.clone();

        let (tape, roots) = SymbolicCoeff::compile_batch([a, b]);
        assert_eq!(roots.len(), 2);
        assert_eq!(roots[0], roots[1]);
        assert_ne!(roots[0], usize::MAX);

        let lut = make_lut(1);
        let results = tape.evaluate_all(&lut);
        let expected = eval(&base, &lut);
        assert!((results[roots[0] as usize] - expected).abs() < 1e-12);
    }

    #[test]
    fn compile_batch_memoizes_shared_prefix_across_many_roots_polynomial_not_linear_in_n() {
        // Mirrors `compile_memoizes_shared_subtrees_polynomial_not_exponential`,
        // but across many independently-compiled roots at once (the shape
        // `run_build` now uses per shard): a shared 50-node prefix branched
        // 20 ways must appear once in the batch tape, not once per branch.
        let mut base = SymbolicCoeff::from_scalar(1.0);
        for p in 0..50u32 {
            let _ = base.apply_rotation(&GateParam::symbolic(p), Complex64::new(0.0, -1.0));
        }

        let n_branches = 20u32;
        let branches: Vec<SymbolicCoeff> = (0..n_branches)
            .map(|i| {
                let mut b = base.clone();
                let _ = b.apply_rotation(&GateParam::symbolic(50 + i), Complex64::new(0.0, -1.0));
                b
            })
            .collect();

        let (tape, roots) = SymbolicCoeff::compile_batch(branches.clone());
        assert_eq!(roots.len(), n_branches as usize);
        assert!(
            tape.len() < 5 * (n_branches as usize),
            "compile_batch should reuse the shared 50-node prefix once, not per branch: {} ops",
            tape.len(),
        );

        let lut = make_lut(70);
        let results = tape.evaluate_all(&lut);
        for (branch, &root) in branches.iter().zip(&roots) {
            assert_ne!(root, usize::MAX);
            let expected = eval(branch, &lut);
            assert!((results[root as usize] - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn compile_batch_empty_coefficient_gets_sentinel_root() {
        let a = SymbolicCoeff::default();
        let mut b = SymbolicCoeff::from_scalar(1.0);
        let _ = b.apply_rotation(&GateParam::symbolic(0), Complex64::new(0.0, -1.0));

        let (tape, roots) = SymbolicCoeff::compile_batch([a, b.clone()]);
        assert_eq!(roots[0], usize::MAX);
        assert_ne!(roots[1], usize::MAX);

        let lut = make_lut(1);
        let results = tape.evaluate_all(&lut);
        assert!((results[roots[1] as usize] - eval(&b, &lut)).abs() < 1e-12);
    }

    #[test]
    fn merge_shards_round_trips_values_across_a_shared_boundary_node() {
        // Two shards, each independently compiled via `compile_batch`, where
        // a node with the same *value* (but built independently per shard --
        // shards never share `Arc` identity, only `merge_shards`' pure
        // index-arithmetic ties them together) sits at a shard boundary.
        // Confirms the offset-shift is applied to operand indices only, and
        // that per-shard root indices become correct once shifted.
        let lut = make_lut(4);

        let mut c1 = SymbolicCoeff::from_scalar(3.0);
        let _ = c1.apply_rotation(&GateParam::symbolic(0), Complex64::new(0.0, -1.0));
        let mut c2 = SymbolicCoeff::from_scalar(3.0);
        let _ = c2.apply_rotation(&GateParam::symbolic(0), Complex64::new(0.0, -1.0));
        let _ = c2.apply_rotation(&GateParam::symbolic(1), Complex64::new(0.0, -1.0));

        let mut c3 = SymbolicCoeff::from_scalar(5.0);
        let _ = c3.apply_rotation(&GateParam::symbolic(2), Complex64::new(0.0, -1.0));
        let mut c4 = SymbolicCoeff::from_scalar(7.0);
        let _ = c4.apply_rotation(&GateParam::symbolic(3), Complex64::new(0.0, -1.0));

        let (shard0, roots0) = SymbolicCoeff::compile_batch([c1.clone(), c2.clone()]);
        let (shard1, roots1) = SymbolicCoeff::compile_batch([c3.clone(), c4.clone()]);
        let shard0_len = shard0.len();

        let (merged, offsets) = CompiledCoeff::merge_shards(vec![shard0, shard1]);
        assert_eq!(offsets, vec![0, shard0_len]);

        let global_roots = [
            roots0[0] + offsets[0],
            roots0[1] + offsets[0],
            roots1[0] + offsets[1],
            roots1[1] + offsets[1],
        ];

        let results = merged.evaluate_all(&lut);
        for (root, coeff) in global_roots.iter().zip([&c1, &c2, &c3, &c4]) {
            let expected = eval(coeff, &lut);
            assert!((results[*root as usize] - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn shift_op_handles_offsets_beyond_u32_max() {
        // Regression test for a real-workload panic: a 4.19M-term model's
        // merged tape exceeded u32::MAX total ops, since each term's own
        // largely-unshared derivation tail still scales with term count
        // even though shared-subtree duplication itself is shard-bounded.
        // `shift_op` is exercised directly here (rather than via
        // `merge_shards` on an actually multi-billion-entry `Vec<CompiledOp>`,
        // which would itself exhaust memory in a test) with synthetic
        // offsets/indices comfortably past `u32::MAX` (4_294_967_295), to
        // prove the arithmetic itself no longer overflows now that operand
        // indices are `usize`.
        let big_offset: usize = u32::MAX as usize + 5_000_000_000;

        assert_eq!(shift_op(CompiledOp::Scalar(3.5), big_offset), CompiledOp::Scalar(3.5));
        assert_eq!(
            shift_op(CompiledOp::Add(10, 20), big_offset),
            CompiledOp::Add(10 + big_offset, 20 + big_offset),
        );
        assert_eq!(
            shift_op(CompiledOp::Scale(2.0, 7), big_offset),
            CompiledOp::Scale(2.0, 7 + big_offset),
        );
        assert_eq!(
            shift_op(CompiledOp::Cos(3, 11), big_offset),
            CompiledOp::Cos(3, 11 + big_offset),
        );
        assert_eq!(
            shift_op(CompiledOp::Sin(4, 12), big_offset),
            CompiledOp::Sin(4, 12 + big_offset),
        );
    }
}
