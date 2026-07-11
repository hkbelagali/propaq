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
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use num_complex::Complex64;
use pyo3::prelude::*;
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
    count: u64,
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
        let count = a.count + b.count;
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
    pub fn monomial_count(&self) -> usize {
        self.0.as_ref().map_or(0, |n| n.count as usize)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    /// Flatten this coefficient's DAG into a `CompiledCoeff`: a linear,
    /// topologically-ordered op tape where every node is emitted exactly
    /// once (memoized by `Arc` pointer identity) regardless of how many
    /// parents reference it, and every op only references earlier indices.
    /// `CompiledCoeff::evaluate` is then a single linear scan -- this is
    /// what keeps compilation polynomial (not exponential) in the presence
    /// of shared subtrees, and is meant to run once per term at build end,
    /// not per gate.
    ///
    /// Iterative (explicit stack), not recursive: a coefficient's DAG can be
    /// as deep as the number of gates that touched it, which for a real
    /// circuit can be in the thousands -- recursion would risk a stack
    /// overflow that an explicit stack doesn't.
    pub fn compile(&self) -> CompiledCoeff {
        let mut ops: Vec<CompiledOp> = Vec::new();
        let Some(root) = &self.0 else { return CompiledCoeff { ops } };

        let mut memo: HashMap<*const Node, u32> = HashMap::new();
        // Tracks every node that has already been pushed onto the work stack
        // (whether or not its `Exit` has run yet), so a node referenced by
        // more than one parent is only ever traversed/compiled once. Without
        // this, two `Enter` frames for the same shared node could both land
        // on the stack before either's subtree finishes, each redundantly
        // re-walking (though not incorrectly -- just wastefully) that whole
        // subtree.
        let mut scheduled: HashSet<*const Node> = HashSet::new();

        enum Frame<'a> {
            Enter(&'a Arc<Node>),
            Exit(&'a Arc<Node>),
        }

        let mut stack: Vec<Frame> = Vec::new();
        scheduled.insert(Arc::as_ptr(root));
        stack.push(Frame::Enter(root));

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
                    memo.insert(Arc::as_ptr(node), (ops.len() - 1) as u32);
                }
            }
        }

        CompiledCoeff { ops }
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

    let mut memo: HashMap<PruneKey, Option<Arc<Node>>> = HashMap::new();
    let mut scheduled: HashSet<PruneKey> = HashSet::new();

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

    /// Monomial count is what actually drives memory/CPU cost for symbolic
    /// coefficients, unlike raw term count.
    #[inline]
    fn size_hint(&self) -> usize {
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

/// One flattened operation in a `CompiledCoeff`'s tape. Indices always refer
/// to earlier positions in the same tape (the tape is topologically ordered
/// by construction -- see `SymbolicCoeff::compile`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum CompiledOp {
    Scalar(f64),
    Add(u32, u32),
    Scale(f64, u32),
    Cos(u32, u32),
    Sin(u32, u32),
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
    /// `2 * param + 1` (`sin`).
    pub fn evaluate(&self, lut: &[f64]) -> f64 {
        if self.ops.is_empty() {
            return 0.0;
        }
        let mut results = vec![0.0f64; self.ops.len()];
        for (i, op) in self.ops.iter().enumerate() {
            results[i] = match *op {
                CompiledOp::Scalar(c) => c,
                CompiledOp::Add(a, b) => results[a as usize] + results[b as usize],
                CompiledOp::Scale(f, inner) => f * results[inner as usize],
                CompiledOp::Cos(p, inner) => lut[2 * p as usize] * results[inner as usize],
                CompiledOp::Sin(p, inner) => lut[2 * p as usize + 1] * results[inner as usize],
            };
        }
        results[self.ops.len() - 1]
    }

    /// Number of ops in the compiled tape (test/diagnostic use).
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Serialize the tape (little-endian): op count, then one tagged record
    /// per op.
    pub fn serialize(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&(self.ops.len() as u64).to_le_bytes());
        for op in &self.ops {
            match *op {
                CompiledOp::Scalar(c) => {
                    buf.push(0);
                    buf.extend_from_slice(&c.to_le_bytes());
                }
                CompiledOp::Add(a, b) => {
                    buf.push(1);
                    buf.extend_from_slice(&a.to_le_bytes());
                    buf.extend_from_slice(&b.to_le_bytes());
                }
                CompiledOp::Scale(f, i) => {
                    buf.push(2);
                    buf.extend_from_slice(&f.to_le_bytes());
                    buf.extend_from_slice(&i.to_le_bytes());
                }
                CompiledOp::Cos(p, i) => {
                    buf.push(3);
                    buf.extend_from_slice(&p.to_le_bytes());
                    buf.extend_from_slice(&i.to_le_bytes());
                }
                CompiledOp::Sin(p, i) => {
                    buf.push(4);
                    buf.extend_from_slice(&p.to_le_bytes());
                    buf.extend_from_slice(&i.to_le_bytes());
                }
            }
        }
    }

    /// Deserialize a tape written by `serialize`, advancing `pos`.
    pub fn deserialize(b: &[u8], pos: &mut usize) -> Self {
        #[inline]
        fn rd_u64(b: &[u8], pos: &mut usize) -> u64 {
            let v = u64::from_le_bytes(b[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            v
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
                1 => CompiledOp::Add(rd_u32(b, pos), rd_u32(b, pos)),
                2 => CompiledOp::Scale(rd_f64(b, pos), rd_u32(b, pos)),
                3 => CompiledOp::Cos(rd_u32(b, pos), rd_u32(b, pos)),
                4 => CompiledOp::Sin(rd_u32(b, pos), rd_u32(b, pos)),
                _ => panic!("corrupt CompiledCoeff tape: unknown op tag {tag}"),
            };
            ops.push(op);
        }
        CompiledCoeff { ops }
    }
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
}
