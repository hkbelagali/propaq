///
/// The symbolic coefficient representation.
///
/// A symbolic coefficient is a persistent DAG of nodes:
///
/// ```text
/// Scalar(c)              -- a numeric leaf
/// Add(a, b)               -- a + b
/// Scale(k, a)             -- k * a, k a real constant
/// Cos(p, a)               -- cos(theta_p) * a
/// Sin(p, a)               -- sin(theta_p) * a
/// ```
///
/// built via `Arc` so that wrapping an existing coefficient (every gate
/// application, every merge) is O(1) regardless of how large its prior
/// history already is. Structural sharing across coefficients is
/// automatic.
///
use std::sync::Arc;

use num_complex::Complex64;
use pyo3::prelude::*;
use rayon::prelude::*;
use propaq_core::coeff::CoeffRepr;

/// One node of a coefficient's history.
struct Node {
    kind: NodeKind,
    count: u128,
    min_freq: u32,
    max_freq: u32,
    upper_scale: f64,
}

enum NodeKind {
    Scalar(f64),
    Add(Arc<Node>, Arc<Node>),
    Scale(f64, Arc<Node>),
    Cos(u32, Arc<Node>),
    Sin(u32, Arc<Node>),
}

impl Drop for Node {
    fn drop(&mut self) {
        if matches!(self.kind, NodeKind::Scalar(_)) {
            return;
        }
        let mut stack: Vec<Arc<Node>> = Vec::new();
        match std::mem::replace(&mut self.kind, NodeKind::Scalar(0.0)) {
            NodeKind::Scalar(_) => {}
            NodeKind::Add(a, b) => {
                stack.push(a);
                stack.push(b);
            }
            NodeKind::Scale(_, inner) | NodeKind::Cos(_, inner) | NodeKind::Sin(_, inner) => {
                stack.push(inner);
            }
        }
        while let Some(arc) = stack.pop() {
            if let Ok(mut node) = Arc::try_unwrap(arc) {
                match std::mem::replace(&mut node.kind, NodeKind::Scalar(0.0)) {
                    NodeKind::Scalar(_) => {}
                    NodeKind::Add(a, b) => {
                        stack.push(a);
                        stack.push(b);
                    }
                    NodeKind::Scale(_, inner) | NodeKind::Cos(_, inner) | NodeKind::Sin(_, inner) => {
                        stack.push(inner);
                    }
                }
            }
        }
    }
}

impl Node {
    fn scalar(c: f64) -> Arc<Node> {
        Arc::new(Node { kind: NodeKind::Scalar(c), count: 1, min_freq: 0, max_freq: 0, upper_scale: c.abs() })
    }

    fn add(a: Arc<Node>, b: Arc<Node>) -> Arc<Node> {
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

    pub fn compile(&self) -> CompiledCoeff {
        let (tape, _roots) = SymbolicCoeff::compile_batch(std::iter::once(self.clone()));
        tape
    }

    pub fn compile_batch(
        coeffs: impl IntoIterator<Item = SymbolicCoeff>,
    ) -> (CompiledCoeff, Vec<usize>) {

        let owned: Vec<SymbolicCoeff> = coeffs.into_iter().collect();

        let mut ops: Vec<CompiledOp> = Vec::new();

        let mut memo: rustc_hash::FxHashMap<*const Node, usize> = rustc_hash::FxHashMap::default();

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

    pub fn prune(&mut self, max_frequency: Option<u32>, coeff_cutoff: Option<f64>) {
        if max_frequency.is_none() && coeff_cutoff.is_none() {
            return;
        }
        let Some(root) = self.0.take() else { return };
        self.0 = prune_node(&root, max_frequency, coeff_cutoff);
    }

    fn apply_rotation_symbolic(&mut self, param: u32, phase: Complex64) -> Self {
        let branch_phase = Complex64::new(0.0, 1.0) * phase;
        debug_assert!(branch_phase.im.abs() < 1e-9, "expected real branch phase: {branch_phase:?}");
        let branch_phase = branch_phase.re;

        let old = self.0.take();
        self.0 = old.clone().map(|n| Node::cos(param, n));
        let sin = old.map(|n| signed(branch_phase, Node::sin(param, n)));
        SymbolicCoeff(sin)
    }

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

    pub fn simplify(&mut self) {
        simplify_batch(std::slice::from_mut(self));
    }
}

#[inline]
fn combine_scale_exp(exp: i32, k: f64) -> i32 {
    let k_exp = k.abs().log2().ceil() as i32;
    exp.saturating_add(k_exp)
}

#[inline]
fn is_doomed_by_coeff(scale_exp: i32, upper_scale: f64, cutoff: f64) -> bool {
    if cutoff <= 0.0 {
        return false;
    }
    (scale_exp as f64) + upper_scale.log2() < cutoff.log2()
}

type PruneKey = (*const Node, u32, i32);

#[inline]
fn prune_key(ptr: *const Node, depth: u32, scale_exp: i32, has_freq: bool, has_coeff: bool) -> PruneKey {
    (ptr, if has_freq { depth } else { 0 }, if has_coeff { scale_exp } else { 0 })
}

fn prune_node(root: &Arc<Node>, max_frequency: Option<u32>, coeff_cutoff: Option<f64>) -> Option<Arc<Node>> {
    let has_freq = max_frequency.is_some();
    let has_coeff = coeff_cutoff.is_some();
    let max_freq_cap = max_frequency.unwrap_or(u32::MAX);
    let cutoff = coeff_cutoff.unwrap_or(0.0);

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

                let provably_safe = has_freq && !has_coeff && depth.saturating_add(node.max_freq) <= max_freq_cap;
                if provably_safe || (!has_freq && !has_coeff) {
                    memo.insert(key, Some(Arc::clone(node)));
                    continue;
                }

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

#[derive(Clone, PartialEq, Eq, Hash, Default)]
struct FactorRun(std::collections::BTreeMap<u32, (u32, u32)>);

impl FactorRun {
    fn increment_in_place(&mut self, param: u32, is_sin: bool) {
        let entry = self.0.entry(param).or_insert((0, 0));
        if is_sin { entry.1 += 1; } else { entry.0 += 1; }
    }
}

type Terms = Vec<(FactorRun, f64)>;

fn group(terms: Terms) -> Terms {
    let mut map: rustc_hash::FxHashMap<FactorRun, f64> =
        rustc_hash::FxHashMap::with_capacity_and_hasher(terms.len(), Default::default());
    for (run, scalar) in terms {
        *map.entry(run).or_insert(0.0) += scalar;
    }
    map.into_iter().collect()
}

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

pub fn simplify_sharded(coeffs: &mut [SymbolicCoeff], n_shards: usize) {
    let chunk = coeffs.len().div_ceil(n_shards.max(1)).max(1);
    coeffs.par_chunks_mut(chunk).for_each(|shard| simplify_batch(shard));
}

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

    #[inline]
    fn is_clifford_param(param: &GateParam, eps: f64) -> bool {
        match param {
            GateParam::Symbolic { .. } => false,
            GateParam::Numeric { angle } => angle.cos().abs() < eps,
        }
    }

    #[inline]
    fn size_hint(&self) -> u128 {
        self.monomial_count()
    }

    fn extract_gate_param(obj: &Bound<'_, PyAny>) -> PyResult<GateParam> {
        let param_index: Option<u32> = obj.getattr("param_index")?.extract()?;
        if let Some(param) = param_index {
            return Ok(GateParam::Symbolic { param });
        }
        let angle: f64 = obj.getattr("angle")?.extract()?;
        Ok(GateParam::Numeric { angle })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum CompiledOp {
    Scalar(f64),
    Add(usize, usize),
    Scale(f64, usize),
    Cos(u32, usize),
    Sin(u32, usize),
}

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

#[derive(Clone, Debug, Default)]
pub struct CompiledCoeff {
    ops: Vec<CompiledOp>,
}

impl CompiledCoeff {
    pub fn evaluate(&self, lut: &[f64]) -> f64 {
        if self.ops.is_empty() {
            return 0.0;
        }
        let results = self.evaluate_all(lut);
        results[self.ops.len() - 1]
    }

    pub fn evaluate_all(&self, lut: &[f64]) -> Vec<f64> {
        let mut results = Vec::new();
        self.evaluate_into(lut, &mut results);
        results
    }

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

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

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

    pub fn serialize(&self, buf: &mut Vec<u8>) {
        serialize_ops(&self.ops, buf);
    }

    /// Deserialize a tape written by `serialize`, advancing `pos`.
    pub fn deserialize(b: &[u8], pos: &mut usize) -> Self {
        CompiledCoeff { ops: deserialize_ops(b, pos) }
    }

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

    pub fn concat(shards: Vec<CompiledCoeff>) -> CompiledCoeff {
        let mut ops = Vec::with_capacity(shards.iter().map(|s| s.ops.len()).sum());
        for shard in shards {
            ops.extend(shard.ops);
        }
        CompiledCoeff { ops }
    }
}

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

        assert!(SymbolicCoeff::is_clifford_param(&GateParam::Numeric { angle: FRAC_PI_2 }, EPS));
        assert!(SymbolicCoeff::is_clifford_param(&GateParam::Numeric { angle: FRAC_PI_2 + PI }, EPS));
        assert!(!SymbolicCoeff::is_clifford_param(&GateParam::Numeric { angle: 0.3 }, EPS));
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
        let lut = make_lut(1);
        let phase = Complex64::new(0.0, -1.0);
        let mut a = SymbolicCoeff::from_scalar(1.0);
        let mut path1 = a.apply_rotation(&GateParam::symbolic(0), phase); // = sin(theta_0)
        let _ = path1.apply_rotation(&GateParam::symbolic(0), phase); // self -> cos(theta_0)*sin(theta_0)

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

        // Symbolic first, then numeric on both resulting branches
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
        let mut c = SymbolicCoeff::from_scalar(3.0);
        let before = eval(&c, &[]);
        c.post_merge();
        assert_eq!(eval(&c, &[]), before);
    }

    #[test]
    fn compile_is_deterministic_and_evaluates_at_scale() {
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
        let lut = make_lut(2);
        let (theta0, theta1) = (0.37f64, 0.74f64);
        let m1 = Node::cos(0, Node::scale(2.0, Node::scalar(3.0)));
        let m2 = Node::sin(1, Node::scalar(0.5));
        let total = SymbolicCoeff(Some(Node::add(m1, m2)));

        let expected_total = 6.0 * theta0.cos() + 0.5 * theta1.sin();
        assert!((eval(&total, &lut) - expected_total).abs() < 1e-12);

        let mut c = total.clone();
        c.prune(Some(0), None);
        assert!(c.is_empty());

        let mut c = total.clone();
        c.prune(Some(1), None);
        assert!((eval(&c, &lut) - expected_total).abs() < 1e-12);

        let mut c = total.clone();
        c.prune(None, Some(1.0));
        assert!((eval(&c, &lut) - 6.0 * theta0.cos()).abs() < 1e-12);

        let mut c = total.clone();
        c.prune(None, Some(10.0));
        assert!(c.is_empty());
    }

    #[test]
    fn prune_memoizes_shared_subtrees_under_coefficient_cutoff() {
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
