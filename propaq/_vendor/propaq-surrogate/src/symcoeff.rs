//!
//! The symbolic coefficient representation.
//!
//! A symbolic coefficient is a persistent DAG of nodes:
//!
//! ```text
//! Scalar(c)      a numeric leaf
//! Add(a, b)      a + b
//! Scale(k, a)    k * a, k a real constant
//! Cos(p, a)      cos(theta_p) * a
//! Sin(p, a)      sin(theta_p) * a
//! ```
//!
use std::sync::Arc;

use num_complex::Complex64;
use propaq_core::coeff::CoeffRepr;
use pyo3::prelude::*;
use rayon::prelude::*;

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
                    NodeKind::Scale(_, inner)
                    | NodeKind::Cos(_, inner)
                    | NodeKind::Sin(_, inner) => {
                        stack.push(inner);
                    }
                }
            }
        }
    }
}

impl Node {
    fn scalar(c: f64) -> Arc<Node> {
        Arc::new(Node {
            kind: NodeKind::Scalar(c),
            count: 1,
            min_freq: 0,
            max_freq: 0,
            upper_scale: c.abs(),
        })
    }

    fn add(a: Arc<Node>, b: Arc<Node>) -> Arc<Node> {
        let count = a.count.saturating_add(b.count);
        let min_freq = a.min_freq.min(b.min_freq);
        let max_freq = a.max_freq.max(b.max_freq);
        let upper_scale = a.upper_scale.max(b.upper_scale);
        Arc::new(Node {
            kind: NodeKind::Add(a, b),
            count,
            min_freq,
            max_freq,
            upper_scale,
        })
    }

    fn scale(factor: f64, inner: Arc<Node>) -> Arc<Node> {
        let count = inner.count;
        let (min_freq, max_freq) = (inner.min_freq, inner.max_freq);
        let upper_scale = factor.abs() * inner.upper_scale;
        Arc::new(Node {
            kind: NodeKind::Scale(factor, inner),
            count,
            min_freq,
            max_freq,
            upper_scale,
        })
    }

    fn cos(param: u32, inner: Arc<Node>) -> Arc<Node> {
        let count = inner.count;
        let min_freq = inner.min_freq.saturating_add(1);
        let max_freq = inner.max_freq.saturating_add(1);
        let upper_scale = inner.upper_scale;
        Arc::new(Node {
            kind: NodeKind::Cos(param, inner),
            count,
            min_freq,
            max_freq,
            upper_scale,
        })
    }

    fn sin(param: u32, inner: Arc<Node>) -> Arc<Node> {
        let count = inner.count;
        let min_freq = inner.min_freq.saturating_add(1);
        let max_freq = inner.max_freq.saturating_add(1);
        let upper_scale = inner.upper_scale;
        Arc::new(Node {
            kind: NodeKind::Sin(param, inner),
            count,
            min_freq,
            max_freq,
            upper_scale,
        })
    }
}

#[inline]
fn signed(sign: f64, node: Arc<Node>) -> Arc<Node> {
    if sign == 1.0 {
        node
    } else {
        Node::scale(sign, node)
    }
}

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

    /// True if this coefficient's DAG is empty (the additive identity).
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    /// Compiles this single coefficient into a flat, evaluable `CompiledCoeff` tape.
    /// See `compile_batch` for compiling many coefficients with a shared, deduplicated tape.
    pub fn compile(&self) -> CompiledCoeff {
        let (tape, _roots) = SymbolicCoeff::compile_batch(std::iter::once(self.clone()));
        tape
    }

    /// Compiles many coefficients into one shared `CompiledCoeff` tape via an iterative
    /// post-order DAG walk, deduplicating structurally shared subtrees by `Arc` pointer
    /// identity. Returns the tape together with each input's root index into it.
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
                        NodeKind::Scale(_, inner)
                        | NodeKind::Cos(_, inner)
                        | NodeKind::Sin(_, inner) => {
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
                        NodeKind::Scale(f, inner) => {
                            CompiledOp::Scale(*f, memo[&Arc::as_ptr(inner)])
                        }
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
        debug_assert!(
            branch_phase.im.abs() < 1e-9,
            "expected real branch phase: {branch_phase:?}"
        );
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
        debug_assert!(
            branch_phase.im.abs() < 1e-9,
            "expected real branch phase: {branch_phase:?}"
        );
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
fn prune_key(
    ptr: *const Node,
    depth: u32,
    scale_exp: i32,
    has_freq: bool,
    has_coeff: bool,
) -> PruneKey {
    (
        ptr,
        if has_freq { depth } else { 0 },
        if has_coeff { scale_exp } else { 0 },
    )
}

fn prune_node(
    root: &Arc<Node>,
    max_frequency: Option<u32>,
    coeff_cutoff: Option<f64>,
) -> Option<Arc<Node>> {
    let has_freq = max_frequency.is_some();
    let has_coeff = coeff_cutoff.is_some();
    let max_freq_cap = max_frequency.unwrap_or(u32::MAX);
    let cutoff = coeff_cutoff.unwrap_or(0.0);

    let mut memo: rustc_hash::FxHashMap<PruneKey, Option<Arc<Node>>> =
        rustc_hash::FxHashMap::default();
    let mut scheduled: rustc_hash::FxHashSet<PruneKey> = rustc_hash::FxHashSet::default();

    enum Frame<'a> {
        Enter {
            node: &'a Arc<Node>,
            depth: u32,
            scale_exp: i32,
        },
        Exit {
            node: &'a Arc<Node>,
            depth: u32,
            scale_exp: i32,
        },
    }

    let mut stack: Vec<Frame> = Vec::new();
    let root_key = prune_key(Arc::as_ptr(root), 0, 0, has_freq, has_coeff);
    scheduled.insert(root_key);
    stack.push(Frame::Enter {
        node: root,
        depth: 0,
        scale_exp: 0,
    });

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Enter {
                node,
                depth,
                scale_exp,
            } => {
                let key = prune_key(Arc::as_ptr(node), depth, scale_exp, has_freq, has_coeff);

                let doomed_by_freq = has_freq && depth.saturating_add(node.min_freq) > max_freq_cap;
                let doomed_by_coeff =
                    has_coeff && is_doomed_by_coeff(scale_exp, node.upper_scale, cutoff);
                if doomed_by_freq || doomed_by_coeff {
                    memo.insert(key, None);
                    continue;
                }

                let provably_safe =
                    has_freq && !has_coeff && depth.saturating_add(node.max_freq) <= max_freq_cap;
                if provably_safe || (!has_freq && !has_coeff) {
                    memo.insert(key, Some(Arc::clone(node)));
                    continue;
                }

                stack.push(Frame::Exit {
                    node,
                    depth,
                    scale_exp,
                });
                match &node.kind {
                    NodeKind::Scalar(_) => {}
                    NodeKind::Add(a, b) => {
                        let kb = prune_key(Arc::as_ptr(b), depth, scale_exp, has_freq, has_coeff);
                        let ka = prune_key(Arc::as_ptr(a), depth, scale_exp, has_freq, has_coeff);
                        if scheduled.insert(kb) {
                            stack.push(Frame::Enter {
                                node: b,
                                depth,
                                scale_exp,
                            });
                        }
                        if scheduled.insert(ka) {
                            stack.push(Frame::Enter {
                                node: a,
                                depth,
                                scale_exp,
                            });
                        }
                    }
                    NodeKind::Scale(k, inner) => {
                        let new_scale_exp = if has_coeff {
                            combine_scale_exp(scale_exp, *k)
                        } else {
                            scale_exp
                        };
                        let ki = prune_key(
                            Arc::as_ptr(inner),
                            depth,
                            new_scale_exp,
                            has_freq,
                            has_coeff,
                        );
                        if scheduled.insert(ki) {
                            stack.push(Frame::Enter {
                                node: inner,
                                depth,
                                scale_exp: new_scale_exp,
                            });
                        }
                    }
                    NodeKind::Cos(_, inner) | NodeKind::Sin(_, inner) => {
                        let new_depth = depth.saturating_add(1);
                        let ki = prune_key(
                            Arc::as_ptr(inner),
                            new_depth,
                            scale_exp,
                            has_freq,
                            has_coeff,
                        );
                        if scheduled.insert(ki) {
                            stack.push(Frame::Enter {
                                node: inner,
                                depth: new_depth,
                                scale_exp,
                            });
                        }
                    }
                }
            }
            Frame::Exit {
                node,
                depth,
                scale_exp,
            } => {
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
                        let new_scale_exp = if has_coeff {
                            combine_scale_exp(scale_exp, *k)
                        } else {
                            scale_exp
                        };
                        let ki = prune_key(
                            Arc::as_ptr(inner),
                            depth,
                            new_scale_exp,
                            has_freq,
                            has_coeff,
                        );
                        memo[&ki].clone().map(|x| {
                            if Arc::ptr_eq(&x, inner) {
                                Arc::clone(node)
                            } else {
                                Node::scale(*k, x)
                            }
                        })
                    }
                    NodeKind::Cos(p, inner) => {
                        let new_depth = depth.saturating_add(1);
                        let ki = prune_key(
                            Arc::as_ptr(inner),
                            new_depth,
                            scale_exp,
                            has_freq,
                            has_coeff,
                        );
                        memo[&ki].clone().map(|x| {
                            if Arc::ptr_eq(&x, inner) {
                                Arc::clone(node)
                            } else {
                                Node::cos(*p, x)
                            }
                        })
                    }
                    NodeKind::Sin(p, inner) => {
                        let new_depth = depth.saturating_add(1);
                        let ki = prune_key(
                            Arc::as_ptr(inner),
                            new_depth,
                            scale_exp,
                            has_freq,
                            has_coeff,
                        );
                        memo[&ki].clone().map(|x| {
                            if Arc::ptr_eq(&x, inner) {
                                Arc::clone(node)
                            } else {
                                Node::sin(*p, x)
                            }
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
        if is_sin {
            entry.1 += 1;
        } else {
            entry.0 += 1;
        }
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

    let mut memo: FxHashMap<*const Node, Terms> = FxHashMap::default();
    let mut scheduled: FxHashSet<*const Node> = FxHashSet::default();

    enum Frame<'a> {
        Enter(&'a Arc<Node>),
        Exit(&'a Arc<Node>),
    }

    #[inline]
    fn take_or_clone(
        memo: &mut FxHashMap<*const Node, Terms>,
        remaining: &mut FxHashMap<*const Node, u32>,
        ptr: *const Node,
    ) -> Terms {
        let r = remaining
            .get_mut(&ptr)
            .expect("remaining must already have an entry from pass 0");
        *r -= 1;
        if *r == 0 {
            memo.remove(&ptr)
                .expect("child must already be deduped by post-order exit ordering")
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
                    NodeKind::Scale(_, inner)
                    | NodeKind::Cos(_, inner)
                    | NodeKind::Sin(_, inner) => {
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
        let new_root = rebuilt
            .entry(ptr)
            .or_insert_with(|| rebuild_balanced(&memo[&ptr]))
            .clone();
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
    coeffs.par_chunks_mut(chunk).for_each(simplify_batch);
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GateParam {
    /// A free variational parameter, resolved later against a lookup table of angles.
    Symbolic { param: u32 },
    /// A fixed, concrete rotation angle.
    Numeric { angle: f64 },
}

impl GateParam {
    #[inline]
    pub fn symbolic(x: u32) -> Self {
        GateParam::Symbolic { param: x }
    }
}

impl CoeffRepr for SymbolicCoeff {
    type GateParam = GateParam;

    #[inline]
    fn from_real(c: f64) -> Self {
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
    fn phase_only_scale(param: &GateParam, eps: f64) -> Option<f64> {
        match param {
            GateParam::Symbolic { .. } => None,
            GateParam::Numeric { angle } => {
                let (sin_t, cos_t) = angle.sin_cos();
                (sin_t.abs() < eps).then_some(cos_t)
            }
        }
    }

    #[inline]
    fn clifford_branch_sign(param: &GateParam, phase: Complex64) -> Option<f64> {
        match param {
            GateParam::Symbolic { .. } => None,
            GateParam::Numeric { angle } => Some(angle.sin() * (-phase.im)),
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

    /// Number of ops in the tape.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// True if the tape has no ops.
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

    /// Serializes the tape to a compact binary format, appended to `buf`. Pairs with
    /// `deserialize`.
    pub fn serialize(&self, buf: &mut Vec<u8>) {
        serialize_ops(&self.ops, buf);
    }

    /// Deserialize a tape written by `serialize`, advancing `pos`.
    pub fn deserialize(b: &[u8], pos: &mut usize) -> Self {
        CompiledCoeff {
            ops: deserialize_ops(b, pos),
        }
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
#[path = "../tests/unit/symcoeff.rs"]
mod tests;
