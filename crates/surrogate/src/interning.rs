/// Global interning tables for the support⊗exponent factored coefficient
/// representation.
///
/// A monomial's factor run `[param:16 | cos_pow:8 | sin_pow:8]*` is split into
/// two independently-shared parts:
///
/// - its **support**: the ascending sequence of parameter indices it touched,
///   interned into a hash-consed [`SupportTrie`] (a prefix of one run is a prefix
///   path in the trie, stored once);
/// - its **exponent pattern**: the positionally-aligned `(cos_pow, sin_pow)`
///   list, interned into an [`ExponentDict`].
///
/// A [`Generation`] bundles the two. During a flush window the propagator holds
/// one **frozen** generation and only *decodes* against it (lock-free reads); at
/// each flush a fresh generation is built by re-interning the live survivors, so
/// truncation-removed structure is garbage-collected for free (no refcounting).
///

use rustc_hash::FxHashMap;

use crate::symcoeff::{factor_cos, factor_param, factor_sin, make_factor};

/// A `(cos_pow, sin_pow)` pair packed into one `u16` (`cos` high byte, `sin`
/// low), the storage unit of an interned exponent pattern.
#[inline]
fn pack_exp(cos_pow: u32, sin_pow: u32) -> u16 {
    ((cos_pow as u16) << 8) | (sin_pow as u16)
}

#[inline]
fn exp_cos(w: u16) -> u32 {
    (w >> 8) as u32
}

#[inline]
fn exp_sin(w: u16) -> u32 {
    (w & 0xff) as u32
}

// Little-endian cursor readers for `Generation::deserialize`.
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
fn rd_u16(b: &[u8], pos: &mut usize) -> u16 {
    let v = u16::from_le_bytes(b[*pos..*pos + 2].try_into().unwrap());
    *pos += 2;
    v
}

/// One trie node: `param` appended to `parent`. Node id `0` is the root (the
/// empty support). Params are strictly ascending root→leaf, matching the
/// canonical run ordering, so a shared prefix is a shared path.
#[derive(Clone, Copy)]
struct SupportNode {
    parent: u32,
    param: u32,
    /// Number of params on the path root→here (root depth 0), so a support's
    /// length is `O(1)`.
    depth: u32,
}

/// Hash-consed trie over ascending parameter sequences.
pub struct SupportTrie {
    /// `nodes[0]` is the root sentinel (empty support).
    nodes: Vec<SupportNode>,
    /// `(parent_id, param) -> child_id`, so appending the same param to the same
    /// parent always returns the same node.
    unique: FxHashMap<(u32, u32), u32>,
}

impl Default for SupportTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl SupportTrie {
    pub fn new() -> Self {
        SupportTrie {
            nodes: vec![SupportNode { parent: 0, param: 0, depth: 0 }],
            unique: FxHashMap::default(),
        }
    }

    /// Number of nodes including the root sentinel.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Length (param count) of the support identified by `id`.
    #[inline]
    pub fn depth(&self, id: u32) -> usize {
        self.nodes[id as usize].depth as usize
    }

    /// Append `param` to the support `parent`, returning the child node id
    /// (shared if this edge already exists).
    #[inline]
    pub fn intern_append(&mut self, parent: u32, param: u32) -> u32 {
        if let Some(&id) = self.unique.get(&(parent, param)) {
            return id;
        }
        let depth = self.nodes[parent as usize].depth + 1;
        let id = self.nodes.len() as u32;
        self.nodes.push(SupportNode { parent, param, depth });
        self.unique.insert((parent, param), id);
        id
    }

    /// Intern a whole ascending param sequence, returning its leaf id.
    pub fn intern_params(&mut self, params: &[u32]) -> u32 {
        let mut cur = 0u32;
        for &p in params {
            cur = self.intern_append(cur, p);
        }
        cur
    }

    /// Append the ascending params of support `id` to `out`.
    pub fn decode_into(&self, id: u32, out: &mut Vec<u32>) {
        let start = out.len();
        let mut cur = id;
        while cur != 0 {
            let node = self.nodes[cur as usize];
            out.push(node.param);
            cur = node.parent;
        }
        // Walked leaf→root (descending); restore ascending order.
        out[start..].reverse();
    }
}

/// Interns whole exponent patterns (the positional `(cos, sin)` power list) to
/// dense ids, with a cached total frequency per pattern.
pub struct ExponentDict {
    /// Flat concatenation of every pattern's packed `u16` words.
    arena: Vec<u16>,
    /// `offs[id]..offs[id + 1]` is pattern `id`'s slice in `arena`; length is
    /// `patterns + 1`.
    offs: Vec<u32>,
    /// Cached `Σ (cos_pow + sin_pow)` per pattern (the monomial frequency).
    freq: Vec<u32>,
    /// Pattern bytes → id.
    unique: FxHashMap<Box<[u16]>, u32>,
}

impl Default for ExponentDict {
    fn default() -> Self {
        Self::new()
    }
}

impl ExponentDict {
    pub fn new() -> Self {
        let mut unique = FxHashMap::default();
        // Id 0 is the empty pattern.
        unique.insert(Box::<[u16]>::from(&[][..]), 0u32);
        ExponentDict {
            arena: Vec::new(),
            offs: vec![0, 0],
            freq: vec![0],
            unique,
        }
    }

    /// Number of distinct patterns (including the empty pattern id 0).
    pub fn pattern_count(&self) -> usize {
        self.freq.len()
    }

    /// Cached frequency (`Σ` powers) of pattern `id`.
    #[inline]
    pub fn freq(&self, id: u32) -> u32 {
        self.freq[id as usize]
    }

    /// Pattern `id`'s packed words.
    #[inline]
    fn pattern(&self, id: u32) -> &[u16] {
        let s = self.offs[id as usize] as usize;
        let e = self.offs[id as usize + 1] as usize;
        &self.arena[s..e]
    }

    /// Intern a packed exponent pattern, returning its id (shared if seen).
    fn intern(&mut self, pattern: &[u16]) -> u32 {
        if let Some(&id) = self.unique.get(pattern) {
            return id;
        }
        let id = (self.offs.len() - 1) as u32;
        self.arena.extend_from_slice(pattern);
        self.offs.push(self.arena.len() as u32);
        let f: u32 = pattern.iter().map(|&w| exp_cos(w) + exp_sin(w)).sum();
        self.freq.push(f);
        self.unique.insert(pattern.into(), id);
        id
    }
}

/// One generation of the two interning tables. Frozen during a flush window,
/// rebuilt at each flush.
#[derive(Default)]
pub struct Generation {
    pub support: SupportTrie,
    pub exp: ExponentDict,
    /// Scratch buffers reused across `intern_run`/`decode` to avoid per-call
    /// allocation during a reconcile pass.
    params_scratch: Vec<u32>,
    exps_scratch: Vec<u16>,
}

impl Generation {
    pub fn new() -> Self {
        Generation::default()
    }

    /// Intern a canonical packed factor run, returning `(support_id, exp_id,
    /// frequency)`.
    pub fn intern_run(&mut self, run: &[u32]) -> (u32, u32, u32) {
        self.params_scratch.clear();
        self.exps_scratch.clear();
        for &f in run {
            self.params_scratch.push(factor_param(f));
            self.exps_scratch.push(pack_exp(factor_cos(f), factor_sin(f)));
        }
        let sid = self.support.intern_params(&self.params_scratch);
        let eid = self.exp.intern(&self.exps_scratch);
        let freq = self.exp.freq(eid);
        (sid, eid, freq)
    }

    /// Length (param count) of the run identified by `(support_id, exp_id)`.
    #[inline]
    pub fn run_len(&self, support_id: u32) -> usize {
        self.support.depth(support_id)
    }

    /// Cached frequency of the run identified by `exp_id`.
    #[inline]
    pub fn run_freq(&self, exp_id: u32) -> u32 {
        self.exp.freq(exp_id)
    }

    /// Append the decoded packed factor run for `(support_id, exp_id)` to `out`.
    pub fn decode_into(&self, support_id: u32, exp_id: u32, out: &mut Vec<u32>) {
        let start = out.len();
        self.support.decode_into(support_id, out);
        let exps = self.exp.pattern(exp_id);
        debug_assert_eq!(
            out.len() - start,
            exps.len(),
            "support and exponent pattern length mismatch"
        );
        for (slot, &w) in out[start..].iter_mut().zip(exps) {
            let param = *slot;
            *slot = make_factor(param, exp_cos(w), exp_sin(w));
        }
    }

    /// Convenience decode into a fresh `Vec` (tests / low-frequency callers).
    pub fn decode_run(&self, support_id: u32, exp_id: u32) -> Vec<u32> {
        let mut out = Vec::new();
        self.decode_into(support_id, exp_id, &mut out);
        out
    }

    /// Serialize the two tables into `out` (little-endian): support nodes as
    /// `(parent, param)` pairs, then exponent patterns as length-prefixed `u16`
    /// runs. `depth` (support) and `freq` (exponents) are recomputed on load, and
    /// the unique tables are not persisted (a loaded generation only decodes).
    pub fn serialize(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.support.nodes.len() as u64).to_le_bytes());
        for n in &self.support.nodes {
            out.extend_from_slice(&n.parent.to_le_bytes());
            out.extend_from_slice(&n.param.to_le_bytes());
        }
        let n_patterns = self.exp.offs.len() - 1;
        out.extend_from_slice(&(n_patterns as u64).to_le_bytes());
        for id in 0..n_patterns {
            let s = self.exp.offs[id] as usize;
            let e = self.exp.offs[id + 1] as usize;
            let pat = &self.exp.arena[s..e];
            out.extend_from_slice(&(pat.len() as u32).to_le_bytes());
            for &w in pat {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
    }

    /// Reconstruct a generation from bytes written by `serialize`, advancing
    /// `pos`. The result decodes correctly but its unique tables are empty, so it
    /// must not be interned into again (loaded models only evaluate).
    pub fn deserialize(b: &[u8], pos: &mut usize) -> Self {
        let n_nodes = rd_u64(b, pos) as usize;
        let mut nodes: Vec<SupportNode> = Vec::with_capacity(n_nodes);
        for _ in 0..n_nodes {
            let parent = rd_u32(b, pos);
            let param = rd_u32(b, pos);
            let depth = if nodes.is_empty() { 0 } else { nodes[parent as usize].depth + 1 };
            nodes.push(SupportNode { parent, param, depth });
        }
        let support = SupportTrie { nodes, unique: FxHashMap::default() };

        let n_patterns = rd_u64(b, pos) as usize;
        let mut arena: Vec<u16> = Vec::new();
        let mut offs: Vec<u32> = Vec::with_capacity(n_patterns + 1);
        offs.push(0);
        let mut freq: Vec<u32> = Vec::with_capacity(n_patterns);
        for _ in 0..n_patterns {
            let len = rd_u32(b, pos) as usize;
            let mut f = 0u32;
            for _ in 0..len {
                let w = rd_u16(b, pos);
                arena.push(w);
                f += exp_cos(w) + exp_sin(w);
            }
            offs.push(arena.len() as u32);
            freq.push(f);
        }
        let exp = ExponentDict { arena, offs, freq, unique: FxHashMap::default() };

        Generation { support, exp, params_scratch: Vec::new(), exps_scratch: Vec::new() }
    }

    /// Product of the base run `(support_id, exp_id)`'s trig factors against
    /// `lut` (indexed `2*param` = cos, `2*param + 1` = sin), computed by walking
    /// the trie path without materializing the run. `1.0` for the empty base.
    pub fn base_product(&self, support_id: u32, exp_id: u32, lut: &[f64]) -> f64 {
        let exps = self.exp.pattern(exp_id);
        debug_assert_eq!(self.support.depth(support_id), exps.len());
        let mut prod = 1.0f64;
        let mut cur = support_id;
        // Walk leaf→root; positions run high→low. Product commutes, so this
        // pairs each param with its exponent by index regardless of order.
        let mut idx = exps.len();
        while cur != 0 {
            idx -= 1;
            let node = self.support.nodes[cur as usize];
            let w = exps[idx];
            let p = node.param as usize;
            let cos_pow = exp_cos(w) as i32;
            let sin_pow = exp_sin(w) as i32;
            if cos_pow > 0 {
                prod *= lut[2 * p].powi(cos_pow);
            }
            if sin_pow > 0 {
                prod *= lut[2 * p + 1].powi(sin_pow);
            }
            cur = node.parent;
        }
        prod
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symcoeff::make_factor;

    /// Build a canonical packed run from `(param, cos, sin)` triples.
    fn run(triples: &[(u32, u32, u32)]) -> Vec<u32> {
        triples.iter().map(|&(p, c, s)| make_factor(p, c, s)).collect()
    }

    #[test]
    fn support_trie_shares_identical_and_prefix_paths() {
        let mut t = SupportTrie::new();
        let a = t.intern_params(&[0, 3, 7]);
        let b = t.intern_params(&[0, 3, 7]);
        assert_eq!(a, b, "identical param sequences intern to the same id");

        let before = t.node_count();
        let _c = t.intern_params(&[0, 3]); // strict prefix, no new nodes
        assert_eq!(t.node_count(), before, "a prefix reuses existing path nodes");

        let _d = t.intern_params(&[0, 5]); // diverges after [0], one new node
        assert_eq!(t.node_count(), before + 1);
    }

    #[test]
    fn support_trie_decode_round_trips_ascending() {
        let mut t = SupportTrie::new();
        let id = t.intern_params(&[1, 4, 9]);
        let mut out = Vec::new();
        t.decode_into(id, &mut out);
        assert_eq!(out, vec![1, 4, 9]);
        assert_eq!(t.depth(id), 3);
        assert_eq!(t.depth(0), 0, "root is the empty support");
    }

    #[test]
    fn exponent_dict_shares_and_caches_freq() {
        let mut d = ExponentDict::new();
        let p1 = d.intern(&[pack_exp(2, 0), pack_exp(1, 1)]);
        let p2 = d.intern(&[pack_exp(2, 0), pack_exp(1, 1)]);
        assert_eq!(p1, p2);
        assert_eq!(d.freq(p1), 2 + 1 + 1);
        assert_eq!(d.freq(0), 0, "empty pattern has zero frequency");

        let p3 = d.intern(&[pack_exp(1, 0)]);
        assert_ne!(p1, p3);
        assert_eq!(d.freq(p3), 1);
    }

    #[test]
    fn generation_run_round_trips() {
        let mut g = Generation::new();
        let r = run(&[(0, 1, 0), (3, 0, 1), (7, 2, 1)]);
        let (sid, eid, freq) = g.intern_run(&r);
        assert_eq!(freq, 1 + 1 + 3);
        assert_eq!(g.run_len(sid), 3);
        assert_eq!(g.decode_run(sid, eid), r);
    }

    #[test]
    fn empty_run_maps_to_root_ids() {
        let mut g = Generation::new();
        let (sid, eid, freq) = g.intern_run(&[]);
        assert_eq!((sid, eid, freq), (0, 0, 0));
        assert!(g.decode_run(sid, eid).is_empty());
    }

    #[test]
    fn outer_product_factoring_shares_axes_independently() {
        let mut g = Generation::new();
        // Same support, different exponents: share support id, differ in exp id.
        let (s1, e1, _) = g.intern_run(&run(&[(0, 1, 0), (3, 1, 0)]));
        let (s2, e2, _) = g.intern_run(&run(&[(0, 2, 0), (3, 0, 1)]));
        assert_eq!(s1, s2, "same param support shares one trie leaf");
        assert_ne!(e1, e2, "different exponents are distinct patterns");

        // Same exponents, different support: share exp id, differ in support id.
        let (s3, e3, _) = g.intern_run(&run(&[(1, 1, 0), (4, 1, 0)]));
        let (s4, e4, _) = g.intern_run(&run(&[(2, 1, 0), (5, 1, 0)]));
        assert_eq!(e3, e4, "same exponent pattern shares one dict entry");
        assert_ne!(s3, s4, "different supports are distinct trie leaves");
    }

    #[test]
    fn serialize_round_trips_and_preserves_decoding() {
        let mut g = Generation::new();
        let runs = [
            run(&[(0, 1, 0)]),
            run(&[(0, 1, 0), (2, 1, 1)]),
            run(&[(0, 2, 0), (2, 1, 1)]),
            run(&[(5, 3, 0)]),
            run(&[]),
        ];
        let ids: Vec<(u32, u32)> = runs.iter().map(|r| { let (s, e, _) = g.intern_run(r); (s, e) }).collect();

        let mut bytes = Vec::new();
        g.serialize(&mut bytes);
        let mut pos = 0usize;
        let g2 = Generation::deserialize(&bytes, &mut pos);
        assert_eq!(pos, bytes.len(), "deserialize consumed the whole blob");

        for (r, &(s, e)) in runs.iter().zip(&ids) {
            assert_eq!(&g2.decode_run(s, e), r, "decoding preserved after round-trip");
            assert_eq!(g2.run_freq(e), g.run_freq(e), "frequency recomputed on load");
        }
        let lut = make_lut(8);
        for &(s, e) in &ids {
            assert!((g2.base_product(s, e, &lut) - g.base_product(s, e, &lut)).abs() < 1e-12);
        }
    }

    fn make_lut(n_params: usize) -> Vec<f64> {
        (0..n_params).flat_map(|i| { let t = 0.31 * (i as f64 + 1.0); [t.cos(), t.sin()] }).collect()
    }

    #[test]
    fn many_runs_reconstruct_exactly() {
        let mut g = Generation::new();
        let runs = [
            run(&[]),
            run(&[(0, 1, 0)]),
            run(&[(0, 1, 0), (2, 1, 1)]),
            run(&[(0, 2, 0), (2, 1, 1)]),
            run(&[(5, 3, 0)]),
            run(&[(0, 1, 0), (2, 1, 1), (5, 0, 2)]),
        ];
        let ids: Vec<(u32, u32)> = runs.iter().map(|r| {
            let (s, e, _) = g.intern_run(r);
            (s, e)
        }).collect();
        for (r, &(s, e)) in runs.iter().zip(&ids) {
            assert_eq!(&g.decode_run(s, e), r);
        }
    }
}
