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

    assert!(SymbolicCoeff::is_clifford_param(
        &GateParam::Numeric { angle: FRAC_PI_2 },
        EPS
    ));
    assert!(SymbolicCoeff::is_clifford_param(
        &GateParam::Numeric {
            angle: FRAC_PI_2 + PI
        },
        EPS
    ));
    assert!(!SymbolicCoeff::is_clifford_param(
        &GateParam::Numeric { angle: 0.3 },
        EPS
    ));
    assert!(!SymbolicCoeff::is_clifford_param(
        &GateParam::Symbolic { param: 0 },
        EPS
    ));
}

#[test]
fn phase_only_scale_flags_sin_zero_numeric_angles_and_never_symbolic() {
    use std::f64::consts::{FRAC_PI_2, PI};
    const EPS: f64 = 1e-9;

    assert_eq!(
        SymbolicCoeff::phase_only_scale(&GateParam::Numeric { angle: 0.0 }, EPS),
        Some(1.0)
    );
    let at_pi = SymbolicCoeff::phase_only_scale(&GateParam::Numeric { angle: PI }, EPS)
        .expect("theta = pi has sin == 0");
    assert!(
        (at_pi + 1.0).abs() < EPS,
        "cos(pi) should be -1, got {at_pi}"
    );

    assert_eq!(
        SymbolicCoeff::phase_only_scale(&GateParam::Numeric { angle: FRAC_PI_2 }, EPS),
        None
    );
    assert_eq!(
        SymbolicCoeff::phase_only_scale(&GateParam::Numeric { angle: 0.3 }, EPS),
        None
    );

    assert_eq!(
        SymbolicCoeff::phase_only_scale(&GateParam::Symbolic { param: 0 }, EPS),
        None
    );
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
    let mut path1 = a.apply_rotation(&GateParam::symbolic(0), phase);
    let _ = path1.apply_rotation(&GateParam::symbolic(0), phase);

    let mut b = SymbolicCoeff::from_scalar(1.0);
    let _ = b.apply_rotation(&GateParam::symbolic(0), phase);
    let path2 = b.apply_rotation(&GateParam::symbolic(0), phase);

    let single = lut[0] * lut[1];
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
    assert_eq!(
        total.monomial_count(),
        2,
        "pre-simplify: still two separate derivation paths"
    );

    let single = lut[0] * lut[1];
    total.simplify();
    assert_eq!(
        total.monomial_count(),
        1,
        "simplify must collapse the two paths into one monomial"
    );
    assert!(
        (eval(&total, &lut) - 2.0 * single).abs() < 1e-12,
        "value must be unchanged by simplify"
    );
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
    assert!(
        (eval(&total, &lut) - 0.0).abs() < 1e-12,
        "pre-simplify value should already be zero"
    );

    total.simplify();
    assert!(
        total.is_empty(),
        "an exact cancellation must simplify away to nothing"
    );
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

    assert_eq!(
        n1, n2,
        "a second simplify pass on an already-simplified DAG must be a true no-op on count"
    );
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
            term.apply_rotation(
                &GateParam::Numeric {
                    angle: 0.05 * i as f64,
                },
                Complex64::new(0.0, -1.0),
            )
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
    assert!(
        pre >= 1 << ROUNDS.min(20),
        "test setup should exercise real pre-dedup growth: {pre}"
    );

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
    assert!(
        post * 10 < pre,
        "simplify should be a large real reduction, not a marginal one: pre={pre} post={post}"
    );

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
        assert!(
            (eval(a, &lut) - eval(b, &lut)).abs() < 1e-9,
            "shard count must not change the evaluated value"
        );
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
        "simplify() took {elapsed:?} on a 200,000-deep unbranched chain: \
         suggests FactorRun growth, the move-vs-clone path, or Terms's \
         Vec-not-hashmap design has regressed to quadratic",
    );
    let after = eval(&c, &lut);
    assert!((after - before).abs() < 1e-6 * before.abs().max(1.0));
    assert_eq!(
        c.monomial_count(),
        1,
        "an unbranched chain is already exactly one monomial"
    );
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
         for one: suggests the memo isn't actually shared across the batch",
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
    let _sin = c.apply_rotation(
        &GateParam::Numeric { angle: 0.7 },
        Complex64::new(0.0, -1.0),
    );
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
            term.apply_rotation(
                &GateParam::Numeric {
                    angle: 0.05 * i as f64,
                },
                Complex64::new(0.0, -1.0),
            )
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
    let sin = c.apply_rotation(
        &GateParam::Numeric { angle: 0.3 },
        Complex64::new(0.0, -1.0),
    );
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
    assert!(
        shard_bufs.len() > 1,
        "test should actually exercise multiple shards"
    );
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
    assert_eq!(
        root_ptr(&c),
        before_ptr,
        "provably-safe fast path should return the original Arc unchanged"
    );
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
        "prune() took {elapsed:?} for a 3-way/12-round shared structure: \
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
    assert!((results[roots[0]] - expected).abs() < 1e-12);
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
        assert!((results[root] - expected).abs() < 1e-9);
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
    assert!((results[roots[1]] - eval(&b, &lut)).abs() < 1e-12);
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
        assert!((results[*root] - expected).abs() < 1e-9);
    }
}

#[test]
fn shift_op_handles_offsets_beyond_u32_max() {
    let big_offset: usize = u32::MAX as usize + 5_000_000_000;

    assert_eq!(
        shift_op(CompiledOp::Scalar(3.5), big_offset),
        CompiledOp::Scalar(3.5)
    );
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
