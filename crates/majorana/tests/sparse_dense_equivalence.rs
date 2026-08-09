///
/// Equivalence checks for the sparse Majorana backend against the retained
/// word-plane algebra it replaced.
///
/// Majorana has no single-word fast path (`local_word` is always `None`), so a
/// randomized circuit here drives the generic sparse `commutes`/`product`
/// kernels end to end. The `Dense` kernel layout decodes each row and calls the
/// word-plane methods, and so stands in as the dense oracle.
///
/// The layout is process-global, so the circuit test serializes on
/// `LAYOUT_LOCK`; this is its own test binary, so nothing else observes the flip.
///
use std::collections::HashMap;
use std::sync::Mutex;

use propaq_core::bitset::Bitset;
use propaq_core::soa::sparse::encode_planes_into;
use propaq_core::soa::{kernels, set_kernel_layout, KernelLayout, Position, SoaBasis, SoaTermSum};
use propaq_majorana::monomial::{MajoranaBasis, MajoranaMonomial};

static LAYOUT_LOCK: Mutex<()> = Mutex::new(());

const N_MODES: usize = 96;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

fn stride() -> usize {
    MajoranaBasis::stride_words(N_MODES)
}

/// Modes drawn from two windows straddling the stride-word boundary: wide
/// enough for multiword keys, narrow enough that monomials overlap and the
/// circuit branches.
fn active_mode(rng: &mut Rng) -> usize {
    let k = rng.below(24) as usize;
    if k < 12 { k } else { 64 + (k - 12) }
}

fn monomial_from_modes(modes: Bitset) -> MajoranaMonomial {
    let (weight, p) = MajoranaMonomial::weight_and_p_for(&modes, N_MODES);
    MajoranaMonomial { modes, n_modes: N_MODES, is_number_preserving: false, weight, p }
}

/// A random monomial with `n_ops` Majorana operators, over the active windows.
fn random_monomial(rng: &mut Rng, n_ops: usize) -> MajoranaMonomial {
    let mut words = vec![0u64; stride()];
    for _ in 0..n_ops {
        let m = active_mode(rng);
        words[m / 64] ^= 1u64 << (m % 64);
    }
    monomial_from_modes(Bitset::from_words(words))
}

/// A random monomial spread over the whole register, for the basis-method
/// differential checks.
fn random_wide_monomial(rng: &mut Rng, n_ops: usize) -> MajoranaMonomial {
    let mut words = vec![0u64; stride()];
    for _ in 0..n_ops {
        let m = rng.below(N_MODES as u64) as usize;
        words[m / 64] ^= 1u64 << (m % 64);
    }
    monomial_from_modes(Bitset::from_words(words))
}

/// A number-preserving monomial: both modes of each occupied site.
fn random_paired_monomial(rng: &mut Rng, n_sites: usize) -> MajoranaMonomial {
    let mut words = vec![0u64; stride()];
    for _ in 0..n_sites {
        let site = active_mode(rng) / 2;
        words[(2 * site) / 64] ^= 1u64 << ((2 * site) % 64);
        words[(2 * site + 1) / 64] ^= 1u64 << ((2 * site + 1) % 64);
    }
    monomial_from_modes(Bitset::from_words(words))
}

fn planes_of(term: &MajoranaMonomial) -> (Vec<u64>, Vec<u64>) {
    let mut modes = vec![0u64; stride()];
    let mut p = vec![0u64; stride()];
    MajoranaBasis::term_into_planes(term, N_MODES, [&mut modes, &mut p]);
    (modes, p)
}

fn sparse_row(planes: [&[u64]; 2]) -> Vec<Position> {
    let mut row = Vec::new();
    encode_planes_into(planes, stride() * 64, &mut row);
    row
}

#[test]
fn sparse_basis_methods_match_the_word_plane_methods() {
    let mut rng = Rng(0xA4093822299F31D0);
    let plane_span = stride() * 64;
    let fock: Vec<u64> = (0..stride()).map(|_| rng.next_u64()).collect();

    for trial in 0..3000 {
        // Mix of arbitrary and number-preserving monomials so both the
        // odd-`single` and even-`single` weight branches are hit.
        let n_term_ops = 1 + rng.below(7) as usize;
        let n_gen_ops = 1 + rng.below(5) as usize;
        let term_mon = if trial % 3 == 0 {
            random_paired_monomial(&mut rng, 1 + n_term_ops % 4)
        } else {
            random_wide_monomial(&mut rng, n_term_ops)
        };
        let gen_mon = random_wide_monomial(&mut rng, n_gen_ops);
        let (tm, tp) = planes_of(&term_mon);
        let (gm, gp) = planes_of(&gen_mon);
        let term = [&tm[..], &tp[..]];
        let gen = [&gm[..], &gp[..]];
        let term_row = sparse_row(term);
        let gen_row = sparse_row(gen);

        assert_eq!(
            MajoranaBasis::weight_sparse(&term_row, plane_span, N_MODES),
            MajoranaBasis::weight(term, N_MODES),
            "weight diverged on trial {trial}"
        );
        assert_eq!(
            MajoranaBasis::trace_sparse(&term_row, plane_span, N_MODES, &fock),
            MajoranaBasis::trace(term, N_MODES, &fock),
            "trace diverged on trial {trial}"
        );
        assert_eq!(
            MajoranaBasis::commutes_sparse(&term_row, &gen_row, plane_span),
            MajoranaBasis::commutes(term, gen),
            "commutation diverged on trial {trial}"
        );
        assert_eq!(
            MajoranaBasis::key_eq_sparse(&term_row, &term_row, plane_span),
            MajoranaBasis::key_eq(term, term),
            "self key equality diverged on trial {trial}"
        );

        let mut got = Vec::new();
        let got_phase = MajoranaBasis::product_sparse(&term_row, &gen_row, plane_span, &mut got);
        let mut wm = vec![0u64; stride()];
        let mut wp = vec![0u64; stride()];
        let want_phase = MajoranaBasis::product(term, gen, [&mut wm, &mut wp]);
        assert_eq!(got_phase, want_phase, "product phase diverged on trial {trial}");
        assert_eq!(got, sparse_row([&wm, &wp]), "product key diverged on trial {trial}");
    }
}

/// Every live term's key (as decoded word planes) mapped to its coefficient.
fn term_values(terms: &SoaTermSum<f64>) -> HashMap<(Vec<u64>, Vec<u64>), f64> {
    let mut buf = vec![0u64; 2 * terms.stride];
    (0..terms.len())
        .map(|i| {
            let planes = terms.decode_row(i, &mut buf);
            ((planes[0].to_vec(), planes[1].to_vec()), *terms.coeff(i))
        })
        .collect()
}

fn run_circuit(seed: u64) -> (HashMap<(Vec<u64>, Vec<u64>), f64>, f64, usize) {
    let s = stride();
    let mut rng = Rng(seed);
    let mut terms = SoaTermSum::<f64>::new(N_MODES, s);

    for _ in 0..3 {
        let (m, p) = planes_of(&random_paired_monomial(&mut rng, 2));
        // Pushed twice so the first merge has a duplicate to fold.
        terms.push([&m, &p], 1.0);
        terms.push([&m, &p], 0.5);
    }

    let fock: Vec<u64> = (0..s).map(|_| rng.next_u64()).collect();

    for step in 0..50u32 {
        let n_gen_ops = 2 + rng.below(2) as usize;
        let gen_mon = random_monomial(&mut rng, n_gen_ops);
        let (gm, gp) = planes_of(&gen_mon);
        match step % 3 {
            0 | 1 => {
                let angle = 0.1 + rng.unit();
                kernels::apply_rotation::<MajoranaBasis, f64>(&mut terms, [&gm, &gp], &angle, false);
            }
            _ => {
                let angle = std::f64::consts::FRAC_PI_2;
                kernels::apply_rotation::<MajoranaBasis, f64>(&mut terms, [&gm, &gp], &angle, true);
            }
        }

        kernels::merge::<MajoranaBasis, f64>(&mut terms);

        if step % 11 == 10 {
            let cfg = propaq_core::truncators::ResolvedConfig {
                weight: Some(10),
                ..Default::default()
            };
            kernels::truncate::<MajoranaBasis, f64>(&mut terms, &cfg);
        }
    }

    kernels::merge::<MajoranaBasis, f64>(&mut terms);
    let expectation = kernels::expectation::<MajoranaBasis, f64>(&terms, &fock);
    (term_values(&terms), expectation, terms.len())
}

#[test]
fn sparse_kernels_match_the_dense_oracle_on_randomized_circuits() {
    let _guard = LAYOUT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    for seed in [0x9E3779B97F4A7C15u64, 0xD1B54A32D192ED03, 0x2545F4914F6CDD1D] {
        set_kernel_layout(KernelLayout::Dense);
        let (want, want_exp, want_len) = run_circuit(seed);
        set_kernel_layout(KernelLayout::Sparse);
        let (got, got_exp, got_len) = run_circuit(seed);
        set_kernel_layout(KernelLayout::Sparse);

        assert!(want_len > 20, "seed {seed:#x}: only {want_len} terms; circuit did not branch enough");
        assert_eq!(got_len, want_len, "seed {seed:#x}: final term count diverged");
        assert_eq!(got.len(), want.len(), "seed {seed:#x}: live key set size diverged");
        for (key, &wanted) in &want {
            let have = got
                .get(key)
                .unwrap_or_else(|| panic!("seed {seed:#x}: key missing from the sparse run"));
            assert!(
                (have - wanted).abs() <= 1e-12 * wanted.abs().max(1.0),
                "seed {seed:#x}: coefficient diverged: sparse={have} dense={wanted}"
            );
        }
        assert!(
            (got_exp - want_exp).abs() <= 1e-10 * want_exp.abs().max(1.0),
            "seed {seed:#x}: expectation diverged: sparse={got_exp} dense={want_exp}"
        );
    }
}
