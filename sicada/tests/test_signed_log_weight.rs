//! The signed log semiring, whose point is that two paths can cancel.

use sicada::algorithms::shortest_distance::{SHORTEST_DELTA, shortest_distance};
use sicada::arc::{Arc, SignedLogArc};
use sicada::fst::MutableFst;
use sicada::fsts::vector_fst::VectorFst;
use sicada::weight::Weight;
use sicada::weights::signed_log_weight::SignedLogWeight;

/// A probability of 0.5, carried as a sign and a negative log.
fn half(sign: f32) -> SignedLogWeight {
    SignedLogWeight::new(sign, std::f32::consts::LN_2)
}

/// `+0.5 ⊕ -0.5` is zero, and `+0.5 ⊗ -0.5` is `-0.25`.
///
/// The first is what an unsigned log semiring cannot express: its ⊕ only ever
/// adds mass, so nothing can come back to zero.
#[test]
fn a_positive_and_a_negative_cancel() {
    let sum = half(1.0).plus(&half(-1.0));
    assert_eq!(
        sum.neg_log_prob,
        f32::INFINITY,
        "a probability of zero is a negative log of infinity"
    );
    assert_eq!(sum, SignedLogWeight::zero());

    let product = half(1.0).times(&half(-1.0));
    assert_eq!(product.sign, -1.0, "one negative factor, so a negative");
    assert!(
        (product.neg_log_prob - (4.0f32).ln()).abs() < 1e-4,
        "0.5 * 0.5 = 0.25, whose negative log is ln 4, not {}",
        product.neg_log_prob
    );
}

/// The same cancellation reached through an FST: two paths of equal and
/// opposite weight leave nothing behind.
#[test]
fn two_paths_of_opposite_sign_cancel_in_an_fst() {
    let mut fst = VectorFst::<SignedLogArc>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, SignedLogWeight::one());
    fst.add_arc(s0, SignedLogArc::new(1, 1, half(1.0), s1));
    fst.add_arc(s0, SignedLogArc::new(2, 2, half(-1.0), s1));

    let total = shortest_distance(&fst, SHORTEST_DELTA).expect("a total weight");
    assert_eq!(
        total.neg_log_prob,
        f32::INFINITY,
        "the two paths weigh +0.5 and -0.5, so the FST as a whole weighs nothing"
    );
}
