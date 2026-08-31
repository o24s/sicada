//! The expectation semiring: a probability paired with a value, where ⊗ is
//! what makes the second component an expectation rather than just a number.

use sicada::algorithms::shortest_distance::{SHORTEST_DELTA, shortest_distance_reverse};
use sicada::arc::{Arc, ArcTpl};
use sicada::fst::MutableFst;
use sicada::fsts::vector_fst::VectorFst;
use sicada::weight::Weight;
use sicada::weights::expectation_weight::ExpectationWeight;
use sicada::weights::float_weight::{RealWeight, TropicalWeight};

/// A probability in the real semiring, with a value alongside it.
type Real = ExpectationWeight<RealWeight, RealWeight>;
/// The same over the tropical semiring, where "probability" is a cost.
type Tropical = ExpectationWeight<TropicalWeight, TropicalWeight>;

fn real(probability: f32, value: f32) -> Real {
    ExpectationWeight::new(RealWeight(probability), RealWeight(value))
}

fn tropical(cost: f32, value: f32) -> Tropical {
    ExpectationWeight::new(TropicalWeight(cost), TropicalWeight(value))
}

/// Over the real semiring both components just add under ⊕, while ⊗ weights
/// each side's value by the other side's probability, which keeps the second
/// component an expectation.
#[test]
fn the_real_expectation_semiring_carries_an_expectation() {
    let lhs = real(0.5, 10.0);
    let rhs = real(0.5, 20.0);

    let sum = lhs.plus(&rhs);
    assert_eq!(*sum.value1(), RealWeight(1.0));
    assert_eq!(*sum.value2(), RealWeight(30.0));

    // p = 0.5 * 0.5, v = 0.5 * 20 + 0.5 * 10.
    let product = lhs.times(&rhs);
    assert_eq!(*product.value1(), RealWeight(0.25));
    assert_eq!(*product.value2(), RealWeight(15.0));

    let reversed = lhs.reverse();
    assert_eq!(*reversed.value1(), RealWeight(0.5));
    assert_eq!(*reversed.value2(), RealWeight(10.0));
}

/// Over the tropical semiring ⊕ is min componentwise, and ⊗ adds the costs
/// while taking the better of the two ways of pairing a cost with a value.
#[test]
fn the_tropical_expectation_semiring_takes_the_better_pairing() {
    let lhs = tropical(1.0, 5.0);
    let rhs = tropical(2.0, 3.0);

    let sum = lhs.plus(&rhs);
    assert_eq!(*sum.value1(), TropicalWeight(1.0));
    assert_eq!(*sum.value2(), TropicalWeight(3.0));

    // cost = 1 + 2, value = min(1 + 3, 2 + 5).
    let product = lhs.times(&rhs);
    assert_eq!(*product.value1(), TropicalWeight(3.0));
    assert_eq!(*product.value2(), TropicalWeight(4.0));
}

/// Two paths through an FST, summed: the probabilities add to one and the
/// values add to the total.
#[test]
fn two_paths_sum_to_the_whole_expectation() {
    type ExpectationRealArc = ArcTpl<Real>;

    let mut fst = VectorFst::<ExpectationRealArc>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, Real::one());
    fst.add_arc(s0, ExpectationRealArc::new(1, 1, real(0.6, 60.0), s1));
    fst.add_arc(s0, ExpectationRealArc::new(2, 2, real(0.4, 20.0), s1));

    // Backwards, so `distance[s0]` is what the whole FST weighs.
    let distance =
        shortest_distance_reverse(&fst, SHORTEST_DELTA).expect("a distance for each state");
    let total = &distance[s0 as usize];

    assert!(
        (total.value1().0 - 1.0).abs() < 1e-5,
        "0.6 + 0.4 should be 1, not {}",
        total.value1().0
    );
    assert!(
        (total.value2().0 - 80.0).abs() < 1e-5,
        "60 + 20 should be 80, not {}",
        total.value2().0
    );
}
