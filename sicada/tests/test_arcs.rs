//! The arc aliases, and that each one's semiring really is the one it names.

use sicada::algorithms::arc_map::{WeightConvertMapper, arc_map_to};
use sicada::algorithms::shortest_distance::{SHORTEST_DELTA, shortest_distance};
use sicada::arc::{Arc, Log64Arc, LogArc, MinMaxArc, Real64Arc, RealArc, StdArc};
use sicada::fst::{ExpandedFst, Fst, MutableFst};
use sicada::fsts::vector_fst::VectorFst;
use sicada::weight::Weight;
use sicada::weights::float_weight::{
    Log64Weight, LogWeight, MinMaxWeight, Real64Weight, RealWeight, TropicalWeight,
};

/// A two-state FST with one arc, over whatever arc type is asked for.
fn one_arc<A: Arc<Label = i32, StateId = i32>>(weight: A::Weight) -> VectorFst<A> {
    let mut fst = VectorFst::<A>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, A::Weight::one());
    fst.add_arc(s0, A::new(1, 1, weight, s1));
    fst
}

/// Every float arc alias builds an FST of its own weight type.
#[test]
fn every_arc_alias_builds_an_fst() {
    let log = one_arc::<LogArc>(LogWeight(2.5));
    assert_eq!(log.num_states(), 2);
    assert_eq!(log.fst_type(), "vector");

    let real64 = one_arc::<Real64Arc>(Real64Weight(0.5));
    assert_eq!(real64.num_states(), 2);

    let minmax = one_arc::<MinMaxArc>(MinMaxWeight(10.0));
    assert_eq!(minmax.num_states(), 2);
}

/// Two arcs from the start to a final state, so the total distance is exactly
/// their ⊕. Each semiring's ⊕ is a different function, and this is where the
/// arc alias has to carry the right one.
#[test]
fn each_arc_alias_carries_its_own_plus() {
    fn total<A>(lhs: A::Weight, rhs: A::Weight) -> A::Weight
    where
        A: Arc<Label = i32, StateId = i32>,
        A::Weight: Weight<ReverseWeight = A::Weight>,
    {
        let mut fst = VectorFst::<A>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, A::Weight::one());
        fst.add_arc(s0, A::new(1, 1, lhs, s1));
        fst.add_arc(s0, A::new(2, 2, rhs, s1));
        shortest_distance(&fst, SHORTEST_DELTA).expect("a total")
    }

    // Tropical: min.
    assert_eq!(
        total::<StdArc>(TropicalWeight(1.0), TropicalWeight(2.0)),
        TropicalWeight(1.0)
    );

    // Log: -ln(e^-x + e^-y).
    let log = total::<LogArc>(LogWeight(1.0), LogWeight(2.0));
    let expected = -((-1.0f32).exp() + (-2.0f32).exp()).ln();
    assert!(
        (log.0 - expected).abs() < 1e-4,
        "log plus is {log}, expected {expected}"
    );

    // Real: ordinary addition.
    assert_eq!(
        total::<RealArc>(RealWeight(1.0), RealWeight(2.0)),
        RealWeight(3.0)
    );

    // MinMax: min, like tropical, but ⊗ is max rather than +.
    assert_eq!(
        total::<MinMaxArc>(MinMaxWeight(1.0), MinMaxWeight(2.0)),
        MinMaxWeight(1.0)
    );
}

/// A weight travels cost -> log -> probability -> cost and comes back to where
/// it started, which is the round trip `WeightConvertMapper` promises.
#[test]
fn a_weight_survives_a_trip_through_three_semirings() {
    let mut cost = VectorFst::<StdArc>::new();
    let s0 = cost.add_state();
    let s1 = cost.add_state();
    cost.set_start(s0);
    cost.set_final(s1, TropicalWeight::one());
    cost.add_arc(s0, StdArc::new(1, 2, TropicalWeight(2.0), s1));

    // Tropical cost -> log cost. Same number, different ⊕.
    let mut in_log = VectorFst::<Log64Arc>::new();
    arc_map_to(
        &cost,
        &mut in_log,
        &mut WeightConvertMapper::<Log64Arc>::new(),
    )
    .expect("a converted FST");

    // Log cost -> real probability: exp(-cost).
    let mut probability = VectorFst::<Real64Arc>::new();
    arc_map_to(
        &in_log,
        &mut probability,
        &mut WeightConvertMapper::<Real64Arc>::new(),
    )
    .expect("a converted FST");
    let arc = probability.arcs(s0).next().expect("the one arc");
    assert!(
        (arc.weight().0 - (-2.0f64).exp()).abs() < 1e-6,
        "cost 2 should be probability e^-2, not {}",
        arc.weight().0
    );
    assert_eq!(probability.final_weight(s1), Real64Weight(1.0));

    // And back: probability -> cost.
    let mut back = VectorFst::<LogArc>::new();
    arc_map_to(
        &probability,
        &mut back,
        &mut WeightConvertMapper::<LogArc>::new(),
    )
    .expect("a converted FST");
    let arc = back.arcs(s0).next().expect("the one arc");
    assert!(
        (arc.weight().0 - 2.0).abs() < 1e-5,
        "the cost should be back to 2, not {}",
        arc.weight().0
    );
    assert_eq!(back.final_weight(s1), LogWeight(0.0));
}

/// Compiling a string does not care which semiring the arcs carry.
#[test]
fn a_string_compiles_over_any_arc_type() {
    use sicada::string::{StringPrinter, TokenType};

    let fst = sicada::fst_linear!(VectorFst<Log64Arc>, "openfst");
    assert_eq!(fst.num_states(), 8);

    let printer = StringPrinter::new(TokenType::Byte);
    let (text, weight) = printer.print_weighted(&fst).expect("a string");
    assert_eq!(text, b"openfst");
    assert_eq!(weight, Log64Weight::one());
}
