//! The algorithm surface, exercised end to end through the public API.
//!
//! These were written before the rewrite, against a method-style `ops` API
//! (`fst.compose(&a, &b, &opts)`), and commented out wholesale when that API
//! went. They are rewritten here against the functions that replaced it; the
//! FSTs and the expected answers are the ones they always had.
//!
//! Two of the thirty-four are not here: they exercised a delayed `ArcMapFst`,
//! which sicada does not have yet. When the delayed FST wrappers land, the eager
//! assertions above are what the lazy ones should also satisfy, since a delayed
//! FST is meant to be indistinguishable from the expanded one.

use sicada::algorithms::arc_map::{
    FromGallicMapper, IdentityArcMapper, InputEpsilonMapper, InvertWeightMapper,
    OutputEpsilonMapper, ToGallicMapper, WeightConvertMapper, arc_map, arc_map_to,
};
use sicada::algorithms::arcsort::{ILabelCompare, OLabelCompare, arc_sort};
use sicada::algorithms::closure::{ClosureType, closure};
use sicada::algorithms::compose::compose;
use sicada::algorithms::concat::concat;
use sicada::algorithms::connect::connect;
use sicada::algorithms::determinize::{DeterminizeOptions, determinize};
use sicada::algorithms::difference::difference;
use sicada::algorithms::disambiguate::{DisambiguateOptions, disambiguate};
use sicada::algorithms::encode::{
    ENCODE_LABELS, ENCODE_WEIGHTS, EncodeMapper, EncodeTable, EncodeType, decode, encode,
};
use sicada::algorithms::epsnormalize::{EpsNormalizeType, eps_normalize};
use sicada::algorithms::equal::{EQUAL_FSTS, equal};
use sicada::algorithms::equivalent::equivalent;
use sicada::algorithms::intersect::intersect;
use sicada::algorithms::invert::{invert, invert_to};
use sicada::algorithms::isomorphic::isomorphic;
use sicada::algorithms::minimize::{DELTA as MINIMIZE_DELTA, minimize};
use sicada::algorithms::project::ProjectType;
use sicada::algorithms::project::{project, project_to};
use sicada::algorithms::prune::{PruneOptions, prune};
use sicada::algorithms::push::{PUSH_REMOVE_TOTAL_WEIGHT, PUSH_WEIGHTS, push_to_initial};
use sicada::algorithms::randequivalent::rand_equivalent_default;
use sicada::algorithms::randgen::{RandGenOptions, Rng, UniformArcSelector, rand_gen};
use sicada::algorithms::relabel::relabel;
use sicada::algorithms::replace::{ReplaceLabelType, ReplaceOptions, replace};
use sicada::algorithms::reverse::reverse;
use sicada::algorithms::reweight::reweight_to_initial;
use sicada::algorithms::rmepsilon::rm_epsilon;
use sicada::algorithms::rmfinalepsilon::rm_final_epsilon;
use sicada::algorithms::shortest_distance::{
    SHORTEST_DELTA, shortest_distance, shortest_distance_forward,
};
use sicada::algorithms::shortest_path::{ShortestPathOptions, shortest_path};
use sicada::algorithms::state_sort::state_sort;
use sicada::algorithms::synchronize::synchronize;
use sicada::algorithms::topsort::top_sort;
use sicada::algorithms::union::union;
use sicada::algorithms::verify::verify;
use sicada::arc::{Arc, GallicArc, Log64Arc, StdArc};
use sicada::arc_filter::AnyArcFilter;
use sicada::fst::{ExpandedFst, Fst, MutableFst};
use sicada::fsts::vector_fst::{Log64VectorFst, StdVectorFst, VectorFst};
use sicada::properties::K_FST_PROPERTIES;
use sicada::symbol_table::SymbolTable;
use sicada::weight::Weight;
use sicada::weights::float_weight::{Log64Weight, TropicalWeight};
use sicada::weights::string_weight::GallicRight;
use std::io::Cursor;

/// The label that consumes nothing.
const EPSILON: i32 = 0;

/// The start state, for a test that has just built one.
fn start(fst: &StdVectorFst) -> i32 {
    fst.start().expect("the FST has a start state")
}

/// The arcs leaving a state, in the order they are stored.
fn arcs(fst: &StdVectorFst, state: i32) -> Vec<StdArc> {
    fst.arcs(state).collect()
}

/// The arcs leaving a state of an FST over any arc type.
fn arcs_of<A: Arc<Label = i32, StateId = i32>>(fst: &VectorFst<A>, state: i32) -> Vec<A> {
    fst.arcs(state).collect()
}

/// Epsilon-free, deterministic and minimal, so that "how many arcs leave the
/// start state" is a question about the language rather than about how the FST
/// happens to have been built.
fn tidy(fst: &StdVectorFst) -> StdVectorFst {
    let mut out = fst.clone();
    rm_epsilon(&mut out, true).expect("epsilons removed");
    let mut determinized = StdVectorFst::new();
    determinize(&out, &mut determinized, &DeterminizeOptions::default())
        .expect("a determinization");
    minimize(&mut determinized, MINIMIZE_DELTA, false).expect("a minimization");
    determinized
}

#[test]
fn sorting_orders_the_arcs_by_whichever_side_is_asked() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s0, TropicalWeight::one());
    fst.add_arc(s0, StdArc::new(3, 1, TropicalWeight::one(), s0));
    fst.add_arc(s0, StdArc::new(1, 3, TropicalWeight::one(), s0));
    fst.add_arc(s0, StdArc::new(2, 2, TropicalWeight::one(), s0));

    arc_sort(&mut fst, &ILabelCompare);
    let sorted = arcs(&fst, s0);
    assert_eq!(sorted.len(), 3);
    assert_eq!(
        sorted.iter().map(|arc| arc.ilabel()).collect::<Vec<_>>(),
        [1, 2, 3]
    );

    arc_sort(&mut fst, &OLabelCompare);
    let sorted = arcs(&fst, s0);
    assert_eq!(
        sorted.iter().map(|arc| arc.olabel()).collect::<Vec<_>>(),
        [1, 2, 3]
    );
}

#[test]
fn inverting_swaps_the_two_sides_and_leaves_the_weight() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());
    fst.add_arc(s0, StdArc::new(1, 2, TropicalWeight(0.5), s1));

    let mut inverted = StdVectorFst::new();
    invert_to(&fst, &mut inverted).expect("an inverted copy");
    let arc = arcs(&inverted, start(&inverted))[0];
    assert_eq!((arc.ilabel(), arc.olabel()), (2, 1));
    assert_eq!(arc.weight(), &TropicalWeight(0.5));

    invert(&mut fst).expect("inverted in place");
    let arc = arcs(&fst, start(&fst))[0];
    assert_eq!((arc.ilabel(), arc.olabel()), (2, 1));
}

#[test]
fn removing_epsilons_folds_their_weight_into_the_arc_that_follows() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::one());
    fst.add_arc(s0, StdArc::new(EPSILON, EPSILON, TropicalWeight(1.0), s1));
    fst.add_arc(s1, StdArc::new(1, 2, TropicalWeight(2.0), s2));

    rm_epsilon(&mut fst, true).expect("epsilons removed");

    let arcs = arcs(&fst, start(&fst));
    assert_eq!(arcs.len(), 1, "the epsilon is gone, leaving one arc");
    assert_eq!((arcs[0].ilabel(), arcs[0].olabel()), (1, 2));
    assert_eq!(
        arcs[0].weight(),
        &TropicalWeight(3.0),
        "the epsilon's 1.0 rides on the arc that took its place"
    );
}

#[test]
fn composing_then_searching_takes_the_cheaper_of_two_ways() {
    // 1:2 twice, at 0.5 and at 5.0, then 2:3.
    let mut first = StdVectorFst::new();
    let s0 = first.add_state();
    let s1 = first.add_state();
    let s2 = first.add_state();
    first.set_start(s0);
    first.set_final(s2, TropicalWeight::one());
    first.add_arc(s0, StdArc::new(1, 2, TropicalWeight(0.5), s1));
    first.add_arc(s0, StdArc::new(1, 2, TropicalWeight(5.0), s1));
    first.add_arc(s1, StdArc::new(2, 3, TropicalWeight(1.0), s2));
    arc_sort(&mut first, &OLabelCompare);

    // 2:4 then 3:5.
    let mut second = StdVectorFst::new();
    let t0 = second.add_state();
    let t1 = second.add_state();
    let t2 = second.add_state();
    second.set_start(t0);
    second.set_final(t2, TropicalWeight::one());
    second.add_arc(t0, StdArc::new(2, 4, TropicalWeight(1.5), t1));
    second.add_arc(t1, StdArc::new(3, 5, TropicalWeight(2.0), t2));
    arc_sort(&mut second, &ILabelCompare);

    let mut composed = StdVectorFst::new();
    compose(&first, &second, &mut composed).expect("a composition");

    let mut best = StdVectorFst::new();
    shortest_path(&composed, &mut best, &ShortestPathOptions::default()).expect("a shortest path");

    let first_arc = arcs(&best, start(&best))[0];
    assert_eq!((first_arc.ilabel(), first_arc.olabel()), (1, 4));
    assert_eq!(
        first_arc.weight(),
        &TropicalWeight(2.0),
        "0.5 from the cheaper arc plus 1.5, not 5.0 plus 1.5"
    );

    let second_arc = arcs(&best, first_arc.nextstate())[0];
    assert_eq!((second_arc.ilabel(), second_arc.olabel()), (2, 5));
    assert_eq!(second_arc.weight(), &TropicalWeight(3.0));
}

#[test]
fn projecting_copies_one_side_onto_the_other() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());
    fst.add_arc(s0, StdArc::new(1, 2, TropicalWeight(0.5), s1));

    let mut on_input = StdVectorFst::new();
    project_to(&fst, &mut on_input, ProjectType::Input).expect("a projected copy");
    let arc = arcs(&on_input, start(&on_input))[0];
    assert_eq!((arc.ilabel(), arc.olabel()), (1, 1));
    assert_eq!(arc.weight(), &TropicalWeight(0.5));

    project(&mut fst, ProjectType::Output).expect("projected in place");
    let arc = arcs(&fst, start(&fst))[0];
    assert_eq!((arc.ilabel(), arc.olabel()), (2, 2));
    assert_eq!(arc.weight(), &TropicalWeight(0.5));
}

#[test]
fn determinizing_leaves_one_arc_per_label_at_the_lighter_weight() {
    // Two ways to read `a` then `b`, at 1+2 and at 3+4.
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    let s3 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s3, TropicalWeight::one());
    fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight(1.0), s1));
    fst.add_arc(s1, StdArc::new(2, 2, TropicalWeight(2.0), s3));
    fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight(3.0), s2));
    fst.add_arc(s2, StdArc::new(2, 2, TropicalWeight(4.0), s3));

    let mut determinized = StdVectorFst::new();
    determinize(&fst, &mut determinized, &DeterminizeOptions::default())
        .expect("a determinization");

    let leaving = arcs(&determinized, start(&determinized));
    assert_eq!(leaving.len(), 1, "one arc for the one label `a`");
    assert_eq!(leaving[0].ilabel(), 1);
    assert_eq!(
        leaving[0].weight(),
        &TropicalWeight(1.0),
        "what the two ways share is pushed forward: min(1, 3)"
    );
}

#[test]
fn minimizing_merges_states_nothing_can_tell_apart() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());
    fst.set_final(s2, TropicalWeight::one());
    fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight(2.0), s1));
    fst.add_arc(s0, StdArc::new(2, 2, TropicalWeight(3.0), s2));
    assert_eq!(fst.num_states(), 3);

    minimize(&mut fst, MINIMIZE_DELTA, false).expect("a minimization");

    assert_eq!(
        fst.num_states(),
        2,
        "the two final states accept the same thing from where they are, so they are one"
    );
}

#[test]
fn union_offers_both_and_concat_puts_one_after_the_other() {
    let mut first = StdVectorFst::new();
    let s0 = first.add_state();
    let s1 = first.add_state();
    first.set_start(s0);
    first.set_final(s1, TropicalWeight::one());
    first.add_arc(s0, StdArc::new(1, 1, TropicalWeight(0.5), s1));

    let mut second = StdVectorFst::new();
    let t0 = second.add_state();
    let t1 = second.add_state();
    second.set_start(t0);
    second.set_final(t1, TropicalWeight::one());
    second.add_arc(t0, StdArc::new(2, 2, TropicalWeight(1.0), t1));

    // "a" or "b": two ways out of the start.
    let mut either = first.clone();
    union(&mut either, &second).expect("a union");
    let either = tidy(&either);
    assert_eq!(
        arcs(&either, start(&either)).len(),
        2,
        "one arc for each of the two words"
    );

    // "a" then "b": one way out, and one way on.
    let mut both = first.clone();
    concat(&mut both, &second).expect("a concatenation");
    let both = tidy(&both);
    let leaving = arcs(&both, start(&both));
    assert_eq!(leaving.len(), 1);
    assert_eq!(leaving[0].ilabel(), 1);

    let following = arcs(&both, leaving[0].nextstate());
    assert_eq!(following.len(), 1);
    assert_eq!(following[0].ilabel(), 2);
    assert_eq!(
        leaving[0].weight().0 + following[0].weight().0,
        1.5,
        "the two words' weights are still there, wherever they sit"
    );
}

#[test]
fn closure_accepts_the_word_any_number_of_times() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());
    fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight(1.0), s1));

    closure(&mut fst, ClosureType::Star);
    let fst = tidy(&fst);

    let start = start(&fst);
    assert_eq!(
        fst.final_weight(start),
        TropicalWeight::one(),
        "zero repetitions is a word, so the start state is final"
    );

    let leaving = arcs(&fst, start);
    assert_eq!(leaving.len(), 1, "one way round the loop");
    assert_eq!(leaving[0].ilabel(), 1);
    assert_eq!(leaving[0].weight(), &TropicalWeight(1.0));
    assert_eq!(
        fst.final_weight(leaving[0].nextstate()),
        TropicalWeight::one(),
        "and one repetition is a word too"
    );
}

#[test]
fn the_four_ways_of_asking_whether_two_fsts_are_the_same() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());
    fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight(0.5), s1));
    fst.properties(K_FST_PROPERTIES, true);
    let clone = fst.clone();

    let delta = 1e-4;
    assert!(
        equal(&fst, &clone, delta, EQUAL_FSTS),
        "a clone is equal state for state"
    );
    assert!(
        isomorphic(&fst, &clone, delta).expect("comparable"),
        "and isomorphic, which is weaker"
    );
    assert!(
        equivalent(&fst, &clone, delta).expect("comparable"),
        "and accepts the same language, which is weaker still"
    );
    assert!(
        rand_equivalent_default(&fst, &clone, 10, &mut Rng::new(1), UniformArcSelector,)
            .expect("comparable"),
        "and agrees on ten random paths"
    );

    // The same FST with its two states numbered the other way round.
    let mut renumbered = StdVectorFst::new();
    let r1 = renumbered.add_state();
    let r0 = renumbered.add_state();
    renumbered.set_start(r0);
    renumbered.set_final(r1, TropicalWeight::one());
    renumbered.add_arc(r0, StdArc::new(1, 1, TropicalWeight(0.5), r1));
    renumbered.properties(K_FST_PROPERTIES, true);

    assert!(
        !equal(&fst, &renumbered, delta, EQUAL_FSTS),
        "equality is state by state, and the states are numbered differently"
    );
    assert!(
        isomorphic(&fst, &renumbered, delta).expect("comparable"),
        "isomorphism does not care how the states are numbered"
    );
    assert!(
        equivalent(&fst, &renumbered, delta).expect("comparable"),
        "and neither does the language"
    );
}

#[test]
fn connecting_drops_what_no_path_goes_through() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    let s3 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::one());
    fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight(0.5), s1));
    fst.add_arc(s1, StdArc::new(2, 2, TropicalWeight(0.5), s2));
    // s3 is reachable but can never finish.
    fst.add_arc(s1, StdArc::new(3, 3, TropicalWeight(1.0), s3));
    assert_eq!(fst.num_states(), 4);

    connect(&mut fst);

    assert_eq!(fst.num_states(), 3, "the state that cannot finish is gone");
    assert_eq!(fst.num_arcs(s1), 1, "and so is the arc that led to it");
}

#[test]
fn a_topological_sort_numbers_the_states_along_the_paths() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    fst.set_start(s2);
    fst.set_final(s0, TropicalWeight::one());
    fst.add_arc(s2, StdArc::new(1, 1, TropicalWeight(1.0), s1));
    fst.add_arc(s1, StdArc::new(2, 2, TropicalWeight(1.0), s0));

    assert!(top_sort(&mut fst).expect("sortable"), "the FST is acyclic");

    assert_eq!(fst.start(), Some(0), "the start state comes first");
    let leaving = arcs(&fst, 0);
    assert_eq!(leaving.len(), 1);
    assert_eq!(leaving[0].nextstate(), 1, "and what it reaches comes next");
}

#[test]
fn a_cyclic_fst_has_no_topological_order() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());
    fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight(1.0), s1));
    fst.add_arc(s1, StdArc::new(2, 2, TropicalWeight(1.0), s0));

    assert!(
        !top_sort(&mut fst).expect("askable"),
        "there is a cycle, so there is no order to put the states in"
    );
}

#[test]
fn the_shortest_distance_is_the_semiring_sum_over_paths() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::one());
    fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight(1.0), s1));
    fst.add_arc(s1, StdArc::new(2, 2, TropicalWeight(2.0), s2));
    fst.add_arc(s0, StdArc::new(3, 3, TropicalWeight(4.0), s2));

    let total = shortest_distance(&fst, SHORTEST_DELTA).expect("a total distance");
    assert_eq!(
        total,
        TropicalWeight(3.0),
        "min(1 + 2, 4): the tropical sum is the lighter path"
    );

    let each = shortest_distance_forward(&fst, SHORTEST_DELTA).expect("a distance for each state");
    assert_eq!(
        each,
        vec![
            TropicalWeight(0.0),
            TropicalWeight(1.0),
            TropicalWeight(3.0)
        ]
    );

    // Over the log semiring the same shape sums probabilities instead: two
    // halves make a whole.
    let mut log = Log64VectorFst::new();
    let l0 = log.add_state();
    let l1 = log.add_state();
    log.set_start(l0);
    log.set_final(l1, Log64Weight::one());
    let half = -0.5f64.ln();
    log.add_arc(l0, Log64Arc::new(1, 1, Log64Weight(half), l1));
    log.add_arc(l0, Log64Arc::new(2, 2, Log64Weight(half), l1));

    let total = shortest_distance(&log, SHORTEST_DELTA).expect("a total distance");
    assert!(
        total.0.abs() < 1e-5,
        "0.5 + 0.5 is 1, whose negative log is 0, not {}",
        total.0
    );
}

#[test]
fn pushing_moves_weight_forward_and_pruning_drops_what_is_too_heavy() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::one());
    fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight(1.0), s1));
    fst.add_arc(s1, StdArc::new(2, 2, TropicalWeight(1.0), s2));
    fst.add_arc(s0, StdArc::new(3, 3, TropicalWeight(5.0), s2));

    let mut pushed = StdVectorFst::new();
    push_to_initial(
        &fst,
        &mut pushed,
        PUSH_WEIGHTS | PUSH_REMOVE_TOTAL_WEIGHT,
        SHORTEST_DELTA,
    )
    .expect("a pushed FST");

    let leaving = arcs(&pushed, start(&pushed));
    let cheap = leaving
        .iter()
        .find(|arc| arc.ilabel() == 1)
        .expect("the light path's first arc");
    assert!(
        cheap.weight().0.abs() < 1e-5,
        "the lighter path costs nothing once the total is taken out, not {}",
        cheap.weight().0
    );
    let dear = leaving
        .iter()
        .find(|arc| arc.ilabel() == 3)
        .expect("the heavy path's arc");
    assert_eq!(
        dear.weight(),
        &TropicalWeight(3.0),
        "and the heavier one costs what it costs above the lighter: 5 - 2"
    );

    // Anything more than 2.5 above the best path goes.
    prune(
        &mut fst,
        &PruneOptions::new(TropicalWeight(2.5), AnyArcFilter),
    )
    .expect("a pruned FST");
    let left = arcs(&fst, start(&fst));
    assert_eq!(left.len(), 1, "the path costing 5 is more than 2 + 2.5");
    assert_eq!(left[0].ilabel(), 1);
}

#[test]
fn intersection_keeps_what_both_accept_and_difference_what_only_one_does() {
    // Accepts 1 or 2.
    let mut left = StdVectorFst::new();
    let a0 = left.add_state();
    let a1 = left.add_state();
    left.set_start(a0);
    left.set_final(a1, TropicalWeight::one());
    left.add_arc(a0, StdArc::new(1, 1, TropicalWeight(0.5), a1));
    left.add_arc(a0, StdArc::new(2, 2, TropicalWeight(0.5), a1));
    arc_sort(&mut left, &ILabelCompare);
    left.properties(K_FST_PROPERTIES, true);

    // Accepts 2 or 3, at a cost.
    let mut right = StdVectorFst::new();
    let b0 = right.add_state();
    let b1 = right.add_state();
    right.set_start(b0);
    right.set_final(b1, TropicalWeight::one());
    right.add_arc(b0, StdArc::new(2, 2, TropicalWeight(1.0), b1));
    right.add_arc(b0, StdArc::new(3, 3, TropicalWeight(1.0), b1));
    arc_sort(&mut right, &ILabelCompare);
    right.properties(K_FST_PROPERTIES, true);

    let mut both = StdVectorFst::new();
    intersect(&left, &right, &mut both).expect("an intersection");
    let leaving = arcs(&both, start(&both));
    assert_eq!(leaving.len(), 1, "only 2 is in both");
    assert_eq!(leaving[0].ilabel(), 2);
    assert_eq!(
        leaving[0].weight(),
        &TropicalWeight(1.5),
        "and it costs what both sides charge"
    );

    // The same alphabet, unweighted and deterministic, so it can be
    // complemented, a difference being an intersection with the complement.
    let mut unweighted = StdVectorFst::new();
    let c0 = unweighted.add_state();
    let c1 = unweighted.add_state();
    unweighted.set_start(c0);
    unweighted.set_final(c1, TropicalWeight::one());
    unweighted.add_arc(c0, StdArc::new(2, 2, TropicalWeight::one(), c1));
    unweighted.add_arc(c0, StdArc::new(3, 3, TropicalWeight::one(), c1));
    arc_sort(&mut unweighted, &ILabelCompare);
    unweighted.properties(K_FST_PROPERTIES, true);

    let mut only_left = StdVectorFst::new();
    difference(&left, &unweighted, &mut only_left).expect("a difference");
    let leaving = arcs(&only_left, start(&only_left));
    assert_eq!(
        leaving.len(),
        1,
        "only 1 is in the first and not the second"
    );
    assert_eq!(leaving[0].ilabel(), 1);
    assert_eq!(leaving[0].weight(), &TropicalWeight(0.5));
}

#[test]
fn reversing_turns_the_paths_round_and_leaves_the_labels_alone() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());
    fst.add_arc(s0, StdArc::new(1, 2, TropicalWeight(1.0), s1));

    let mut reversed = StdVectorFst::new();
    reverse(&fst, &mut reversed, false);

    let leaving = arcs(&reversed, start(&reversed));
    assert_eq!(leaving.len(), 1);
    assert_eq!(
        (leaving[0].ilabel(), leaving[0].olabel()),
        (1, 2),
        "an arc keeps both its labels; it is the paths that run the other way"
    );
}

#[test]
fn relabelling_rewrites_each_side_by_its_own_table() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());
    fst.add_arc(s0, StdArc::new(1, 2, TropicalWeight::one(), s1));

    relabel(&mut fst, &[(1, 10)], &[(2, 20)]).expect("relabelled");

    let arc = arcs(&fst, s0)[0];
    assert_eq!((arc.ilabel(), arc.olabel()), (10, 20));
}

#[test]
fn replacing_a_non_terminal_splices_in_what_it_stands_for() {
    let mut symbols = SymbolTable::new("vocab");
    let the = symbols.add_symbol("the", 1) as i32;
    let cat = symbols.add_symbol("cat", 2) as i32;
    let dog = symbols.add_symbol("dog", 3) as i32;
    let noun = symbols.add_symbol("<NOUN>", 10) as i32;

    // "the" then whatever <NOUN> is.
    let mut root = StdVectorFst::new();
    let r0 = root.add_state();
    let r1 = root.add_state();
    let r2 = root.add_state();
    root.set_start(r0);
    root.set_final(r2, TropicalWeight::one());
    root.add_arc(r0, StdArc::new(the, the, TropicalWeight(0.5), r1));
    root.add_arc(r1, StdArc::new(noun, noun, TropicalWeight::one(), r2));

    // <NOUN> is "cat" or "dog".
    let mut nouns = StdVectorFst::new();
    let n0 = nouns.add_state();
    let n1 = nouns.add_state();
    nouns.set_start(n0);
    nouns.set_final(n1, TropicalWeight::one());
    nouns.add_arc(n0, StdArc::new(cat, cat, TropicalWeight(1.0), n1));
    nouns.add_arc(n0, StdArc::new(dog, dog, TropicalWeight(2.0), n1));

    const ROOT: i32 = 0;
    let mut replaced = StdVectorFst::new();
    replace(
        &[(ROOT, &root), (noun, &nouns)],
        &mut replaced,
        &ReplaceOptions {
            call_label_type: ReplaceLabelType::Neither,
            return_label_type: ReplaceLabelType::Neither,
            ..ReplaceOptions::new(ROOT)
        },
    )
    .expect("a replacement");

    let replaced = tidy(&replaced);
    let leaving = arcs(&replaced, start(&replaced));
    assert_eq!(leaving.len(), 1);
    assert_eq!(leaving[0].ilabel(), the);

    let mut following = arcs(&replaced, leaving[0].nextstate());
    following.sort_by_key(|arc| arc.ilabel());
    assert_eq!(following.len(), 2, "the non-terminal branches into two");
    assert_eq!(following[0].ilabel(), cat);
    assert_eq!(following[1].ilabel(), dog);
    assert_eq!(leaving[0].weight().0 + following[0].weight().0, 1.5);
    assert_eq!(leaving[0].weight().0 + following[1].weight().0, 2.5);
}

#[test]
fn encoding_makes_an_acceptor_and_decoding_gives_the_transducer_back() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());
    fst.add_arc(s0, StdArc::new(1, 2, TropicalWeight(0.5), s1));

    let mut mapper = EncodeMapper::<StdArc>::new(ENCODE_LABELS | ENCODE_WEIGHTS);
    encode(&mut fst, &mut mapper).expect("an encoding");

    let arc = arcs(&fst, s0)[0];
    assert_eq!(
        arc.ilabel(),
        arc.olabel(),
        "the two labels and the weight are one label now, so it is an acceptor"
    );
    assert_eq!(
        arc.weight(),
        &TropicalWeight::one(),
        "and the weight has moved into that label"
    );

    decode(&mut fst, &mapper).expect("a decoding");
    let arc = arcs(&fst, s0)[0];
    assert_eq!((arc.ilabel(), arc.olabel()), (1, 2));
    assert_eq!(arc.weight(), &TropicalWeight(0.5));
}

#[test]
fn an_encode_table_can_be_saved_and_used_to_decode_later() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());
    fst.add_arc(s0, StdArc::new(1, 2, TropicalWeight(0.5), s1));

    let mut mapper = EncodeMapper::<StdArc>::new(ENCODE_LABELS | ENCODE_WEIGHTS);
    encode(&mut fst, &mut mapper).expect("an encoding");

    let mut bytes = Vec::new();
    mapper
        .table()
        .borrow()
        .write(&mut bytes, StdArc::type_name())
        .expect("the table written");
    let table =
        EncodeTable::<i32, TropicalWeight>::read(&mut Cursor::new(bytes)).expect("the table read");
    let loaded = EncodeMapper::<StdArc>::from_table(
        std::rc::Rc::new(std::cell::RefCell::new(table)),
        EncodeType::Encode,
    );
    assert_eq!(loaded.flags(), ENCODE_LABELS | ENCODE_WEIGHTS);

    decode(&mut fst, &loaded).expect("a decoding through the table that was saved");
    let arc = arcs(&fst, s0)[0];
    assert_eq!((arc.ilabel(), arc.olabel()), (1, 2));
    assert_eq!(arc.weight(), &TropicalWeight(0.5));
}

#[test]
fn the_arc_mappers_each_change_the_one_thing_they_name() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight(2.0));
    fst.add_arc(s0, StdArc::new(1, 2, TropicalWeight(3.0), s1));

    let mut same = StdVectorFst::new();
    arc_map_to(&fst, &mut same, &mut IdentityArcMapper).expect("a copy");
    assert_eq!(arcs(&same, s0)[0].ilabel(), 1);

    let mut no_input = StdVectorFst::new();
    arc_map_to(&fst, &mut no_input, &mut InputEpsilonMapper).expect("a mapped FST");
    let arc = arcs(&no_input, s0)[0];
    assert_eq!((arc.ilabel(), arc.olabel()), (0, 2));

    let mut no_output = StdVectorFst::new();
    arc_map_to(&fst, &mut no_output, &mut OutputEpsilonMapper).expect("a mapped FST");
    let arc = arcs(&no_output, s0)[0];
    assert_eq!((arc.ilabel(), arc.olabel()), (1, 0));

    // Over the tropical semiring, inverting a weight negates it.
    arc_map(&mut fst, &mut InvertWeightMapper).expect("a mapped FST");
    assert_eq!(arcs(&fst, s0)[0].weight(), &TropicalWeight(-3.0));
}

#[test]
fn an_fst_can_be_carried_into_another_semiring_and_back() {
    let mut tropical = StdVectorFst::new();
    let s0 = tropical.add_state();
    let s1 = tropical.add_state();
    tropical.set_start(s0);
    tropical.set_final(s1, TropicalWeight::one());
    tropical.add_arc(s0, StdArc::new(1, 2, TropicalWeight(2.0), s1));

    let mut log = Log64VectorFst::new();
    arc_map_to(
        &tropical,
        &mut log,
        &mut WeightConvertMapper::<Log64Arc>::new(),
    )
    .expect("a converted FST");
    assert_eq!(log.num_states(), 2);
    assert_eq!(arcs_of(&log, s0)[0].weight(), &Log64Weight(2.0));

    let mut back = StdVectorFst::new();
    arc_map_to(&log, &mut back, &mut WeightConvertMapper::<StdArc>::new())
        .expect("a converted FST");
    assert_eq!(back.num_states(), 2);
    assert_eq!(arcs(&back, s0)[0].weight(), &TropicalWeight(2.0));
}

#[test]
fn the_output_side_can_be_carried_in_the_weight_and_brought_back() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());
    fst.add_arc(s0, StdArc::new(1, 2, TropicalWeight(2.0), s1));

    // Into the gallic semiring: the output label rides in the weight, so the
    // result is an acceptor and a shortest distance sums over labels too.
    let mut gallic = VectorFst::<GallicArc<StdArc, GallicRight>>::new();
    arc_map_to(&fst, &mut gallic, &mut ToGallicMapper::<GallicRight>::new())
        .expect("an FST over gallic weights");
    assert_eq!(gallic.num_states(), 2);

    let mut back = StdVectorFst::new();
    arc_map_to(
        &gallic,
        &mut back,
        &mut FromGallicMapper::<i32, GallicRight>::new(),
    )
    .expect("the transducer back");

    assert_eq!(back.num_states(), 2);
    let arc = arcs(&back, start(&back))[0];
    assert_eq!((arc.ilabel(), arc.olabel()), (1, 2));
    assert_eq!(arc.weight(), &TropicalWeight(2.0));
}

#[test]
fn reweighting_moves_weight_between_arcs_without_changing_any_path() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight(2.0));
    fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight(1.0), s1));

    // A potential per state. Each arc gains its destination's and loses its
    // source's; the start state's is given back on the way out, so no path
    // changes weight.
    let potential = [TropicalWeight(3.0), TropicalWeight(5.0)];
    reweight_to_initial(&mut fst, &potential);

    let arc = arcs(&fst, s0)[0];
    assert_eq!(
        arc.weight(),
        &TropicalWeight(6.0),
        "1 + 5 - 3, and then the start's 3 back again"
    );
    assert_eq!(fst.final_weight(s1), TropicalWeight(-3.0), "2 - 5");
    assert_eq!(
        arc.weight().0 + fst.final_weight(s1).0,
        3.0,
        "the one path still weighs 1 + 2"
    );
}

#[test]
fn synchronizing_pairs_the_two_sides_up_as_soon_as_it_can() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::one());
    // Reads 1 and writes nothing, then reads nothing and writes 2.
    fst.add_arc(s0, StdArc::new(1, EPSILON, TropicalWeight(0.5), s1));
    fst.add_arc(s1, StdArc::new(EPSILON, 2, TropicalWeight(1.0), s2));

    let mut synchronized = StdVectorFst::new();
    synchronize(&fst, &mut synchronized);

    // The 1 is held back until there is a 2 to pair it with.
    let first = arcs(&synchronized, start(&synchronized))[0];
    assert_eq!((first.ilabel(), first.olabel()), (EPSILON, EPSILON));
    assert_eq!(first.weight(), &TropicalWeight(0.5));

    let second = arcs(&synchronized, first.nextstate())[0];
    assert_eq!(
        (second.ilabel(), second.olabel()),
        (1, 2),
        "and then both come out together"
    );
    assert_eq!(second.weight(), &TropicalWeight(1.0));
}

#[test]
fn epsilon_normalising_moves_the_labelled_arcs_first() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::one());
    // Writes 1 without reading, then reads 2 without writing.
    fst.add_arc(s0, StdArc::new(EPSILON, 1, TropicalWeight(0.5), s1));
    fst.add_arc(s1, StdArc::new(2, EPSILON, TropicalWeight(1.0), s2));

    let mut on_input = StdVectorFst::new();
    eps_normalize(&fst, &mut on_input, EpsNormalizeType::Input, GallicRight)
        .expect("normalised on the input side");
    let arc = arcs(&on_input, start(&on_input))[0];
    assert_eq!(
        (arc.ilabel(), arc.olabel()),
        (2, 1),
        "no input epsilon comes before an input label any more"
    );

    let mut on_output = StdVectorFst::new();
    eps_normalize(&fst, &mut on_output, EpsNormalizeType::Output, GallicRight)
        .expect("normalised on the output side");
    let arc = arcs(&on_output, start(&on_output))[0];
    assert_eq!(
        (arc.ilabel(), arc.olabel()),
        (EPSILON, 1),
        "the output side was already normal, so nothing moved"
    );
}

#[test]
fn drawing_random_paths_gives_paths_the_fst_has() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::one());
    fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight(0.5), s1));
    fst.add_arc(s0, StdArc::new(2, 2, TropicalWeight(0.5), s2));
    fst.add_arc(s1, StdArc::new(3, 3, TropicalWeight(1.0), s2));

    let mut drawn = StdVectorFst::new();
    rand_gen(
        &fst,
        &mut drawn,
        &mut Rng::new(7),
        &RandGenOptions {
            npath: 5,
            weighted: false,
            ..RandGenOptions::new(UniformArcSelector)
        },
    )
    .expect("five paths");

    assert!(drawn.num_states() >= 2, "five paths make at least one");
    for state in 0..drawn.num_states() as i32 {
        for arc in drawn.arcs(state) {
            assert!(
                // An unweighted result cannot carry a final weight, so a path
                // that stops does it with an epsilon arc to a superfinal state.
                [EPSILON, 1, 2, 3].contains(&arc.ilabel()),
                "every label drawn is one the FST has, not {}",
                arc.ilabel()
            );
        }
    }
}

#[test]
fn a_final_state_reached_only_by_epsilon_folds_back_into_its_predecessor() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight(1.0));
    fst.set_final(s2, TropicalWeight(2.0));
    fst.add_arc(s0, StdArc::new(1, 2, TropicalWeight(0.5), s1));
    fst.add_arc(s1, StdArc::new(EPSILON, EPSILON, TropicalWeight(1.5), s2));

    rm_final_epsilon(&mut fst);

    assert_eq!(
        fst.final_weight(s1),
        TropicalWeight(1.0),
        "min(1, 1.5 + 2): stopping at s1 was already the cheaper way to finish"
    );
    assert_eq!(fst.num_arcs(s1), 0, "and the epsilon that led on is gone");
}

#[test]
fn sorting_the_states_renumbers_them_and_the_arcs_that_point_at_them() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());
    fst.add_arc(s0, StdArc::new(1, 2, TropicalWeight(0.5), s1));

    // `order[old] == new`: the two states swap.
    state_sort(&mut fst, &[1, 0]).expect("a state sort");

    assert_eq!(fst.start(), Some(1));
    assert_eq!(fst.final_weight(0), TropicalWeight::one());
    let arc = arcs(&fst, 1)[0];
    assert_eq!(arc.nextstate(), 0);
    assert_eq!(arc.weight(), &TropicalWeight(0.5));
}

#[test]
fn verification_names_what_is_wrong_with_an_fst() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    fst.set_start(s0);
    fst.add_arc(s0, StdArc::new(-10, 1, TropicalWeight(1.0), s0));

    let Err(err) = verify(&fst, false) else {
        panic!("a negative label is not a label")
    };
    assert!(
        format!("{err}").contains("negative"),
        "the error should say which way it is wrong: {err}"
    );

    assert!(
        verify(&fst, true).is_ok(),
        "unless the caller says negative labels are meant, as ρ and σ matchers do"
    );
}

#[test]
fn disambiguating_leaves_one_path_per_input_string() {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    let s3 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s3, TropicalWeight::one());
    // Two ways to read 1 then 2, at 3 and at 7.
    fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight(1.0), s1));
    fst.add_arc(s1, StdArc::new(2, 2, TropicalWeight(2.0), s3));
    fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight(3.0), s2));
    fst.add_arc(s2, StdArc::new(2, 2, TropicalWeight(4.0), s3));
    fst.properties(K_FST_PROPERTIES, true);

    let mut unambiguous = StdVectorFst::new();
    disambiguate(&fst, &mut unambiguous, &DisambiguateOptions::default())
        .expect("a disambiguation");

    assert!(unambiguous.num_states() > 0);
    assert_eq!(
        arcs(&unambiguous, start(&unambiguous)).len(),
        1,
        "one way to read the first label, where there were two"
    );
}
