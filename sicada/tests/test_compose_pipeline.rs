//! Composition feeding a shortest-path search: the pipeline a transducer
//! cascade is actually used through.
//!
//! The lazy half of this file is still waiting: upstream's `ComposeFst`
//! expands states on demand, and sicada has no delayed FST wrappers yet. When
//! they land, the same assertions should hold of the delayed form, since it is
//! supposed to be indistinguishable from the expanded one.

use sicada::AtomicRc;
use sicada::algorithms::arcsort::{ILabelCompare, OLabelCompare, arc_sort};
use sicada::algorithms::compose::compose;
use sicada::algorithms::shortest_path::{ShortestPathOptions, shortest_path};
use sicada::arc::{Arc, StdArc};
use sicada::fst::{ExpandedFst, Fst, MutableFst};
use sicada::fsts::vector_fst::StdVectorFst;
use sicada::symbol_table::SymbolTable;
use sicada::weight::Weight;
use sicada::weights::float_weight::TropicalWeight;

/// `a:x` at 0.5, and `x:y` at 1.5, over one shared alphabet. Composed they are
/// `a:y` at 2.0, and there is no other path.
fn cascade() -> (StdVectorFst, StdVectorFst) {
    let mut symbols = SymbolTable::new("vocab");
    symbols.add_symbol("eps", 0);
    symbols.add_symbol("a", 1);
    symbols.add_symbol("x", 2);
    symbols.add_symbol("y", 3);
    let symbols = AtomicRc::new(symbols);

    let mut first = StdVectorFst::new();
    first.set_input_symbols(Some(AtomicRc::clone(&symbols)));
    first.set_output_symbols(Some(AtomicRc::clone(&symbols)));
    let s0 = first.add_state();
    let s1 = first.add_state();
    first.set_start(s0);
    first.set_final(s1, TropicalWeight::one());
    first.add_arc(s0, StdArc::new(1, 2, TropicalWeight(0.5), s1));
    arc_sort(&mut first, &OLabelCompare);

    let mut second = StdVectorFst::new();
    second.set_input_symbols(Some(AtomicRc::clone(&symbols)));
    second.set_output_symbols(Some(symbols));
    let s0 = second.add_state();
    let s1 = second.add_state();
    second.set_start(s0);
    second.set_final(s1, TropicalWeight::one());
    second.add_arc(s0, StdArc::new(2, 3, TropicalWeight(1.5), s1));
    arc_sort(&mut second, &ILabelCompare);

    (first, second)
}

#[test]
fn composing_then_searching_finds_the_one_path() {
    let (first, second) = cascade();

    let mut composed = StdVectorFst::new();
    compose(&first, &second, &mut composed).expect("a composition");
    assert!(
        composed.num_states() > 0,
        "the first writes x and the second reads it, so they do meet"
    );

    let mut best = StdVectorFst::new();
    shortest_path(&composed, &mut best, &ShortestPathOptions::default()).expect("a shortest path");

    let start = best.start().expect("a path was found");
    let arc = best.arcs(start).next().expect("the path's first arc");
    assert_eq!(
        arc.ilabel(),
        1,
        "the input side is what the first side read"
    );
    assert_eq!(
        arc.olabel(),
        3,
        "the output side is what the second side wrote"
    );
    assert_eq!(
        arc.weight(),
        &TropicalWeight(2.0),
        "the two arcs' weights multiply: 0.5 + 1.5 in the tropical semiring"
    );
    assert_eq!(
        best.final_weight(arc.nextstate()),
        TropicalWeight::one(),
        "and the path ends there for nothing more"
    );
}
