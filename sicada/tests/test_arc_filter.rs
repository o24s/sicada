use sicada::arc::{Arc, StdArc};
use sicada::float_weight::TropicalWeight;
use sicada::fst::{Fst, MutableFst};
use sicada::vector_fst::StdVectorFst;
use sicada::weight::Weight;

use sicada::arc_filter::{
    AnyArcFilter, ArcFilter, EpsilonArcFilter, InputEpsilonArcFilter, LabelArcFilter,
    MultiLabelArcFilter, OutputEpsilonArcFilter,
};

fn build_test_fst() -> StdVectorFst {
    let mut fst = StdVectorFst::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());

    // Epsilons (0 = EPS)
    fst.add_arc(s0, StdArc::new(0, 0, TropicalWeight(1.0), s1));
    fst.add_arc(s0, StdArc::new(0, 5, TropicalWeight(2.0), s1));
    fst.add_arc(s0, StdArc::new(5, 0, TropicalWeight(3.0), s1));

    // Normal labels
    fst.add_arc(s0, StdArc::new(10, 20, TropicalWeight(4.0), s1));
    fst.add_arc(s0, StdArc::new(30, 40, TropicalWeight(5.0), s1));

    fst
}

#[test]
fn test_any_and_epsilon_filters() {
    let fst = build_test_fst();
    let s0 = fst.start().unwrap();

    // AnyArcFilter
    let filter_any = AnyArcFilter;
    let count_any = fst.arcs(s0).filter(|a| filter_any.call(a)).count();
    assert_eq!(count_any, 5);

    // EpsilonArcFilter (ilabel == 0 AND olabel == 0)
    let filter_eps = EpsilonArcFilter;
    let eps_arcs: Vec<_> = fst.arcs(s0).filter(|a| filter_eps.call(a)).collect();
    assert_eq!(eps_arcs.len(), 1);
    assert_eq!(eps_arcs[0].weight().value(), 1.0);

    // InputEpsilonArcFilter (ilabel == 0)
    let filter_ieps = InputEpsilonArcFilter;
    let ieps_arcs: Vec<_> = fst.arcs(s0).filter(|a| filter_ieps.call(a)).collect();
    assert_eq!(ieps_arcs.len(), 2, "Should match (0,0) and (0,5)");

    // OutputEpsilonArcFilter (olabel == 0)
    let filter_oeps = OutputEpsilonArcFilter;
    let oeps_arcs: Vec<_> = fst.arcs(s0).filter(|a| filter_oeps.call(a)).collect();
    assert_eq!(oeps_arcs.len(), 2, "Should match (0,0) and (5,0)");
}

#[test]
fn test_label_filters() {
    let fst = build_test_fst();
    let s0 = fst.start().unwrap();

    let filter_label10 = LabelArcFilter::new(10); // default: match_input = true
    let arcs_10: Vec<_> = fst.arcs(s0).filter(|a| filter_label10.call(a)).collect();
    assert_eq!(arcs_10.len(), 1);
    assert_eq!(arcs_10[0].olabel(), 20);

    let filter_exclude_out_40 = LabelArcFilter::with_options(40, false, false);
    let count_exclude = fst
        .arcs(s0)
        .filter(|a| filter_exclude_out_40.call(a))
        .count();
    assert_eq!(
        count_exclude, 4,
        "Should keep 4 arcs, excluding the one with olabel 40"
    );
}

#[test]
fn test_multi_label_filters() {
    let fst = build_test_fst();
    let s0 = fst.start().unwrap();

    let mut multi_filter = MultiLabelArcFilter::new();
    multi_filter.add_label(10);
    multi_filter.add_label(30);

    let matched: Vec<_> = fst.arcs(s0).filter(|a| multi_filter.call(a)).collect();
    assert_eq!(matched.len(), 2);
    assert_eq!(matched[0].ilabel(), 10);
    assert_eq!(matched[1].ilabel(), 30);
}
