//! A decoding graph, built the way a recogniser builds one, against OpenFst's.
//!
//! H o L o G: the graph a monophone or CTC system decodes with, without the
//! context-dependency transducer C. It is the end-to-end exercise for the
//! algorithms, since it runs compose, epsilon removal, determinization and
//! minimization over weighted transducers with output epsilons and shared
//! prefixes, which is what they exist for.
//!
//! The lexicon has the two shapes that make it non-trivial: "car" is a prefix
//! of "cart", and "two" and "to" are the same phones. Both need disambiguation
//! symbols, without which L o G does not determinize.
//!
//! `tests/fixtures/openfst-hlg.fst` is the same graph built by the real OpenFst
//! (`tests/oracles/hlg-reference.cc`). The result here has to be isomorphic to
//! it: state numbering is not part of the answer, everything else is.

use sicada::algorithms::arcsort::{ILabelCompare, OLabelCompare, arc_sort};
use sicada::algorithms::compose::compose;
use sicada::algorithms::connect::connect;
use sicada::algorithms::determinize::{DeterminizeOptions, determinize};
use sicada::algorithms::isomorphic::isomorphic;
use sicada::algorithms::minimize::minimize;
use sicada::algorithms::rmepsilon::rm_epsilon;
use sicada::arc::{Arc as _, StdArc};
use sicada::fst::{ExpandedFst, Fst, FstReadOptions, MutableFst};
use sicada::fsts::vector_fst::VectorFst;
use sicada::weight::Weight;
use sicada::weights::float_weight::TropicalWeight;

type F = VectorFst<StdArc>;
fn w(v: f32) -> TropicalWeight {
    TropicalWeight(v)
}
fn one() -> TropicalWeight {
    TropicalWeight::one()
}

fn build(
    n: usize,
    start: i32,
    finals: &[(i32, TropicalWeight)],
    arcs: &[(i32, i32, i32, TropicalWeight, i32)],
) -> F {
    let mut f = F::new();
    for _ in 0..n {
        f.add_state();
    }
    f.set_start(start);
    for &(s, fw) in finals {
        f.set_final(s, fw);
    }
    for &(s, il, ol, wt, next) in arcs {
        f.add_arc(s, StdArc::new(il, ol, wt, next));
    }
    f
}

fn shape(name: &str, f: &F) {
    let arcs: usize = (0..f.num_states() as i32).map(|s| f.num_arcs(s)).sum();
    println!("{name:<6} {} states {} arcs", f.num_states(), arcs);
}

#[test]
fn the_pipeline_builds_the_graph_openfst_builds() {
    // G: (cat|car|cart)*
    let g = build(
        1,
        0,
        &[(0, one())],
        &[
            (0, 1, 1, w(1.0), 0),
            (0, 2, 2, w(2.0), 0),
            (0, 3, 3, w(1.5), 0),
            (0, 4, 4, w(0.5), 0),
            (0, 5, 5, w(0.25), 0),
        ],
    );
    // L: phones -> words, with #1 on "car" because it is a prefix of "cart"
    let mut l = build(
        15,
        0,
        &[(0, one())],
        &[
            (0, 1, 1, one(), 1),
            (1, 2, 0, one(), 2),
            (2, 3, 0, one(), 0),
            (0, 1, 2, one(), 3),
            (3, 2, 0, one(), 4),
            (4, 4, 0, one(), 5),
            (5, 5, 0, one(), 0),
            (0, 1, 3, one(), 6),
            (6, 2, 0, one(), 7),
            (7, 4, 0, one(), 8),
            (8, 3, 0, one(), 0),
            (0, 3, 4, one(), 9),
            (9, 6, 0, one(), 10),
            (10, 7, 0, one(), 0),
            (0, 3, 5, one(), 12),
            (12, 6, 0, one(), 13),
            (13, 8, 0, one(), 0),
        ],
    );
    // H: transition ids -> phones
    let mut h = F::new();
    h.add_state();
    h.set_start(0);
    h.set_final(0, one());
    for p in [1i32, 2, 3, 4, 6] {
        let s = h.add_state();
        h.add_arc(0, StdArc::new(10 * p + 1, p, one(), s));
        h.add_arc(s, StdArc::new(10 * p + 2, 0, one(), 0));
    }
    for d in [5i32, 7, 8] {
        h.add_arc(0, StdArc::new(d, d, one(), 0));
    }
    shape("G", &g);
    shape("L", &l);
    shape("H", &h);

    arc_sort(&mut l, &OLabelCompare);
    let mut lg = F::new();
    compose(&l, &g, &mut lg).expect("L o G");
    shape("L.G", &lg);
    rm_epsilon(&mut lg, true).expect("rmeps");
    shape("rmeps", &lg);
    let mut dlg = F::new();
    determinize(&lg, &mut dlg, &DeterminizeOptions::default()).expect("det LG");
    shape("det", &dlg);

    arc_sort(&mut h, &OLabelCompare);
    arc_sort(&mut dlg, &ILabelCompare);
    let mut hlg = F::new();
    compose(&h, &dlg, &mut hlg).expect("H o LG");
    shape("H.LG", &hlg);
    rm_epsilon(&mut hlg, true).expect("rmeps");
    let mut dhlg = F::new();
    determinize(&hlg, &mut dhlg, &DeterminizeOptions::default()).expect("det HLG");
    shape("det", &dhlg);
    minimize(&mut dhlg, 1e-6, false).expect("minimize");
    shape("min", &dhlg);
    connect(&mut dhlg);
    shape("HLG", &dhlg);

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/openfst-hlg.fst"
    );
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let theirs: F = VectorFst::read(&mut bytes.as_slice(), &FstReadOptions::default()).unwrap();
    shape("ref", &theirs);
    assert!(
        isomorphic(&dhlg, &theirs, 1e-6).expect("isomorphic"),
        "sicada's HLG is not isomorphic to OpenFst's"
    );
    println!("→ isomorphic to OpenFst's HLG");
}
