//! What two acceptors both accept.
//!
//! Port of OpenFst's `intersect.h`. Intersection is composition read on
//! acceptors: an acceptor's two sides carry the same label, so running one into
//! the other keeps exactly the strings both have.

use crate::algorithms::compose::compose;
use crate::arc::Arc;
use crate::data_structures::bi_table::BiTableId;
use crate::error::OpenFstError;
use crate::fst::{ExpandedFst, Fst, MutableFst};
use crate::properties::K_ACCEPTOR;

/// The acceptor accepting what `fst1` and `fst2` both accept.
///
/// SICADA-DIVERGE: upstream reports a non-acceptor argument by setting `kError`
/// on the result, which a caller that does not look is free to use. Composition
/// of two transducers is a different operation with a different meaning, so
/// asking for their intersection is an error here.
pub fn intersect<A, F1, F2, FO>(fst1: &F1, fst2: &F2, ofst: &mut FO) -> Result<(), OpenFstError>
where
    A: Arc,
    A::StateId: BiTableId,
    F1: Fst<A> + ExpandedFst<A>,
    F2: Fst<A> + ExpandedFst<A>,
    FO: MutableFst<A> + ExpandedFst<A>,
{
    for (which, fst_props) in [
        ("1st", fst1.properties(K_ACCEPTOR, true)),
        ("2nd", fst2.properties(K_ACCEPTOR, true)),
    ] {
        if fst_props & K_ACCEPTOR == 0 {
            return Err(OpenFstError::InvalidOperation(format!(
                "Intersect: the {which} argument is not an acceptor"
            )));
        }
    }
    compose(fst1, fst2, ofst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{Rng, random_acyclic_fst, string_weights, visible_paths};
    use crate::arc::StdArc;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::K_FST_PROPERTIES;
    use crate::weight::Weight;
    use crate::weights::float_weight::TropicalWeight;

    /// A deterministic acceptor over the given strings.
    fn acceptor(strings: &[&[i32]]) -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        let start = fst.add_state();
        fst.set_start(start);
        for labels in strings {
            let mut state = start;
            for label in *labels {
                let existing = fst.arcs(state).find(|arc| arc.ilabel() == *label);
                state = match existing {
                    Some(arc) => arc.nextstate(),
                    None => {
                        let next = fst.add_state();
                        fst.add_arc(
                            state,
                            StdArc::new(*label, *label, TropicalWeight::one(), next),
                        );
                        next
                    }
                };
            }
            fst.set_final(state, TropicalWeight::one());
        }
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    fn strings(fst: &StdVectorFst) -> Vec<Vec<i32>> {
        let mut out: Vec<Vec<i32>> = string_weights(visible_paths(fst, 16))
            .into_iter()
            .map(|(ilabels, _, _)| ilabels)
            .collect();
        out.sort();
        out
    }

    fn both(fst1: &StdVectorFst, fst2: &StdVectorFst) -> StdVectorFst {
        let mut out = StdVectorFst::new();
        intersect(fst1, fst2, &mut out).unwrap();
        out
    }

    #[test]
    fn the_intersection_is_what_both_accept() {
        let a = acceptor(&[&[1, 2], &[3], &[4, 5]]);
        let b = acceptor(&[&[3], &[4, 5], &[9]]);
        assert_eq!(strings(&both(&a, &b)), vec![vec![3], vec![4, 5]]);
    }

    #[test]
    fn an_intersection_with_nothing_in_common_is_empty() {
        let a = acceptor(&[&[1]]);
        let b = acceptor(&[&[2]]);
        assert!(strings(&both(&a, &b)).is_empty());
    }

    /// The weights of the two are multiplied along the shared path.
    #[test]
    fn the_weights_of_both_are_kept() {
        let mut a = acceptor(&[&[1]]);
        a.set_final(1, TropicalWeight(2.0));
        a.properties(K_FST_PROPERTIES, true);
        let mut b = acceptor(&[&[1]]);
        b.set_final(1, TropicalWeight(3.0));
        b.properties(K_FST_PROPERTIES, true);

        let out = both(&a, &b);
        let weights: Vec<String> = string_weights(visible_paths(&out, 16))
            .into_iter()
            .map(|(_, _, w)| w)
            .collect();
        assert_eq!(weights, vec!["5.0000".to_string()]);
    }

    /// Intersecting an acceptor with itself gives it back.
    #[test]
    fn intersecting_with_itself_gives_the_same_language() {
        let a = acceptor(&[&[1, 2], &[3]]);
        assert_eq!(strings(&both(&a, &a)), strings(&a));
    }

    /// A transducer has no intersection to speak of.
    #[test]
    fn a_transducer_is_refused() {
        let acceptor = acceptor(&[&[1]]);
        let mut transducer = StdVectorFst::new();
        for _ in 0..2 {
            transducer.add_state();
        }
        transducer.set_start(0);
        transducer.add_arc(0, StdArc::new(1, 2, TropicalWeight::one(), 1));
        transducer.set_final(1, TropicalWeight::one());
        transducer.properties(K_FST_PROPERTIES, true);

        let mut out = StdVectorFst::new();
        assert!(intersect(&acceptor, &transducer, &mut out).is_err());
        assert!(intersect(&transducer, &acceptor, &mut out).is_err());
    }

    /// The intersection holds exactly the strings both hold, over random
    /// acceptors.
    #[test]
    fn the_intersection_is_the_intersection_of_the_string_sets() {
        let mut rng = Rng::new(0x000A_0DEF_u64);
        let mut checked = 0;
        for round in 0..100 {
            // Random acceptors over a small alphabet, so they share strings.
            let make = |rng: &mut Rng| {
                let count = 1 + rng.below(5);
                let strings: Vec<Vec<i32>> = (0..count)
                    .map(|_| {
                        let len = 1 + rng.below(3);
                        (0..len).map(|_| 1 + rng.below(3) as i32).collect()
                    })
                    .collect();
                let refs: Vec<&[i32]> = strings.iter().map(|s| s.as_slice()).collect();
                acceptor(&refs)
            };
            let a = make(&mut rng);
            let b = make(&mut rng);

            let sa = strings(&a);
            let sb = strings(&b);
            let want: Vec<Vec<i32>> = sa.iter().filter(|s| sb.contains(s)).cloned().collect();
            if !want.is_empty() {
                checked += 1;
            }
            assert_eq!(strings(&both(&a, &b)), want, "round {round}");
        }
        assert!(checked > 20, "only {checked} intersections had anything");
        let _ = random_acyclic_fst;
    }
}
