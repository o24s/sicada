//! Putting the arcs leaving each state into order.
//!
//! Port of OpenFst's `arcsort.h`. Sorting lets a matcher find the arcs carrying
//! a given label by binary search instead of a scan, which composition is built
//! on.

use std::cmp::Ordering;

use crate::arc::{Arc, ArcStateId};
use crate::fst::MutableFst;
use crate::properties::{
    K_ACCEPTOR, K_ARC_SORT_PROPERTIES, K_FST_PROPERTIES, K_I_LABEL_SORTED, K_O_LABEL_SORTED,
};

/// An order on the arcs leaving one state.
///
/// A comparison is usable here, rather than any closure, because of the
/// properties method: sorting establishes something about the FST, and the FST
/// has to be told what.
pub trait ArcCompare<A: Arc> {
    /// Which of two arcs comes first.
    fn compare(&self, lhs: &A, rhs: &A) -> Ordering;

    /// The properties the sorted FST has, given those it had before.
    fn properties(&self, props: u64) -> u64;
}

/// Orders arcs by input label, then by output label.
#[derive(Debug, Clone, Copy, Default)]
pub struct ILabelCompare;

impl<A: Arc> ArcCompare<A> for ILabelCompare {
    #[inline]
    fn compare(&self, lhs: &A, rhs: &A) -> Ordering {
        (lhs.ilabel(), lhs.olabel()).cmp(&(rhs.ilabel(), rhs.olabel()))
    }

    fn properties(&self, props: u64) -> u64 {
        // On an acceptor the two sides carry the same label, so sorting by one
        // sorts by the other.
        (props & K_ARC_SORT_PROPERTIES)
            | K_I_LABEL_SORTED
            | if props & K_ACCEPTOR != 0 {
                K_O_LABEL_SORTED
            } else {
                0
            }
    }
}

/// Orders arcs by output label, then by input label.
#[derive(Debug, Clone, Copy, Default)]
pub struct OLabelCompare;

impl<A: Arc> ArcCompare<A> for OLabelCompare {
    #[inline]
    fn compare(&self, lhs: &A, rhs: &A) -> Ordering {
        (lhs.olabel(), lhs.ilabel()).cmp(&(rhs.olabel(), rhs.ilabel()))
    }

    fn properties(&self, props: u64) -> u64 {
        (props & K_ARC_SORT_PROPERTIES)
            | K_O_LABEL_SORTED
            | if props & K_ACCEPTOR != 0 {
                K_I_LABEL_SORTED
            } else {
                0
            }
    }
}

/// Sorts the arcs leaving every state of `fst`.
///
/// The sort is stable, so arcs that compare equal keep the order they were in.
/// That is not a detail: two arcs alike in labels but differing in weight or
/// destination are equal to both comparisons here, and an operation reading
/// them back has to see the same order every time.
///
/// Complexity: time `O(V d log d)`, space `O(d)`, for `V` states of out-degree
/// at most `d`.
///
/// SICADA-DIVERGE: upstream routes this through `StateMap` with an
/// `ArcSortMapper` that holds a `const Fst<Arc>&` to the very FST `StateMap` is
/// writing through, so the mapper reads each state's arcs just before they are
/// replaced. That aliasing lets the delayed `ArcSortFst` share the mapper; the
/// eager path pays the whole state-map machinery for what is a sort in place.
/// Sorting directly is the same work without the alias.
pub fn arc_sort<A, F, C>(fst: &mut F, comp: &C)
where
    A: Arc,
    F: MutableFst<A>,
    C: ArcCompare<A>,
{
    let props = comp.properties(fst.properties(K_FST_PROPERTIES, false));
    // In place through `arcs_mut`. Copying each state's arcs out, sorting the
    // copy, deleting the originals and adding them back made four passes over
    // every state's arcs where one will do, and every `add_arc` re-derived the
    // property bits from the arc before it, work thrown away by the
    // `set_properties` below, which is how the sort really updates them.
    for index in 0..fst.num_states() {
        let state = A::StateId::from_usize(index);
        let arcs = fst.arcs_mut(state);
        if arcs.len() < 2 {
            continue;
        }
        arcs.sort_by(|lhs, rhs| comp.compare(lhs, rhs));
    }
    fst.set_properties(props, K_FST_PROPERTIES);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{Rng, paths, random_acyclic_fst, sorted};
    use crate::arc::StdArc;
    use crate::fst::Fst;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::weight::Weight;
    use crate::weights::float_weight::TropicalWeight;

    fn fst_with(arcs: &[(i32, i32, f32, i32)]) -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..4 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.set_final(3, TropicalWeight::one());
        for (ilabel, olabel, weight, nextstate) in arcs {
            fst.add_arc(
                0,
                StdArc::new(*ilabel, *olabel, TropicalWeight(*weight), *nextstate),
            );
        }
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    fn labels(fst: &StdVectorFst) -> Vec<(i32, i32)> {
        fst.arcs(0).map(|a| (a.ilabel(), a.olabel())).collect()
    }

    #[test]
    fn sorting_by_input_label_breaks_ties_on_the_output_label() {
        let mut fst = fst_with(&[
            (3, 1, 0.0, 1),
            (1, 9, 0.0, 1),
            (1, 2, 0.0, 1),
            (2, 5, 0.0, 1),
        ]);
        arc_sort(&mut fst, &ILabelCompare);
        assert_eq!(labels(&fst), vec![(1, 2), (1, 9), (2, 5), (3, 1)]);
    }

    #[test]
    fn sorting_by_output_label_breaks_ties_on_the_input_label() {
        let mut fst = fst_with(&[
            (3, 1, 0.0, 1),
            (1, 9, 0.0, 1),
            (1, 2, 0.0, 1),
            (2, 5, 0.0, 1),
        ]);
        arc_sort(&mut fst, &OLabelCompare);
        assert_eq!(labels(&fst), vec![(3, 1), (1, 2), (2, 5), (1, 9)]);
    }

    /// Arcs alike in both labels keep the order they were in, so the result
    /// does not depend on how the sort happens to move equal elements.
    #[test]
    fn arcs_that_compare_equal_keep_their_order() {
        let mut fst = fst_with(&[
            (1, 1, 3.0, 1),
            (0, 0, 0.0, 2),
            (1, 1, 2.0, 3),
            (1, 1, 1.0, 2),
        ]);
        arc_sort(&mut fst, &ILabelCompare);
        let after: Vec<(f32, i32)> = fst
            .arcs(0)
            .skip(1)
            .map(|a| (a.weight().value(), a.nextstate()))
            .collect();
        assert_eq!(after, vec![(3.0, 1), (2.0, 3), (1.0, 2)]);
    }

    /// Sorting claims the FST is sorted, and on an acceptor it claims both
    /// sides are, since the two labels are the same.
    #[test]
    fn the_claimed_properties_are_the_ones_the_result_has() {
        let mut transducer = fst_with(&[(3, 1, 0.0, 1), (1, 9, 0.0, 1)]);
        arc_sort(&mut transducer, &ILabelCompare);
        let props = transducer.properties(K_FST_PROPERTIES, false);
        assert_ne!(props & K_I_LABEL_SORTED, 0);
        assert_eq!(
            props & K_O_LABEL_SORTED,
            0,
            "the output side is not sorted, and the FST must not say it is"
        );
        assert_eq!(
            transducer.properties(K_I_LABEL_SORTED | K_O_LABEL_SORTED, true),
            props & (K_I_LABEL_SORTED | K_O_LABEL_SORTED),
            "the claim has to match what a recomputation finds"
        );

        let mut acceptor = fst_with(&[(3, 3, 0.0, 1), (1, 1, 0.0, 1)]);
        arc_sort(&mut acceptor, &ILabelCompare);
        let props = acceptor.properties(K_FST_PROPERTIES, false);
        assert_ne!(props & K_I_LABEL_SORTED, 0);
        assert_ne!(
            props & K_O_LABEL_SORTED,
            0,
            "on an acceptor, sorting one side sorts the other"
        );
    }

    /// Sorting reorders arcs and changes nothing else.
    #[test]
    fn sorting_does_not_change_what_the_fst_accepts() {
        let mut rng = Rng::new(0x0050_A7ED_u64);
        for round in 0..200 {
            let fst = random_acyclic_fst(&mut rng, 6);
            let before = sorted(paths(&fst, 10));

            for by_input in [true, false] {
                let mut copy = fst.clone();
                if by_input {
                    arc_sort(&mut copy, &ILabelCompare);
                } else {
                    arc_sort(&mut copy, &OLabelCompare);
                }
                assert_eq!(sorted(paths(&copy, 10)), before, "round {round}");

                // And the arcs really are in order.
                for state in copy.states() {
                    let arcs: Vec<StdArc> = copy.arcs(state).collect();
                    assert!(
                        arcs.windows(2).all(|w| {
                            let key = |a: &StdArc| {
                                if by_input {
                                    (a.ilabel(), a.olabel())
                                } else {
                                    (a.olabel(), a.ilabel())
                                }
                            };
                            key(&w[0]) <= key(&w[1])
                        }),
                        "round {round}, state {state}"
                    );
                }
            }
        }
    }

    /// A state with fewer than two arcs is already sorted, and an FST with no
    /// states is nothing to sort.
    #[test]
    fn there_is_nothing_to_do_for_a_state_with_one_arc() {
        let mut empty = StdVectorFst::new();
        arc_sort(&mut empty, &ILabelCompare);
        assert_eq!(empty.start(), None);

        let mut one = fst_with(&[(5, 6, 1.0, 1)]);
        arc_sort(&mut one, &ILabelCompare);
        assert_eq!(labels(&one), vec![(5, 6)]);
        assert_ne!(
            one.properties(K_I_LABEL_SORTED, false) & K_I_LABEL_SORTED,
            0
        );
    }
}
