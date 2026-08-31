//! Removing final states that are reached only by epsilon.
//!
//! Port of OpenFst's `rmfinalepsilon.h`.
//!
//! A final state with nothing useful after it contributes only its final
//! weight. If the only way in is an epsilon arc, that arc consumes no symbol
//! either, so the whole thing can be folded into the final weight of the state
//! before it and the state dropped.

use rustc_hash::FxHashSet;

use crate::algorithms::cc_visitors::SccVisitor;
use crate::algorithms::connect::connect;
use crate::algorithms::dfs_visit::dfs_visit_any;
use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::data_structures::bit_set::GrowableBitSet;
use crate::fst::MutableFst;
use crate::weight::{LEFT_SEMIRING, Weight};

/// Removes the final states whose only incoming arcs are epsilons and which
/// lead nowhere useful.
///
/// The final weights they carried move onto the arcs that reached them.
pub fn rm_final_epsilon<A: Arc, F: MutableFst<A>>(fst: &mut F) {
    let mut coaccess = GrowableBitSet::new();
    let mut props = 0;
    {
        let mut visitor = SccVisitor::new(&*fst, None, None, Some(&mut coaccess), &mut props);
        dfs_visit_any(&*fst, &mut visitor);
    }

    // The candidates: final states with no future worth keeping, meaning every
    // arc out of them leads somewhere that reaches no final state.
    let epsilon = A::Label::epsilon();
    let zero = A::Weight::zero();
    let mut removable: FxHashSet<A::StateId> = FxHashSet::default();
    for s in fst.states() {
        if fst.final_weight(s) == zero {
            continue;
        }
        let future_coaccess = fst
            .arcs(s)
            .any(|arc| coaccess.contains(arc.nextstate().as_usize()));
        if !future_coaccess {
            removable.insert(s);
        }
    }

    // Whether several such arcs can be folded into one final weight at once, or
    // only one. Adding a second means distributing a product over a sum on the
    // left, which not every semiring allows, so the check is on the weight's
    // own declared properties rather than a bound, since upstream supports both
    // and simply does less work in the weaker case.
    let left_distributive = A::Weight::properties() & LEFT_SEMIRING != 0;

    let states: Vec<A::StateId> = fst.states().collect();
    let mut kept: Vec<A> = Vec::new();
    for s in states {
        let mut weight = fst.final_weight(s);
        kept.clear();
        let narcs = fst.num_arcs(s);

        for arc in fst.arcs(s) {
            let folds = removable.contains(&arc.nextstate())
                && arc.ilabel() == epsilon
                && arc.olabel() == epsilon
                && (weight == zero || left_distributive);
            if folds {
                let through = arc.weight().times(&fst.final_weight(arc.nextstate()));
                weight = through.plus(&weight);
            } else {
                kept.push(arc);
            }
        }

        if kept.len() < narcs {
            fst.delete_arcs(s);
            fst.set_final(s, weight);
            for arc in kept.drain(..) {
                fst.add_arc(s, arc);
            }
        }
    }

    // The states that were folded away are now unreachable, or lead nowhere.
    connect(fst);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{Rng, random_acyclic_fst, string_weights, visible_paths};
    use crate::arc::StdArc;
    use crate::fst::{ExpandedFst as _, Fst as _};
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::weights::float_weight::TropicalWeight;

    /// A chain 0 → 1 → 2 where 2 is final and reached only by an epsilon.
    fn chain_with_epsilon_tail() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(1.0), 1));
        fst.add_arc(1, StdArc::new(0, 0, TropicalWeight(2.0), 2));
        fst.set_final(2, TropicalWeight(3.0));
        fst
    }

    #[test]
    fn an_epsilon_only_final_state_is_folded_into_the_state_before_it() {
        let mut fst = chain_with_epsilon_tail();
        rm_final_epsilon(&mut fst);

        // State 2 is gone; state 1 carries what the path through it came to.
        assert_eq!(fst.num_states(), 2);
        assert_eq!(fst.final_weight(1), TropicalWeight(5.0));
        assert_eq!(fst.num_arcs(1), 0);
    }

    /// A final state reached by a labelled arc stays: dropping it would lose
    /// the symbol.
    #[test]
    fn a_final_state_reached_by_a_labelled_arc_is_kept() {
        let mut fst = StdVectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(7, 7, TropicalWeight(1.0), 1));
        fst.set_final(1, TropicalWeight(2.0));

        rm_final_epsilon(&mut fst);
        assert_eq!(fst.num_states(), 2);
        assert_eq!(fst.final_weight(1), TropicalWeight(2.0));
    }

    /// A final state with a future that reaches another final state is not a
    /// candidate: it is on a longer path as well as ending one.
    #[test]
    fn a_final_state_with_a_useful_future_is_kept() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(0, 0, TropicalWeight(1.0), 1));
        fst.add_arc(1, StdArc::new(5, 5, TropicalWeight(1.0), 2));
        fst.set_final(1, TropicalWeight(2.0));
        fst.set_final(2, TropicalWeight(3.0));

        rm_final_epsilon(&mut fst);
        assert_eq!(fst.num_states(), 3, "state 1 leads on to state 2");
    }

    /// The point of the operation: what the FST accepts does not change.
    #[test]
    fn removing_final_epsilons_preserves_the_language() {
        let mut rng = Rng::new(0xF1A1_EB50);
        for round in 0..200 {
            let mut fst = random_acyclic_fst(&mut rng, 6);
            // Make some arcs epsilon so there is something to remove.
            for s in 0..fst.num_states() as i32 {
                if rng.below(2) == 0 {
                    fst.mutate_arcs(s, |arc| {
                        *arc = StdArc::new(0, 0, *arc.weight(), arc.nextstate());
                    });
                }
            }

            // Compared by what each string weighs rather than path by path:
            // folding a final state away merges two paths for the same string
            // into one, which is exactly what it is supposed to do.
            let before = string_weights(visible_paths(&fst, 8));
            rm_final_epsilon(&mut fst);
            let after = string_weights(visible_paths(&fst, 8));
            assert_eq!(after, before, "round {round}");
        }
    }

    #[test]
    fn an_fst_with_nothing_to_remove_is_left_alone() {
        let mut fst = StdVectorFst::new();
        fst.add_state();
        fst.set_start(0);
        fst.set_final(0, TropicalWeight(1.0));

        rm_final_epsilon(&mut fst);
        assert_eq!(fst.num_states(), 1);
        assert_eq!(fst.final_weight(0), TropicalWeight(1.0));
    }
}
