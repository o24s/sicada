//! Renumbering the states of an FST.
//!
//! Port of OpenFst's `statesort.h`.

use crate::arc::{Arc, ArcStateId};
use crate::data_structures::bit_set::DenseBitSet;
use crate::error::OpenFstError;
use crate::fst::MutableFst;
use crate::properties::{K_ERROR, K_FST_PROPERTIES, K_STATE_SORT_PROPERTIES};
use crate::weight::Weight;

/// Renumbers the states of an FST in place.
///
/// `order[i]` is the number state `i` takes, so `order` has to be a permutation
/// of the FST's state IDs. What the FST accepts does not change; only the
/// numbering does.
///
/// SICADA-DIVERGE: upstream checks the length of `order` and takes the rest of
/// its contract on trust, so an entry outside the state range reads past the end
/// of a vector. Checking that it is a permutation costs one pass over states an
/// algorithm already walks in full, and it is the difference between an error
/// and a corrupted FST.
pub fn state_sort<A: Arc, F: MutableFst<A>>(
    fst: &mut F,
    order: &[A::StateId],
) -> Result<(), OpenFstError> {
    let nstates = fst.num_states();
    if order.len() != nstates {
        fst.set_properties(K_ERROR, K_ERROR);
        return Err(OpenFstError::InvalidOperation(format!(
            "StateSort: Bad order vector size. Expected {nstates}, but got {}",
            order.len()
        )));
    }
    let mut seen = DenseBitSet::new_empty(nstates);
    for (from, to) in order.iter().enumerate() {
        let to = to.as_usize();
        if to >= nstates || !seen.insert(to) {
            fst.set_properties(K_ERROR, K_ERROR);
            return Err(OpenFstError::InvalidOperation(format!(
                "StateSort: order is not a permutation: state {from} maps to {to}"
            )));
        }
    }

    let start = match fst.start() {
        Some(s) => s,
        None => return Ok(()),
    };

    let props = fst.properties(K_STATE_SORT_PROPERTIES, false);

    let mut done = DenseBitSet::new_empty(order.len());
    let mut arcsa: Vec<A> = Vec::new();
    let mut arcsb: Vec<A> = Vec::new();

    fst.set_start(order[start.as_usize()]);

    for s1_idx in 0..order.len() {
        if done.contains(s1_idx) {
            continue;
        }

        let mut s1 = s1_idx;
        let mut final1 = fst.final_weight(A::StateId::from_usize(s1));
        let mut final2 = A::Weight::zero();

        arcsa.clear();
        arcsa.extend(fst.arcs(A::StateId::from_usize(s1)));

        // Follow the cycle of permutations
        while !done.contains(s1) {
            let s2 = order[s1].as_usize();
            if !done.contains(s2) {
                final2 = fst.final_weight(A::StateId::from_usize(s2));
                arcsb.clear();
                arcsb.extend(fst.arcs(A::StateId::from_usize(s2)));
            }

            let s2_id = A::StateId::from_usize(s2);
            fst.set_final(s2_id, final1.clone());
            fst.delete_arcs(s2_id);

            // Re-add arcs with mapped nextstate
            for arc in arcsa.drain(..) {
                let nextstate_new = order[arc.nextstate().as_usize()];
                let new_arc = A::new(
                    arc.ilabel(),
                    arc.olabel(),
                    arc.weight().clone(),
                    nextstate_new,
                );
                fst.add_arc(s2_id, new_arc);
            }

            done.insert(s1);
            s1 = s2;
            final1 = final2.clone();
            std::mem::swap(&mut arcsa, &mut arcsb);
        }
    }

    fst.set_properties(props, K_FST_PROPERTIES);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::fst::{ExpandedFst as _, Fst, MutableFst};
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::weights::float_weight::TropicalWeight;

    #[test]
    fn test_state_sort() {
        let mut fst = StdVectorFst::new();
        let s0 = fst.add_state(); // old id = 0
        let s1 = fst.add_state(); // old id = 1
        let s2 = fst.add_state(); // old id = 2

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight(2.0));

        fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight(0.5), s1));
        fst.add_arc(s1, StdArc::new(2, 2, TropicalWeight(1.0), s2));

        // We want to reorder such that:
        // old state 0 -> becomes new state 2
        // old state 1 -> becomes new state 0
        // old state 2 -> becomes new state 1
        let order = vec![2, 0, 1];

        assert!(state_sort(&mut fst, &order).is_ok());

        // Start state should now be 2
        assert_eq!(fst.start().unwrap(), 2);

        // Check final weights:
        // Old state 2 is now state 1, so state 1 should be the only final state.
        assert_eq!(fst.final_weight(1), TropicalWeight(2.0));
        assert_eq!(fst.final_weight(0), TropicalWeight::zero());
        assert_eq!(fst.final_weight(2), TropicalWeight::zero());

        // Check arcs:
        // Old state 0 (now 2) should have an arc to old state 1 (now 0)
        let mut arcs_from_2 = fst.arcs(2);
        let arc = arcs_from_2.next().unwrap();
        assert_eq!(arc.ilabel(), 1);
        assert_eq!(arc.nextstate(), 0);

        // Old state 1 (now 0) should have an arc to old state 2 (now 1)
        let mut arcs_from_0 = fst.arcs(0);
        let arc = arcs_from_0.next().unwrap();
        assert_eq!(arc.ilabel(), 2);
        assert_eq!(arc.nextstate(), 1);
    }

    #[test]
    fn test_state_sort_error() {
        let mut fst = StdVectorFst::new();
        fst.add_state();
        fst.add_state();

        let order = vec![0]; // Incorrect length (should be 2)

        let result = state_sort(&mut fst, &order);
        assert!(result.is_err());
        // Verify K_ERROR property is set
        assert_ne!(fst.properties(K_ERROR, false) & K_ERROR, 0);
    }

    #[test]
    fn an_order_that_is_not_a_permutation_is_refused() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);

        // Two states sent to the same number, so one of them would be lost.
        assert!(state_sort(&mut fst, &[0, 0, 2]).is_err());
        // A number outside the state range, which upstream would index with.
        let mut fst2 = StdVectorFst::new();
        for _ in 0..3 {
            fst2.add_state();
        }
        fst2.set_start(0);
        assert!(state_sort(&mut fst2, &[0, 1, 7]).is_err());
    }

    /// Renumbering must not change the FST: every state's final weight and arcs
    /// have to come back under its new number, with the arcs' destinations
    /// renumbered to match.
    #[test]
    fn renumbering_moves_every_state_intact() {
        let mut rng = 0x51ED_F00Du64;
        let mut next = |bound: usize| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as usize) % bound
        };

        for round in 0..200 {
            let nstates = 1 + next(6);
            let mut fst = StdVectorFst::new();
            for _ in 0..nstates {
                fst.add_state();
            }
            fst.set_start(next(nstates) as i32);
            for s in 0..nstates {
                for _ in 0..next(3) {
                    let label = next(4) as i32;
                    fst.add_arc(
                        s as i32,
                        StdArc::new(
                            label,
                            label,
                            TropicalWeight(next(5) as f32),
                            next(nstates) as i32,
                        ),
                    );
                }
                if next(3) == 0 {
                    fst.set_final(s as i32, TropicalWeight(next(5) as f32));
                }
            }

            // A random permutation, by Fisher-Yates.
            let mut order: Vec<i32> = (0..nstates as i32).collect();
            for i in (1..nstates).rev() {
                order.swap(i, next(i + 1));
            }

            let before: Vec<(TropicalWeight, Vec<StdArc>)> = (0..nstates)
                .map(|s| {
                    (
                        fst.final_weight(s as i32),
                        fst.arcs(s as i32).collect::<Vec<_>>(),
                    )
                })
                .collect();
            let old_start = fst.start().unwrap();

            state_sort(&mut fst, &order).unwrap();

            assert_eq!(fst.num_states(), nstates, "round {round}");
            assert_eq!(
                fst.start(),
                Some(order[old_start as usize]),
                "round {round}"
            );
            for (old, (final_weight, arcs)) in before.iter().enumerate() {
                let new = order[old];
                assert_eq!(
                    fst.final_weight(new),
                    *final_weight,
                    "round {round}, state {old}"
                );
                let want: Vec<StdArc> = arcs
                    .iter()
                    .map(|arc| {
                        StdArc::new(
                            arc.ilabel(),
                            arc.olabel(),
                            *arc.weight(),
                            order[arc.nextstate() as usize],
                        )
                    })
                    .collect();
                assert_eq!(
                    fst.arcs(new).collect::<Vec<_>>(),
                    want,
                    "round {round}, state {old} became {new}"
                );
            }
        }
    }

    /// The identity permutation leaves everything where it is, including the
    /// self-mapped states the cycle-following loop has to handle.
    #[test]
    fn the_identity_permutation_changes_nothing() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(1);
        fst.set_final(2, TropicalWeight(3.0));
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight::one(), 2));
        fst.add_arc(1, StdArc::new(3, 3, TropicalWeight::one(), 1));

        let before: Vec<Vec<StdArc>> = (0..3).map(|s| fst.arcs(s).collect()).collect();
        state_sort(&mut fst, &[0, 1, 2]).unwrap();

        assert_eq!(fst.start(), Some(1));
        assert_eq!(fst.final_weight(2), TropicalWeight(3.0));
        for s in 0..3 {
            assert_eq!(fst.arcs(s).collect::<Vec<_>>(), before[s as usize]);
        }
    }

    #[test]
    fn an_fst_with_no_start_state_is_left_alone() {
        let mut fst = StdVectorFst::new();
        fst.add_state();
        fst.add_state();
        assert_eq!(fst.start(), None);
        state_sort(&mut fst, &[1, 0]).unwrap();
        assert_eq!(fst.start(), None);
        assert_eq!(fst.num_states(), 2);
    }
}
