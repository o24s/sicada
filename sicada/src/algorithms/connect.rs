use crate::algorithms::cc_visitors::SccVisitor;
use crate::algorithms::dfs_visit::dfs_visit_any;
use crate::arc::{Arc, ArcStateId};
use crate::data_structures::bit_set::GrowableBitSet;
use crate::fst::{Fst, MutableFst};
use crate::properties::{K_ACCESSIBLE, K_ACYCLIC, K_CO_ACCESSIBLE, K_INITIAL_ACYCLIC};
use crate::weight::Weight;

/// Trims an FST, removing states and arcs that are not on successful paths.
/// This version modifies its input.
///
/// Complexity:
///   Time:  O(V + E)
///   Space: O(V + E)
/// where V = # of states and E = # of arcs.
pub fn connect<A: Arc, F: MutableFst<A>>(fst: &mut F) {
    let mut access = GrowableBitSet::new();
    let mut coaccess = GrowableBitSet::new();
    let mut props = 0;

    {
        let mut visitor = SccVisitor::new(
            &*fst,
            None,
            Some(&mut access),
            Some(&mut coaccess),
            &mut props,
        );
        dfs_visit_any(&*fst, &mut visitor);
    }

    // Over the FST's own states rather than however far the two sets happen to
    // reach: a state that is neither accessible nor coaccessible is exactly the
    // kind that leaves no mark on either, and it is the kind being looked for.
    let nstates = fst.num_states();
    let mut dstates = Vec::with_capacity(nstates);
    for s in 0..nstates {
        if !access.contains(s) || !coaccess.contains(s) {
            dstates.push(A::StateId::from_usize(s));
        }
    }

    fst.delete_states(&dstates);

    // Deleting states narrows the property bits on general grounds, but what is
    // left after this is by construction on a successful path.
    fst.set_properties(
        K_ACCESSIBLE | K_CO_ACCESSIBLE,
        K_ACCESSIBLE | K_CO_ACCESSIBLE,
    );
}

/// Returns an acyclic FST where each SCC in the input FST has been condensed to
/// a single state with transitions between SCCs retained and within SCCs dropped.
/// Also populates `scc` with a mapping from input to output states.
pub fn condense<A: Arc, F1: Fst<A>, F2: MutableFst<A>>(
    ifst: &F1,
    ofst: &mut F2,
    scc: &mut Vec<A::StateId>,
) {
    ofst.delete_all_states();
    let mut props = 0;

    {
        let mut visitor = SccVisitor::new(ifst, Some(scc), None, None, &mut props);
        dfs_visit_any(ifst, &mut visitor);
    }

    let max_scc = scc.iter().max();
    if max_scc.is_none() || *max_scc.unwrap() == A::StateId::no_state() {
        return;
    }

    let num_condensed_states = max_scc.unwrap().as_usize() + 1;
    ofst.reserve_states(num_condensed_states);
    for _ in 0..num_condensed_states {
        ofst.add_state();
    }

    for s in 0..scc.len() {
        let c = scc[s];
        if c == A::StateId::no_state() {
            continue;
        }

        let state_id = A::StateId::from_usize(s);

        if Some(state_id) == ifst.start() {
            ofst.set_start(c);
        }

        let weight = ifst.final_weight(state_id);
        if weight.is_member() && weight != A::Weight::zero() {
            let curr_final = ofst.final_weight(c);
            ofst.set_final(c, curr_final.plus(&weight));
        }

        for arc in ifst.arcs(state_id) {
            let next_idx = arc.nextstate().as_usize();
            let nextc = scc[next_idx];
            if nextc != c {
                let condensed_arc = A::new(arc.ilabel(), arc.olabel(), arc.weight().clone(), nextc);
                ofst.add_arc(c, condensed_arc);
            }
        }
    }

    ofst.set_properties(K_ACYCLIC | K_INITIAL_ACYCLIC, K_ACYCLIC | K_INITIAL_ACYCLIC);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::{Arc, StdArc};
    use crate::float_weight::TropicalWeight;
    use crate::fst::{ExpandedFst as _, Fst, MutableFst};
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::weight::Weight;

    #[test]
    fn test_connect() {
        let mut fst = StdVectorFst::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state(); // dead end
        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // 0 -> 1 -> 2 (successful path)
        fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s1, StdArc::new(2, 2, TropicalWeight::one(), s2));

        // 1 -> 3 (dead end)
        fst.add_arc(s1, StdArc::new(3, 3, TropicalWeight::one(), s3));

        assert_eq!(fst.num_states(), 4);

        connect(&mut fst);

        // State 3 should be removed.
        assert_eq!(fst.num_states(), 3);
        // Properties check for K_ACCESSIBLE & K_CO_ACCESSIBLE
        let target_props = K_ACCESSIBLE | K_CO_ACCESSIBLE;
        assert_eq!(
            fst.properties(target_props, false) & target_props,
            target_props
        );
    }

    /// A state nothing reaches, numbered above every state that is reached.
    #[test]
    fn an_unreachable_last_state_is_removed() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.set_final(1, TropicalWeight::one());
        // State 2 is reached by nothing and reaches nothing.

        connect(&mut fst);
        assert_eq!(fst.num_states(), 2);
    }

    /// The definition, checked against a brute-force one: a state survives
    /// exactly when the start reaches it and it reaches a final state.
    #[test]
    fn connect_keeps_exactly_the_states_on_a_successful_path() {
        let mut rng = 0x1234_5678u64;
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
                    fst.add_arc(
                        s as i32,
                        StdArc::new(1, 1, TropicalWeight::one(), next(nstates) as i32),
                    );
                }
                if next(3) == 0 {
                    fst.set_final(s as i32, TropicalWeight::one());
                }
            }

            // Reachability both ways, by transitive closure.
            let mut reaches = vec![vec![false; nstates]; nstates];
            for (s, row) in reaches.iter_mut().enumerate() {
                row[s] = true;
                for arc in fst.arcs(s as i32) {
                    row[arc.nextstate() as usize] = true;
                }
            }
            for k in 0..nstates {
                let through_k = reaches[k].clone();
                for row in reaches.iter_mut() {
                    if row[k] {
                        for (dest, &r) in row.iter_mut().zip(&through_k) {
                            *dest |= r;
                        }
                    }
                }
            }
            let start = fst.start().unwrap() as usize;
            let kept: Vec<usize> = (0..nstates)
                .filter(|&s| {
                    reaches[start][s]
                        && (0..nstates).any(|f| {
                            reaches[s][f] && fst.final_weight(f as i32) != TropicalWeight::zero()
                        })
                })
                .collect();

            // Remember what survives, so the result can be identified after
            // the renumbering delete_states does.
            let kept_finals: Vec<bool> = kept
                .iter()
                .map(|&s| fst.final_weight(s as i32) != TropicalWeight::zero())
                .collect();

            connect(&mut fst);
            assert_eq!(fst.num_states(), kept.len(), "round {round}");
            let finals: Vec<bool> = (0..fst.num_states())
                .map(|s| fst.final_weight(s as i32) != TropicalWeight::zero())
                .collect();
            assert_eq!(finals, kept_finals, "round {round}");
        }
    }

    /// Condensing collapses each strongly connected component to one state, so
    /// what comes out has no cycles at all, and two states of the input share a
    /// state of the output exactly when each can reach the other.
    #[test]
    fn condensing_collapses_each_component_to_one_acyclic_state() {
        let mut fst = StdVectorFst::new();
        for _ in 0..5 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.set_final(4, TropicalWeight::one());
        // 1 and 2 form a cycle; 3 loops on itself; 0 and 4 stand alone.
        for &(from, to) in &[(0, 1), (1, 2), (2, 1), (2, 3), (3, 3), (3, 4)] {
            fst.add_arc(from, StdArc::new(1, 1, TropicalWeight::one(), to));
        }

        let mut condensed = StdVectorFst::new();
        let mut scc = Vec::new();
        condense(&fst, &mut condensed, &mut scc);

        assert_eq!(scc.len(), 5);
        assert_eq!(scc[1], scc[2], "the cycle is one component");
        for (a, b) in [(0, 1), (0, 3), (0, 4), (1, 3), (1, 4), (3, 4)] {
            assert_ne!(
                scc[a], scc[b],
                "states {a} and {b} are not mutually reachable"
            );
        }
        assert_eq!(condensed.num_states(), 4);

        // No arc of the output stays inside a component, so none can cycle.
        for s in 0..condensed.num_states() as i32 {
            for arc in condensed.arcs(s) {
                assert_ne!(arc.nextstate(), s, "a self-loop survived condensing");
            }
        }
        assert_eq!(condensed.start(), Some(scc[0]));
        assert_eq!(condensed.final_weight(scc[4]), TropicalWeight::one());
    }

    /// Condensing an FST with no states leaves the output empty rather than
    /// reaching into a component vector that was never filled.
    #[test]
    fn condensing_nothing_produces_nothing() {
        let fst = StdVectorFst::new();
        let mut condensed = StdVectorFst::new();
        condensed.add_state();
        let mut scc = Vec::new();
        condense(&fst, &mut condensed, &mut scc);
        assert_eq!(condensed.num_states(), 0);
        assert!(scc.is_empty());
    }
}
