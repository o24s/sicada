//! Renumbering an acyclic FST so every arc goes forwards.
//!
//! Port of OpenFst's `topsort.h`.

use crate::algorithms::dfs_visit::{DfsVisitor, dfs_visit_any};
use crate::algorithms::state_sort::state_sort;
use crate::arc::{Arc, ArcStateId};
use crate::error::OpenFstError;
use crate::fst::{Fst, MutableFst};
use crate::properties::{K_ACYCLIC, K_CYCLIC, K_INITIAL_ACYCLIC, K_NOT_TOP_SORTED, K_TOP_SORTED};

/// Collects a topological order from a depth-first search.
///
/// The order is the reverse of the order states finish in, which is the
/// standard fact about depth-first search: a state finishes only after
/// everything reachable from it has, so reversing the finishing order puts
/// every state before everything it can reach.
pub struct TopOrderVisitor<A: Arc> {
    /// States in the order they finished.
    finish: Vec<A::StateId>,
    acyclic: bool,
}

impl<A: Arc> Default for TopOrderVisitor<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Arc> TopOrderVisitor<A> {
    /// Creates a visitor that has seen nothing yet.
    pub fn new() -> Self {
        Self {
            finish: Vec::new(),
            acyclic: true,
        }
    }

    /// Whether the FST turned out to have no cycles.
    pub fn acyclic(&self) -> bool {
        self.acyclic
    }

    /// The position each state takes, or `None` if the FST had a cycle and so
    /// has no topological order.
    ///
    /// SICADA-DIVERGE: upstream writes into a vector the caller owns and leaves
    /// it untouched when the FST is cyclic, so the caller has to remember to
    /// check the separate `acyclic` flag before reading it. An `Option` makes
    /// the two inseparable.
    pub fn order(&self) -> Option<Vec<A::StateId>> {
        if !self.acyclic {
            return None;
        }
        let mut order = vec![A::StateId::no_state(); self.finish.len()];
        for (position, state) in self.finish.iter().rev().enumerate() {
            order[state.as_usize()] = A::StateId::from_usize(position);
        }
        Some(order)
    }
}

impl<A: Arc> DfsVisitor<A> for TopOrderVisitor<A> {
    fn init_visit<F: Fst<A>>(&mut self, _fst: &F) {
        self.finish.clear();
        self.acyclic = true;
    }

    #[inline]
    fn init_state(&mut self, _s: A::StateId, _root: A::StateId) -> bool {
        true
    }

    #[inline]
    fn tree_arc(&mut self, _s: A::StateId, _arc: &A) -> bool {
        true
    }

    /// An arc to a state still on the stack closes a cycle.
    ///
    /// Upstream returns the flag it just cleared, which aborts the search, so
    /// the finishing order it has collected so far is discarded anyway.
    fn back_arc(&mut self, _s: A::StateId, _arc: &A) -> bool {
        self.acyclic = false;
        false
    }

    #[inline]
    fn forward_or_cross_arc(&mut self, _s: A::StateId, _arc: &A) -> bool {
        true
    }

    #[inline]
    fn finish_state(&mut self, s: A::StateId, _parent: Option<A::StateId>, _arc: Option<&A>) {
        self.finish.push(s);
    }

    fn finish_visit(&mut self) {}
}

/// Renumbers `fst` so that every arc goes from a lower state to a higher one.
///
/// Returns whether the FST was acyclic. A cyclic FST has no such numbering and
/// is left exactly as it was.
///
/// Time and space are both `O(V + E)`.
pub fn top_sort<A: Arc, F: MutableFst<A>>(fst: &mut F) -> Result<bool, OpenFstError> {
    let mut visitor = TopOrderVisitor::<A>::new();
    dfs_visit_any(&*fst, &mut visitor);

    match visitor.order() {
        // SICADA-BUGFIX: the search covers every state when there is one to
        // start from, and none at all when there is not, so a short order
        // means the FST has no start state. Upstream hands the short order to
        // StateSort, which rejects it and sets `kError` on an FST that was
        // fine, and then claims `kAcyclic | kInitialAcyclic | kTopSorted`
        // regardless, having examined no arcs. An FST with a cycle among its
        // states comes out marked acyclic. Nothing is reachable here, so there
        // is nothing to renumber, and no structural claim has been checked.
        Some(order) if order.len() != fst.num_states() => Ok(true),
        Some(order) => {
            state_sort(fst, &order)?;
            fst.set_properties(
                K_ACYCLIC | K_INITIAL_ACYCLIC | K_TOP_SORTED,
                K_ACYCLIC | K_INITIAL_ACYCLIC | K_TOP_SORTED,
            );
            Ok(true)
        }
        None => {
            fst.set_properties(K_CYCLIC | K_NOT_TOP_SORTED, K_CYCLIC | K_NOT_TOP_SORTED);
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{Rng, random_acyclic_fst};
    use crate::arc::StdArc;
    use crate::fst::ExpandedFst as _;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::{K_ERROR, K_FST_PROPERTIES};
    use crate::weight::Weight;
    use crate::weights::float_weight::TropicalWeight;

    fn build(nstates: usize, edges: &[(i32, i32)], start: i32) -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..nstates {
            fst.add_state();
        }
        fst.set_start(start);
        for &(from, to) in edges {
            fst.add_arc(from, StdArc::new(1, 1, TropicalWeight::one(), to));
        }
        fst
    }

    /// What topological sorting is for: afterwards every arc goes forwards.
    #[test]
    fn every_arc_goes_forwards_afterwards() {
        let mut rng = Rng::new(0x7095_0077);
        for round in 0..200 {
            let mut fst = random_acyclic_fst(&mut rng, 6);
            // Scramble the numbering so the FST is not already sorted.
            let nstates = fst.num_states();
            let mut order: Vec<i32> = (0..nstates as i32).collect();
            for i in (1..nstates).rev() {
                order.swap(i, rng.below(i + 1));
            }
            crate::algorithms::state_sort::state_sort(&mut fst, &order).unwrap();

            assert!(top_sort(&mut fst).unwrap(), "round {round}");
            for s in 0..fst.num_states() as i32 {
                for arc in fst.arcs(s) {
                    assert!(
                        arc.nextstate() > s,
                        "round {round}: {s} -> {}",
                        arc.nextstate()
                    );
                }
            }
        }
    }

    /// A cyclic FST has no topological order, so it is left exactly as it was.
    #[test]
    fn a_cyclic_fst_is_left_alone() {
        let mut fst = build(3, &[(0, 1), (1, 2), (2, 1)], 0);
        fst.set_final(2, TropicalWeight::one());
        let before: Vec<Vec<StdArc>> = (0..3).map(|s| fst.arcs(s).collect()).collect();

        assert!(!top_sort(&mut fst).unwrap());
        for s in 0..3 {
            assert_eq!(fst.arcs(s).collect::<Vec<_>>(), before[s as usize]);
        }
        let props = fst.properties(K_FST_PROPERTIES, false);
        assert_ne!(props & K_CYCLIC, 0);
        assert_ne!(props & K_NOT_TOP_SORTED, 0);
        assert_eq!(props & K_TOP_SORTED, 0);
    }

    /// A self-loop is a cycle.
    #[test]
    fn a_self_loop_is_a_cycle() {
        let mut fst = build(2, &[(0, 1), (1, 1)], 0);
        fst.set_final(1, TropicalWeight::one());
        assert!(!top_sort(&mut fst).unwrap());
    }

    /// Sorting does not change what the FST accepts, only the numbering.
    #[test]
    fn sorting_preserves_the_states_and_their_arcs() {
        let mut fst = build(4, &[(0, 2), (2, 1), (1, 3), (0, 3)], 0);
        fst.set_final(3, TropicalWeight(2.5));

        assert!(top_sort(&mut fst).unwrap());
        assert_eq!(fst.num_states(), 4);
        // The start state has no arcs coming in, so it sorts first.
        assert_eq!(fst.start(), Some(0));
        // The only final state has none going out, so it sorts last.
        assert_eq!(fst.final_weight(3), TropicalWeight(2.5));
        assert_eq!(fst.count_arcs(), 4);

        let props = fst.properties(K_FST_PROPERTIES, false);
        assert_ne!(props & K_TOP_SORTED, 0);
        assert_ne!(props & K_ACYCLIC, 0);
    }

    /// An FST with states but no start state reaches nothing, so there is
    /// nothing to renumber, and nothing has been examined, so nothing is
    /// claimed. Upstream marks such an FST as being in error and then asserts
    /// that it is acyclic and top-sorted anyway.
    #[test]
    fn an_fst_with_no_start_state_is_left_alone_and_nothing_is_claimed() {
        let mut fst = StdVectorFst::new();
        fst.add_state();
        fst.add_state();
        // A cycle among the unreachable states, which nothing has looked at.
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(1, StdArc::new(1, 1, TropicalWeight::one(), 0));
        let before = fst.properties(K_FST_PROPERTIES, false);

        assert!(top_sort(&mut fst).unwrap());
        assert_eq!(fst.num_states(), 2);
        assert_eq!(fst.count_arcs(), 2);
        assert_eq!(
            fst.properties(K_FST_PROPERTIES, false),
            before,
            "no property was touched"
        );
        assert_eq!(fst.properties(K_ERROR, false) & K_ERROR, 0);
        assert_eq!(fst.properties(K_TOP_SORTED, false) & K_TOP_SORTED, 0);
        assert_eq!(fst.properties(K_ACYCLIC, false) & K_ACYCLIC, 0);
    }

    /// The order a visitor reports is only meaningful when there was no cycle,
    /// which is why the two come together.
    #[test]
    fn no_order_is_reported_for_a_cyclic_fst() {
        let fst = build(2, &[(0, 1), (1, 0)], 0);
        let mut visitor = TopOrderVisitor::<StdArc>::new();
        dfs_visit_any(&fst, &mut visitor);
        assert!(!visitor.acyclic());
        assert!(visitor.order().is_none());
    }
}
