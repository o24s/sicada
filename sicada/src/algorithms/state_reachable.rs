//! Which final states a state can reach.
//!
//! Port of OpenFst's `state-reachable.h`.
//!
//! The question "can state `s` reach final state `f`?" is asked over and over
//! by lookahead matching, so the answer is precomputed. The trick is that in an
//! acyclic FST the final states can be numbered in depth-first pre-order, and
//! then the set of final states reachable from any state is a small union of
//! contiguous ranges; see [`IntervalSet`].

use crate::algorithms::connect::condense;
use crate::algorithms::dfs_visit::{DfsVisitor, dfs_visit_any};
use crate::arc::{Arc, ArcStateId};
use crate::data_structures::interval_set::{IntInterval, IntervalSet};
use crate::error::OpenFstError;
use crate::fst::Fst;
use crate::fsts::vector_fst::VectorFst;
use crate::properties::K_ACYCLIC;
use crate::weight::Weight;

/// The number a final state is known by.
///
/// SICADA-DIVERGE: upstream leaves this a template parameter defaulting to the
/// state ID type. It indexes the final states rather than the states, so it is
/// not a state ID, and `i64` holds any count a state ID could.
pub type Index = i64;

/// The value meaning "this state is not final, and so has no index".
const NO_INDEX: Index = -1;

/// Numbers the final states in pre-order and collects, for each state, the
/// range of final-state numbers it can reach.
///
/// Only meaningful on an acyclic FST; a cycle makes the ranges ill-defined,
/// which is why [`StateReachable`] condenses first.
struct IntervalReachVisitor<'f, A: Arc, F: Fst<A>> {
    fst: &'f F,
    isets: Vec<IntervalSet<Index>>,
    state2index: Vec<Index>,
    /// The next number to hand out, or `None` when the numbering was supplied
    /// by the caller and must be used as given.
    next_index: Option<Index>,
    error: Option<&'static str>,
    _marker: std::marker::PhantomData<A>,
}

impl<'f, A: Arc, F: Fst<A>> IntervalReachVisitor<'f, A, F> {
    fn new(fst: &'f F) -> Self {
        Self {
            fst,
            isets: Vec::new(),
            state2index: Vec::new(),
            next_index: Some(1),
            error: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Grows both tables to cover `s`.
    fn ensure(&mut self, s: usize) {
        if self.isets.len() <= s {
            self.isets.resize_with(s + 1, IntervalSet::new);
        }
        if self.state2index.len() <= s {
            self.state2index.resize(s + 1, NO_INDEX);
        }
    }
}

impl<A: Arc, F: Fst<A>> DfsVisitor<A> for IntervalReachVisitor<'_, A, F> {
    fn init_visit<G: Fst<A>>(&mut self, _fst: &G) {
        self.error = None;
    }

    fn init_state(&mut self, s: A::StateId, _root: A::StateId) -> bool {
        let idx = s.as_usize();
        self.ensure(idx);
        if self.fst.final_weight(s) == A::Weight::zero() {
            return true;
        }
        // A final state opens a range at the number it is given. The range is
        // closed again when the state finishes, by which point every final
        // state below it has been numbered, so the range covers exactly the
        // final states reachable through it.
        match self.next_index {
            Some(index) => {
                self.isets[idx]
                    .intervals_mut()
                    .push(IntInterval::new(index, index + 1));
                self.state2index[idx] = index;
                self.next_index = Some(index + 1);
            }
            None => {
                if self.fst.num_arcs(s) > 0 {
                    self.error =
                        Some("a supplied numbering requires the final states to have no arcs");
                    return false;
                }
                let index = self.state2index[idx];
                if index == NO_INDEX {
                    self.error = Some("the supplied numbering is incomplete");
                    return false;
                }
                self.isets[idx]
                    .intervals_mut()
                    .push(IntInterval::new(index, index + 1));
            }
        }
        true
    }

    #[inline]
    fn tree_arc(&mut self, _s: A::StateId, _arc: &A) -> bool {
        true
    }

    fn back_arc(&mut self, _s: A::StateId, _arc: &A) -> bool {
        self.error = Some("the FST has a cycle");
        false
    }

    fn forward_or_cross_arc(&mut self, s: A::StateId, arc: &A) -> bool {
        // The destination is already finished, so its ranges are settled and
        // can be taken as they are.
        let (from, to) = (s.as_usize(), arc.nextstate().as_usize());
        self.ensure(from.max(to));
        let reached = std::mem::take(&mut self.isets[to]);
        self.isets[from].union(&reached);
        self.isets[to] = reached;
        true
    }

    fn finish_state(&mut self, s: A::StateId, parent: Option<A::StateId>, _arc: Option<&A>) {
        let idx = s.as_usize();
        self.ensure(idx);
        if let Some(index) = self.next_index
            && self.fst.final_weight(s) != A::Weight::zero()
        {
            // Close the range: everything numbered while this state was open
            // is reachable from it.
            self.isets[idx].intervals_mut()[0].end = index;
        }
        self.isets[idx].normalize();
        if let Some(parent) = parent {
            let parent_idx = parent.as_usize();
            self.ensure(parent_idx);
            let reached = std::mem::take(&mut self.isets[idx]);
            self.isets[parent_idx].union(&reached);
            self.isets[idx] = reached;
        }
    }

    fn finish_visit(&mut self) {}
}

/// Answers whether one state can reach a given final state.
pub struct StateReachable {
    isets: Vec<IntervalSet<Index>>,
    state2index: Vec<Index>,
}

impl StateReachable {
    /// Precomputes the reachability of every final state from every state.
    ///
    /// A cyclic FST is condensed first, so the answer is about the components
    /// rather than the states, which is the same answer, since everything in a
    /// component reaches everything else in it.
    ///
    /// SICADA-DIVERGE: upstream reports every failure through an `Error()` flag
    /// the caller has to remember to check, and its copy constructor sets that
    /// flag rather than copying. Here a failure is a `Result`, and the type is
    /// cloneable because there is nothing stopping it.
    pub fn new<A: Arc, F: Fst<A>>(fst: &F) -> Result<Self, OpenFstError> {
        if fst.properties(K_ACYCLIC, true) & K_ACYCLIC != 0 {
            Self::acyclic(fst)
        } else {
            Self::cyclic(fst)
        }
    }

    fn acyclic<A: Arc, F: Fst<A>>(fst: &F) -> Result<Self, OpenFstError> {
        let mut visitor = IntervalReachVisitor::new(fst);
        dfs_visit_any(fst, &mut visitor);
        if let Some(reason) = visitor.error {
            return Err(OpenFstError::InvalidOperation(format!(
                "StateReachable: {reason}"
            )));
        }
        Ok(Self {
            isets: visitor.isets,
            state2index: visitor.state2index,
        })
    }

    fn cyclic<A: Arc, F: Fst<A>>(fst: &F) -> Result<Self, OpenFstError> {
        let mut condensed = VectorFst::<A>::new();
        let mut scc: Vec<A::StateId> = Vec::new();
        condense(fst, &mut condensed, &mut scc);
        let reachable = Self::new(&condensed)?;

        // How many states each component swallowed, so that a final state
        // sitting inside a cycle can be caught: its own range would then stand
        // for the whole component.
        let mut component_size: Vec<usize> = Vec::new();
        for &c in &scc {
            let c = c.as_usize();
            if component_size.len() <= c {
                component_size.resize(c + 1, 0);
            }
            component_size[c] += 1;
        }

        let mut isets = vec![IntervalSet::new(); scc.len()];
        let mut state2index = vec![NO_INDEX; scc.len()];
        for (s, &c) in scc.iter().enumerate() {
            let c = c.as_usize();
            isets[s] = reachable.isets[c].clone();
            state2index[s] = reachable.state2index[c];
            if condensed.final_weight(A::StateId::from_usize(c)) != A::Weight::zero()
                && component_size[c] > 1
            {
                return Err(OpenFstError::InvalidOperation(
                    "StateReachable: a final state is contained in a cycle".to_string(),
                ));
            }
        }
        Ok(Self { isets, state2index })
    }

    /// Whether `to`, which must be a final state, can be reached from `from`.
    ///
    /// A state that is not final has no number and so is never reported
    /// reachable, since reaching it is not the same as reaching a final state.
    pub fn reach<S: ArcStateId>(&self, from: S, to: S) -> bool {
        let (from, to) = (from.as_usize(), to.as_usize());
        let Some(&index) = self.state2index.get(to) else {
            return false;
        };
        if index == NO_INDEX {
            return false;
        }
        self.isets.get(from).is_some_and(|iset| iset.member(index))
    }

    /// The number each final state was given, or `NO_INDEX` for a state that
    /// is not final.
    pub fn state2index(&self) -> &[Index] {
        &self.state2index
    }

    /// The final-state numbers each state can reach.
    pub fn interval_sets(&self) -> &[IntervalSet<Index>] {
        &self.isets
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::fst::{ExpandedFst as _, MutableFst};
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::weights::float_weight::TropicalWeight;

    fn build(nstates: usize, edges: &[(i32, i32)], finals: &[i32]) -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..nstates {
            fst.add_state();
        }
        fst.set_start(0);
        for &(from, to) in edges {
            fst.add_arc(from, StdArc::new(1, 1, TropicalWeight::one(), to));
        }
        for &s in finals {
            fst.set_final(s, TropicalWeight::one());
        }
        fst
    }

    /// Reachability computed the direct way, for comparison.
    fn brute_force(fst: &StdVectorFst, from: usize, to: usize) -> bool {
        let n = fst.num_states();
        let mut seen = vec![false; n];
        let mut stack = vec![from];
        seen[from] = true;
        while let Some(s) = stack.pop() {
            if s == to {
                return true;
            }
            for arc in fst.arcs(s as i32) {
                let next = arc.nextstate() as usize;
                if !seen[next] {
                    seen[next] = true;
                    stack.push(next);
                }
            }
        }
        false
    }

    /// The answer has to match the definition, for every pair of states.
    fn assert_matches_reachability(fst: &StdVectorFst) {
        let reachable = StateReachable::new(fst).expect("acyclic or condensable");
        let n = fst.num_states();
        for from in 0..n {
            for to in 0..n {
                let is_final = fst.final_weight(to as i32) != TropicalWeight::zero();
                let want = is_final && brute_force(fst, from, to);
                assert_eq!(
                    reachable.reach(from as i32, to as i32),
                    want,
                    "{from} -> {to}"
                );
            }
        }
    }

    #[test]
    fn a_chain_reaches_the_final_states_after_it() {
        // 0 → 1 → 2 → 3, with 1 and 3 final.
        assert_matches_reachability(&build(4, &[(0, 1), (1, 2), (2, 3)], &[1, 3]));
    }

    #[test]
    fn a_branching_fst_reaches_the_final_states_down_each_branch() {
        // 0 branches to 1 and 2; 1 leads to 3, 2 to 4; 3 and 4 are final.
        assert_matches_reachability(&build(5, &[(0, 1), (0, 2), (1, 3), (2, 4)], &[3, 4]));
    }

    /// A state reached from two places has to be reachable from both, which is
    /// what the non-tree arcs are for.
    #[test]
    fn a_shared_final_state_is_reachable_from_both_sides() {
        assert_matches_reachability(&build(4, &[(0, 1), (0, 2), (1, 3), (2, 3)], &[3]));
    }

    #[test]
    fn a_state_that_is_not_final_is_never_reported_reachable() {
        let fst = build(3, &[(0, 1), (1, 2)], &[2]);
        let reachable = StateReachable::new(&fst).unwrap();
        assert!(reachable.reach(0, 2));
        // State 1 is on the way, but it is not a final state.
        assert!(!reachable.reach(0, 1));
        assert_eq!(reachable.state2index()[1], NO_INDEX);
    }

    /// A cyclic FST is condensed first: everything in a component reaches
    /// everything else in it, so the answer is the component's.
    #[test]
    fn a_cyclic_fst_is_answered_through_its_components() {
        // 0 → 1 → 2 → 1 is a cycle; 3 is final and reached from 2.
        let fst = build(4, &[(0, 1), (1, 2), (2, 1), (2, 3)], &[3]);
        let reachable = StateReachable::new(&fst).unwrap();
        assert!(reachable.reach(0, 3));
        assert!(reachable.reach(1, 3));
        assert!(reachable.reach(2, 3));
        // A final state reaches itself by the empty path, which its own number
        // being inside its own range records.
        assert!(reachable.reach(3, 3));
        assert!(!reachable.reach(3, 0), "nothing leads back out of state 3");
    }

    /// A final state inside a cycle cannot be numbered: its range would have to
    /// stand for every state of the component.
    #[test]
    fn a_final_state_inside_a_cycle_is_refused() {
        let fst = build(3, &[(0, 1), (1, 2), (2, 1)], &[1]);
        assert!(StateReachable::new(&fst).is_err());
    }

    #[test]
    fn an_fst_with_no_final_states_reaches_nothing() {
        let fst = build(3, &[(0, 1), (1, 2)], &[]);
        let reachable = StateReachable::new(&fst).unwrap();
        for from in 0..3 {
            for to in 0..3 {
                assert!(!reachable.reach(from, to));
            }
        }
    }
}
