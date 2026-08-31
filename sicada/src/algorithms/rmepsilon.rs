//! Removing the arcs that read and write nothing.
//!
//! Port of OpenFst's `rmepsilon.h`. An epsilon arc consumes no input and
//! produces no output, so what it really does is let a state stand in for
//! everything reachable from it by epsilons, its *epsilon closure*. Removing
//! them means giving each state the arcs of its whole closure, weighted by what
//! it costs to get there.
//!
//! > Mohri, M. 2002. Generic epsilon-removal and input epsilon-normalization
//! > algorithms for weighted transducers. *International Journal of Foundations
//! > of Computer Science* 13(1): 129-143.

use std::cell::RefCell;
use std::rc::Rc;

use crate::algorithms::cc_visitors::SccVisitor;
use crate::algorithms::connect::connect;
use crate::algorithms::dfs_visit::dfs_visit;
use crate::algorithms::prune::{PruneOptions, prune};
use crate::algorithms::shortest_distance::{
    Distance, SHORTEST_DELTA, ShortestDistanceOptions, ShortestDistanceState,
};
use crate::algorithms::topsort::TopOrderVisitor;
use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::arc_filter::{ArcFilter, EpsilonArcFilter};
use crate::data_structures::bit_set::GrowableBitSet;
use crate::error::OpenFstError;
use crate::fst::{ExpandedFst, Fst, MutableFst};
use crate::properties::{K_ACYCLIC, K_FST_PROPERTIES, K_TOP_SORTED, rm_epsilon_properties};
use crate::queue::{AutoQueue, natural_less_unchecked, state_weight_compare};
use crate::weight::{IdempotentWeight, PATH, PathWeight, Weight};
use hashbrown::HashMap;

/// How to remove epsilons.
#[derive(Debug, Clone)]
pub struct RmEpsilonOptions<W> {
    /// Whether to drop the states nothing can reach afterwards.
    pub connect: bool,
    /// Prune the result to this much above the shortest path.
    /// [`Weight::zero`] keeps everything.
    pub weight_threshold: W,
    /// Prune the result to this many states.
    pub state_threshold: Option<usize>,
    /// How closely the closure distances have to converge.
    pub delta: f32,
}

impl<W: Weight> Default for RmEpsilonOptions<W> {
    fn default() -> Self {
        Self {
            connect: true,
            weight_threshold: W::zero(),
            state_threshold: None,
            delta: SHORTEST_DELTA,
        }
    }
}

/// The arcs a state should have once its epsilon closure is folded in, and the
/// final weight it should carry.
struct Closure<A: Arc> {
    /// The arcs, with duplicates merged.
    arcs: Vec<A>,
    /// Where each (ilabel, olabel, nextstate) sits in `arcs`, and which
    /// expansion put it there.
    seen: HashMap<(A::Label, A::Label, A::StateId), (usize, usize)>,
    /// Which expansion is running, so that `seen` need not be cleared.
    expansion: usize,
    /// The final weight.
    final_weight: A::Weight,
    /// Which states the closure walk has reached.
    visited: GrowableBitSet,
    /// Those states, so that only they are cleared.
    visited_states: Vec<A::StateId>,
    /// The closure walk's worklist.
    queue: Vec<A::StateId>,
}

impl<A: Arc> Closure<A> {
    fn new() -> Self {
        Self {
            arcs: Vec::new(),
            seen: HashMap::new(),
            expansion: 0,
            final_weight: A::Weight::zero(),
            visited: GrowableBitSet::new(),
            visited_states: Vec::new(),
            queue: Vec::new(),
        }
    }

    /// Works out what `source` should look like once its closure is folded in.
    ///
    /// `distance` must already hold the distance from `source` to each state
    /// over the epsilon graph.
    fn expand<F: Fst<A>>(&mut self, fst: &F, source: A::StateId, distance: &[A::Weight]) {
        self.arcs.clear();
        self.final_weight = A::Weight::zero();
        self.queue.clear();
        self.queue.push(source);
        let zero = A::Weight::zero();

        while let Some(state) = self.queue.pop() {
            let index = state.as_usize();
            if self.visited.contains(index) {
                continue;
            }
            self.visited.insert(index);
            self.visited_states.push(state);

            let here = distance.get(index).cloned().unwrap_or_else(A::Weight::zero);
            for arc in fst.arcs(state) {
                let weight = here.times(arc.weight());
                if EpsilonArcFilter.call(&arc) {
                    // Still inside the closure: nothing to emit, keep walking.
                    if !self.visited.contains(arc.nextstate().as_usize()) {
                        self.queue.push(arc.nextstate());
                    }
                    continue;
                }
                let key = (arc.ilabel(), arc.olabel(), arc.nextstate());
                match self.seen.get_mut(&key) {
                    // An arc the same expansion already emitted: the two ways
                    // of reaching it are one arc at their sum.
                    Some((expansion, at)) if *expansion == self.expansion => {
                        let merged = self.arcs[*at].weight().plus(&weight);
                        self.arcs[*at] =
                            A::new(arc.ilabel(), arc.olabel(), merged, arc.nextstate());
                    }
                    // Left over from an earlier expansion, so it is reused.
                    Some(entry) => {
                        *entry = (self.expansion, self.arcs.len());
                        self.arcs
                            .push(A::new(arc.ilabel(), arc.olabel(), weight, arc.nextstate()));
                    }
                    None => {
                        self.seen.insert(key, (self.expansion, self.arcs.len()));
                        self.arcs
                            .push(A::new(arc.ilabel(), arc.olabel(), weight, arc.nextstate()));
                    }
                }
            }
            let final_here = fst.final_weight(state);
            if final_here != zero {
                self.final_weight = self.final_weight.plus(&here.times(&final_here));
            }
        }

        for state in self.visited_states.drain(..) {
            self.visited.remove(state.as_usize());
        }
        self.expansion += 1;
    }
}

/// The order to expand states in, so that a state is dealt with after whatever
/// it can reach by epsilons.
fn expansion_order<A, F>(fst: &F) -> Vec<A::StateId>
where
    A: Arc,
    F: Fst<A> + ExpandedFst<A>,
{
    let nstates = fst.num_states();
    let props = fst.properties(K_TOP_SORTED | K_ACYCLIC, false);
    if props & K_TOP_SORTED != 0 {
        return (0..nstates).map(A::StateId::from_usize).collect();
    }
    if props & K_ACYCLIC != 0 {
        let mut visitor = TopOrderVisitor::<A>::new();
        dfs_visit(fst, &mut visitor, EpsilonArcFilter, false);
        if let Some(order) = visitor.order() {
            // `order[state]` is the position; the walk wants the inverse.
            let mut states = vec![A::StateId::from_usize(0); order.len()];
            for (state, position) in order.iter().enumerate() {
                states[position.as_usize()] = A::StateId::from_usize(state);
            }
            return states;
        }
    }
    // Cyclic: group the states by strongly connected component of the epsilon
    // graph, which is as close to a topological order as there is.
    let mut scc: Vec<A::StateId> = Vec::new();
    let mut props = 0;
    {
        let mut visitor = SccVisitor::new(fst, Some(&mut scc), None, None, &mut props);
        dfs_visit(fst, &mut visitor, EpsilonArcFilter, false);
    }
    let ncomponents = scc.iter().map(|id| id.as_usize() + 1).max().unwrap_or(0);
    let mut grouped: Vec<Vec<A::StateId>> = vec![Vec::new(); ncomponents];
    for (state, component) in scc.iter().enumerate() {
        grouped[component.as_usize()].push(A::StateId::from_usize(state));
    }
    grouped.into_iter().flatten().collect()
}

/// Removes the epsilon arcs of `fst` in place.
///
/// `connect` drops the states that nothing reaches once the epsilons are gone.
pub fn rm_epsilon<A, F>(fst: &mut F, connect_after: bool) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: IdempotentWeight,
    F: MutableFst<A> + ExpandedFst<A>,
{
    rm_epsilon_inner(fst, connect_after, SHORTEST_DELTA)
}

/// As [`rm_epsilon`], with pruning.
///
/// Pruning needs the path property and a reverse shortest distance, which is
/// why it takes the reverse arc type; [`rm_epsilon`] does not.
pub fn rm_epsilon_with<A, F>(
    fst: &mut F,
    opts: &RmEpsilonOptions<A::Weight>,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: IdempotentWeight + PathWeight + crate::weight::Divide,
    F: MutableFst<A> + ExpandedFst<A>,
    <A::Weight as Weight>::ReverseWeight: Weight<ReverseWeight = A::Weight>,
{
    let pruning = opts.weight_threshold != A::Weight::zero() || opts.state_threshold.is_some();
    // Connecting is folded into pruning, which drops everything useless anyway.
    rm_epsilon_inner(fst, opts.connect && !pruning, opts.delta)?;
    if pruning {
        prune(
            fst,
            &PruneOptions {
                state_threshold: opts.state_threshold,
                ..PruneOptions::threshold(opts.weight_threshold.clone())
            },
        )?;
    }
    Ok(())
}

fn rm_epsilon_inner<A, F>(fst: &mut F, connect_after: bool, delta: f32) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: IdempotentWeight,
    F: MutableFst<A> + ExpandedFst<A>,
{
    let Some(start) = fst.start() else {
        return Ok(());
    };
    let nstates = fst.num_states();

    // A state nothing reaches other than by an epsilon does not need its own
    // arcs: whoever reaches it will have folded them in already.
    let mut reached_by_label = GrowableBitSet::new();
    reached_by_label.insert(start.as_usize());
    let epsilon = A::Label::epsilon();
    for state in fst.states() {
        for arc in fst.arcs(state) {
            if arc.ilabel() != epsilon || arc.olabel() != epsilon {
                reached_by_label.insert(arc.nextstate().as_usize());
            }
        }
    }

    let order = expansion_order(fst);
    let mut closure = Closure::<A>::new();
    let distance: Distance<A::Weight> = Rc::new(RefCell::new(Vec::new()));

    // One queue for the whole run, as upstream's `RmEpsilon` builds one and
    // hands it to every expansion. Choosing a discipline reads the FST's
    // properties and, for a cyclic one, decomposes it into components; doing
    // that once per state made the pass quadratic in the number of states.
    // Which discipline suits the epsilon graph does not depend on where the
    // walk starts, so there is nothing to redo.
    let comp = state_weight_compare::<A::StateId, A::Weight, _>(
        Rc::clone(&distance),
        natural_less_unchecked::<A::Weight>,
    );
    let comp = (A::Weight::properties() & PATH != 0).then_some(comp);
    let mut queue = AutoQueue::new(&*fst, comp);
    // The distance buffers likewise: this runs once per state, and a fresh set
    // of buffers would be filled from index zero every time, which is quadratic
    // in the number of states however few epsilon arcs there are.
    let mut sd = ShortestDistanceState::<A::Weight, A::StateId>::retained();

    // Working through the order backwards means a state is expanded after the
    // states it can reach by epsilons, so their arcs are already in place.
    for state in order.into_iter().rev() {
        if !reached_by_label.contains(state.as_usize()) && connect_after {
            continue;
        }
        // How far each state is from this one over the epsilon graph alone.
        let opts = ShortestDistanceOptions {
            source: Some(state.as_usize()),
            delta,
            ..ShortestDistanceOptions::new(EpsilonArcFilter)
        };
        sd.run(&*fst, &distance, &mut queue, &opts)?;

        closure.expand(&*fst, state, &distance.borrow());

        fst.set_final(state, closure.final_weight.clone());
        fst.delete_arcs(state);
        for arc in closure.arcs.drain(..) {
            fst.add_arc(state, arc);
        }
    }

    if connect_after {
        // The states that were skipped keep arcs into a graph that no longer
        // describes anything; dropping those arcs lets `connect` remove them.
        for index in 0..nstates {
            if !reached_by_label.contains(index) {
                fst.delete_arcs(A::StateId::from_usize(index));
            }
        }
    }

    let props = rm_epsilon_properties(fst.properties(K_FST_PROPERTIES, false), false);
    fst.set_properties(props, K_FST_PROPERTIES);
    if connect_after {
        connect(fst);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{Rng, random_acyclic_fst, string_weights, visible_paths};
    use crate::arc::StdArc;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::weights::float_weight::TropicalWeight;

    /// What the FST transduces, with epsilons already invisible.
    fn language(fst: &StdVectorFst, max_len: usize) -> Vec<(Vec<i32>, Vec<i32>, String)> {
        string_weights(visible_paths(fst, max_len))
    }

    fn has_epsilons(fst: &StdVectorFst) -> bool {
        fst.states()
            .any(|s| fst.arcs(s).any(|a| a.ilabel() == 0 && a.olabel() == 0))
    }

    /// 0 -eps/2-> 1 -a/3-> 2, final: the epsilon's weight moves onto the arc.
    #[test]
    fn an_epsilon_arc_folds_into_what_follows_it() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(0, 0, TropicalWeight(2.0), 1));
        fst.add_arc(1, StdArc::new(1, 1, TropicalWeight(3.0), 2));
        fst.set_final(2, TropicalWeight(1.0));
        fst.properties(K_FST_PROPERTIES, true);

        let before = language(&fst, 8);
        rm_epsilon(&mut fst, true).unwrap();
        assert!(!has_epsilons(&fst));
        assert_eq!(language(&fst, 8), before);
        assert_eq!(
            language(&fst, 8),
            vec![(vec![1], vec![1], "6.0000".to_string())]
        );
    }

    /// An epsilon into a final state makes its source final.
    #[test]
    fn an_epsilon_into_a_final_state_makes_the_source_final() {
        let mut fst = StdVectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(0, 0, TropicalWeight(2.0), 1));
        fst.set_final(1, TropicalWeight(3.0));
        fst.properties(K_FST_PROPERTIES, true);

        rm_epsilon(&mut fst, true).unwrap();
        assert_eq!(fst.final_weight(fst.start().unwrap()), TropicalWeight(5.0));
        assert!(!has_epsilons(&fst));
    }

    /// Two epsilon routes to the same arc become one arc at their sum.
    #[test]
    fn two_routes_to_the_same_arc_become_one() {
        let mut fst = StdVectorFst::new();
        for _ in 0..4 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(0, 0, TropicalWeight(1.0), 1));
        fst.add_arc(0, StdArc::new(0, 0, TropicalWeight(4.0), 2));
        fst.add_arc(1, StdArc::new(5, 5, TropicalWeight::one(), 3));
        fst.add_arc(2, StdArc::new(5, 5, TropicalWeight::one(), 3));
        fst.set_final(3, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        rm_epsilon(&mut fst, true).unwrap();
        let start = fst.start().unwrap();
        let arcs: Vec<StdArc> = fst.arcs(start).collect();
        assert_eq!(arcs.len(), 1, "{arcs:?}");
        assert_eq!(
            *arcs[0].weight(),
            TropicalWeight(1.0),
            "the lighter of the two routes"
        );
    }

    /// A cycle of epsilons has to settle rather than go round forever.
    #[test]
    fn a_cycle_of_epsilons_settles() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(0, 0, TropicalWeight(1.0), 1));
        fst.add_arc(1, StdArc::new(0, 0, TropicalWeight(1.0), 0));
        fst.add_arc(1, StdArc::new(7, 7, TropicalWeight::one(), 2));
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let before = language(&fst, 10);
        rm_epsilon(&mut fst, true).unwrap();
        assert!(!has_epsilons(&fst));
        assert_eq!(language(&fst, 10), before);
    }

    /// An FST with no epsilons at all comes back unchanged in what it says.
    #[test]
    fn an_fst_without_epsilons_is_unchanged() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(1.0), 1));
        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight(2.0), 2));
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let before = language(&fst, 8);
        rm_epsilon(&mut fst, true).unwrap();
        assert_eq!(language(&fst, 8), before);
    }

    /// Whatever the FST, removing epsilons leaves none and changes nothing
    /// about what it transduces.
    #[test]
    fn removing_epsilons_leaves_none_and_keeps_the_language() {
        let mut rng = Rng::new(0x0E95_0000_u64);
        let mut checked = 0;
        for round in 0..200 {
            let mut fst = random_acyclic_fst(&mut rng, 6);
            // Turn some arcs into epsilons.
            let states: Vec<i32> = fst.states().collect();
            for state in states {
                fst.mutate_arcs(state, |arc| {
                    if arc.ilabel() % 3 == 1 {
                        *arc = StdArc::new(0, 0, *arc.weight(), arc.nextstate());
                    }
                });
            }
            fst.properties(K_FST_PROPERTIES, true);
            if !has_epsilons(&fst) {
                continue;
            }
            checked += 1;

            let before = language(&fst, 12);
            rm_epsilon(&mut fst, true).unwrap();
            assert!(!has_epsilons(&fst), "round {round}");
            assert_eq!(language(&fst, 12), before, "round {round}");
        }
        assert!(checked > 50, "only {checked} FSTs had epsilons");
    }

    /// Not connecting keeps the states that nothing labelled reaches.
    #[test]
    fn without_connecting_the_unreachable_states_stay() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(0, 0, TropicalWeight(1.0), 1));
        fst.add_arc(1, StdArc::new(5, 5, TropicalWeight::one(), 2));
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let mut kept = fst.clone();
        rm_epsilon(&mut kept, false).unwrap();
        assert_eq!(kept.num_states(), 3);

        let mut connected = fst;
        rm_epsilon(&mut connected, true).unwrap();
        assert!(connected.num_states() < 3, "state 1 is no longer reachable");
        assert_eq!(language(&connected, 8), language(&kept, 8));
    }

    /// Pruning after removal keeps only what a good enough path goes through.
    #[test]
    fn removing_epsilons_can_prune_as_it_goes() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(0, 0, TropicalWeight(1.0), 1));
        fst.add_arc(0, StdArc::new(2, 2, TropicalWeight(9.0), 2));
        fst.add_arc(1, StdArc::new(3, 3, TropicalWeight::one(), 2));
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        rm_epsilon_with(
            &mut fst,
            &RmEpsilonOptions {
                weight_threshold: TropicalWeight(2.0),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!has_epsilons(&fst));
        let strings: Vec<Vec<i32>> = language(&fst, 8).into_iter().map(|(i, _, _)| i).collect();
        assert_eq!(strings, vec![vec![3]], "the weight-9 branch is far out");
    }
}
