//! Throwing away what no good path goes through.
//!
//! Port of OpenFst's `prune.h`. A state or arc is kept when some accepting path
//! through it weighs no more than the shortest path times a threshold; the rest
//! go. That is how a lattice becomes the part of it worth searching.
//!
//! The weight has to have the path property, so that "shortest" means
//! something, and every cycle has to be bounded, `w ⊕ 1 = 1`, or the distances
//! the decision rests on do not settle.

use std::cell::RefCell;
use std::rc::Rc;

use crate::algorithms::shortest_distance::{SHORTEST_DELTA, shortest_distance_reverse};
use crate::arc::{Arc, ArcStateId};
use crate::arc_filter::{AnyArcFilter, ArcFilter};
use crate::data_structures::indexed_heap::IndexedHeap;
use crate::error::OpenFstError;
use crate::fst::{ExpandedFst, Fst, MutableFst};
use crate::weight::{IdempotentWeight, PathWeight, Weight, natural_less};

/// What to keep.
#[derive(Debug, Clone)]
pub struct PruneOptions<W, F> {
    /// How much heavier than the shortest path a path may be and still count.
    /// [`Weight::zero`] keeps everything reachable.
    pub weight_threshold: W,
    /// At most this many states, or `None` for no limit.
    pub state_threshold: Option<usize>,
    /// Which arcs to consider; the rest are left alone.
    pub filter: F,
    /// The distance from each state to the final states, if it is already
    /// known.
    pub distance: Option<Vec<W>>,
    /// How close the distances have to converge.
    pub delta: f32,
    /// Whether the threshold multiplies the shortest weight on the left rather
    /// than the right, which only matters where ⊗ does not commute.
    pub threshold_initial: bool,
}

impl<W: Weight, F> PruneOptions<W, F> {
    /// Keeps everything within `weight_threshold` of the shortest path.
    pub fn new(weight_threshold: W, filter: F) -> Self {
        Self {
            weight_threshold,
            state_threshold: None,
            filter,
            distance: None,
            delta: SHORTEST_DELTA,
            threshold_initial: false,
        }
    }
}

impl<W: Weight> PruneOptions<W, AnyArcFilter> {
    /// As [`new`](PruneOptions::new), considering every arc.
    pub fn threshold(weight_threshold: W) -> Self {
        Self::new(weight_threshold, AnyArcFilter)
    }
}

/// Orders states by the weight of the best path through them, lightest first.
///
/// SICADA-DIVERGE: upstream's `PruneCompare` holds bare references to two
/// vectors, one of which the algorithm rewrites as it goes. Sharing is the
/// point, so it is spelled out.
fn best_through<S, W>(
    idistance: Rc<RefCell<Vec<W>>>,
    fdistance: Rc<RefCell<Vec<W>>>,
) -> impl Fn(&S, &S) -> bool + Clone
where
    S: ArcStateId,
    W: Weight + IdempotentWeight,
{
    move |x: &S, y: &S| {
        let idistance = idistance.borrow();
        let fdistance = fdistance.borrow();
        let through = |s: &S| {
            let index = s.as_usize();
            let to = idistance.get(index).cloned().unwrap_or_else(W::zero);
            let from = fdistance.get(index).cloned().unwrap_or_else(W::zero);
            to.times(&from)
        };
        natural_less(&through(x), &through(y))
    }
}

/// The distance from each state to the final states, computed if it was not
/// supplied.
fn final_distance<A, F>(
    fst: &F,
    supplied: &Option<Vec<A::Weight>>,
    delta: f32,
) -> Result<Vec<A::Weight>, OpenFstError>
where
    A: Arc,
    F: Fst<A>,
    <A::Weight as Weight>::ReverseWeight: Weight<ReverseWeight = A::Weight>,
{
    match supplied {
        Some(distance) => Ok(distance.clone()),
        None => shortest_distance_reverse::<A, F>(fst, delta),
    }
}

/// Removes from `fst` everything no path within the threshold goes through.
///
/// SICADA-DIVERGE: upstream redirects a pruned arc to a state it adds for the
/// purpose and deletes that state at the end, because its `MutableArcIterator`
/// can rewrite an arc but not remove one. `mutate_arcs` has the same
/// restriction, so the arcs that survive are collected and written back: one
/// pass over each state's arcs either way, and no state that exists only to be
/// deleted.
pub fn prune<A, F, AF>(fst: &mut F, opts: &PruneOptions<A::Weight, AF>) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: PathWeight + IdempotentWeight,
    F: MutableFst<A> + ExpandedFst<A>,
    AF: ArcFilter<A>,
    <A::Weight as Weight>::ReverseWeight: Weight<ReverseWeight = A::Weight>,
{
    let nstates = fst.num_states();
    if nstates == 0 {
        return Ok(());
    }
    let fdistance = final_distance::<A, F>(fst, &opts.distance, opts.delta)?;

    let Some(start) = fst.start() else {
        fst.delete_all_states();
        return Ok(());
    };
    let start_index = start.as_usize();
    let reachable = fdistance
        .get(start_index)
        .is_some_and(|weight| *weight != A::Weight::zero());
    if opts.state_threshold == Some(0) || !reachable {
        // Nothing accepts, so nothing is worth keeping.
        fst.delete_all_states();
        return Ok(());
    }

    let idistance = Rc::new(RefCell::new(vec![A::Weight::zero(); nstates]));
    let fdistance = Rc::new(RefCell::new(fdistance));
    let mut heap = IndexedHeap::new(best_through::<A::StateId, A::Weight>(
        Rc::clone(&idistance),
        Rc::clone(&fdistance),
    ));
    let mut keys: Vec<Option<usize>> = vec![None; nstates];
    let mut visited = vec![false; nstates];

    let limit = {
        let fdistance = fdistance.borrow();
        let shortest = &fdistance[start_index];
        if opts.threshold_initial {
            opts.weight_threshold.times(shortest)
        } else {
            shortest.times(&opts.weight_threshold)
        }
    };

    let mut num_visited = 0usize;
    if !natural_less(&limit, &fdistance.borrow()[start_index]) {
        idistance.borrow_mut()[start_index] = A::Weight::one();
        keys[start_index] = Some(heap.insert(start));
        num_visited += 1;
    }

    let mut kept: Vec<A> = Vec::new();
    while let Some(state) = heap.pop() {
        let index = state.as_usize();
        keys[index] = None;
        visited[index] = true;

        let here = idistance.borrow()[index].clone();
        if natural_less(&limit, &here.times(&fst.final_weight(state))) {
            fst.set_final(state, A::Weight::zero());
        }

        kept.clear();
        let mut dropped = 0usize;
        for arc in fst.arcs(state) {
            if !opts.filter.call(&arc) {
                kept.push(arc);
                continue;
            }
            let next = arc.nextstate().as_usize();
            let to_here = here.times(arc.weight());
            let onward = fdistance
                .borrow()
                .get(next)
                .cloned()
                .unwrap_or_else(A::Weight::zero);
            if natural_less(&limit, &to_here.times(&onward)) {
                dropped += 1;
                continue;
            }
            kept.push(arc.clone());

            {
                let mut idistance = idistance.borrow_mut();
                if natural_less(&to_here, &idistance[next]) {
                    idistance[next] = to_here;
                }
            }
            if visited[next] {
                continue;
            }
            if opts
                .state_threshold
                .is_some_and(|limit| num_visited >= limit)
            {
                continue;
            }
            match keys[next] {
                None => {
                    keys[next] = Some(heap.insert(arc.nextstate()));
                    num_visited += 1;
                }
                Some(key) => heap.update(key, arc.nextstate()),
            }
        }
        if dropped > 0 {
            fst.delete_arcs(state);
            for arc in kept.drain(..) {
                fst.add_arc(state, arc);
            }
        }
    }

    let dead: Vec<A::StateId> = (0..nstates)
        .filter(|index| !visited[*index])
        .map(A::StateId::from_usize)
        .collect();
    fst.delete_states(&dead);
    Ok(())
}

/// As [`prune`], writing the result to `ofst` and leaving the input alone.
///
/// `state_map`, when given, comes back holding the state of `ofst` each state
/// of `ifst` became, or `None` for one that was dropped.
pub fn prune_to<A, F1, F2, AF>(
    ifst: &F1,
    ofst: &mut F2,
    opts: &PruneOptions<A::Weight, AF>,
    mut state_map: Option<&mut Vec<Option<A::StateId>>>,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: PathWeight + IdempotentWeight,
    F1: Fst<A>,
    F2: MutableFst<A>,
    AF: ArcFilter<A>,
    <A::Weight as Weight>::ReverseWeight: Weight<ReverseWeight = A::Weight>,
{
    ofst.delete_all_states();
    ofst.set_input_symbols(ifst.input_symbols());
    ofst.set_output_symbols(ifst.output_symbols());
    if let Some(map) = state_map.as_deref_mut() {
        map.clear();
    }

    let Some(start) = ifst.start() else {
        return Ok(());
    };
    // A threshold lighter than One would keep nothing at all.
    if natural_less(&opts.weight_threshold, &A::Weight::one()) || opts.state_threshold == Some(0) {
        return Ok(());
    }
    let fdistance = final_distance::<A, F1>(ifst, &opts.distance, opts.delta)?;
    let start_index = start.as_usize();
    if fdistance
        .get(start_index)
        .is_none_or(|weight| *weight == A::Weight::zero())
    {
        return Ok(());
    }

    let idistance: Rc<RefCell<Vec<A::Weight>>> = Rc::new(RefCell::new(Vec::new()));
    let fdistance = Rc::new(RefCell::new(fdistance));
    let mut heap = IndexedHeap::new(best_through::<A::StateId, A::Weight>(
        Rc::clone(&idistance),
        Rc::clone(&fdistance),
    ));

    let mut local = Vec::new();
    let copy = match state_map {
        Some(map) => map,
        None => &mut local,
    };
    let mut keys: Vec<Option<usize>> = Vec::new();
    let mut visited: Vec<bool> = Vec::new();

    /// Grows a vector until `index` is inside it.
    fn ensure<T: Clone>(vector: &mut Vec<T>, index: usize, fill: T) {
        while vector.len() <= index {
            vector.push(fill.clone());
        }
    }

    let limit = {
        let fdistance = fdistance.borrow();
        let shortest = &fdistance[start_index];
        if opts.threshold_initial {
            opts.weight_threshold.times(shortest)
        } else {
            shortest.times(&opts.weight_threshold)
        }
    };

    ensure(copy, start_index, None);
    copy[start_index] = Some(ofst.add_state());
    ofst.set_start(copy[start_index].expect("just added"));
    ensure(&mut idistance.borrow_mut(), start_index, A::Weight::zero());
    idistance.borrow_mut()[start_index] = A::Weight::one();
    ensure(&mut keys, start_index, None);
    ensure(&mut visited, start_index, false);
    keys[start_index] = Some(heap.insert(start));

    while let Some(state) = heap.pop() {
        let index = state.as_usize();
        keys[index] = None;
        visited[index] = true;
        let here = idistance.borrow()[index].clone();
        let from = copy[index].expect("a state on the heap has been copied");

        if !natural_less(&limit, &here.times(&ifst.final_weight(state))) {
            ofst.set_final(from, ifst.final_weight(state));
        }

        for arc in ifst.arcs(state) {
            if !opts.filter.call(&arc) {
                continue;
            }
            let next = arc.nextstate().as_usize();
            let to_here = here.times(arc.weight());
            let onward = fdistance
                .borrow()
                .get(next)
                .cloned()
                .unwrap_or_else(A::Weight::zero);
            if natural_less(&limit, &to_here.times(&onward)) {
                continue;
            }
            if opts
                .state_threshold
                .is_some_and(|limit| ofst.num_states() >= limit)
            {
                continue;
            }
            {
                let mut idistance = idistance.borrow_mut();
                ensure(&mut idistance, next, A::Weight::zero());
                if natural_less(&to_here, &idistance[next]) {
                    idistance[next] = to_here;
                }
            }
            ensure(copy, next, None);
            let to = *copy[next].get_or_insert_with(|| ofst.add_state());
            ofst.add_arc(
                from,
                A::new(arc.ilabel(), arc.olabel(), arc.weight().clone(), to),
            );

            ensure(&mut keys, next, None);
            ensure(&mut visited, next, false);
            if visited[next] {
                continue;
            }
            match keys[next] {
                None => keys[next] = Some(heap.insert(arc.nextstate())),
                Some(key) => heap.update(key, arc.nextstate()),
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{Rng, paths, random_acyclic_fst, sorted};
    use crate::arc::StdArc;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::K_FST_PROPERTIES;
    use crate::weights::float_weight::TropicalWeight;

    /// Three paths from 0 to 3 weighing 1, 3 and 6.
    fn fan() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(1.0), 1));
        fst.add_arc(0, StdArc::new(2, 2, TropicalWeight(3.0), 1));
        fst.add_arc(0, StdArc::new(3, 3, TropicalWeight(6.0), 2));
        fst.set_final(1, TropicalWeight::one());
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    fn kept(fst: &StdVectorFst) -> Vec<(Vec<i32>, Vec<i32>, String)> {
        sorted(paths(fst, 12))
    }

    fn prune_copy(fst: &StdVectorFst, threshold: f32) -> StdVectorFst {
        let mut out = StdVectorFst::new();
        prune_to(
            fst,
            &mut out,
            &PruneOptions::threshold(TropicalWeight(threshold)),
            None,
        )
        .unwrap();
        out
    }

    fn prune_in_place(fst: &StdVectorFst, threshold: f32) -> StdVectorFst {
        let mut copy = fst.clone();
        prune(
            &mut copy,
            &PruneOptions::threshold(TropicalWeight(threshold)),
        )
        .unwrap();
        copy
    }

    /// The threshold is measured from the shortest path, so a threshold of 2
    /// keeps everything up to weight 3.
    #[test]
    fn a_path_further_than_the_threshold_goes() {
        let fst = fan();
        assert_eq!(kept(&prune_copy(&fst, 2.0)).len(), 2, "1 and 3 survive");
        assert_eq!(kept(&prune_copy(&fst, 0.0)).len(), 1, "only the shortest");
        assert_eq!(kept(&prune_copy(&fst, 5.0)).len(), 3, "all three");
    }

    /// Pruning in place and pruning into a new FST keep the same paths.
    #[test]
    fn pruning_in_place_keeps_what_pruning_into_a_copy_keeps() {
        let fst = fan();
        for threshold in [0.0, 1.0, 2.0, 5.0, 100.0] {
            assert_eq!(
                kept(&prune_in_place(&fst, threshold)),
                kept(&prune_copy(&fst, threshold)),
                "threshold {threshold}"
            );
        }
    }

    /// Pruning keeps a state or arc exactly when the best path through it is
    /// within the limit, which is stronger than "the paths within the limit
    /// survive", since an arc shared by a good path and a bad one keeps both.
    #[test]
    fn an_arc_survives_exactly_when_the_best_path_through_it_does() {
        use crate::algorithms::shortest_distance::shortest_distance_forward;

        let mut rng = Rng::new(0x_0BAD_C0DE_u64);
        let mut checked = 0;
        for round in 0..200 {
            let fst = random_acyclic_fst(&mut rng, 6);
            let forward = shortest_distance_forward(&fst, SHORTEST_DELTA).unwrap();
            let backward = shortest_distance_reverse(&fst, SHORTEST_DELTA).unwrap();
            let Some(start) = fst.start() else { continue };
            let shortest = backward
                .get(start as usize)
                .copied()
                .unwrap_or_else(TropicalWeight::zero);
            if shortest == TropicalWeight::zero() {
                continue; // Nothing accepts.
            }
            checked += 1;

            for threshold in [0.0f32, 1.0, 3.0] {
                let limit = shortest.times(&TropicalWeight(threshold));
                let at = |distance: &[TropicalWeight], state: usize| {
                    distance
                        .get(state)
                        .copied()
                        .unwrap_or_else(TropicalWeight::zero)
                };

                let mut out = StdVectorFst::new();
                let mut map = Vec::new();
                prune_to(
                    &fst,
                    &mut out,
                    &PruneOptions::threshold(TropicalWeight(threshold)),
                    Some(&mut map),
                )
                .unwrap();

                for state in fst.states() {
                    let index = state as usize;
                    let here = at(&forward, index);
                    let want: Vec<StdArc> = fst
                        .arcs(state)
                        .filter(|arc| {
                            let through = here
                                .times(arc.weight())
                                .times(&at(&backward, arc.nextstate() as usize));
                            !natural_less(&limit, &through)
                        })
                        .collect();

                    let Some(Some(copied)) = map.get(index).copied() else {
                        assert!(
                            want.is_empty(),
                            "round {round}, threshold {threshold}: state {state} was dropped but \
                             {} of its arcs are within the limit",
                            want.len()
                        );
                        continue;
                    };
                    let got: Vec<(i32, i32, TropicalWeight)> = out
                        .arcs(copied)
                        .map(|arc| (arc.ilabel(), arc.olabel(), *arc.weight()))
                        .collect();
                    let want: Vec<(i32, i32, TropicalWeight)> = want
                        .iter()
                        .map(|arc| (arc.ilabel(), arc.olabel(), *arc.weight()))
                        .collect();
                    assert_eq!(
                        got, want,
                        "round {round}, threshold {threshold}, state {state}"
                    );
                }
            }
        }
        assert!(checked > 50, "only {checked} FSTs had an accepting path");
    }

    /// Every path within the limit is still there afterwards.
    #[test]
    fn no_path_within_the_threshold_is_lost() {
        let mut rng = Rng::new(0x_00FF_1CE5_u64);
        for round in 0..200 {
            let fst = random_acyclic_fst(&mut rng, 6);
            let before = paths(&fst, 12);
            let Some(shortest) = before
                .iter()
                .map(|(_, _, weight)| weight.value())
                .min_by(f32::total_cmp)
            else {
                continue;
            };

            for threshold in [0.0f32, 1.0, 3.0] {
                let limit = shortest + threshold + 1e-6;
                let want = sorted(
                    before
                        .iter()
                        .filter(|(_, _, weight)| weight.value() <= limit)
                        .cloned()
                        .collect(),
                );
                let after = sorted(paths(&prune_copy(&fst, threshold), 12));
                for path in &want {
                    assert!(
                        after.contains(path),
                        "round {round}, threshold {threshold}: {path:?} was lost"
                    );
                }
                for path in &after {
                    assert!(
                        sorted(before.clone()).contains(path),
                        "round {round}, threshold {threshold}: {path:?} is not a path of the input"
                    );
                }
            }
        }
    }

    /// The state limit caps how much is kept, whatever the weights say.
    #[test]
    fn the_state_threshold_caps_the_result() {
        let fst = fan();
        let mut out = StdVectorFst::new();
        let opts = PruneOptions {
            state_threshold: Some(2),
            ..PruneOptions::threshold(TropicalWeight(100.0))
        };
        prune_to(&fst, &mut out, &opts, None).unwrap();
        assert!(out.num_states() <= 2, "{} states", out.num_states());

        // A limit of zero keeps nothing at all.
        let opts = PruneOptions {
            state_threshold: Some(0),
            ..PruneOptions::threshold(TropicalWeight(100.0))
        };
        let mut out = StdVectorFst::new();
        prune_to(&fst, &mut out, &opts, None).unwrap();
        assert_eq!(out.num_states(), 0);
    }

    /// An FST nothing accepts prunes to nothing.
    #[test]
    fn an_fst_with_no_accepting_path_prunes_away() {
        let mut fst = StdVectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(1.0), 1));
        fst.properties(K_FST_PROPERTIES, true);

        assert_eq!(prune_copy(&fst, 5.0).num_states(), 0);
        assert_eq!(prune_in_place(&fst, 5.0).num_states(), 0);
    }

    /// An empty FST is nothing to prune.
    #[test]
    fn an_empty_fst_is_left_empty() {
        let empty = StdVectorFst::new();
        assert_eq!(prune_copy(&empty, 1.0).num_states(), 0);
        assert_eq!(prune_in_place(&empty, 1.0).num_states(), 0);
    }

    /// The map says which state of the result each input state became.
    #[test]
    fn the_state_map_says_where_each_state_went() {
        let fst = fan();
        let mut out = StdVectorFst::new();
        let mut map = Vec::new();
        prune_to(
            &fst,
            &mut out,
            &PruneOptions::threshold(TropicalWeight(2.0)),
            Some(&mut map),
        )
        .unwrap();
        assert_eq!(map[0], Some(out.start().unwrap()));
        assert!(map[1].is_some(), "state 1 is on a surviving path");
        assert!(
            map.get(2).copied().flatten().is_none(),
            "state 2 is only reached by the path that was pruned"
        );
    }

    /// A precomputed distance is used as given rather than recomputed.
    #[test]
    fn a_supplied_distance_is_used() {
        let fst = fan();
        let distance = shortest_distance_reverse(&fst, SHORTEST_DELTA).unwrap();
        let opts = PruneOptions {
            distance: Some(distance),
            ..PruneOptions::threshold(TropicalWeight(2.0))
        };
        let mut out = StdVectorFst::new();
        prune_to(&fst, &mut out, &opts, None).unwrap();
        assert_eq!(kept(&out), kept(&prune_copy(&fst, 2.0)));
    }
}
