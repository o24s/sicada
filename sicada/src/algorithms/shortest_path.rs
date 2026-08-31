//! The lightest path through an FST, and the next lightest after it.
//!
//! Port of OpenFst's `shortest-path.h`.
//!
//! > Mohri, M. and Riley, M. 2002. An efficient algorithm for the n-best-strings
//! > problem. In *Proc. ICSLP*.
//!
//! One path is found by a shortest-first walk that remembers how each state was
//! reached, and then reading those back from the best final state. `n` paths
//! take the reverse FST and a best-first search over *paths* rather than
//! states, guided by the shortest distance already computed, which brings the
//! cost down to `n` times that of one path rather than an exponential.

use std::cell::RefCell;
use std::rc::Rc;

use crate::algorithms::connect::connect;
use crate::algorithms::reverse::reverse;
use crate::algorithms::shortest_distance::{
    Distance, SHORTEST_DELTA, ShortestDistanceOptions, shortest_distance_with,
};
use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::arc_filter::AnyArcFilter;
use crate::data_structures::indexed_heap::IndexedHeap;
use crate::error::OpenFstError;
use crate::fst::{ExpandedFst, Fst, MutableFst};
use crate::fsts::vector_fst::VectorFst;
use crate::properties::{K_FST_PROPERTIES, shortest_path_properties};
use crate::queue::{AutoQueue, Queue, natural_less_unchecked, state_weight_compare};
use crate::weight::{PATH, PathWeight, Weight, natural_less};

/// How to search.
#[derive(Debug, Clone)]
pub struct ShortestPathOptions<W> {
    /// How many paths to return.
    pub nshortest: usize,
    /// Whether to return only paths whose input strings differ.
    ///
    /// SICADA-DIVERGE: not implemented. Upstream gets it by determinizing the
    /// reverse FST while carrying the distances through, which sicada's
    /// determinization does not do yet.
    pub unique: bool,
    /// How closely the distances have to converge.
    pub delta: f32,
    /// Stop at the first final state reached, which is the shortest path only
    /// when the queue is shortest-first and every weight lies between
    /// [`Weight::one`] and [`Weight::zero`] in the natural order.
    pub first_path: bool,
    /// Drop paths more than this much heavier than the shortest.
    pub weight_threshold: W,
    /// Stop once this many states have been built.
    pub state_threshold: Option<usize>,
}

impl<W: Weight> Default for ShortestPathOptions<W> {
    fn default() -> Self {
        Self {
            nshortest: 1,
            unique: false,
            delta: SHORTEST_DELTA,
            first_path: false,
            weight_threshold: W::zero(),
            state_threshold: None,
        }
    }
}

/// Where a state was reached from, and by which of that state's arcs.
///
/// The absent case is [`ArcStateId::no_state`] and `u32::MAX` rather than two
/// `Option`s: this is written once per arc that improves a distance, so for
/// `i32` state ids it is the difference between an 8-byte store and a 24-byte
/// one. Same reasoning as [`Weight::no_weight`](crate::weight::Weight::no_weight).
#[derive(Clone, Copy)]
struct Parent<S> {
    /// The state this one was reached from.
    from: S,
    /// Which of `from`'s arcs was taken, counted from the first.
    position: u32,
}

/// No arc, for a state nothing has reached yet.
const NO_POSITION: u32 = u32::MAX;

impl<S: ArcStateId> Parent<S> {
    /// The entry for a state nothing has reached.
    fn none() -> Self {
        Self {
            from: S::no_state(),
            position: NO_POSITION,
        }
    }

    /// The state this one was reached from, if anything did.
    fn from(&self) -> Option<S> {
        (self.from != S::no_state()).then_some(self.from)
    }
}

/// The lightest path to each state, remembering how it got there.
///
/// Returns the final state the best path ends at, if there is one.
fn single_shortest_path<A, F, Q>(
    ifst: &F,
    distance: &Distance<A::Weight>,
    queue: &mut Q,
    parent: &mut Vec<Parent<A::StateId>>,
    first_path: bool,
) -> Result<Option<A::StateId>, OpenFstError>
where
    A: Arc,
    A::Weight: PathWeight,
    F: Fst<A>,
    Q: Queue<A::StateId>,
{
    parent.clear();
    distance.borrow_mut().clear();
    let Some(source) = ifst.start() else {
        return Ok(None);
    };
    queue.clear();

    let mut enqueued: Vec<bool> = Vec::new();
    // How far the three parallel vectors have been grown. Only `ensure` grows
    // them, so keeping the length here turns the check that runs once per arc
    // into a comparison instead of a `RefCell` borrow.
    let mut grown = 0usize;
    let ensure = |distance: &Distance<A::Weight>,
                  parent: &mut Vec<Parent<A::StateId>>,
                  enqueued: &mut Vec<bool>,
                  grown: &mut usize,
                  index: usize| {
        if index < *grown {
            return;
        }
        let mut distance = distance.borrow_mut();
        while distance.len() <= index {
            distance.push(A::Weight::zero());
            parent.push(Parent::none());
            enqueued.push(false);
        }
        *grown = distance.len();
    };

    let source_index = source.as_usize();
    ensure(distance, parent, &mut enqueued, &mut grown, source_index);
    distance.borrow_mut()[source_index] = A::Weight::one();
    enqueued[source_index] = true;
    queue.enqueue(source);

    let zero = A::Weight::zero();
    let mut best_final: Option<A::StateId> = None;
    let mut best_distance = zero.clone();
    let mut final_seen = false;

    while let Some(state) = queue.dequeue() {
        let index = state.as_usize();
        ensure(distance, parent, &mut enqueued, &mut grown, index);
        enqueued[index] = false;
        let here = distance.borrow()[index].clone();

        // With a shortest-first queue nothing still waiting can beat what has
        // already reached a final state.
        if first_path && final_seen && !natural_less(&here, &best_distance) {
            break;
        }

        let final_here = ifst.final_weight(state);
        if final_here != zero {
            let through = here.times(&final_here);
            let sum = best_distance.plus(&through);
            if sum != best_distance {
                best_distance = sum;
                best_final = Some(state);
            }
            if !best_distance.is_member() {
                return Err(OpenFstError::InvalidOperation(
                    "ShortestPath: the best distance left the semiring".into(),
                ));
            }
            final_seen = true;
        }

        for (position, arc) in ifst.arcs(state).enumerate() {
            let next = arc.nextstate().as_usize();
            ensure(distance, parent, &mut enqueued, &mut grown, next);
            let weight = here.times(arc.weight());
            // One borrow for the read and the write together: the queue is only
            // touched after it is dropped, so nothing else can be looking.
            let sum = {
                let mut distance = distance.borrow_mut();
                let current = &distance[next];
                let sum = current.plus(&weight);
                if sum == *current {
                    continue;
                }
                distance[next] = sum.clone();
                sum
            };
            if !sum.is_member() {
                return Err(OpenFstError::InvalidOperation(
                    "ShortestPath: a distance left the semiring".into(),
                ));
            }
            parent[next] = Parent {
                from: state,
                position: position as u32,
            };
            if enqueued[next] {
                queue.update(arc.nextstate());
            } else {
                queue.enqueue(arc.nextstate());
                enqueued[next] = true;
            }
        }
    }
    Ok(best_final)
}

/// Reads the remembered path back from `best_final`, forwards.
fn backtrace<A, F1, F2>(
    ifst: &F1,
    ofst: &mut F2,
    parent: &[Parent<A::StateId>],
    best_final: Option<A::StateId>,
) where
    A: Arc,
    F1: Fst<A>,
    F2: MutableFst<A>,
{
    ofst.delete_all_states();
    ofst.set_input_symbols(ifst.input_symbols());
    ofst.set_output_symbols(ifst.output_symbols());
    let Some(best_final) = best_final else {
        return;
    };

    // Walking back from the final state builds the path in reverse, so each
    // state made points at the one made before it.
    let mut here: Option<A::StateId> = None;
    let mut previous: Option<A::StateId>;
    let mut state = Some(best_final);
    let mut came_from: Option<A::StateId> = None;
    while let Some(at) = state {
        previous = here;
        let made = ofst.add_state();
        here = Some(made);
        match came_from {
            None => ofst.set_final(made, ifst.final_weight(best_final)),
            Some(from) => {
                let position = parent[from.as_usize()].position;
                debug_assert_ne!(position, NO_POSITION, "a state with a parent has an arc");
                if let Some(arc) = ifst.arcs(at).nth(position as usize) {
                    ofst.add_arc(
                        made,
                        A::new(
                            arc.ilabel(),
                            arc.olabel(),
                            arc.weight().clone(),
                            previous.expect("the state reached before this one"),
                        ),
                    );
                }
            }
        }
        came_from = Some(at);
        state = parent[at.as_usize()].from();
    }
    if let Some(start) = here {
        ofst.set_start(start);
    }
    let props = shortest_path_properties(ofst.properties(K_FST_PROPERTIES, false), true);
    ofst.set_properties(props, K_FST_PROPERTIES);
}

/// A state of the result: which state of the reverse FST a path reached, and
/// what the path weighs.
///
/// `None` is the superfinal state, which is where every path starts in the
/// reverse.
type Pair<S, W> = (Option<S>, W);

/// The `n` lightest paths, over the reverse of the input.
///
/// `distance` is the distance to each state of the *reverse*, with the extra
/// entry for its superinitial state at the front.
#[allow(clippy::too_many_arguments)]
fn n_shortest_path<A, F1, F2>(
    rfst: &F1,
    ofst: &mut F2,
    distance: &[A::Weight],
    nshortest: usize,
    delta: f32,
    weight_threshold: &A::Weight,
    state_threshold: Option<usize>,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: PathWeight,
    A::Reverse: Arc<Label = A::Label, StateId = A::StateId>,
    <<A::Reverse as Arc>::Weight as Weight>::ReverseWeight: Into<A::Weight>,
    F1: Fst<A::Reverse> + ExpandedFst<A::Reverse>,
    F2: MutableFst<A> + ExpandedFst<A>,
{
    ofst.delete_all_states();
    ofst.set_input_symbols(rfst.input_symbols());
    ofst.set_output_symbols(rfst.output_symbols());
    if nshortest == 0 || state_threshold == Some(0) {
        return Ok(());
    }
    let Some(rstart) = rfst.start() else {
        return Ok(());
    };
    let start_index = rstart.as_usize();
    if distance.len() <= start_index || distance[start_index] == A::Weight::zero() {
        return Ok(());
    }
    if natural_less(weight_threshold, &A::Weight::one()) {
        return Ok(());
    }

    // Each state of the result stands for one path: where it reached, and what
    // it weighs.
    let pairs: Rc<RefCell<Vec<Pair<A::StateId, A::Weight>>>> = Rc::new(RefCell::new(Vec::new()));
    let owned_distance: Vec<A::Weight> = distance.to_vec();

    // Upstream's comparison, which reports that `x` is *worse* than `y`,
    // since it is written for a max-heap. A complete path is penalized at a tie
    // so that inexact weights cannot make the search wander.
    let worse = {
        let pairs = Rc::clone(&pairs);
        let distance = owned_distance.clone();
        move |x: &usize, y: &usize| -> bool {
            let pairs = pairs.borrow();
            let at = |index: usize| -> A::Weight {
                let (state, weight): &Pair<A::StateId, A::Weight> = &pairs[index];
                let d = match state {
                    None => A::Weight::one(),
                    Some(state) => distance
                        .get(state.as_usize())
                        .cloned()
                        .unwrap_or_else(A::Weight::zero),
                };
                d.times(weight)
            };
            let (wx, wy) = (at(*x), at(*y));
            let x_complete = pairs[*x].0.is_none();
            let y_complete = pairs[*y].0.is_none();
            match (x_complete, y_complete) {
                (true, false) => natural_less(&wy, &wx) || wx.approx_equal(&wy, delta),
                (false, true) => natural_less(&wy, &wx) && !wx.approx_equal(&wy, delta),
                _ => natural_less(&wy, &wx),
            }
        }
    };
    // The heap hands back the best, so the comparison is turned around.
    let mut heap = IndexedHeap::new(move |x: &usize, y: &usize| worse(y, x));

    let start = ofst.add_state();
    ofst.set_start(start);
    let final_state = ofst.add_state();
    ofst.set_final(final_state, A::Weight::one());
    {
        let mut pairs = pairs.borrow_mut();
        while pairs.len() <= final_state.as_usize() {
            pairs.push((None, A::Weight::zero()));
        }
        pairs[final_state.as_usize()] = (Some(rstart), A::Weight::one());
    }
    heap.insert(final_state.as_usize());

    let limit = distance[start_index].times(weight_threshold);
    // How many paths to each state have been taken so far; the entry at 0 is
    // the superfinal state's.
    let mut taken: Vec<usize> = Vec::new();
    let count_index = |state: &Option<A::StateId>| match state {
        None => 0,
        Some(state) => state.as_usize() + 1,
    };

    while let Some(state) = heap.pop() {
        let pair = pairs.borrow()[state].clone();
        let d = match &pair.0 {
            None => A::Weight::one(),
            Some(at) => owned_distance
                .get(at.as_usize())
                .cloned()
                .unwrap_or_else(A::Weight::zero),
        };
        if natural_less(&limit, &d.times(&pair.1))
            || state_threshold.is_some_and(|limit| ofst.num_states() >= limit)
        {
            continue;
        }
        let index = count_index(&pair.0);
        while taken.len() <= index {
            taken.push(0);
        }
        taken[index] += 1;

        if pair.0.is_none() {
            // A complete path: it becomes one of the answers.
            ofst.add_arc(
                start,
                A::new(
                    A::Label::epsilon(),
                    A::Label::epsilon(),
                    A::Weight::one(),
                    A::StateId::from_usize(state),
                ),
            );
            if taken[index] == nshortest {
                break;
            }
            continue;
        }
        if taken[index] > nshortest {
            continue;
        }
        let at = pair.0.expect("just checked");

        for arc in rfst.arcs(at) {
            let weight = pair.1.times(&arc.weight().reverse().into());
            let next = ofst.add_state();
            pairs.borrow_mut().push((Some(arc.nextstate()), weight));
            ofst.add_arc(
                next,
                A::new(
                    arc.ilabel(),
                    arc.olabel(),
                    arc.weight().reverse().into(),
                    A::StateId::from_usize(state),
                ),
            );
            heap.insert(next.as_usize());
        }
        let final_weight: A::Weight = rfst.final_weight(at).reverse().into();
        if final_weight != A::Weight::zero() {
            let weight = pair.1.times(&final_weight);
            let next = ofst.add_state();
            pairs.borrow_mut().push((None, weight));
            ofst.add_arc(
                next,
                A::new(
                    A::Label::epsilon(),
                    A::Label::epsilon(),
                    final_weight,
                    A::StateId::from_usize(state),
                ),
            );
            heap.insert(next.as_usize());
        }
    }

    connect(ofst);
    let props = shortest_path_properties(ofst.properties(K_FST_PROPERTIES, false), false);
    ofst.set_properties(props, K_FST_PROPERTIES);
    Ok(())
}

/// The `opts.nshortest` lightest paths of `ifst`, into `ofst`.
///
/// The first arc leaving the result's start state begins the lightest path, the
/// second the next lightest, and so on; apart from those, the result is a tree
/// rooted at its one final state.
pub fn shortest_path<A, F1, F2>(
    ifst: &F1,
    ofst: &mut F2,
    opts: &ShortestPathOptions<A::Weight>,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: PathWeight,
    F1: Fst<A> + ExpandedFst<A>,
    F2: MutableFst<A> + ExpandedFst<A>,
    <A::Weight as Weight>::ReverseWeight: Weight<ReverseWeight = A::Weight>,
    <<A::Reverse as Arc>::Weight as Weight>::ReverseWeight: Into<A::Weight>,
{
    if opts.unique {
        return Err(OpenFstError::InvalidOperation(
            "ShortestPath: returning only paths with distinct input strings is not implemented; \
             it needs determinization to carry distances through"
                .into(),
        ));
    }
    if opts.nshortest == 0 {
        ofst.delete_all_states();
        return Ok(());
    }

    let distance: Distance<A::Weight> = Rc::new(RefCell::new(Vec::new()));
    let comp = state_weight_compare::<A::StateId, A::Weight, _>(
        Rc::clone(&distance),
        natural_less_unchecked::<A::Weight>,
    );
    let comp = (A::Weight::properties() & PATH != 0).then_some(comp);

    if opts.nshortest == 1 {
        let mut queue = AutoQueue::new(ifst, comp);
        let mut parent: Vec<Parent<A::StateId>> = Vec::new();
        let best_final =
            single_shortest_path(ifst, &distance, &mut queue, &mut parent, opts.first_path)?;
        backtrace(ifst, ofst, &parent, best_final);
        return Ok(());
    }

    // More than one path: the search runs over the reverse, guided by the
    // distance from each state of the original to a final state.
    {
        let mut queue = AutoQueue::new(ifst, comp);
        let sd_opts = ShortestDistanceOptions {
            delta: opts.delta,
            ..ShortestDistanceOptions::new(AnyArcFilter)
        };
        shortest_distance_with(ifst, &distance, &mut queue, &sd_opts)?;
    }
    let forward = distance.borrow().clone();

    let mut rfst: VectorFst<A::Reverse> = VectorFst::new();
    reverse(ifst, &mut rfst, true);

    // Reversing added a superinitial state, whose distance is what the arcs out
    // of it can reach; every other state of the reverse is one of the original,
    // shifted by one.
    let mut shifted: Vec<A::Weight> = Vec::with_capacity(forward.len() + 1);
    let mut superinitial = A::Weight::zero();
    if let Some(rstart) = rfst.start() {
        for arc in rfst.arcs(rstart) {
            let state = arc.nextstate().as_usize().wrapping_sub(1);
            if let Some(weight) = forward.get(state) {
                let reversed: A::Weight = arc.weight().reverse();
                superinitial = superinitial.plus(&reversed.times(weight));
            }
        }
    }
    shifted.push(superinitial);
    shifted.extend(forward);

    n_shortest_path(
        &rfst,
        ofst,
        &shifted,
        opts.nshortest,
        opts.delta,
        &opts.weight_threshold,
        opts.state_threshold,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::shortest_distance::shortest_distance;
    use crate::algorithms::test_support::{Rng, paths, random_acyclic_fst};
    use crate::arc::StdArc;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::K_FST_PROPERTIES;
    use crate::weights::float_weight::TropicalWeight;

    fn best(fst: &StdVectorFst, n: usize) -> StdVectorFst {
        let mut out = StdVectorFst::new();
        shortest_path(
            fst,
            &mut out,
            &ShortestPathOptions {
                nshortest: n,
                ..Default::default()
            },
        )
        .unwrap();
        out
    }

    /// The weights of the paths a result holds, lightest first.
    fn weights(fst: &StdVectorFst) -> Vec<f32> {
        let mut out: Vec<f32> = paths(fst, 24)
            .into_iter()
            .map(|(_, _, weight)| weight.value())
            .collect();
        out.sort_by(f32::total_cmp);
        out
    }

    /// Three paths from 0, weighing 1, 3 and 6.
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

    #[test]
    fn the_shortest_path_is_the_lightest_one() {
        let out = best(&fan(), 1);
        assert_eq!(weights(&out), vec![1.0]);
        let labels: Vec<i32> = paths(&out, 24)
            .into_iter()
            .flat_map(|(ilabels, _, _)| ilabels)
            .collect();
        assert_eq!(labels, vec![1]);
    }

    /// Over a diamond, the shortest path is the lighter way round.
    #[test]
    fn the_shortest_path_goes_the_cheaper_way_round() {
        let mut fst = StdVectorFst::new();
        for _ in 0..4 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(1.0), 1));
        fst.add_arc(0, StdArc::new(2, 2, TropicalWeight(4.0), 2));
        fst.add_arc(1, StdArc::new(3, 3, TropicalWeight(5.0), 3));
        fst.add_arc(2, StdArc::new(4, 4, TropicalWeight(1.0), 3));
        fst.set_final(3, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let out = best(&fst, 1);
        let labels: Vec<i32> = paths(&out, 24)
            .into_iter()
            .flat_map(|(ilabels, _, _)| ilabels)
            .collect();
        assert_eq!(labels, vec![2, 4], "4 + 1 beats 1 + 5");
        assert_eq!(weights(&out), vec![5.0]);
    }

    /// The n lightest paths come out in order.
    #[test]
    fn the_n_shortest_paths_are_the_n_lightest() {
        let fst = fan();
        assert_eq!(weights(&best(&fst, 2)), vec![1.0, 3.0]);
        assert_eq!(weights(&best(&fst, 3)), vec![1.0, 3.0, 6.0]);
        // Asking for more than there are gives what there is.
        assert_eq!(weights(&best(&fst, 10)), vec![1.0, 3.0, 6.0]);
    }

    /// Asking for none gives none.
    #[test]
    fn asking_for_no_paths_gives_none() {
        assert_eq!(best(&fan(), 0).num_states(), 0);
    }

    /// An FST with no accepting path has no shortest path.
    #[test]
    fn an_fst_with_no_accepting_path_has_none() {
        let mut fst = StdVectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.properties(K_FST_PROPERTIES, true);
        assert!(weights(&best(&fst, 1)).is_empty());
        assert!(weights(&best(&fst, 3)).is_empty());

        assert_eq!(best(&StdVectorFst::new(), 1).num_states(), 0);
    }

    /// The shortest path weighs what the shortest distance says it does.
    #[test]
    fn the_shortest_path_weighs_what_the_shortest_distance_says() {
        let mut rng = Rng::new(0x0000_5B47_u64);
        let mut checked = 0;
        for round in 0..200 {
            let fst = random_acyclic_fst(&mut rng, 6);
            let total = shortest_distance(&fst, SHORTEST_DELTA).unwrap();
            if total == TropicalWeight::zero() {
                continue;
            }
            checked += 1;
            let got = weights(&best(&fst, 1));
            assert_eq!(got.len(), 1, "round {round}");
            assert!(
                (got[0] - total.value()).abs() < 1e-4,
                "round {round}: {} against {}",
                got[0],
                total.value()
            );
        }
        assert!(checked > 50, "only {checked} FSTs accepted anything");
    }

    /// The n lightest paths are the n lightest of all the paths there are.
    #[test]
    fn the_n_shortest_paths_are_the_n_lightest_of_all_of_them() {
        let mut rng = Rng::new(0x0000_9E57_u64);
        let mut checked = 0;
        for round in 0..200 {
            let fst = random_acyclic_fst(&mut rng, 6);
            let mut all: Vec<f32> = paths(&fst, 16)
                .into_iter()
                .map(|(_, _, weight)| weight.value())
                .collect();
            if all.is_empty() {
                continue;
            }
            all.sort_by(f32::total_cmp);
            checked += 1;

            for n in [1usize, 2, 3, 5] {
                let want: Vec<f32> = all.iter().take(n).copied().collect();
                let got = weights(&best(&fst, n));
                assert_eq!(got.len(), want.len(), "round {round}, n = {n}");
                for (got, want) in got.iter().zip(&want) {
                    assert!(
                        (got - want).abs() < 1e-4,
                        "round {round}, n = {n}: {got} against {want}"
                    );
                }
            }
        }
        assert!(checked > 50, "only {checked} FSTs accepted anything");
    }

    /// A threshold drops the paths too far behind the best.
    ///
    /// The limit is the shortest path times the threshold, so with the paths
    /// weighing 1, 3 and 6 a threshold of 2.5 admits everything up to 3.5.
    #[test]
    fn a_weight_threshold_drops_what_is_too_heavy() {
        let with = |threshold: f32| {
            let mut out = StdVectorFst::new();
            shortest_path(
                &fan(),
                &mut out,
                &ShortestPathOptions {
                    nshortest: 5,
                    weight_threshold: TropicalWeight(threshold),
                    ..Default::default()
                },
            )
            .unwrap();
            weights(&out)
        };
        assert_eq!(with(2.5), vec![1.0, 3.0], "the limit is 3.5");
        assert_eq!(with(1.0), vec![1.0], "the limit is 2.0");
        assert_eq!(with(10.0), vec![1.0, 3.0, 6.0], "the limit is 11.0");
    }

    /// The unimplemented option says so rather than quietly doing something
    /// else.
    #[test]
    fn asking_for_distinct_inputs_is_refused() {
        let mut out = StdVectorFst::new();
        let err = shortest_path(
            &fan(),
            &mut out,
            &ShortestPathOptions {
                nshortest: 2,
                unique: true,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(format!("{err}").contains("not implemented"), "{err}");
    }
}
