//! The shortest distance from a state to every other, in any semiring.
//!
//! Port of OpenFst's `shortest-distance.h`, which is Mohri's generic
//! single-source shortest-distance algorithm:
//!
//! > Mohri, M. 2002. Semiring framework and algorithms for shortest-distance
//! > problems, *Journal of Automata, Languages and Combinatorics* 7(3):
//! > 321-350.
//!
//! "Shortest" is the semiring's ⊕: over the tropical semiring it is the
//! lightest path, over the log semiring the ⊕-sum of every path. What makes one
//! algorithm cover both is the residual `r`, which holds what a state has
//! gained since it was last expanded, so that a state can be relaxed again
//! without recounting what its neighbours already have.
//!
//! The weights must be right distributive and *k*-closed, meaning
//! `1 ⊕ x ⊕ … ⊕ x^(k+1) = 1 ⊕ x ⊕ … ⊕ x^k`, or the relaxation does not settle.

use std::cell::RefCell;
use std::rc::Rc;

use crate::arc::{Arc, ArcStateId};
use crate::arc_filter::{AnyArcFilter, ArcFilter};
use crate::error::OpenFstError;
use crate::fst::Fst;
use crate::fsts::vector_fst::VectorFst;
use crate::properties::K_ERROR;
use crate::queue::{AutoQueue, Queue, natural_less_unchecked, state_weight_compare};
use crate::weight::{Adder, PATH, RIGHT_SEMIRING, Weight};

/// The convergence threshold upstream uses, and the smallest that is
/// representable for the purpose.
pub const SHORTEST_DELTA: f32 = 1e-6;

/// The distance vector, shared with the queue that orders states by it.
///
/// SICADA-DIVERGE: upstream passes the queue a bare pointer to the vector the
/// algorithm is writing, leaving the caller to keep the two in step. Sharing is
/// the point, so it is spelled out.
pub type Distance<W> = Rc<RefCell<Vec<W>>>;

/// The knobs upstream's `ShortestDistanceOptions` offers, less the queue and
/// the distance vector, which are passed separately because they are shared.
#[derive(Debug, Clone)]
pub struct ShortestDistanceOptions<F> {
    /// Which arcs to follow; [`AnyArcFilter`] follows all of them.
    pub arc_filter: F,
    /// Where to start, or `None` for the FST's own start state.
    pub source: Option<usize>,
    /// How close two weights have to be before relaxation stops.
    pub delta: f32,
    /// Stop at the first final state reached.
    ///
    /// Only meaningful on a semiring with the path property, and only gives the
    /// shortest path when the queue is shortest-first, every final weight is
    /// the same, and every weight lies between `One` and `Zero` in the natural
    /// order.
    pub first_path: bool,
}

impl<F> ShortestDistanceOptions<F> {
    /// The defaults: follow every arc, start at the FST's start state, and run
    /// to convergence.
    pub fn new(arc_filter: F) -> Self {
        Self {
            arc_filter,
            source: None,
            delta: SHORTEST_DELTA,
            first_path: false,
        }
    }
}

/// The buffers a shortest-distance run needs, kept so that several runs from
/// different sources can share them.
///
/// Port of upstream's `internal::ShortestDistanceState`. Without this,
/// [`rm_epsilon`](super::rmepsilon::rm_epsilon), which computes the distance
/// from *every* state over the epsilon arcs, pays to build and fill a
/// full-length vector once per state, and the pass is quadratic in the number
/// of states however few epsilon arcs there are.
///
/// A run marks each entry it touches with the number of the run that touched
/// it; an entry left from an earlier run is indistinguishable from an untouched
/// one, so nothing has to be cleared in between.
pub struct ShortestDistanceState<W: Weight, S> {
    /// Sums the distances accurately.
    adder: Vec<Adder<W>>,
    /// What each state has gained since it was last expanded.
    residual: Vec<Adder<W>>,
    enqueued: Vec<bool>,
    /// Which run last wrote each entry, or [`NO_RUN`] for none.
    sources: Vec<usize>,
    /// How far the buffers have been grown.
    ///
    /// Only [`ensure`](Self::ensure) grows them, so keeping the length here
    /// turns the check that runs once per arc into a comparison instead of a
    /// `RefCell` borrow.
    grown: usize,
    /// The number of the run in progress.
    run: usize,
    /// Whether the buffers survive a run.
    retain: bool,
    _marker: std::marker::PhantomData<fn() -> S>,
}

/// The mark on an entry no run has touched.
const NO_RUN: usize = usize::MAX;

impl<W: Weight, S: ArcStateId> ShortestDistanceState<W, S> {
    /// Buffers for one run, thrown away afterwards.
    pub fn new() -> Self {
        Self {
            adder: Vec::new(),
            residual: Vec::new(),
            enqueued: Vec::new(),
            sources: Vec::new(),
            grown: 0,
            run: 0,
            retain: false,
            _marker: std::marker::PhantomData,
        }
    }

    /// Buffers that survive, for a caller that runs from many sources.
    ///
    /// The distance vector is *not* cleared between runs: an entry that the
    /// current run has not written holds whatever the last one left, and only
    /// the states this run reached mean anything. Every caller of this reads
    /// the distance through its own walk of the same arcs, so it never looks at
    /// an entry the run did not write.
    pub fn retained() -> Self {
        Self {
            retain: true,
            ..Self::new()
        }
    }

    /// Grows the buffers so that `index` is a valid entry.
    #[inline]
    fn ensure(&mut self, distance: &Distance<W>, index: usize) {
        if index < self.grown {
            return;
        }
        self.grow(distance, index);
    }

    /// The uncommon half of [`ensure`](Self::ensure), kept out of line so that
    /// the common one is a comparison.
    #[cold]
    fn grow(&mut self, distance: &Distance<W>, index: usize) {
        let mut distance = distance.borrow_mut();
        while distance.len() <= index {
            distance.push(W::zero());
            self.adder.push(Adder::new());
            self.residual.push(Adder::new());
            self.enqueued.push(false);
        }
        self.grown = distance.len();
        if self.retain {
            while self.sources.len() <= index {
                self.sources.push(NO_RUN);
            }
        }
    }

    /// Wipes an entry an earlier run left behind, if that is what it is.
    fn claim(&mut self, distance: &Distance<W>, index: usize) {
        if !self.retain || self.sources[index] == self.run {
            return;
        }
        distance.borrow_mut()[index] = W::zero();
        self.adder[index].reset(W::zero());
        self.residual[index].reset(W::zero());
        self.enqueued[index] = false;
        self.sources[index] = self.run;
    }

    /// Runs the algorithm from `opts.source`, or from the FST's start state.
    pub fn run<A, F, Q, AF>(
        &mut self,
        fst: &F,
        distance: &Distance<A::Weight>,
        queue: &mut Q,
        opts: &ShortestDistanceOptions<AF>,
    ) -> Result<(), OpenFstError>
    where
        A: Arc<Weight = W, StateId = S>,
        F: Fst<A>,
        Q: Queue<A::StateId>,
        AF: ArcFilter<A>,
    {
        let Some(start) = fst.start() else {
            return if fst.properties(K_ERROR, false) & K_ERROR != 0 {
                Err(OpenFstError::InvalidOperation(
                    "ShortestDistance: the FST is marked as being in error".into(),
                ))
            } else {
                // No start state means no paths, and every distance is Zero.
                if !self.retain {
                    distance.borrow_mut().clear();
                }
                Ok(())
            };
        };
        if W::properties() & RIGHT_SEMIRING == 0 {
            return Err(OpenFstError::InvalidOperation(format!(
                "ShortestDistance: the weight has to be right distributive: {}",
                W::type_name()
            )));
        }
        if opts.first_path && W::properties() & PATH == 0 {
            return Err(OpenFstError::InvalidOperation(format!(
                "ShortestDistance: first_path needs a weight with the path property: {}",
                W::type_name()
            )));
        }

        queue.clear();
        if !self.retain {
            distance.borrow_mut().clear();
            self.adder.clear();
            self.residual.clear();
            self.enqueued.clear();
            self.grown = 0;
        }
        if let Some(nstates) = fst.num_states_if_known() {
            distance.borrow_mut().reserve(nstates);
            self.adder.reserve(nstates);
            self.residual.reserve(nstates);
            self.enqueued.reserve(nstates);
        }

        let source = opts.source.map_or(start, A::StateId::from_usize);
        let source_index = source.as_usize();
        self.ensure(distance, source_index);
        self.claim(distance, source_index);
        distance.borrow_mut()[source_index] = W::one();
        self.adder[source_index].reset(W::one());
        self.residual[source_index].reset(W::one());
        self.enqueued[source_index] = true;
        queue.enqueue(source);

        let zero = W::zero();
        while let Some(state) = queue.dequeue() {
            let index = state.as_usize();
            self.ensure(distance, index);
            if opts.first_path && fst.final_weight(state) != zero {
                break;
            }
            self.enqueued[index] = false;
            let r = self.residual[index].sum();
            self.residual[index].reset(W::zero());

            for arc in fst.arcs(state) {
                if !opts.arc_filter.call(&arc) {
                    continue;
                }
                let next = arc.nextstate().as_usize();
                self.ensure(distance, next);
                self.claim(distance, next);
                let weight = r.times(arc.weight());
                // One borrow for the read and the write together: the queue is
                // only touched after it is dropped, so nothing else can be
                // looking at the distances meanwhile.
                let updated = {
                    let mut distance = distance.borrow_mut();
                    let current = &distance[next];
                    if current.approx_equal(&current.plus(&weight), opts.delta) {
                        // Converged as far as `delta` asks.
                        continue;
                    }
                    self.adder[next].add(&weight);
                    self.residual[next].add(&weight);
                    let updated = self.adder[next].sum();
                    distance[next] = updated.clone();
                    updated
                };
                if !updated.is_member() || !self.residual[next].sum().is_member() {
                    return Err(OpenFstError::InvalidOperation(
                        "ShortestDistance: the relaxation left the semiring".into(),
                    ));
                }
                if self.enqueued[next] {
                    queue.update(arc.nextstate());
                } else {
                    queue.enqueue(arc.nextstate());
                    self.enqueued[next] = true;
                }
            }
        }
        self.run = self.run.wrapping_add(1);
        if self.run == NO_RUN {
            // Wrapped round onto the mark that means "no run"; start again with
            // every entry unclaimed.
            self.sources.iter_mut().for_each(|source| *source = NO_RUN);
            self.run = 0;
        }

        if fst.properties(K_ERROR, false) & K_ERROR != 0 {
            return Err(OpenFstError::InvalidOperation(
                "ShortestDistance: the FST is marked as being in error".into(),
            ));
        }
        Ok(())
    }
}

impl<W: Weight, S: ArcStateId> Default for ShortestDistanceState<W, S> {
    fn default() -> Self {
        Self::new()
    }
}

/// The generic shortest-distance algorithm, over one queue and one arc filter.
///
/// Fills `distance[s]` with the ⊕-sum over every path from the source to `s`.
/// Mohri's formulation: each state carries a residual `r`, being what it has
/// gained since it was last expanded, and relaxes its neighbours with that
/// rather than with its whole distance, so no weight is counted twice.
///
/// The queue is handed in because its discipline is the algorithm's main tuning
/// knob, and because a shortest-first queue has to read `distance` as it goes,
/// which is why that is shared rather than owned.
///
/// A caller running this from many sources should keep a
/// [`ShortestDistanceState::retained`] instead, which reuses its buffers.
///
/// # Errors
///
/// SICADA-DIVERGE: upstream signals a failure by overwriting the whole distance
/// vector with a single `NoWeight`, so a caller that does not check the length
/// reads a distance for state 0 that is not a weight at all. Failures are
/// returned as an `Err` here, and the distance vector is left as it stands.
pub fn shortest_distance_with<A, F, Q, AF>(
    fst: &F,
    distance: &Distance<A::Weight>,
    queue: &mut Q,
    opts: &ShortestDistanceOptions<AF>,
) -> Result<(), OpenFstError>
where
    A: Arc,
    F: Fst<A>,
    Q: Queue<A::StateId>,
    AF: ArcFilter<A>,
{
    ShortestDistanceState::<A::Weight, A::StateId>::new().run(fst, distance, queue, opts)
}

/// Builds the queue the FST's shape calls for and runs the algorithm forwards
/// from the start state.
pub fn shortest_distance_forward<A, F>(fst: &F, delta: f32) -> Result<Vec<A::Weight>, OpenFstError>
where
    A: Arc,
    F: Fst<A>,
{
    let distance: Distance<A::Weight> = Rc::new(RefCell::new(Vec::new()));
    // The natural order is only an order where the semiring has the path
    // property; without it, `AutoQueue` falls back to disciplines that do not
    // need one.
    let comp = state_weight_compare::<A::StateId, A::Weight, _>(
        Rc::clone(&distance),
        natural_less_unchecked::<A::Weight>,
    );
    let comp = (A::Weight::properties() & PATH != 0).then_some(comp);
    let mut queue = AutoQueue::new(fst, comp);
    let opts = ShortestDistanceOptions {
        delta,
        ..ShortestDistanceOptions::new(AnyArcFilter)
    };
    shortest_distance_with(fst, &distance, &mut queue, &opts)?;
    Ok(Rc::try_unwrap(distance)
        .map(RefCell::into_inner)
        .unwrap_or_else(|shared| shared.borrow().clone()))
}

/// The distance from each state to the final states, rather than from the start
/// state to each.
///
/// Computed by reversing the FST and running forwards, which is why the weight
/// has to be *left* distributive for this direction: the reverse of a right
/// distributive semiring.
pub fn shortest_distance_reverse<A, F>(fst: &F, delta: f32) -> Result<Vec<A::Weight>, OpenFstError>
where
    A: Arc,
    F: Fst<A>,
    <A::Weight as Weight>::ReverseWeight: Weight<ReverseWeight = A::Weight>,
{
    let mut reversed = VectorFst::<A::Reverse>::new();
    crate::algorithms::reverse::reverse(fst, &mut reversed, true);
    let rdistance = shortest_distance_forward::<A::Reverse, _>(&reversed, delta)?;
    // Reversing added a superinitial state at 0, so state `s` of the original
    // is state `s + 1` of the reverse.
    Ok(rdistance.iter().skip(1).map(Weight::reverse).collect())
}

/// The ⊕-sum of the weights of every accepting path.
///
/// Over the tropical semiring that is the weight of the lightest path; over the
/// log semiring it is the ⊕-sum over all of them, as a probability requires.
///
/// Which direction this runs depends on the semiring: summing the forward
/// distances against the final weights needs right distributivity, so a
/// left-only semiring is run backwards instead.
pub fn shortest_distance<A, F>(fst: &F, delta: f32) -> Result<A::Weight, OpenFstError>
where
    A: Arc,
    F: Fst<A>,
    <A::Weight as Weight>::ReverseWeight: Weight<ReverseWeight = A::Weight>,
{
    if A::Weight::properties() & RIGHT_SEMIRING != 0 {
        let distance = shortest_distance_forward(fst, delta)?;
        let mut total = Adder::<A::Weight>::new();
        for (index, weight) in distance.iter().enumerate() {
            total.add(&weight.times(&fst.final_weight(A::StateId::from_usize(index))));
        }
        Ok(total.sum())
    } else {
        let distance = shortest_distance_reverse::<A, F>(fst, delta)?;
        Ok(match fst.start() {
            Some(start) => distance
                .get(start.as_usize())
                .cloned()
                .unwrap_or_else(A::Weight::zero),
            None => A::Weight::zero(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{Rng, paths, random_acyclic_fst};
    use crate::arc::{LogArc, StdArc};
    use crate::fst::{ExpandedFst as _, MutableFst};
    use crate::fsts::vector_fst::{StdVectorFst, VectorFst};
    use crate::properties::K_FST_PROPERTIES;
    use crate::queue::{FifoQueue, LifoQueue, StateOrderQueue};
    use crate::weights::float_weight::{LogWeight, TropicalWeight};

    fn forward(fst: &StdVectorFst) -> Vec<TropicalWeight> {
        shortest_distance_forward(fst, SHORTEST_DELTA).unwrap()
    }

    /// Reusing the buffers has to give what a fresh set would, for every state
    /// the run actually reached.
    ///
    /// The states it did not reach are what the reuse trades away: their
    /// entries hold whatever an earlier run left, which is why only a caller
    /// that walks the same arcs itself may keep a retained state.
    #[test]
    fn a_retained_run_agrees_with_a_fresh_one_wherever_it_reached() {
        let mut rng = Rng::new(0x5D_1571);
        let fst: StdVectorFst = random_acyclic_fst(&mut rng, 40);
        let zero = TropicalWeight::zero();

        let mut retained = ShortestDistanceState::<TropicalWeight, i32>::retained();
        let shared: Distance<TropicalWeight> = Rc::new(RefCell::new(Vec::new()));

        // Deliberately not in order, and with sources revisited, so that a run
        // meets entries left by both earlier and later ones.
        for source in [0usize, 17, 3, 17, 39, 0, 12] {
            let opts = ShortestDistanceOptions {
                source: Some(source),
                ..ShortestDistanceOptions::new(AnyArcFilter)
            };

            let mut queue = StateOrderQueue::new();
            retained.run(&fst, &shared, &mut queue, &opts).unwrap();
            let reused = shared.borrow().clone();

            let fresh_distance: Distance<TropicalWeight> = Rc::new(RefCell::new(Vec::new()));
            let mut queue = StateOrderQueue::new();
            shortest_distance_with(&fst, &fresh_distance, &mut queue, &opts).unwrap();
            let fresh = fresh_distance.borrow().clone();

            let mut reached = 0;
            for (state, expected) in fresh.iter().enumerate() {
                if *expected == zero {
                    continue;
                }
                reached += 1;
                assert_eq!(
                    reused.get(state),
                    Some(expected),
                    "source {source}, state {state}"
                );
            }
            assert!(reached > 0, "source {source} reached nothing to compare");
        }
    }

    /// The distance to each state by enumerating every path to it, which is the
    /// definition the algorithm is an efficient way of computing.
    fn by_enumeration(fst: &StdVectorFst, max_len: usize) -> Vec<TropicalWeight> {
        let mut distance = vec![TropicalWeight::zero(); fst.num_states()];
        let Some(start) = fst.start() else {
            return distance;
        };
        // Every path of at most `max_len` arcs, walked in full.
        let mut frontier = vec![(start, TropicalWeight::one())];
        distance[start.as_usize()] = TropicalWeight::one();
        for _ in 0..max_len {
            let mut next = Vec::new();
            for (state, weight) in frontier.drain(..) {
                for arc in fst.arcs(state) {
                    let through = weight.times(arc.weight());
                    let at = &mut distance[arc.nextstate().as_usize()];
                    let sum = at.plus(&through);
                    if sum != *at {
                        *at = sum;
                        next.push((arc.nextstate(), through));
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        distance
    }

    /// 0 --1--> 1 --2--> 2, with a second way round costing more.
    fn diamond() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..4 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(1.0), 1));
        fst.add_arc(0, StdArc::new(2, 2, TropicalWeight(4.0), 2));
        fst.add_arc(1, StdArc::new(3, 3, TropicalWeight(2.0), 3));
        fst.add_arc(2, StdArc::new(4, 4, TropicalWeight(1.0), 3));
        fst.set_final(3, TropicalWeight(1.0));
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    #[test]
    fn the_distance_is_the_lightest_way_to_each_state() {
        let fst = diamond();
        let distance = forward(&fst);
        assert_eq!(
            distance,
            vec![
                TropicalWeight::one(),
                TropicalWeight(1.0),
                TropicalWeight(4.0),
                // 1 + 2 = 3 through state 1, against 4 + 1 = 5 through state 2.
                TropicalWeight(3.0),
            ]
        );
    }

    /// The ⊕-sum over accepting paths: the lightest one, over the tropical
    /// semiring.
    #[test]
    fn the_total_is_the_lightest_accepting_path() {
        let total = shortest_distance(&diamond(), SHORTEST_DELTA).unwrap();
        assert_eq!(total, TropicalWeight(4.0), "1 + 2 to state 3, then 1");
    }

    /// The backward distance is what each state can still reach a final state
    /// for.
    #[test]
    fn the_reverse_distance_is_the_way_out_to_a_final_state() {
        let distance = shortest_distance_reverse(&diamond(), SHORTEST_DELTA).unwrap();
        assert_eq!(
            distance,
            vec![
                TropicalWeight(4.0), // the whole lightest path
                TropicalWeight(3.0), // 2 to state 3, then 1
                TropicalWeight(2.0), // 1 to state 3, then 1
                TropicalWeight(1.0), // already there
            ]
        );
    }

    /// An FST with no start state has no paths, so every distance is Zero.
    #[test]
    fn an_fst_with_no_start_state_has_no_distances() {
        let mut fst = StdVectorFst::new();
        fst.add_state();
        assert!(forward(&fst).is_empty());
        assert_eq!(
            shortest_distance(&fst, SHORTEST_DELTA).unwrap(),
            TropicalWeight::zero()
        );
    }

    /// A cycle contributes as many times round as it takes to converge, which
    /// over the tropical semiring with a non-negative cycle is once.
    #[test]
    fn a_cycle_settles() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(1.0), 1));
        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight(2.0), 2));
        fst.add_arc(2, StdArc::new(3, 3, TropicalWeight(3.0), 1));
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        assert_eq!(
            forward(&fst),
            vec![
                TropicalWeight::one(),
                TropicalWeight(1.0),
                TropicalWeight(3.0)
            ]
        );
    }

    /// Over the log semiring the distance is the ⊕-sum of every path rather
    /// than the lightest, which is where a diamond's two branches both count.
    #[test]
    fn the_log_semiring_sums_over_every_path() {
        let mut fst: VectorFst<LogArc> = VectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, LogArc::new(1, 1, LogWeight(1.0), 1));
        fst.add_arc(0, LogArc::new(2, 2, LogWeight(1.0), 1));
        fst.set_final(1, LogWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let distance = shortest_distance_forward(&fst, SHORTEST_DELTA).unwrap();
        // -log(e^-1 + e^-1) = 1 - log 2.
        let want = 1.0 - 2.0f32.ln();
        assert!(
            (distance[1].value() - want).abs() < 1e-5,
            "{:?} against {want}",
            distance[1]
        );
    }

    /// The queue discipline decides how soon the algorithm converges, never
    /// what it converges to.
    #[test]
    fn the_queue_discipline_never_changes_the_answer() {
        let mut rng = Rng::new(0x0D15_7A11_u64);
        for round in 0..200 {
            let fst = random_acyclic_fst(&mut rng, 7);
            let want = forward(&fst);

            for name in ["fifo", "lifo", "state-order"] {
                let distance: Distance<TropicalWeight> = Rc::new(RefCell::new(Vec::new()));
                let opts = ShortestDistanceOptions::new(AnyArcFilter);
                match name {
                    "fifo" => {
                        shortest_distance_with(&fst, &distance, &mut FifoQueue::<i32>::new(), &opts)
                    }
                    "lifo" => {
                        shortest_distance_with(&fst, &distance, &mut LifoQueue::<i32>::new(), &opts)
                    }
                    _ => shortest_distance_with(
                        &fst,
                        &distance,
                        &mut StateOrderQueue::<i32>::new(),
                        &opts,
                    ),
                }
                .unwrap();
                assert_eq!(distance.borrow().as_slice(), want, "round {round}, {name}");
            }
        }
    }

    /// The distance really is the ⊕-sum over every path to a state, checked
    /// against walking them all.
    #[test]
    fn the_distance_agrees_with_enumerating_every_path() {
        let mut rng = Rng::new(0x_0EEF_5D15_u64);
        for round in 0..200 {
            let fst = random_acyclic_fst(&mut rng, 7);
            let got = forward(&fst);
            let want = by_enumeration(&fst, 16);
            // Unvisited states past the last one reached are simply absent.
            assert!(got.len() <= want.len(), "round {round}");
            for (state, weight) in got.iter().enumerate() {
                assert_eq!(*weight, want[state], "round {round}, state {state}");
            }
        }
    }

    /// The total is the ⊕-sum over accepting paths, checked against
    /// enumerating them.
    #[test]
    fn the_total_agrees_with_enumerating_accepting_paths() {
        let mut rng = Rng::new(0x0007_07A1_u64);
        for round in 0..200 {
            let fst = random_acyclic_fst(&mut rng, 7);
            let got = shortest_distance(&fst, SHORTEST_DELTA).unwrap();
            let want = paths(&fst, 16)
                .into_iter()
                .fold(TropicalWeight::zero(), |total, (_, _, weight)| {
                    total.plus(&weight)
                });
            assert_eq!(got, want, "round {round}");
        }
    }

    /// Stopping at the first final state is only allowed where the semiring
    /// has the path property.
    #[test]
    fn stopping_at_the_first_final_state_needs_the_path_property() {
        let fst = diamond();
        let distance: Distance<TropicalWeight> = Rc::new(RefCell::new(Vec::new()));
        let opts = ShortestDistanceOptions {
            first_path: true,
            ..ShortestDistanceOptions::new(AnyArcFilter)
        };
        assert!(
            shortest_distance_with(&fst, &distance, &mut FifoQueue::<i32>::new(), &opts).is_ok()
        );

        let mut log: VectorFst<LogArc> = VectorFst::new();
        log.add_state();
        log.set_start(0);
        log.set_final(0, LogWeight::one());
        let distance: Distance<LogWeight> = Rc::new(RefCell::new(Vec::new()));
        let err = shortest_distance_with(&log, &distance, &mut FifoQueue::<i32>::new(), &opts)
            .unwrap_err();
        assert!(format!("{err}").contains("path property"), "{err}");
    }

    /// Starting somewhere other than the start state gives the distances from
    /// there.
    #[test]
    fn the_source_can_be_any_state() {
        let fst = diamond();
        let distance: Distance<TropicalWeight> = Rc::new(RefCell::new(Vec::new()));
        let opts = ShortestDistanceOptions {
            source: Some(1),
            ..ShortestDistanceOptions::new(AnyArcFilter)
        };
        shortest_distance_with(&fst, &distance, &mut FifoQueue::<i32>::new(), &opts).unwrap();
        let got = distance.borrow().clone();
        assert_eq!(got[1], TropicalWeight::one());
        assert_eq!(got[3], TropicalWeight(2.0));
        assert_eq!(got[0], TropicalWeight::zero(), "state 0 is not reachable");
    }

    /// A filter that follows only epsilon arcs sees only the epsilon graph, as
    /// epsilon removal requires.
    #[test]
    fn an_arc_filter_limits_which_arcs_are_followed() {
        use crate::arc_filter::InputEpsilonArcFilter;

        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(0, 0, TropicalWeight(1.0), 1));
        fst.add_arc(0, StdArc::new(5, 5, TropicalWeight(1.0), 2));
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let distance: Distance<TropicalWeight> = Rc::new(RefCell::new(Vec::new()));
        let opts = ShortestDistanceOptions::new(InputEpsilonArcFilter);
        shortest_distance_with(&fst, &distance, &mut FifoQueue::<i32>::new(), &opts).unwrap();
        let got = distance.borrow().clone();
        assert_eq!(got[1], TropicalWeight(1.0));
        assert_eq!(
            got.get(2).copied().unwrap_or_else(TropicalWeight::zero),
            TropicalWeight::zero(),
            "the labelled arc was not followed"
        );
    }
}
