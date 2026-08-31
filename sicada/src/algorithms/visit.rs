//! Visitation in the order a queue chooses.
//!
//! Port of OpenFst's `visit.h`. Where [`dfs_visit`](super::dfs_visit) fixes the
//! order, this takes any of the disciplines in [`crate::queue`]: a
//! [`LifoQueue`](crate::queue::LifoQueue) gives depth-first, a
//! [`FifoQueue`](crate::queue::FifoQueue) gives breadth-first, and a
//! [`ShortestFirstQueue`](crate::queue::ShortestFirstQueue) gives what
//! shortest-path needs.

use crate::arc::{Arc, ArcStateId};
use crate::arc_filter::{AnyArcFilter, ArcFilter};
use crate::fst::{Fst, MutableFst};
use crate::properties::K_EXPANDED;
use std::marker::PhantomData;

pub use crate::queue::Queue;

/// Visitor Interface: class determining actions taken during a visit. If any of
/// the boolean member functions return false, the visit is aborted by first
/// calling FinishState() on all unfinished (grey) states and then calling FinishVisit().
pub trait Visitor<A: Arc> {
    /// Invoked before visit.
    fn init_visit<F: Fst<A>>(&mut self, fst: &F);
    /// Invoked when state discovered (2nd arg is visitation root).
    fn init_state(&mut self, s: A::StateId, root: A::StateId) -> bool;
    /// Invoked when arc to white/undiscovered state examined.
    fn white_arc(&mut self, s: A::StateId, arc: &A) -> bool;
    /// Invoked when arc to grey/unfinished state examined.
    fn grey_arc(&mut self, s: A::StateId, arc: &A) -> bool;
    /// Invoked when arc to black/finished state examined.
    fn black_arc(&mut self, s: A::StateId, arc: &A) -> bool;
    /// Invoked when state finished.
    fn finish_state(&mut self, s: A::StateId);
    /// Invoked after visit.
    fn finish_visit(&mut self);
}

const WHITE_STATE: u8 = 0x01; // Undiscovered.
const GREY_STATE: u8 = 0x02; // Discovered & unfinished.
const BLACK_STATE: u8 = 0x04; // Finished.
const ARC_ITER_DONE: u8 = 0x08;

/// Performs queue-dependent visitation. Visitor class argument determines
/// actions and contains any return data. ArcFilter determines arcs that are
/// considered. If 'access_only' is true, performs visitation only to states
/// accessible from the initial state.
pub fn visit<'a, F, V, Q, Filter, A>(
    fst: &'a F,
    visitor: &mut V,
    queue: &mut Q,
    filter: Filter,
    access_only: bool,
) where
    A: Arc,
    F: Fst<A>,
    V: Visitor<A>,
    Q: Queue<A::StateId>,
    Filter: ArcFilter<A>,
{
    visitor.init_visit(fst);
    let start = match fst.start() {
        Some(s) => s,
        None => {
            visitor.finish_visit();
            return;
        }
    };

    let start_idx = start.as_usize();
    // SICADA-OPT: upstream keeps arc iterators in a `MemoryPool` and stores raw
    // pointers to them, because a C++ `ArcIterator` cannot live in a vector.
    // These can, and `None` marks the same "iterator finished, dropped" state
    // its null pointer does.
    let mut state_status: Vec<u8> = Vec::new();
    let mut arc_iterator: Vec<Option<std::iter::Peekable<F::ArcIter<'a>>>> = Vec::new();

    let mut nstates = fst.num_states_if_known().unwrap_or(start_idx + 1);
    let expanded = (fst.properties(K_EXPANDED, false) & K_EXPANDED) != 0;

    state_status.resize(nstates, WHITE_STATE);
    arc_iterator.resize_with(nstates, || None);
    let mut siter = fst.states();

    let mut do_visit = true;
    let mut root_idx = start_idx;

    while do_visit && root_idx < nstates {
        let root = A::StateId::from_usize(root_idx);
        do_visit = visitor.init_state(root, root);
        state_status[root_idx] = GREY_STATE;
        queue.enqueue(root);

        while !queue.is_empty() {
            let state = queue.head().unwrap();
            let s_idx = state.as_usize();

            if s_idx >= state_status.len() {
                nstates = s_idx + 1;
                state_status.resize(nstates, WHITE_STATE);
                arc_iterator.resize_with(nstates, || None);
            }

            // Create arc iterator if needed
            if arc_iterator[s_idx].is_none()
                && (state_status[s_idx] & ARC_ITER_DONE) == 0
                && do_visit
            {
                arc_iterator[s_idx] = Some(fst.arcs(state).peekable());
            }

            // Check if iterator is done or visit was aborted
            let is_done = if let Some(aiter) = &mut arc_iterator[s_idx] {
                aiter.peek().is_none()
            } else {
                true
            };

            if is_done || !do_visit {
                arc_iterator[s_idx] = None;
                state_status[s_idx] |= ARC_ITER_DONE;
            }

            // Dequeue state and mark black if done
            if (state_status[s_idx] & ARC_ITER_DONE) != 0 {
                queue.dequeue();
                visitor.finish_state(state);
                state_status[s_idx] = BLACK_STATE;
                continue;
            }

            // We safely unwrap because we just checked `is_done`
            let arc = arc_iterator[s_idx]
                .as_mut()
                .unwrap()
                .peek()
                .unwrap()
                .clone();
            let next_idx = arc.nextstate().as_usize();

            if next_idx >= state_status.len() {
                nstates = next_idx + 1;
                state_status.resize(nstates, WHITE_STATE);
                arc_iterator.resize_with(nstates, || None);
            }

            if filter.call(&arc) {
                let next_color = state_status[next_idx] & !ARC_ITER_DONE;
                if next_color == WHITE_STATE {
                    do_visit = visitor.white_arc(state, &arc);
                    if do_visit {
                        do_visit = visitor.init_state(arc.nextstate(), root);
                        state_status[next_idx] = GREY_STATE;
                        queue.enqueue(arc.nextstate());
                    }
                } else if next_color == BLACK_STATE {
                    do_visit = visitor.black_arc(state, &arc);
                } else {
                    do_visit = visitor.grey_arc(state, &arc);
                }
            }

            // Advance the iterator
            let aiter = arc_iterator[s_idx].as_mut().unwrap();
            aiter.next();
            if aiter.peek().is_none() {
                arc_iterator[s_idx] = None;
                state_status[s_idx] |= ARC_ITER_DONE;
            }
        }

        if access_only {
            break;
        }

        root_idx = if root_idx == start_idx {
            0
        } else {
            root_idx + 1
        };
        while root_idx < nstates && state_status[root_idx] != WHITE_STATE {
            root_idx += 1;
        }

        if !expanded && root_idx == nstates {
            for state_val in &mut siter {
                if state_val.as_usize() == nstates {
                    nstates += 1;
                    state_status.push(WHITE_STATE);
                    arc_iterator.push(None);
                    break;
                }
            }
        }
    }

    visitor.finish_visit();
}

/// Convenience wrapper for Visit that uses an AnyArcFilter.
#[inline(always)]
pub fn visit_any<F, V, Q, A>(fst: &F, visitor: &mut V, queue: &mut Q)
where
    A: Arc,
    F: Fst<A>,
    V: Visitor<A>,
    Q: Queue<A::StateId>,
{
    visit(fst, visitor, queue, AnyArcFilter, false);
}

/// Copies input FST to mutable FST following queue order.
pub struct CopyVisitor<'a, A: Arc, F1: Fst<A>, F2: MutableFst<A>> {
    ifst: &'a F1,
    ofst: &'a mut F2,
    _marker: PhantomData<A>,
}

impl<'a, A: Arc, F1: Fst<A>, F2: MutableFst<A>> CopyVisitor<'a, A, F1, F2> {
    pub fn new(ifst: &'a F1, ofst: &'a mut F2) -> Self {
        Self {
            ifst,
            ofst,
            _marker: PhantomData,
        }
    }
}

impl<'a, A: Arc, F1: Fst<A>, F2: MutableFst<A>> Visitor<A> for CopyVisitor<'a, A, F1, F2> {
    fn init_visit<F: Fst<A>>(&mut self, _fst: &F) {
        self.ofst.delete_all_states();
        if let Some(start) = self.ifst.start() {
            self.ofst.set_start(start);
        }
    }

    fn init_state(&mut self, state: A::StateId, _root: A::StateId) -> bool {
        while self.ofst.num_states() <= state.as_usize() {
            self.ofst.add_state();
        }
        true
    }

    fn white_arc(&mut self, state: A::StateId, arc: &A) -> bool {
        self.ofst.add_arc(state, arc.clone());
        true
    }

    fn grey_arc(&mut self, state: A::StateId, arc: &A) -> bool {
        self.ofst.add_arc(state, arc.clone());
        true
    }

    fn black_arc(&mut self, state: A::StateId, arc: &A) -> bool {
        self.ofst.add_arc(state, arc.clone());
        true
    }

    fn finish_state(&mut self, state: A::StateId) {
        self.ofst.set_final(state, self.ifst.final_weight(state));
    }

    fn finish_visit(&mut self) {}
}

/// Visits input FST up to a state limit following queue order.
pub struct PartialVisitor<'a, A: Arc, F1: Fst<A>> {
    _fst: &'a F1,
    maxvisit: usize,
    ninit: usize,
    nfinish: usize,
    _marker: PhantomData<A>,
}

impl<'a, A: Arc, F1: Fst<A>> PartialVisitor<'a, A, F1> {
    pub fn new(fst: &'a F1, maxvisit: usize) -> Self {
        Self {
            _fst: fst,
            maxvisit,
            ninit: 0,
            nfinish: 0,
            _marker: PhantomData,
        }
    }

    pub fn num_initialized(&self) -> usize {
        self.ninit
    }

    pub fn num_finished(&self) -> usize {
        self.nfinish
    }
}

impl<'a, A: Arc, F1: Fst<A>> Visitor<A> for PartialVisitor<'a, A, F1> {
    fn init_visit<F: Fst<A>>(&mut self, _fst: &F) {
        self.ninit = 0;
        self.nfinish = 0;
    }

    fn init_state(&mut self, _state: A::StateId, _root: A::StateId) -> bool {
        self.ninit += 1;
        self.ninit <= self.maxvisit
    }

    fn white_arc(&mut self, _state: A::StateId, _arc: &A) -> bool {
        true
    }

    fn grey_arc(&mut self, _state: A::StateId, _arc: &A) -> bool {
        true
    }

    fn black_arc(&mut self, _state: A::StateId, _arc: &A) -> bool {
        true
    }

    fn finish_state(&mut self, _state: A::StateId) {
        self.nfinish += 1;
    }

    fn finish_visit(&mut self) {}
}

/// Copies input FST to mutable FST up to a state limit following queue order.
pub struct PartialCopyVisitor<'a, A: Arc, F1: Fst<A>, F2: MutableFst<A>> {
    copy_visitor: CopyVisitor<'a, A, F1, F2>,
    maxvisit: usize,
    ninit: usize,
    nfinish: usize,
    copy_grey: bool,
    copy_black: bool,
}

impl<'a, A: Arc, F1: Fst<A>, F2: MutableFst<A>> PartialCopyVisitor<'a, A, F1, F2> {
    pub fn new(
        ifst: &'a F1,
        ofst: &'a mut F2,
        maxvisit: usize,
        copy_grey: bool,
        copy_black: bool,
    ) -> Self {
        Self {
            copy_visitor: CopyVisitor::new(ifst, ofst),
            maxvisit,
            ninit: 0,
            nfinish: 0,
            copy_grey,
            copy_black,
        }
    }

    pub fn num_initialized(&self) -> usize {
        self.ninit
    }

    pub fn num_finished(&self) -> usize {
        self.nfinish
    }
}

impl<'a, A: Arc, F1: Fst<A>, F2: MutableFst<A>> Visitor<A> for PartialCopyVisitor<'a, A, F1, F2> {
    fn init_visit<F: Fst<A>>(&mut self, fst: &F) {
        self.copy_visitor.init_visit(fst);
        self.ninit = 0;
        self.nfinish = 0;
    }

    fn init_state(&mut self, state: A::StateId, root: A::StateId) -> bool {
        self.copy_visitor.init_state(state, root);
        self.ninit += 1;
        self.ninit <= self.maxvisit
    }

    fn white_arc(&mut self, state: A::StateId, arc: &A) -> bool {
        self.copy_visitor.white_arc(state, arc)
    }

    fn grey_arc(&mut self, state: A::StateId, arc: &A) -> bool {
        if self.copy_grey {
            self.copy_visitor.grey_arc(state, arc)
        } else {
            true
        }
    }

    fn black_arc(&mut self, state: A::StateId, arc: &A) -> bool {
        if self.copy_black {
            self.copy_visitor.black_arc(state, arc)
        } else {
            true
        }
    }

    fn finish_state(&mut self, state: A::StateId) {
        self.copy_visitor.finish_state(state);
        self.nfinish += 1;
    }

    fn finish_visit(&mut self) {
        self.copy_visitor.finish_visit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{Rng, random_acyclic_fst};
    use crate::arc::StdArc;
    use crate::float_weight::TropicalWeight;
    use crate::fst::{ExpandedFst as _, Fst, MutableFst};
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::queue::{FifoQueue, LifoQueue};
    use crate::weight::Weight as _;

    #[test]
    fn test_visit_copy_dfs() {
        let mut ifst = StdVectorFst::new();
        let s0 = ifst.add_state();
        let s1 = ifst.add_state();
        let s2 = ifst.add_state();
        ifst.set_start(s0);
        ifst.set_final(s2, TropicalWeight(1.0));

        ifst.add_arc(s0, StdArc::new(1, 1, TropicalWeight(1.0), s1));
        ifst.add_arc(s1, StdArc::new(2, 2, TropicalWeight(2.0), s2));
        ifst.add_arc(s2, StdArc::new(3, 3, TropicalWeight(3.0), s0)); // cycle

        let mut ofst = StdVectorFst::new();
        let mut queue: LifoQueue<i32> = LifoQueue::new(); // DFS LIFO queue

        let mut visitor = CopyVisitor::new(&ifst, &mut ofst);
        visit_any(&ifst, &mut visitor, &mut queue);

        assert_eq!(ofst.num_states(), 3);
        assert_eq!(ofst.count_arcs(), 3);
        assert_eq!(ofst.final_weight(s2), TropicalWeight(1.0));
    }

    #[test]
    fn test_visit_copy_bfs() {
        let mut ifst = StdVectorFst::new();
        let s0 = ifst.add_state();
        let s1 = ifst.add_state();
        let s2 = ifst.add_state();
        ifst.set_start(s0);
        ifst.set_final(s2, TropicalWeight(1.0));

        ifst.add_arc(s0, StdArc::new(1, 1, TropicalWeight(1.0), s1));
        ifst.add_arc(s0, StdArc::new(2, 2, TropicalWeight(1.5), s2));
        ifst.add_arc(s1, StdArc::new(3, 3, TropicalWeight(2.0), s2));

        let mut ofst = StdVectorFst::new();
        let mut queue: FifoQueue<i32> = FifoQueue::new(); // BFS FIFO queue

        let mut visitor = CopyVisitor::new(&ifst, &mut ofst);
        visit_any(&ifst, &mut visitor, &mut queue);

        assert_eq!(ofst.num_states(), 3);
        assert_eq!(ofst.count_arcs(), 3);
        assert_eq!(ofst.final_weight(s2), TropicalWeight(1.0));
    }

    /// Visiting with a copying visitor has to reproduce the FST, every state,
    /// arc and weight of it, whichever order the queue chooses.
    #[test]
    fn copying_through_a_visit_reproduces_the_fst() {
        let mut rng = Rng::new(0x1357_9BDF);
        for round in 0..200 {
            let fst = random_acyclic_fst(&mut rng, 6);

            for lifo in [false, true] {
                let mut copy = StdVectorFst::new();
                {
                    let mut visitor = CopyVisitor::new(&fst, &mut copy);
                    if lifo {
                        let mut queue: LifoQueue<i32> = LifoQueue::new();
                        visit(&fst, &mut visitor, &mut queue, AnyArcFilter, false);
                    } else {
                        let mut queue: FifoQueue<i32> = FifoQueue::new();
                        visit(&fst, &mut visitor, &mut queue, AnyArcFilter, false);
                    }
                }

                assert_eq!(copy.num_states(), fst.num_states(), "round {round}");
                assert_eq!(copy.start(), fst.start(), "round {round}");
                for s in 0..fst.num_states() as i32 {
                    assert_eq!(
                        copy.final_weight(s),
                        fst.final_weight(s),
                        "round {round}, state {s}"
                    );
                    assert_eq!(
                        copy.arcs(s).collect::<Vec<_>>(),
                        fst.arcs(s).collect::<Vec<_>>(),
                        "round {round}, state {s}, lifo={lifo}"
                    );
                }
            }
        }
    }

    /// Records what it was told, and can be made to refuse at a chosen point.
    #[derive(Default)]
    struct Recorder {
        initialized: Vec<i32>,
        finished: Vec<i32>,
        stop_after: Option<usize>,
    }

    impl Visitor<StdArc> for Recorder {
        fn init_visit<F: Fst<StdArc>>(&mut self, _fst: &F) {}

        fn init_state(&mut self, s: i32, _root: i32) -> bool {
            self.initialized.push(s);
            self.stop_after.is_none_or(|n| self.initialized.len() <= n)
        }

        fn white_arc(&mut self, _s: i32, _arc: &StdArc) -> bool {
            true
        }

        fn grey_arc(&mut self, _s: i32, _arc: &StdArc) -> bool {
            true
        }

        fn black_arc(&mut self, _s: i32, _arc: &StdArc) -> bool {
            true
        }

        fn finish_state(&mut self, s: i32) {
            self.finished.push(s);
        }

        fn finish_visit(&mut self) {}
    }

    fn chain(n: usize) -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..n {
            fst.add_state();
        }
        fst.set_start(0);
        for s in 0..n - 1 {
            fst.add_arc(
                s as i32,
                StdArc::new(1, 1, TropicalWeight::one(), s as i32 + 1),
            );
        }
        fst.set_final(n as i32 - 1, TropicalWeight::one());
        fst
    }

    /// Aborting is an unwind here too: every state still on the queue is
    /// finished before the visit ends.
    #[test]
    fn aborting_finishes_every_state_still_queued() {
        let fst = chain(6);
        let mut visitor = Recorder {
            stop_after: Some(3),
            ..Default::default()
        };
        let mut queue: FifoQueue<i32> = FifoQueue::new();
        visit(&fst, &mut visitor, &mut queue, AnyArcFilter, false);

        // Refusing the fourth state does not un-discover it: upstream marks it
        // grey and queues it before the refusal takes effect, so it is one of
        // the states the unwind then finishes.
        assert_eq!(visitor.initialized, vec![0, 1, 2, 3]);
        let mut finished = visitor.finished.clone();
        finished.sort_unstable();
        assert_eq!(finished, vec![0, 1, 2, 3], "what was started is finished");
    }

    /// Without an abort, every state is initialized once and finished once.
    #[test]
    fn every_state_is_initialized_once_and_finished_once() {
        let mut rng = Rng::new(0xF00D_1234);
        for _ in 0..100 {
            let fst = random_acyclic_fst(&mut rng, 6);
            let mut visitor = Recorder::default();
            let mut queue: FifoQueue<i32> = FifoQueue::new();
            visit(&fst, &mut visitor, &mut queue, AnyArcFilter, false);

            let mut initialized = visitor.initialized.clone();
            initialized.sort_unstable();
            let mut finished = visitor.finished.clone();
            finished.sort_unstable();
            let all: Vec<i32> = (0..fst.num_states() as i32).collect();
            assert_eq!(initialized, all);
            assert_eq!(finished, all);
        }
    }

    /// The partial visitor is the abort mechanism given a count.
    ///
    /// It counts the state it refuses: upstream increments before testing the
    /// limit, so a limit of three lets a fourth state be discovered and, since
    /// discovery queues it before the refusal is acted on, finished as well.
    #[test]
    fn a_partial_visit_stops_at_its_limit() {
        let fst = chain(8);
        let mut visitor = PartialVisitor::new(&fst, 3);
        let mut queue: FifoQueue<i32> = FifoQueue::new();
        visit(&fst, &mut visitor, &mut queue, AnyArcFilter, false);

        assert_eq!(visitor.num_initialized(), 4);
        assert_eq!(visitor.num_finished(), 4, "what was started is finished");

        // A limit past the end of the FST stops nothing.
        let mut visitor = PartialVisitor::new(&fst, 100);
        let mut queue: FifoQueue<i32> = FifoQueue::new();
        visit(&fst, &mut visitor, &mut queue, AnyArcFilter, false);
        assert_eq!(visitor.num_initialized(), 8);
        assert_eq!(visitor.num_finished(), 8);
    }

    /// Copying up to a limit gives the states the limit allowed, and whether
    /// arcs to already-seen states come along is the caller's choice.
    #[test]
    fn a_partial_copy_stops_at_its_limit() {
        // A cycle, so that grey and black arcs both occur.
        let mut fst = StdVectorFst::new();
        for _ in 0..4 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.set_final(3, TropicalWeight::one());
        for &(from, to) in &[(0, 1), (1, 2), (2, 0), (2, 3)] {
            fst.add_arc(from, StdArc::new(1, 1, TropicalWeight::one(), to));
        }

        for (copy_grey, copy_black) in [(true, true), (false, false)] {
            let mut copy = StdVectorFst::new();
            let initialized = {
                let mut visitor =
                    PartialCopyVisitor::new(&fst, &mut copy, 3, copy_grey, copy_black);
                let mut queue: FifoQueue<i32> = FifoQueue::new();
                visit(&fst, &mut visitor, &mut queue, AnyArcFilter, false);
                visitor.num_initialized()
            };
            assert_eq!(initialized, 4, "the refused state is counted");
            assert_eq!(copy.num_states(), 4);

            // Out of state 2: the arc to 3, which is white and always copied,
            // and the arc back to 0, which is black by then and copied only if
            // asked for.
            let from_two = copy.arcs(2).count();
            assert_eq!(from_two, if copy_black { 2 } else { 1 });
            let _ = copy_grey;
        }
    }
}
