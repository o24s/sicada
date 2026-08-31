//! Depth-first visitation.
//!
//! Port of OpenFst's `dfs-visit.h`. See [`visit`](super::visit) for the more
//! general search-queue disciplines.

use crate::arc::{Arc, ArcStateId};
use crate::arc_filter::{AnyArcFilter, ArcFilter};
use crate::fst::Fst;
use crate::properties::K_EXPANDED;

/// Determines what a depth-first search does, and holds whatever it produces.
///
/// Returning `false` from any of the `bool` methods aborts the search. Aborting
/// is not the same as stopping: [`finish_state`](Self::finish_state) is still
/// called for every state left on the stack, deepest first, and then
/// [`finish_visit`](Self::finish_visit). A visitor that accumulates per-state
/// results in `finish_state` therefore sees a consistent picture either way.
pub trait DfsVisitor<A: Arc> {
    /// Invoked before the search begins.
    fn init_visit<F: Fst<A>>(&mut self, fst: &F);

    /// Invoked when a state is discovered. `root` is the root of the tree it
    /// was discovered under.
    fn init_state(&mut self, s: A::StateId, root: A::StateId) -> bool;

    /// Invoked on an arc to a white (undiscovered) state.
    fn tree_arc(&mut self, s: A::StateId, arc: &A) -> bool;

    /// Invoked on an arc to a grey (discovered but unfinished) state.
    fn back_arc(&mut self, s: A::StateId, arc: &A) -> bool;

    /// Invoked on an arc to a black (finished) state.
    fn forward_or_cross_arc(&mut self, s: A::StateId, arc: &A) -> bool;

    /// Invoked when a state is finished.
    ///
    /// `parent` and `arc` are `None` when `s` is the root of its tree, and
    /// otherwise are the state above it on the stack and the tree arc that
    /// reached it.
    fn finish_state(&mut self, s: A::StateId, parent: Option<A::StateId>, arc: Option<&A>);

    /// Invoked after the search ends, however it ended.
    fn finish_visit(&mut self);
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum StateColor {
    /// Undiscovered.
    White = 0,
    /// Discovered but unfinished.
    Grey = 1,
    /// Finished.
    Black = 2,
}

/// One frame of the search: a state, and how far its arcs have been walked.
///
/// SICADA-OPT: upstream allocates these out of a `MemoryPool` because it
/// `new`s one per frame and needs the allocation traffic absorbed. They are
/// pushed and popped in strict last-in-first-out order, which is to say the
/// pool is a stack, so here they live in the stack itself and cost no
/// allocation and no indirection at all.
struct DfsState<'a, F: Fst<A> + 'a, A: Arc> {
    state_id: A::StateId,
    arc_iter: F::ArcIter<'a>,
    /// The arc the parent descended on, and which state it left.
    ///
    /// SICADA-OPT: upstream keeps the parent's arc iterator resting on the tree
    /// arc and reads it back when the child finishes, which needs a `Peekable`,
    /// and an `ArcIter` that yields arcs by value has to *copy*
    /// each arc into the peek slot and read it out again, once per arc of the
    /// whole FST. Handing the arc to the child instead means each arc is taken
    /// from the iterator exactly once and moved exactly once, and the reports
    /// are identical.
    entered_by: Option<(A::StateId, A)>,
}

/// Visits `fst` depth first, calling `visitor` as states and arcs are reached.
///
/// Only arcs `filter` accepts are followed. With `access_only`, the search
/// covers the states reachable from the initial state and stops; otherwise it
/// continues over the remaining trees of the search forest.
pub fn dfs_visit<'a, F, V, Filter, A>(
    fst: &'a F,
    visitor: &mut V,
    filter: Filter,
    access_only: bool,
) where
    A: Arc,
    F: Fst<A> + 'a,
    V: DfsVisitor<A>,
    Filter: ArcFilter<A>,
{
    visitor.init_visit(fst);

    let Some(start) = fst.start() else {
        visitor.finish_visit();
        return;
    };
    let start_idx = start.as_usize();

    // Exact if the FST knows its size, a lower bound otherwise; the search
    // grows it as it discovers states beyond it.
    let mut nstates = fst.num_states_if_known().unwrap_or(start_idx + 1);
    let expanded = (fst.properties(K_EXPANDED, false) & K_EXPANDED) != 0;

    let mut state_color = vec![StateColor::White; nstates];
    let mut stack: Vec<DfsState<'a, F, A>> = Vec::new();
    let mut siter = fst.states();
    let mut dfs = true;
    let mut root_idx = start_idx;

    while dfs && root_idx < nstates {
        state_color[root_idx] = StateColor::Grey;
        let root = A::StateId::from_usize(root_idx);
        stack.push(DfsState {
            state_id: root,
            arc_iter: fst.arcs(root),
            entered_by: None,
        });
        dfs = visitor.init_state(root, root);

        while let Some(frame) = stack.last_mut() {
            let s = frame.state_id;

            // Finished, either because the visitor asked to stop or because the
            // arcs ran out. Unwinding through here rather than breaking out is
            // what gives every state left on the stack its `finish_state`.
            let next = if dfs { frame.arc_iter.next() } else { None };
            let Some(arc) = next else {
                let s_idx = s.as_usize();
                if s_idx >= state_color.len() {
                    nstates = s_idx + 1;
                    state_color.resize(nstates, StateColor::White);
                }
                state_color[s_idx] = StateColor::Black;
                let frame = stack.pop().expect("the frame just borrowed");
                match frame.entered_by {
                    Some((parent, arc)) => visitor.finish_state(s, Some(parent), Some(&arc)),
                    None => visitor.finish_state(s, None, None),
                }
                continue;
            };

            let nextstate = arc.nextstate();
            let nextstate_idx = nextstate.as_usize();
            if nextstate_idx >= state_color.len() {
                nstates = nextstate_idx + 1;
                state_color.resize(nstates, StateColor::White);
            }

            if !filter.call(&arc) {
                continue;
            }

            match state_color[nextstate_idx] {
                StateColor::White => {
                    dfs = visitor.tree_arc(s, &arc);
                    if dfs {
                        state_color[nextstate_idx] = StateColor::Grey;
                        stack.push(DfsState {
                            state_id: nextstate,
                            arc_iter: fst.arcs(nextstate),
                            entered_by: Some((s, arc)),
                        });
                        dfs = visitor.init_state(nextstate, root);
                    }
                }
                StateColor::Grey => dfs = visitor.back_arc(s, &arc),
                StateColor::Black => dfs = visitor.forward_or_cross_arc(s, &arc),
            }
        }

        if access_only {
            break;
        }

        // Next tree root: the first state still white. The start state was
        // taken first, so the sweep begins at zero.
        root_idx = if root_idx == start_idx {
            0
        } else {
            root_idx + 1
        };
        while root_idx < nstates && state_color[root_idx] != StateColor::White {
            root_idx += 1;
        }

        // An FST that does not know how many states it has may still have one
        // past everything seen so far.
        if !expanded && root_idx == nstates {
            for state in &mut siter {
                if state.as_usize() == nstates {
                    nstates += 1;
                    state_color.push(StateColor::White);
                    break;
                }
            }
        }
    }

    visitor.finish_visit();
}

/// Visits `fst` depth first, following every arc.
pub fn dfs_visit_any<'a, F, V, A>(fst: &'a F, visitor: &mut V)
where
    A: Arc,
    F: Fst<A> + 'a,
    V: DfsVisitor<A>,
{
    dfs_visit(fst, visitor, AnyArcFilter, false);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::arc_filter::InputEpsilonArcFilter;
    use crate::fst::MutableFst;
    use crate::fsts::vector_fst::VectorFst;
    use crate::weight::Weight;
    use crate::weights::float_weight::TropicalWeight;

    /// What the visitor was told, in order.
    #[derive(Debug, PartialEq, Eq, Clone)]
    enum Event {
        InitVisit,
        InitState(i32, i32),
        Tree(i32, i32),
        Back(i32, i32),
        ForwardOrCross(i32, i32),
        /// State, parent, and the label of the tree arc that reached it.
        Finish(i32, Option<i32>, Option<i32>),
        FinishVisit,
    }

    /// Records every call, and can be told to refuse at a chosen point.
    #[derive(Default)]
    struct Recorder {
        events: Vec<Event>,
        /// Stop when this event is about to be recorded.
        stop_at: Option<Event>,
    }

    impl Recorder {
        fn stopping_at(event: Event) -> Self {
            Self {
                events: Vec::new(),
                stop_at: Some(event),
            }
        }

        fn record(&mut self, event: Event) -> bool {
            let go_on = self.stop_at.as_ref() != Some(&event);
            self.events.push(event);
            go_on
        }

        fn finished(&self) -> Vec<(i32, Option<i32>)> {
            self.events
                .iter()
                .filter_map(|e| match e {
                    Event::Finish(s, parent, _) => Some((*s, *parent)),
                    _ => None,
                })
                .collect()
        }
    }

    impl DfsVisitor<StdArc> for Recorder {
        fn init_visit<F: Fst<StdArc>>(&mut self, _fst: &F) {
            self.events.push(Event::InitVisit);
        }

        fn init_state(&mut self, s: i32, root: i32) -> bool {
            self.record(Event::InitState(s, root))
        }

        fn tree_arc(&mut self, s: i32, arc: &StdArc) -> bool {
            self.record(Event::Tree(s, arc.nextstate()))
        }

        fn back_arc(&mut self, s: i32, arc: &StdArc) -> bool {
            self.record(Event::Back(s, arc.nextstate()))
        }

        fn forward_or_cross_arc(&mut self, s: i32, arc: &StdArc) -> bool {
            self.record(Event::ForwardOrCross(s, arc.nextstate()))
        }

        fn finish_state(&mut self, s: i32, parent: Option<i32>, arc: Option<&StdArc>) {
            self.events
                .push(Event::Finish(s, parent, arc.map(|a| a.ilabel())));
        }

        fn finish_visit(&mut self) {
            self.events.push(Event::FinishVisit);
        }
    }

    /// Builds an FST from an edge list; each arc's input label is its index, so
    /// a reported tree arc can be told from its siblings.
    fn build(states: usize, edges: &[(i32, i32)]) -> VectorFst<StdArc> {
        let mut fst = VectorFst::new();
        for _ in 0..states {
            fst.add_state();
        }
        if states > 0 {
            fst.set_start(0);
        }
        for (i, &(from, to)) in edges.iter().enumerate() {
            fst.add_arc(from, StdArc::new(i as i32, 1, TropicalWeight::one(), to));
        }
        fst
    }

    fn visit(fst: &VectorFst<StdArc>, mut visitor: Recorder) -> Recorder {
        dfs_visit_any(fst, &mut visitor);
        visitor
    }

    /// 0 → 1 → 2, with 2 → 0 closing a cycle, 0 → 2 reaching a finished state,
    /// and a second arc 1 → 2 that is a forward arc by the time it is seen.
    fn diamond() -> VectorFst<StdArc> {
        build(3, &[(0, 1), (0, 2), (1, 2), (2, 0)])
    }

    #[test]
    fn arcs_are_classified_by_the_colour_of_where_they_lead() {
        let events = visit(&diamond(), Recorder::default()).events;
        assert_eq!(
            events,
            vec![
                Event::InitVisit,
                Event::InitState(0, 0),
                // 0 -> 1 is the first arc out of 0, and 1 is undiscovered.
                Event::Tree(0, 1),
                Event::InitState(1, 0),
                Event::Tree(1, 2),
                Event::InitState(2, 0),
                // 2 -> 0, and 0 is still on the stack.
                Event::Back(2, 0),
                Event::Finish(2, Some(1), Some(2)),
                Event::Finish(1, Some(0), Some(0)),
                // 0 -> 2, reached after 2 was finished.
                Event::ForwardOrCross(0, 2),
                Event::Finish(0, None, None),
                Event::FinishVisit,
            ]
        );
    }

    /// Aborting is an unwind, not a jump: every state still on the stack is
    /// finished, deepest first. A visitor that tallies results in `finish_state`
    /// would otherwise be left with a partial tally and no way to know.
    #[test]
    fn aborting_still_finishes_every_state_on_the_stack() {
        // 0 → 1 → 2 → 3, and out of 2 also a back arc to 0 and a second,
        // forward arc to 3, so every kind of refusal has a place to happen.
        let fst = build(4, &[(0, 1), (1, 2), (2, 3), (2, 0), (2, 3)]);

        for (stop, expected) in [
            // Refused before 3 is ever entered.
            (Event::Tree(2, 3), vec![2, 1, 0]),
            // Entered, refused, and finished on the way back out.
            (Event::InitState(3, 0), vec![3, 2, 1, 0]),
            // Both of these come after 3 has finished on its own.
            (Event::Back(2, 0), vec![3, 2, 1, 0]),
            (Event::ForwardOrCross(2, 3), vec![3, 2, 1, 0]),
        ] {
            let recorder = visit(&fst, Recorder::stopping_at(stop.clone()));
            assert!(
                recorder.events.contains(&stop),
                "the search never reached {stop:?}"
            );

            let finished = recorder.finished();
            let states: Vec<i32> = finished.iter().map(|&(s, _)| s).collect();
            assert_eq!(states, expected, "stopping at {stop:?}");
            assert_eq!(
                finished.last().map(|&(_, parent)| parent),
                Some(None),
                "the root is finished with no parent, stopping at {stop:?}"
            );
            assert_eq!(
                recorder.events.last(),
                Some(&Event::FinishVisit),
                "stopping at {stop:?}"
            );
            // Nothing is examined after the refusal: the unwind only finishes.
            let after: Vec<&Event> = recorder
                .events
                .iter()
                .skip_while(|e| **e != stop)
                .skip(1)
                .filter(|e| !matches!(e, Event::Finish(..) | Event::FinishVisit))
                .collect();
            assert!(after.is_empty(), "kept searching after {stop:?}: {after:?}");
        }
    }

    #[test]
    fn refusing_the_root_finishes_it_and_stops() {
        let recorder = visit(&diamond(), Recorder::stopping_at(Event::InitState(0, 0)));
        assert_eq!(
            recorder.events,
            vec![
                Event::InitVisit,
                Event::InitState(0, 0),
                Event::Finish(0, None, None),
                Event::FinishVisit,
            ]
        );
    }

    /// States unreachable from the start are roots of their own trees, and the
    /// sweep for them begins at zero because the start was taken out of order.
    #[test]
    fn unreachable_states_become_further_roots() {
        // 2 → 3 is a component of its own; 0 → 1 is reachable.
        let mut fst = build(4, &[(0, 1), (2, 3)]);
        fst.set_start(1);
        let recorder = visit(&fst, Recorder::default());

        let roots: Vec<i32> = recorder
            .events
            .iter()
            .filter_map(|e| match e {
                Event::InitState(s, root) if s == root => Some(*s),
                _ => None,
            })
            .collect();
        assert_eq!(roots, vec![1, 0, 2]);
        assert_eq!(recorder.finished().len(), 4, "every state is visited");
    }

    #[test]
    fn access_only_stops_after_the_first_tree() {
        let mut fst = build(4, &[(0, 1), (2, 3)]);
        fst.set_start(0);
        let mut recorder = Recorder::default();
        dfs_visit(&fst, &mut recorder, AnyArcFilter, true);

        let states: Vec<i32> = recorder.finished().iter().map(|&(s, _)| s).collect();
        assert_eq!(states, vec![1, 0], "2 and 3 are not reachable from 0");
    }

    /// A filtered arc is not classified at all: the visitor never hears about
    /// it, and the state it leads to stays undiscovered.
    #[test]
    fn filtered_arcs_are_not_followed_or_reported() {
        let mut fst = VectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        // Only the epsilon-input arc survives the filter.
        fst.add_arc(0, StdArc::new(0, 1, TropicalWeight::one(), 1));
        fst.add_arc(0, StdArc::new(7, 1, TropicalWeight::one(), 2));

        let mut recorder = Recorder::default();
        dfs_visit(&fst, &mut recorder, InputEpsilonArcFilter, true);

        assert!(
            !recorder
                .events
                .iter()
                .any(|e| matches!(e, Event::Tree(_, 2))),
            "the filtered arc was followed"
        );
        let states: Vec<i32> = recorder.finished().iter().map(|&(s, _)| s).collect();
        assert_eq!(states, vec![1, 0]);
    }

    #[test]
    fn an_fst_with_no_start_state_is_visited_trivially() {
        let fst: VectorFst<StdArc> = VectorFst::new();
        let recorder = visit(&fst, Recorder::default());
        assert_eq!(recorder.events, vec![Event::InitVisit, Event::FinishVisit]);
    }

    /// Every state is reported finished exactly once, and always after every
    /// state below it in its tree.
    #[test]
    fn each_state_is_finished_once_and_after_its_descendants() {
        let fst = build(
            8,
            &[
                (0, 1),
                (0, 2),
                (1, 3),
                (1, 4),
                (2, 5),
                (3, 6),
                (4, 6),
                (5, 7),
                (6, 0),
                (7, 2),
            ],
        );
        let recorder = visit(&fst, Recorder::default());
        let finished = recorder.finished();

        let mut seen: Vec<i32> = finished.iter().map(|&(s, _)| s).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..8).collect::<Vec<_>>());

        // A parent is finished after the child that reported it as parent.
        let order: Vec<i32> = finished.iter().map(|&(s, _)| s).collect();
        for &(state, parent) in &finished {
            if let Some(parent) = parent {
                let child_at = order.iter().position(|&s| s == state).unwrap();
                let parent_at = order.iter().position(|&s| s == parent).unwrap();
                assert!(child_at < parent_at, "{state} finished after {parent}");
            }
        }
    }
}
