//! Asking, before following an arc, whether anything can still be matched.
//!
//! Port of OpenFst's `label-reachable.h`. Composition wastes most of its work
//! on states that turn out to lead nowhere; a lookahead matcher avoids that by
//! asking of one FST, at each of its states, "is there any path from here that
//! starts with this label?" Answering that per state and label would take a
//! table the size of states times labels, which is why the answer is kept as a
//! *set of intervals* instead.
//!
//! The trick is a relabelling. Every arc carrying label `l` is redirected to a
//! state that stands for `l` alone, and [`StateReachable`] then numbers those
//! states so that what any state reaches is a union of few ranges of numbers.
//! Asking whether a label is reachable is then asking whether its number falls
//! in one of them.

use hashbrown::HashMap;

use crate::algorithms::accumulator::{DefaultAccumulator, WeightAccumulator};
use crate::algorithms::state_reachable::StateReachable;
use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::data_structures::interval_set::IntervalSet;
use crate::error::OpenFstError;
use crate::fst::{ExpandedFst, Fst, MutableFst};
use crate::fsts::vector_fst::VectorFst;
use crate::properties::{K_FST_PROPERTIES, K_I_LABEL_SORTED, K_O_LABEL_SORTED};
use crate::weight::Weight;

/// The number a label was given. The interval sets are over these.
///
/// The same numbering [`StateReachable`] hands out, so the two agree by
/// construction rather than by conversion.
pub type Index = crate::algorithms::state_reachable::Index;

/// The label standing for "a final state", which has no label of its own.
///
/// Kept apart from the real labels so that "can this state still finish?" is
/// the same question as "can it still reach this label?".
const FINAL_LABEL: Index = -1;

/// What labels each state can still reach, and the numbering that says so.
///
/// Shared between the copies of a [`LabelReachable`], since it is what the
/// construction costs and it never changes afterwards.
#[derive(Debug, Clone)]
pub struct LabelReachableData {
    /// Whether the input or the output side was indexed.
    reach_input: bool,
    /// The number given to each label.
    label2index: HashMap<i64, Index>,
    /// The number given to "final".
    final_index: Index,
    /// The numbers each state can reach.
    interval_sets: Vec<IntervalSet<Index>>,
}

impl LabelReachableData {
    /// Whether the input side was indexed.
    pub fn reach_input(&self) -> bool {
        self.reach_input
    }

    /// The number a label was given, or `None` if it was never seen.
    pub fn index_of<L: ArcLabel>(&self, label: L) -> Option<Index> {
        self.label2index.get(&label.to_i64()?).copied()
    }

    /// The number standing for "final".
    pub fn final_index(&self) -> Index {
        self.final_index
    }

    /// What the given state can reach.
    pub fn interval_set(&self, state: usize) -> Option<&IntervalSet<Index>> {
        self.interval_sets.get(state)
    }

    /// Rebuilds the index from what a file held.
    pub fn from_parts(
        reach_input: bool,
        label2index: HashMap<i64, Index>,
        final_index: Index,
        interval_sets: Vec<IntervalSet<Index>>,
    ) -> Self {
        Self {
            reach_input,
            label2index,
            final_index,
            interval_sets,
        }
    }

    /// Every label and the number it was given.
    pub fn label_indices(&self) -> impl Iterator<Item = (i64, Index)> + '_ {
        self.label2index
            .iter()
            .map(|(label, index)| (*label, *index))
    }

    /// How many labels were numbered.
    pub fn num_labels(&self) -> usize {
        self.label2index.len()
    }

    /// What every state can reach, in state order.
    pub fn interval_sets(&self) -> &[IntervalSet<Index>] {
        &self.interval_sets
    }

    /// How many states were indexed.
    pub fn len(&self) -> usize {
        self.interval_sets.len()
    }

    /// Whether nothing was indexed.
    pub fn is_empty(&self) -> bool {
        self.interval_sets.is_empty()
    }
}

/// Answers "can this state still reach that label?" for one FST.
pub struct LabelReachable<A: Arc, Acc = DefaultAccumulator> {
    data: std::sync::Arc<LabelReachableData>,
    /// The state being asked about.
    state: Option<usize>,
    /// Sums the weights of the arcs a lookahead matched.
    accumulator: Acc,
    /// Which side of the *other* FST's arcs to read in
    /// [`reach_range`](Self::reach_range).
    reach_fst_input: bool,
    /// Where the last matched run began and ended.
    reach_begin: Option<usize>,
    reach_end: Option<usize>,
    /// What that run weighed.
    reach_weight: A::Weight,
    _marker: std::marker::PhantomData<A>,
}

impl<A: Arc, Acc: Clone> Clone for LabelReachable<A, Acc> {
    fn clone(&self) -> Self {
        Self {
            data: std::sync::Arc::clone(&self.data),
            state: self.state,
            accumulator: self.accumulator.clone(),
            reach_fst_input: self.reach_fst_input,
            reach_begin: self.reach_begin,
            reach_end: self.reach_end,
            reach_weight: self.reach_weight.clone(),
            _marker: std::marker::PhantomData,
        }
    }
}

/// Builds the FST whose reachability is the question being asked.
///
/// Every arc carrying a label is redirected to a state standing for that label,
/// every final weight to a state standing for "final", and a superinitial state
/// is added over everything nothing else reaches. What a state of the original
/// can reach is then exactly which of those label states it can reach.
fn transform<A, F>(fst: &F, reach_input: bool) -> (VectorFst<A>, HashMap<i64, usize>)
where
    A: Arc,
    F: Fst<A> + ExpandedFst<A>,
{
    let mut out: VectorFst<A> = VectorFst::new();
    let nstates = fst.num_states();
    out.add_states(nstates);
    let mut label2state: HashMap<i64, usize> = HashMap::new();
    let mut indegree: Vec<usize> = vec![0; nstates];
    let zero = A::Weight::zero();
    let epsilon = A::Label::epsilon();

    let state_for = |label: i64,
                     out: &mut VectorFst<A>,
                     label2state: &mut HashMap<i64, usize>,
                     indegree: &mut Vec<usize>| {
        *label2state.entry(label).or_insert_with(|| {
            let state = out.add_state().as_usize();
            indegree.push(0);
            state
        })
    };

    for state in fst.states() {
        for arc in fst.arcs(state) {
            let label = if reach_input {
                arc.ilabel()
            } else {
                arc.olabel()
            };
            let nextstate = if label != epsilon {
                let key = label.to_i64().unwrap_or(FINAL_LABEL);
                state_for(key, &mut out, &mut label2state, &mut indegree)
            } else {
                arc.nextstate().as_usize()
            };
            indegree[nextstate] += 1;
            out.add_arc(
                state,
                A::new(
                    arc.ilabel(),
                    arc.olabel(),
                    arc.weight().clone(),
                    A::StateId::from_usize(nextstate),
                ),
            );
        }
        // Being final is treated as reaching one more label, so that "can this
        // state still finish?" is the same question as any other.
        let final_weight = fst.final_weight(state);
        if final_weight != zero {
            let nextstate = state_for(FINAL_LABEL, &mut out, &mut label2state, &mut indegree);
            indegree[nextstate] += 1;
            out.add_arc(
                state,
                A::new(
                    A::Label::no_label(),
                    A::Label::no_label(),
                    final_weight,
                    A::StateId::from_usize(nextstate),
                ),
            );
        }
    }

    // The label states are where a walk ends.
    for &state in label2state.values() {
        out.set_final(A::StateId::from_usize(state), A::Weight::one());
    }
    // Everything nothing else reaches has to be reachable from somewhere, or
    // the numbering would not see it.
    let start = out.add_state();
    out.set_start(start);
    for state in 0..start.as_usize() {
        if indegree.get(state).copied().unwrap_or(0) == 0 {
            out.add_arc(
                start,
                A::new(
                    epsilon,
                    epsilon,
                    A::Weight::one(),
                    A::StateId::from_usize(state),
                ),
            );
        }
    }
    out.properties(K_FST_PROPERTIES, true);
    (out, label2state)
}

impl<A: Arc> LabelReachable<A, DefaultAccumulator> {
    /// Indexes `fst`, on its input side when `reach_input`.
    pub fn new<F>(fst: &F, reach_input: bool) -> Result<Self, OpenFstError>
    where
        F: Fst<A> + ExpandedFst<A>,
    {
        Self::with_accumulator(fst, reach_input, DefaultAccumulator)
    }
}

impl<A: Arc, Acc> LabelReachable<A, Acc>
where
    Acc: WeightAccumulator<A>,
{
    /// As [`new`](LabelReachable::new), summing matched weights with
    /// `accumulator`.
    pub fn with_accumulator<F>(
        fst: &F,
        reach_input: bool,
        accumulator: Acc,
    ) -> Result<Self, OpenFstError>
    where
        F: Fst<A> + ExpandedFst<A>,
    {
        let nstates = fst.num_states();
        let (transformed, label2state) = transform(fst, reach_input);
        let reachable = StateReachable::new(&transformed)?;

        let state2index = reachable.state2index();
        let mut label2index = HashMap::with_capacity(label2state.len());
        let mut final_index = FINAL_LABEL;
        for (label, state) in label2state {
            let index = state2index[state];
            if label == FINAL_LABEL {
                final_index = index;
            }
            label2index.insert(label, index);
        }

        // Only the states of the original FST are ever asked about; the label
        // states and the superinitial one were scaffolding.
        let mut interval_sets = reachable.interval_sets().to_vec();
        interval_sets.truncate(nstates);
        interval_sets.resize_with(nstates, IntervalSet::new);

        Ok(Self {
            data: std::sync::Arc::new(LabelReachableData {
                reach_input,
                label2index,
                final_index,
                interval_sets,
            }),
            state: None,
            accumulator,
            reach_fst_input: false,
            reach_begin: None,
            reach_end: None,
            reach_weight: A::Weight::zero(),
            _marker: std::marker::PhantomData,
        })
    }

    /// Reuses an index already built.
    pub fn from_data(data: std::sync::Arc<LabelReachableData>, accumulator: Acc) -> Self {
        Self {
            data,
            state: None,
            accumulator,
            reach_fst_input: false,
            reach_begin: None,
            reach_end: None,
            reach_weight: A::Weight::zero(),
            _marker: std::marker::PhantomData,
        }
    }

    /// The index, which copies share.
    pub fn data(&self) -> &std::sync::Arc<LabelReachableData> {
        &self.data
    }

    /// Says which state the questions are about.
    pub fn set_state(&mut self, state: A::StateId) {
        self.state = Some(state.as_usize());
    }

    /// The number a label was given, which is the argument
    /// [`reach`](Self::reach) takes.
    ///
    /// A label the indexed FST never carries has no number, and nothing can
    /// reach it.
    pub fn relabel(&self, label: A::Label) -> Option<Index> {
        self.data.index_of(label)
    }

    /// Whether the label `index` stands for is one the current state can read
    /// *next*, after any epsilons.
    ///
    /// Not "does that label turn up somewhere ahead": the transform sends every
    /// labelled arc to the state standing for its label, so a walk stops at the
    /// first label it reads. That is the question a look-ahead matcher asks:
    /// whether the arc about to be taken can meet anything.
    ///
    /// SICADA-DIVERGE: upstream's `Reach` takes an already-relabelled `Label`
    /// of the arc's own type, so passing a raw label instead of a relabelled
    /// one type-checks and quietly answers about the wrong thing. The index is
    /// its own type here, and [`relabel`](Self::relabel) is the only way to
    /// get one.
    pub fn reach(&self, index: Index) -> bool {
        if index == 0 {
            return false;
        }
        self.data
            .interval_set(self.state.unwrap_or(usize::MAX))
            .is_some_and(|set| set.member(index))
    }

    /// Whether the current state can finish here, after any epsilons.
    pub fn reach_final(&self) -> bool {
        self.data
            .interval_set(self.state.unwrap_or(usize::MAX))
            .is_some_and(|set| set.member(self.data.final_index))
    }

    /// Prepares to look ahead over the arcs of `fst`, reading its input side
    /// when `reach_input`.
    pub fn reach_init<F>(&mut self, fst: &F, reach_input: bool) -> Result<(), OpenFstError>
    where
        F: Fst<A>,
    {
        self.reach_fst_input = reach_input;
        let sorted = if reach_input {
            K_I_LABEL_SORTED
        } else {
            K_O_LABEL_SORTED
        };
        if fst.properties(sorted, true) & sorted == 0 {
            return Err(OpenFstError::InvalidOperation(
                "LabelReachable: the FST looked ahead over is not sorted on the side being read"
                    .into(),
            ));
        }
        self.accumulator.init(fst)?;
        Ok(())
    }

    /// Whether any of the arcs `begin..end` carries a label the current state
    /// can reach.
    ///
    /// The arcs must be sorted on the side [`reach_init`](Self::reach_init) was
    /// told to read. When `compute_weight`, the matched arcs' weights are
    /// summed and left in [`reach_weight`](Self::reach_weight).
    pub fn reach_range<I>(
        &mut self,
        arcs: I,
        begin: usize,
        end: usize,
        compute_weight: bool,
    ) -> bool
    where
        I: Iterator<Item = A> + Clone,
    {
        self.reach_begin = None;
        self.reach_end = None;
        self.reach_weight = A::Weight::zero();
        if end <= begin {
            return false;
        }
        if let Some(state) = self.state {
            self.accumulator.set_state(A::StateId::from_usize(state));
        }

        // Checking each arc against the intervals, which is the cheaper way
        // round when the range is short. Upstream switches to walking the
        // intervals instead once the range is more than twice their number;
        // that path needs a lower-bound search over the arcs, namely
        // `LabelLowerBound`, and is left for the lookahead matcher to bring.
        let mut matched: Option<(usize, usize)> = None;
        for (offset, arc) in arcs.clone().skip(begin).take(end - begin).enumerate() {
            let label = if self.reach_fst_input {
                arc.ilabel()
            } else {
                arc.olabel()
            };
            let Some(index) = self.relabel(label) else {
                continue;
            };
            if !self.reach(index) {
                continue;
            }
            let position = begin + offset;
            matched = Some(match matched {
                None => (position, position + 1),
                Some((first, _)) => (first, position + 1),
            });
        }

        let Some((first, last)) = matched else {
            return false;
        };
        self.reach_begin = Some(first);
        self.reach_end = Some(last);
        if compute_weight {
            // Only the arcs that matched count, so the run is summed one arc at
            // a time rather than as a range: a range sum would include the arcs
            // in between that did not match.
            let mut weight = A::Weight::zero();
            for arc in arcs.skip(first).take(last - first) {
                let label = if self.reach_fst_input {
                    arc.ilabel()
                } else {
                    arc.olabel()
                };
                if self.relabel(label).is_some_and(|index| self.reach(index)) {
                    weight = self.accumulator.sum(&weight, arc.weight());
                }
            }
            self.reach_weight = weight;
        }
        true
    }

    /// Where the last matched run began.
    pub fn reach_begin(&self) -> Option<usize> {
        self.reach_begin
    }

    /// Where it ended.
    pub fn reach_end(&self) -> Option<usize> {
        self.reach_end
    }

    /// What it weighed, when the last [`reach_range`](Self::reach_range) was
    /// asked for one.
    pub fn reach_weight(&self) -> &A::Weight {
        &self.reach_weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::arcsort::{ILabelCompare, arc_sort};
    use crate::algorithms::test_support::{Rng, random_acyclic_fst};
    use crate::arc::StdArc;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::weights::float_weight::TropicalWeight;

    /// 0 -1-> 1 -2-> 2 (final), and 0 -3-> 3 (final).
    fn sample() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..4 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight::one(), 2));
        fst.add_arc(0, StdArc::new(3, 3, TropicalWeight::one(), 3));
        fst.set_final(2, TropicalWeight::one());
        fst.set_final(3, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    /// The labels a state can read next, following epsilons and stopping at the
    /// first labelled arc. The interval sets have to agree with this.
    fn reachable_by_walking(fst: &StdVectorFst, from: i32) -> (Vec<i32>, bool) {
        let mut seen = vec![false; fst.num_states()];
        let mut stack = vec![from];
        let mut labels = Vec::new();
        let mut can_finish = false;
        while let Some(state) = stack.pop() {
            if seen[state as usize] {
                continue;
            }
            seen[state as usize] = true;
            if fst.final_weight(state) != TropicalWeight::zero() {
                can_finish = true;
            }
            for arc in fst.arcs(state) {
                if arc.ilabel() == 0 {
                    stack.push(arc.nextstate());
                } else {
                    labels.push(arc.ilabel());
                }
            }
        }
        labels.sort_unstable();
        labels.dedup();
        (labels, can_finish)
    }

    #[test]
    fn a_state_reaches_the_labels_that_can_be_read_next() {
        let fst = sample();
        let mut reachable = LabelReachable::<StdArc>::new(&fst, true).unwrap();

        for state in fst.states() {
            reachable.set_state(state);
            let (labels, can_finish) = reachable_by_walking(&fst, state);
            for label in 1..=4 {
                let want = labels.contains(&label);
                let got = reachable
                    .relabel(label)
                    .is_some_and(|index| reachable.reach(index));
                assert_eq!(got, want, "state {state}, label {label}");
            }
            assert_eq!(reachable.reach_final(), can_finish, "state {state}");
        }
    }

    /// A label the FST never carries has no number, so nothing can read it.
    #[test]
    fn a_label_the_fst_never_carries_is_reachable_from_nowhere() {
        let fst = sample();
        let reachable = LabelReachable::<StdArc>::new(&fst, true).unwrap();
        assert!(reachable.relabel(99).is_none());
    }

    /// Whichever FST, the intervals say exactly what a walk says.
    #[test]
    fn the_intervals_agree_with_walking_on_any_fst() {
        let mut rng = Rng::new(0x000A_BE01_u64);
        for round in 0..100 {
            let fst = random_acyclic_fst(&mut rng, 6);
            if fst.num_states() == 0 {
                continue;
            }
            let mut reachable = LabelReachable::<StdArc>::new(&fst, true).unwrap();
            for state in fst.states() {
                reachable.set_state(state);
                let (labels, can_finish) = reachable_by_walking(&fst, state);
                for label in 1..=4 {
                    let want = labels.contains(&label);
                    let got = reachable
                        .relabel(label)
                        .is_some_and(|index| reachable.reach(index));
                    assert_eq!(got, want, "round {round}, state {state}, label {label}");
                }
                assert_eq!(
                    reachable.reach_final(),
                    can_finish,
                    "round {round}, state {state}"
                );
            }
        }
    }

    /// Looking ahead over another FST's arcs finds the run that matches.
    #[test]
    fn looking_ahead_finds_the_arcs_that_can_be_matched() {
        let index = sample();
        let mut reachable = LabelReachable::<StdArc>::new(&index, true).unwrap();

        // The other side, whose arcs are looked ahead over.
        let mut other = StdVectorFst::new();
        for _ in 0..2 {
            other.add_state();
        }
        other.set_start(0);
        for label in [1, 2, 3, 9] {
            other.add_arc(
                0,
                StdArc::new(label, label, TropicalWeight(label as f32), 1),
            );
        }
        other.set_final(1, TropicalWeight::one());
        arc_sort(&mut other, &ILabelCompare);
        other.properties(K_FST_PROPERTIES, true);

        reachable.reach_init(&other, true).unwrap();

        // From state 0 of the index, labels 1 and 3 are reachable, 2 and 9 are
        // not; so the run spans the arcs carrying 1 through 3.
        reachable.set_state(0);
        assert!(reachable.reach_range(other.arcs(0), 0, 4, true));
        assert_eq!(reachable.reach_begin(), Some(0));
        assert_eq!(reachable.reach_end(), Some(3));
        assert_eq!(
            *reachable.reach_weight(),
            TropicalWeight(1.0),
            "the lighter of the arcs carrying 1 and 3"
        );

        // From state 1, only label 2 is reachable.
        reachable.set_state(1);
        assert!(reachable.reach_range(other.arcs(0), 0, 4, false));
        assert_eq!(reachable.reach_begin(), Some(1));
        assert_eq!(reachable.reach_end(), Some(2));

        // From state 2, nothing is.
        reachable.set_state(2);
        assert!(!reachable.reach_range(other.arcs(0), 0, 4, false));
        assert_eq!(reachable.reach_begin(), None);
    }

    /// The FST looked ahead over has to be sorted, or the run is not a run.
    #[test]
    fn an_unsorted_fst_is_refused() {
        let index = sample();
        let mut reachable = LabelReachable::<StdArc>::new(&index, true).unwrap();

        let mut unsorted = StdVectorFst::new();
        for _ in 0..2 {
            unsorted.add_state();
        }
        unsorted.set_start(0);
        unsorted.add_arc(0, StdArc::new(3, 3, TropicalWeight::one(), 1));
        unsorted.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        unsorted.set_final(1, TropicalWeight::one());
        unsorted.properties(K_FST_PROPERTIES, true);

        assert!(reachable.reach_init(&unsorted, true).is_err());
    }

    /// Indexing the output side answers about output labels.
    #[test]
    fn the_output_side_can_be_indexed_instead() {
        let mut fst = StdVectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 7, TropicalWeight::one(), 1));
        fst.set_final(1, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let mut inputs = LabelReachable::<StdArc>::new(&fst, true).unwrap();
        inputs.set_state(0);
        assert!(inputs.relabel(1).is_some_and(|i| inputs.reach(i)));
        assert!(inputs.relabel(7).is_none());

        let mut outputs = LabelReachable::<StdArc>::new(&fst, false).unwrap();
        outputs.set_state(0);
        assert!(outputs.relabel(7).is_some_and(|i| outputs.reach(i)));
        assert!(outputs.relabel(1).is_none());
        assert!(!outputs.data().reach_input());
    }
}
