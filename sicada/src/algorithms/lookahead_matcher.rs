//! Matchers that can say, before an arc is followed, whether it leads anywhere.
//!
//! Port of OpenFst's `lookahead-matcher.h`. Composition builds a state for
//! every pair it can reach, and most of them turn out to lead nowhere; a
//! look-ahead matcher is asked "from this state of mine, can anything still
//! match what is over there?" before the pair is built at all.
//!
//! The flags describe how well it answers: whether it can say anything about
//! the input side, the output side, the weight of what it found, or the single
//! arc that must be taken next.

use crate::algorithms::accumulator::{DefaultAccumulator, WeightAccumulator};
use crate::algorithms::label_reachable::{LabelReachable, LabelReachableData};
use crate::arc::{Arc, ArcLabel};
use crate::error::OpenFstError;
use crate::fst::{ExpandedFst, Fst, MatchType};
use crate::matcher::Matcher;
use crate::weight::Weight;

/// The matcher can look ahead on the input side.
pub const INPUT_LOOKAHEAD_MATCHER: u32 = 0x0000_0010;
/// The matcher can look ahead on the output side.
pub const OUTPUT_LOOKAHEAD_MATCHER: u32 = 0x0000_0020;
/// It reports what the arcs it found weigh together.
pub const LOOKAHEAD_WEIGHT: u32 = 0x0000_0040;
/// It reports the one arc that has to be taken, when there is only one.
pub const LOOKAHEAD_PREFIX: u32 = 0x0000_0080;
/// It looks ahead over the arcs carrying a label.
pub const LOOKAHEAD_NON_EPSILONS: u32 = 0x0000_0100;
/// It looks ahead over the epsilon arcs.
pub const LOOKAHEAD_EPSILONS: u32 = 0x0000_0200;
/// A prefix arc is only reported for a non-epsilon match.
pub const LOOKAHEAD_NON_EPSILON_PREFIX: u32 = 0x0000_0400;
/// The relabelling the index was built with is kept, so it can be applied to
/// the other side.
pub const LOOKAHEAD_KEEP_RELABEL_DATA: u32 = 0x0000_0800;
/// Every flag above.
pub const LOOKAHEAD_FLAGS: u32 = 0x0000_0ff0;

/// The flags a label look-ahead matcher has by default.
pub const DEFAULT_LABEL_LOOKAHEAD_FLAGS: u32 = LOOKAHEAD_EPSILONS
    | LOOKAHEAD_WEIGHT
    | LOOKAHEAD_PREFIX
    | LOOKAHEAD_NON_EPSILON_PREFIX
    | LOOKAHEAD_KEEP_RELABEL_DATA;

/// A matcher that can be asked about the future before it is walked.
///
/// SICADA-DIVERGE: upstream keeps the answer in mutable members of a base class
/// (`SetLookAheadWeight`, `SetLookAheadPrefix`) that the caller reads afterwards
/// through `LookAheadWeight()` / `LookAheadPrefix(&arc)`, so what those return
/// depends on which call was made last, and nothing in the types says so.
/// [`look_ahead`](Self::look_ahead) returns what it found.
pub trait LookAheadMatcher<'f, A: Arc>: Matcher<'f, A> {
    /// What this matcher can say; see the flag constants.
    fn lookahead_flags(&self) -> u32;

    /// Whether anything can still match, from the matcher's current state
    /// against state `state` of `fst`.
    fn look_ahead<L: Fst<A> + ExpandedFst<A>>(
        &mut self,
        fst: &L,
        state: A::StateId,
    ) -> LookAhead<A>;

    /// Whether `label` could still be matched from the current state, without
    /// looking at the other FST at all.
    fn look_ahead_label(&mut self, label: A::Label) -> bool;
}

/// What a look-ahead found.
#[derive(Debug, Clone)]
pub struct LookAhead<A: Arc> {
    /// Whether anything can still match.
    pub reachable: bool,
    /// What the arcs that matched weigh together, when the matcher reports it.
    pub weight: A::Weight,
    /// The one arc that has to be taken next, when there is exactly one and the
    /// matcher reports it.
    pub prefix: Option<A>,
}

impl<A: Arc> LookAhead<A> {
    /// Nothing can match.
    fn nothing() -> Self {
        Self {
            reachable: false,
            weight: A::Weight::one(),
            prefix: None,
        }
    }

    /// Something can match, with nothing more said about it.
    fn anything() -> Self {
        Self {
            reachable: true,
            weight: A::Weight::one(),
            prefix: None,
        }
    }
}

/// Says the future looks good without looking.
///
/// The matcher to use where composition wants to be written once against a
/// look-ahead interface but the FST has no index to answer from.
#[derive(Clone)]
pub struct TrivialLookAheadMatcher<M> {
    inner: M,
}

impl<M> TrivialLookAheadMatcher<M> {
    /// Wraps `inner`.
    pub fn new(inner: M) -> Self {
        Self { inner }
    }

    /// The matcher underneath.
    pub fn inner(&self) -> &M {
        &self.inner
    }
}

impl<'f, A, M> Matcher<'f, A> for TrivialLookAheadMatcher<M>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    type Fst = M::Fst;

    fn new(fst: &'f Self::Fst, match_type: MatchType) -> Result<Self, OpenFstError> {
        Ok(Self {
            inner: M::new(fst, match_type)?,
        })
    }

    fn match_type(&self) -> MatchType {
        self.inner.match_type()
    }

    fn set_state(&mut self, state: A::StateId) {
        self.inner.set_state(state);
    }

    fn find(&mut self, label: A::Label) -> bool {
        self.inner.find(label)
    }

    fn done(&self) -> bool {
        self.inner.done()
    }

    fn value(&self) -> A {
        self.inner.value()
    }

    fn next(&mut self) {
        self.inner.next();
    }

    fn priority(&mut self, state: A::StateId) -> isize {
        self.inner.priority(state)
    }
}

impl<'f, A, M> LookAheadMatcher<'f, A> for TrivialLookAheadMatcher<M>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    fn lookahead_flags(&self) -> u32 {
        INPUT_LOOKAHEAD_MATCHER | OUTPUT_LOOKAHEAD_MATCHER
    }

    fn look_ahead<L: Fst<A> + ExpandedFst<A>>(
        &mut self,
        _fst: &L,
        _state: A::StateId,
    ) -> LookAhead<A> {
        LookAhead::anything()
    }

    fn look_ahead_label(&mut self, _label: A::Label) -> bool {
        true
    }
}

/// Looks ahead one arc at a time, by asking the matcher about each arc leaving
/// the other side's state.
///
/// Needs nothing precomputed, and so costs the other state's out-degree per
/// question; [`LabelLookAheadMatcher`] answers from an index instead.
pub struct ArcLookAheadMatcher<'f, F, M, A: Arc> {
    inner: M,
    /// The matcher's own FST, which the look-ahead reads final weights from --
    /// upstream's holds one for the same reason.
    fst: &'f F,
    state: Option<A::StateId>,
    flags: u32,
}

impl<F, M: Clone, A: Arc> Clone for ArcLookAheadMatcher<'_, F, M, A> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            fst: self.fst,
            state: self.state,
            flags: self.flags,
        }
    }
}

impl<'f, F, M, A: Arc> ArcLookAheadMatcher<'f, F, M, A> {
    /// Wraps `inner` over `fst`, reporting `flags`.
    pub fn with_flags(fst: &'f F, inner: M, flags: u32) -> Self {
        Self {
            inner,
            fst,
            state: None,
            flags,
        }
    }
}

impl<'f, A, F, M> Matcher<'f, A> for ArcLookAheadMatcher<'f, F, M, A>
where
    A: Arc,
    F: Fst<A>,
    M: Matcher<'f, A, Fst = F>,
{
    type Fst = F;

    fn new(fst: &'f Self::Fst, match_type: MatchType) -> Result<Self, OpenFstError> {
        Ok(Self {
            inner: M::new(fst, match_type)?,
            fst,
            state: None,
            flags: LOOKAHEAD_WEIGHT
                | LOOKAHEAD_PREFIX
                | LOOKAHEAD_EPSILONS
                | LOOKAHEAD_NON_EPSILONS,
        })
    }

    fn match_type(&self) -> MatchType {
        self.inner.match_type()
    }

    fn set_state(&mut self, state: A::StateId) {
        self.state = Some(state);
        self.inner.set_state(state);
    }

    fn find(&mut self, label: A::Label) -> bool {
        self.inner.find(label)
    }

    fn done(&self) -> bool {
        self.inner.done()
    }

    fn value(&self) -> A {
        self.inner.value()
    }

    fn next(&mut self) {
        self.inner.next();
    }

    fn priority(&mut self, state: A::StateId) -> isize {
        self.inner.priority(state)
    }
}

impl<'f, A, F, M> LookAheadMatcher<'f, A> for ArcLookAheadMatcher<'f, F, M, A>
where
    A: Arc,
    F: Fst<A>,
    M: Matcher<'f, A, Fst = F>,
{
    fn lookahead_flags(&self) -> u32 {
        self.flags | INPUT_LOOKAHEAD_MATCHER | OUTPUT_LOOKAHEAD_MATCHER
    }

    fn look_ahead<L: Fst<A> + ExpandedFst<A>>(
        &mut self,
        fst: &L,
        state: A::StateId,
    ) -> LookAhead<A> {
        let flags = self.flags;
        let detailed = flags & (LOOKAHEAD_WEIGHT | LOOKAHEAD_PREFIX) != 0;
        let mut found = LookAhead::<A>::nothing();
        let zero = A::Weight::zero();
        // SICADA-DIVERGE: upstream accumulates the weight of what it found into
        // `LookAheadWeight()`, which `ClearLookAheadWeight()` has just set to
        // `One()`, so what it reports is `One ⊕ (what it found)`. Over the
        // tropical semiring that is `min(0, …)`, which is `One` for any
        // non-negative future: sound, since the filter above divides back out
        // whatever it was given, but never anything a search could prune on.
        // A sum starts from `Zero`; nothing reads it unless something was
        // found, and the two places upstream assigns rather than accumulates
        // are unaffected.
        let mut sum = A::Weight::zero();
        // How many ways forward there are: a prefix arc is only worth reporting
        // when there is exactly one.
        let mut nprefix = 0usize;

        // Both sides can stop here, which is a way forward in itself.
        let here_final = self
            .state
            .map_or_else(A::Weight::zero, |here| self.fst.final_weight(here));
        if here_final != zero && fst.final_weight(state) != zero {
            if !detailed {
                return LookAhead::anything();
            }
            nprefix += 1;
            if flags & LOOKAHEAD_WEIGHT != 0 {
                sum = sum.plus(&here_final.times(&fst.final_weight(state)));
            }
            found.reachable = true;
        }

        // An implicit match, the one the matcher offers for "no label", goes
        // forward without the other side moving.
        if self.inner.find(A::Label::no_label()) {
            if !detailed {
                return LookAhead::anything();
            }
            nprefix += 1;
            if flags & LOOKAHEAD_WEIGHT != 0 {
                while !self.inner.done() {
                    let value = self.inner.value();
                    sum = sum.plus(value.weight());
                    self.inner.next();
                }
            }
            found.reachable = true;
        }

        for arc in fst.arcs(state) {
            // The label to look for is the one that meets this matcher's side.
            let label = match self.inner.match_type() {
                MatchType::Input => arc.olabel(),
                MatchType::Output => arc.ilabel(),
                _ => return LookAhead::anything(),
            };
            if label == A::Label::epsilon() {
                if !detailed {
                    return LookAhead::anything();
                }
                if flags & LOOKAHEAD_NON_EPSILON_PREFIX == 0 {
                    nprefix += 1;
                }
                if flags & LOOKAHEAD_WEIGHT != 0 {
                    sum = sum.plus(arc.weight());
                }
                found.reachable = true;
                continue;
            }
            if !self.inner.find(label) {
                continue;
            }
            if !detailed {
                return LookAhead::anything();
            }
            while !self.inner.done() {
                let value = self.inner.value();
                nprefix += 1;
                if flags & LOOKAHEAD_WEIGHT != 0 {
                    sum = sum.plus(&arc.weight().times(value.weight()));
                }
                if flags & LOOKAHEAD_PREFIX != 0 && nprefix == 1 {
                    found.prefix = Some(arc.clone());
                }
                self.inner.next();
            }
            found.reachable = true;
        }

        if flags & LOOKAHEAD_WEIGHT != 0 && found.reachable {
            found.weight = sum;
        }
        if flags & LOOKAHEAD_PREFIX != 0 {
            if nprefix == 1 {
                // The prefix says everything; reporting the weight too would
                // count it twice.
                found.weight = A::Weight::one();
            } else {
                found.prefix = None;
            }
        }
        found
    }

    fn look_ahead_label(&mut self, label: A::Label) -> bool {
        if label == A::Label::epsilon() {
            return true;
        }
        self.inner.find(label)
    }
}

/// Looks ahead from an index of which labels each state can still reach.
///
/// One question costs a walk over the other state's arcs against a handful of
/// intervals, rather than a search per arc. That is the question
/// [`LabelReachable`] is built to answer.
pub struct LabelLookAheadMatcher<A: Arc, M, Acc = DefaultAccumulator> {
    inner: M,
    reachable: Option<LabelReachable<A, Acc>>,
    flags: u32,
    state: Option<A::StateId>,
    /// Whether the index has been told which state the questions are about.
    reach_state_set: bool,
    /// Which FST the index was last prepared for.
    prepared: bool,
}

impl<A: Arc, M: Clone, Acc: Clone> Clone for LabelLookAheadMatcher<A, M, Acc> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            reachable: self.reachable.clone(),
            flags: self.flags,
            state: self.state,
            reach_state_set: false,
            prepared: false,
        }
    }
}

impl<A: Arc, M, Acc> LabelLookAheadMatcher<A, M, Acc>
where
    Acc: WeightAccumulator<A>,
{
    /// Builds the index over `fst` and wraps `inner`.
    ///
    /// `flags` says what the matcher will report; it has to name exactly one of
    /// [`INPUT_LOOKAHEAD_MATCHER`] and [`OUTPUT_LOOKAHEAD_MATCHER`], which is
    /// the side the index is built over.
    pub fn new<F>(
        fst: &F,
        inner: M,
        match_type: MatchType,
        flags: u32,
        accumulator: Acc,
    ) -> Result<Self, OpenFstError>
    where
        F: Fst<A> + ExpandedFst<A>,
    {
        let input = flags & INPUT_LOOKAHEAD_MATCHER != 0;
        let output = flags & OUTPUT_LOOKAHEAD_MATCHER != 0;
        if input == output {
            return Err(OpenFstError::InvalidOperation(
                "LabelLookAheadMatcher: the flags have to name exactly one of the input and \
                 output sides"
                    .into(),
            ));
        }
        // The index is over the side the matcher matches on: a matcher looking
        // up input labels asks what input labels are still reachable.
        let reach_input = match_type == MatchType::Input;
        let reachable = if (reach_input && input) || (!reach_input && output) {
            Some(LabelReachable::with_accumulator(
                fst,
                reach_input,
                accumulator,
            )?)
        } else {
            None
        };
        Ok(Self {
            inner,
            reachable,
            flags,
            state: None,
            reach_state_set: false,
            prepared: false,
        })
    }

    /// Reuses an index already built.
    pub fn from_data(
        data: std::sync::Arc<LabelReachableData>,
        inner: M,
        flags: u32,
        accumulator: Acc,
    ) -> Self {
        Self {
            inner,
            reachable: Some(LabelReachable::from_data(data, accumulator)),
            flags,
            state: None,
            reach_state_set: false,
            prepared: false,
        }
    }

    /// The index, which copies of this matcher can share.
    pub fn data(&self) -> Option<&std::sync::Arc<LabelReachableData>> {
        self.reachable.as_ref().map(|r| r.data())
    }
}

impl<'f, A, M, Acc> Matcher<'f, A> for LabelLookAheadMatcher<A, M, Acc>
where
    A: Arc,
    M: Matcher<'f, A>,
    Acc: WeightAccumulator<A> + Clone,
    LabelReachable<A, Acc>: Clone,
{
    type Fst = M::Fst;

    fn new(fst: &'f Self::Fst, match_type: MatchType) -> Result<Self, OpenFstError> {
        // Without an index this is a plain matcher that answers "yes" to every
        // look-ahead; `LabelLookAheadMatcher::new` builds the index.
        Ok(Self {
            inner: M::new(fst, match_type)?,
            reachable: None,
            flags: DEFAULT_LABEL_LOOKAHEAD_FLAGS,
            state: None,
            reach_state_set: false,
            prepared: false,
        })
    }

    fn match_type(&self) -> MatchType {
        self.inner.match_type()
    }

    fn set_state(&mut self, state: A::StateId) {
        self.state = Some(state);
        self.reach_state_set = false;
        self.inner.set_state(state);
    }

    fn find(&mut self, label: A::Label) -> bool {
        self.inner.find(label)
    }

    fn done(&self) -> bool {
        self.inner.done()
    }

    fn value(&self) -> A {
        self.inner.value()
    }

    fn next(&mut self) {
        self.inner.next();
    }

    fn priority(&mut self, state: A::StateId) -> isize {
        self.inner.priority(state)
    }
}

impl<'f, A, M, Acc> LookAheadMatcher<'f, A> for LabelLookAheadMatcher<A, M, Acc>
where
    A: Arc,
    M: Matcher<'f, A>,
    Acc: WeightAccumulator<A> + Clone,
    LabelReachable<A, Acc>: Clone,
{
    fn lookahead_flags(&self) -> u32 {
        self.flags
    }

    fn look_ahead<L: Fst<A> + ExpandedFst<A>>(
        &mut self,
        fst: &L,
        state: A::StateId,
    ) -> LookAhead<A> {
        let Some(reachable) = self.reachable.as_mut() else {
            return LookAhead::anything();
        };
        let Some(here) = self.state else {
            return LookAhead::anything();
        };
        if !self.prepared {
            // The other side has to be sorted the way the index reads it.
            let reach_input = self.inner.match_type() == MatchType::Output;
            if reachable.reach_init(fst, reach_input).is_err() {
                return LookAhead::anything();
            }
            self.prepared = true;
        }
        reachable.set_state(here);
        self.reach_state_set = true;

        let mut compute_weight = self.flags & LOOKAHEAD_WEIGHT != 0;
        let compute_prefix = self.flags & LOOKAHEAD_PREFIX != 0;
        let narcs = fst.num_arcs(state);
        let reach_arc = reachable.reach_range(fst.arcs(state), 0, narcs, compute_weight);

        let lfinal = fst.final_weight(state);
        let reach_final = lfinal != A::Weight::zero() && reachable.reach_final();

        let mut found = LookAhead::<A>::nothing();
        found.reachable = reach_arc || reach_final;
        if reach_arc {
            let begin = reachable.reach_begin().unwrap_or(0);
            let end = reachable.reach_end().unwrap_or(0);
            if compute_prefix && end - begin == 1 && !reach_final {
                // Exactly one way forward: the arc says everything, so the
                // weight would be counting it twice.
                found.prefix = fst.arcs(state).nth(begin);
                compute_weight = false;
            } else if compute_weight {
                found.weight = reachable.reach_weight().clone();
            }
        }
        if reach_final && compute_weight {
            found.weight = if reach_arc {
                found.weight.plus(&lfinal)
            } else {
                lfinal
            };
        }
        found
    }

    fn look_ahead_label(&mut self, label: A::Label) -> bool {
        if label == A::Label::epsilon() {
            return true;
        }
        let Some(here) = self.state else {
            return true;
        };
        let Some(reachable) = self.reachable.as_mut() else {
            return true;
        };
        if !self.reach_state_set {
            reachable.set_state(here);
            self.reach_state_set = true;
        }
        match reachable.relabel(label) {
            Some(index) => reachable.reach(index),
            None => false,
        }
    }
}

/// A multi-epsilon matcher looks ahead exactly as the matcher it wraps does:
/// the extra labels it treats as epsilon change what `Find` returns, not what
/// is reachable from the state.
impl<'f, A, M> LookAheadMatcher<'f, A> for crate::matcher::MultiEpsMatcher<'f, M, A>
where
    A: Arc,
    M: LookAheadMatcher<'f, A>,
{
    fn lookahead_flags(&self) -> u32 {
        self.matcher().lookahead_flags()
    }

    fn look_ahead<L: Fst<A> + ExpandedFst<A>>(
        &mut self,
        fst: &L,
        state: A::StateId,
    ) -> LookAhead<A> {
        self.matcher_mut().look_ahead(fst, state)
    }

    fn look_ahead_label(&mut self, label: A::Label) -> bool {
        self.matcher_mut().look_ahead_label(label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::arcsort::{ILabelCompare, arc_sort};
    use crate::arc::StdArc;
    use crate::fst::MutableFst;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::matcher::SortedMatcher;
    use crate::properties::K_FST_PROPERTIES;
    use crate::weights::float_weight::TropicalWeight;

    /// The side being looked ahead from: 0 -1-> 1 -2-> 2 (final).
    fn indexed() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight::one(), 2));
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    /// The other side, whose states are asked about.
    fn other(labels: &[i32]) -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        for label in labels {
            fst.add_arc(
                0,
                StdArc::new(*label, *label, TropicalWeight(*label as f32), 1),
            );
        }
        fst.set_final(1, TropicalWeight::one());
        arc_sort(&mut fst, &ILabelCompare);
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    /// The trivial matcher says yes without looking, which is its purpose.
    #[test]
    fn the_trivial_matcher_says_yes_to_everything() {
        let fst = indexed();
        let inner = SortedMatcher::new(&fst, MatchType::Output).unwrap();
        let mut matcher = TrivialLookAheadMatcher::new(inner);
        matcher.set_state(2);
        let elsewhere = other(&[9]);
        assert!(matcher.look_ahead(&elsewhere, 0).reachable);
        assert!(matcher.look_ahead_label(99));
    }

    /// The label matcher answers from the index: which label can be read
    /// *next*, after any epsilons. That is the question composition asks,
    /// namely whether the arc about to be taken can meet anything, rather than
    /// whether the label turns up somewhere further on.
    #[test]
    fn the_label_matcher_answers_from_the_index() {
        let fst = indexed();
        let inner = SortedMatcher::new(&fst, MatchType::Input).unwrap();
        let mut matcher = LabelLookAheadMatcher::new(
            &fst,
            inner,
            MatchType::Input,
            DEFAULT_LABEL_LOOKAHEAD_FLAGS | INPUT_LOOKAHEAD_MATCHER,
            DefaultAccumulator,
        )
        .unwrap();

        matcher.set_state(0);
        assert!(matcher.look_ahead_label(1));
        assert!(
            !matcher.look_ahead_label(2),
            "2 comes after 1, so it is not what is read next"
        );
        assert!(!matcher.look_ahead_label(9));

        matcher.set_state(1);
        assert!(!matcher.look_ahead_label(1), "1 is behind, not next");
        assert!(matcher.look_ahead_label(2));

        matcher.set_state(2);
        assert!(!matcher.look_ahead_label(1));
        assert!(!matcher.look_ahead_label(2));
    }

    /// Looking ahead against another FST's state finds whether anything there
    /// can be matched.
    #[test]
    fn looking_ahead_against_another_state_finds_what_can_match() {
        let fst = indexed();
        let inner = SortedMatcher::new(&fst, MatchType::Input).unwrap();
        let mut matcher = LabelLookAheadMatcher::new(
            &fst,
            inner,
            MatchType::Input,
            DEFAULT_LABEL_LOOKAHEAD_FLAGS | INPUT_LOOKAHEAD_MATCHER,
            DefaultAccumulator,
        )
        .unwrap();

        // From state 0, label 1 is what comes next; label 9 is not.
        let reachable = other(&[1, 9]);
        matcher.set_state(0);
        assert!(matcher.look_ahead(&reachable, 0).reachable);

        let unreachable = other(&[9]);
        matcher.set_state(0);
        assert!(!matcher.look_ahead(&unreachable, 0).reachable);

        // From state 2 nothing is ahead at all.
        matcher.set_state(2);
        assert!(!matcher.look_ahead(&reachable, 0).reachable);
    }

    /// Exactly one way forward is reported as the arc that has to be taken.
    #[test]
    fn one_way_forward_is_reported_as_a_prefix() {
        let fst = indexed();
        let inner = SortedMatcher::new(&fst, MatchType::Input).unwrap();
        let mut matcher = LabelLookAheadMatcher::new(
            &fst,
            inner,
            MatchType::Input,
            DEFAULT_LABEL_LOOKAHEAD_FLAGS | INPUT_LOOKAHEAD_MATCHER,
            DefaultAccumulator,
        )
        .unwrap();

        // Only label 1 of these can be matched from state 0.
        let one_way = other(&[1, 9]);
        matcher.set_state(0);
        let found = matcher.look_ahead(&one_way, 0);
        assert!(found.reachable);
        assert_eq!(
            found.prefix.map(|arc| arc.ilabel()),
            Some(1),
            "the one arc that can be taken"
        );
    }

    /// The arc matcher needs no index, and finds the same answers.
    #[test]
    fn the_arc_matcher_agrees_without_an_index() {
        let fst = indexed();
        let inner = SortedMatcher::new(&fst, MatchType::Input).unwrap();
        let mut matcher = ArcLookAheadMatcher::with_flags(
            &fst,
            inner,
            LOOKAHEAD_WEIGHT | LOOKAHEAD_PREFIX | LOOKAHEAD_NON_EPSILONS,
        );

        // From state 0 the matcher can take label 1.
        matcher.set_state(0);
        assert!(matcher.look_ahead(&other(&[1, 9]), 0).reachable);

        matcher.set_state(0);
        assert!(
            !matcher.look_ahead(&other(&[9]), 0).reachable,
            "nothing at state 0 carries 9"
        );

        // And it agrees with the index that 2 is not what comes next from
        // state 0.
        matcher.set_state(0);
        assert!(!matcher.look_ahead(&other(&[2]), 0).reachable);
    }

    /// What was found weighs what the arcs that matched weigh: a sum starting
    /// from `Zero`, not from `One`.
    ///
    /// See the `SICADA-DIVERGE` note in `look_ahead`: accumulating from `One`
    /// makes the answer `min(0, …)` over the tropical semiring, which is `One`
    /// for every non-negative future and so says nothing.
    #[test]
    fn what_was_found_weighs_what_it_weighs() {
        let fst = indexed();
        let inner = SortedMatcher::new(&fst, MatchType::Output).unwrap();
        // No prefix: with a single way forward the arc is reported instead, and
        // its weight deliberately left off so as not to count it twice.
        let mut matcher =
            ArcLookAheadMatcher::with_flags(&fst, inner, LOOKAHEAD_WEIGHT | LOOKAHEAD_NON_EPSILONS);
        // From state 0 the only match is label 1, which the other side carries
        // at weight 1 and this side at weight 0.
        matcher.set_state(0);
        let found = matcher.look_ahead(&other(&[1, 9]), 0);
        assert!(found.reachable);
        assert_eq!(found.weight, TropicalWeight(1.0));

        // Nothing found says nothing about weight.
        matcher.set_state(0);
        let found = matcher.look_ahead(&other(&[9]), 0);
        assert!(!found.reachable);
        assert_eq!(found.weight, TropicalWeight::one());
    }

    /// The flags have to name exactly one side to index.
    #[test]
    fn the_flags_have_to_name_one_side() {
        let fst = indexed();
        let inner = SortedMatcher::new(&fst, MatchType::Input).unwrap();
        let Err(err) = LabelLookAheadMatcher::new(
            &fst,
            inner.clone(),
            MatchType::Input,
            DEFAULT_LABEL_LOOKAHEAD_FLAGS,
            DefaultAccumulator,
        ) else {
            panic!("naming neither side has to be refused")
        };
        assert!(format!("{err}").contains("exactly one"), "{err}");

        assert!(
            LabelLookAheadMatcher::new(
                &fst,
                inner,
                MatchType::Input,
                DEFAULT_LABEL_LOOKAHEAD_FLAGS | INPUT_LOOKAHEAD_MATCHER | OUTPUT_LOOKAHEAD_MATCHER,
                DefaultAccumulator,
            )
            .is_err()
        );
    }
}
