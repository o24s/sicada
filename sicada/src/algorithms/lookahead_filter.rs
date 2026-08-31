//! Composition filters that ask the matcher about the future.
//!
//! Port of OpenFst's `lookahead-filter.h`. A [compose
//! filter](super::compose_filter::ComposeFilter) decides which pairs of arcs
//! composition is allowed to take; these three ask a
//! [`LookAheadMatcher`] what lies
//! past the pair and act on the answer:
//!
//! - [`LookAheadComposeFilter`] refuses a pair whose destination has no future.
//! - [`PushWeightsComposeFilter`] moves the weight of that future onto the arc,
//!   so a search over the result sees it sooner.
//! - [`PushLabelsComposeFilter`] does the same for the one label that must come
//!   next, so it is matched as early as it can be.
//!
//! Each wraps the one before it, in that order, which is how upstream stacks
//! them too.
//!
//! SICADA-DIVERGE: upstream reaches the right matcher through a
//! `LookAheadSelector` templated on the match type, which exists to pick one of
//! two members and one of two FSTs. Which one it is, is fixed when the filter
//! is built, so it is a field here and there is no selector type.

use std::marker::PhantomData;

use crate::algorithms::compose_filter::ComposeFilter;
use crate::algorithms::filter_state::{LabelFilterState, PairFilterState, WeightFilterState};
use crate::algorithms::lookahead_matcher::{
    INPUT_LOOKAHEAD_MATCHER, LOOKAHEAD_EPSILONS, LOOKAHEAD_NON_EPSILON_PREFIX,
    LOOKAHEAD_NON_EPSILONS, LOOKAHEAD_PREFIX, LOOKAHEAD_WEIGHT, LookAhead, LookAheadMatcher,
    OUTPUT_LOOKAHEAD_MATCHER,
};
use crate::arc::{Arc, ArcLabel};
use crate::error::OpenFstError;
use crate::fst::{ExpandedFst, Fst, MatchType};
use crate::matcher::{Matcher, MultiEpsMatcher};
use crate::properties::{
    K_ERROR, K_I_LABEL_INVARIANT_PROPERTIES, K_O_LABEL_INVARIANT_PROPERTIES,
    K_WEIGHT_INVARIANT_PROPERTIES,
};
use crate::weight::{DELTA, Divide, DivideType, Weight};

/// The weight of a filter's arc type.
type FilterWeight<F> = <<F as ComposeFilter>::Arc as Arc>::Weight;
/// The label of a filter's arc type.
type FilterLabel<F> = <<F as ComposeFilter>::Arc as Arc>::Label;
/// The state id of a filter's arc type.
type FilterStateId<F> = <<F as ComposeFilter>::Arc as Arc>::StateId;

/// Which side the look-ahead is done on.
///
/// The matcher over the first FST looks ahead on output labels and the one over
/// the second on input labels, so this says which of the two can be asked;
/// [`MatchType::None`] means neither can.
pub fn lookahead_match_type<'f, A, M1, M2>(matcher1: &M1, matcher2: &M2) -> MatchType
where
    A: Arc,
    M1: LookAheadMatcher<'f, A>,
    M2: LookAheadMatcher<'f, A>,
{
    if matcher1.match_type() == MatchType::Output
        && matcher1.lookahead_flags() & OUTPUT_LOOKAHEAD_MATCHER != 0
    {
        return MatchType::Output;
    }
    if matcher2.match_type() == MatchType::Input
        && matcher2.lookahead_flags() & INPUT_LOOKAHEAD_MATCHER != 0
    {
        return MatchType::Input;
    }
    MatchType::None
}

/// What a filter stacked on a look-ahead filter needs to read back from it.
///
/// SICADA-DIVERGE: upstream's `LookAheadArc()` reports whether the last
/// `FilterArc` looked ahead, and the caller then reads the matcher's
/// `LookAheadWeight()`/`LookAheadPrefix()`, two calls that only mean anything
/// together. [`last_lookahead`](Self::last_lookahead) is `None` exactly when
/// `LookAheadArc()` would be false, so there is nothing to read out of order.
pub trait HasLookAhead: ComposeFilter {
    /// What the matcher doing the asking can report; see the flag constants in
    /// [`lookahead_matcher`](super::lookahead_matcher).
    fn lookahead_flags(&self) -> u32;

    /// Whether the look-ahead is done on the output side.
    fn lookahead_output(&self) -> bool;

    /// What the last [`ComposeFilter::filter_arc`] found, or `None` if it
    /// did not look.
    fn last_lookahead(&self) -> Option<&LookAhead<Self::Arc>>;

    /// Whether `label` could still be matched from `state` on the look-ahead
    /// side, without looking at the other FST at all.
    ///
    /// Answered on the filter's own copy of the matcher, for the reason given
    /// on [`LookAheadComposeFilter`]'s fields.
    fn look_ahead_label_from(
        &mut self,
        state: <Self::Arc as Arc>::StateId,
        label: <Self::Arc as Arc>::Label,
    ) -> bool;
}

/// Refuses the arc pairs whose destination the look-ahead says leads nowhere.
///
/// Composition otherwise builds a state for every pair it can reach and finds
/// out only later that most of them are dead ends; this asks first.
pub struct LookAheadComposeFilter<'f, Inner: ComposeFilter, F> {
    inner: Inner,
    /// The filter's own copies of the two matchers.
    ///
    /// Looking ahead moves a matcher, and composition is in the middle of
    /// iterating one of them when the filter is consulted: it calls
    /// `SetState(s1)` once and then `Find` for each arc of the other side, with
    /// the filter run in between. Moving the composition's matcher leaves the
    /// next `Find` pointed at the wrong state, and paths quietly go missing --
    /// 8 of 37 words on a pair of 300-state acceptors, before this was a copy.
    /// Upstream's `LookAheadSelector` holds `matcher1->Copy()` for exactly this
    /// reason.
    ahead1: Inner::Matcher1,
    ahead2: Inner::Matcher2,
    /// The FST looked into: the *other* side from the matcher doing the asking.
    lookahead_fst: &'f F,
    /// Which side is asked.
    lookahead_type: MatchType,
    /// What that matcher can say.
    flags: u32,
    /// What the last look-ahead found, when one was made.
    last: Option<LookAhead<Inner::Arc>>,
}

impl<'f, Inner, F> LookAheadComposeFilter<'f, Inner, F>
where
    Inner: ComposeFilter,
    Inner::Matcher1: LookAheadMatcher<'f, Inner::Arc>,
    Inner::Matcher2: LookAheadMatcher<'f, Inner::Arc>,
    F: Fst<Inner::Arc> + ExpandedFst<Inner::Arc>,
{
    /// Wraps `inner`, looking ahead into `lookahead_fst`.
    ///
    /// `lookahead_fst` is the side the asking matcher is *not* over: with the
    /// look-ahead on output labels, matcher 1 asks about states of the second
    /// FST, and the other way round on input labels.
    pub fn new(inner: Inner, lookahead_fst: &'f F) -> Result<Self, OpenFstError> {
        let lookahead_type = lookahead_match_type(inner.matcher1(), inner.matcher2());
        if lookahead_type == MatchType::None {
            return Err(OpenFstError::InvalidOperation(
                "LookAheadComposeFilter: the 1st argument cannot match/look-ahead on output \
                 labels and the 2nd cannot on input labels"
                    .into(),
            ));
        }
        let flags = if lookahead_type == MatchType::Output {
            inner.matcher1().lookahead_flags()
        } else {
            inner.matcher2().lookahead_flags()
        };
        let ahead1 = inner.matcher1().clone();
        let ahead2 = inner.matcher2().clone();
        Ok(Self {
            inner,
            ahead1,
            ahead2,
            lookahead_fst,
            lookahead_type,
            flags,
            last: None,
        })
    }

    /// The filter underneath.
    pub fn inner(&self) -> &Inner {
        &self.inner
    }

    /// Asks the matcher whether the pair leads anywhere.
    ///
    /// `arca` is the arc on the look-ahead side and `arcb` the one whose
    /// destination is asked about.
    fn look(&mut self, arca: &Inner::Arc, arcb: &Inner::Arc) -> bool {
        let label = if self.lookahead_output() {
            arca.olabel()
        } else {
            arca.ilabel()
        };
        let epsilon = FilterLabel::<Inner>::epsilon();
        // Only the kinds of arc the matcher said it looks at are asked about;
        // for the others the pair stands as the inner filter left it.
        if label != epsilon && self.flags & LOOKAHEAD_NON_EPSILONS == 0 {
            return true;
        }
        if label == epsilon && self.flags & LOOKAHEAD_EPSILONS == 0 {
            return true;
        }
        let found = if self.lookahead_output() {
            self.ahead1.set_state(arca.nextstate());
            self.ahead1.look_ahead(self.lookahead_fst, arcb.nextstate())
        } else {
            self.ahead2.set_state(arca.nextstate());
            self.ahead2.look_ahead(self.lookahead_fst, arcb.nextstate())
        };
        let reachable = found.reachable;
        self.last = Some(found);
        reachable
    }
}

impl<'f, Inner, F> ComposeFilter for LookAheadComposeFilter<'f, Inner, F>
where
    Inner: ComposeFilter,
    Inner::Matcher1: LookAheadMatcher<'f, Inner::Arc>,
    Inner::Matcher2: LookAheadMatcher<'f, Inner::Arc>,
    F: Fst<Inner::Arc> + ExpandedFst<Inner::Arc>,
{
    type Arc = Inner::Arc;
    type FilterState = Inner::FilterState;
    type Matcher1 = Inner::Matcher1;
    type Matcher2 = Inner::Matcher2;

    fn start(&self) -> Self::FilterState {
        self.inner.start()
    }

    fn set_state(
        &mut self,
        s1: FilterStateId<Self>,
        s2: FilterStateId<Self>,
        fs: &Self::FilterState,
    ) {
        self.inner.set_state(s1, s2, fs);
    }

    fn filter_arc(
        &mut self,
        arc1: &mut Self::Arc,
        arc2: &mut Self::Arc,
    ) -> Option<Self::FilterState> {
        self.look_ahead_filter_arc(arc1, arc2)
    }

    fn filter_final(&self, w1: &mut FilterWeight<Self>, w2: &mut FilterWeight<Self>) {
        self.inner.filter_final(w1, w2);
    }

    fn matcher1(&self) -> &Self::Matcher1 {
        self.inner.matcher1()
    }

    fn matcher1_mut(&mut self) -> &mut Self::Matcher1 {
        self.inner.matcher1_mut()
    }

    fn matcher2(&self) -> &Self::Matcher2 {
        self.inner.matcher2()
    }

    fn matcher2_mut(&mut self) -> &mut Self::Matcher2 {
        self.inner.matcher2_mut()
    }

    fn properties(&self, props: u64) -> u64 {
        let out = self.inner.properties(props);
        if self.lookahead_type == MatchType::None {
            out | K_ERROR
        } else {
            out
        }
    }
}

impl<'f, Inner, F> LookAheadComposeFilter<'f, Inner, F>
where
    Inner: ComposeFilter,
    Inner::Matcher1: LookAheadMatcher<'f, Inner::Arc>,
    Inner::Matcher2: LookAheadMatcher<'f, Inner::Arc>,
    F: Fst<Inner::Arc> + ExpandedFst<Inner::Arc>,
{
    fn look_ahead_filter_arc(
        &mut self,
        arc1: &mut <Self as ComposeFilter>::Arc,
        arc2: &mut <Self as ComposeFilter>::Arc,
    ) -> Option<<Self as ComposeFilter>::FilterState> {
        self.last = None;
        let fs = self.inner.filter_arc(arc1, arc2)?;
        let (arca, arcb) = if self.lookahead_output() {
            (arc1.clone(), arc2.clone())
        } else {
            (arc2.clone(), arc1.clone())
        };
        self.look(&arca, &arcb).then_some(fs)
    }
}

impl<'f, Inner, F> HasLookAhead for LookAheadComposeFilter<'f, Inner, F>
where
    Inner: ComposeFilter,
    Inner::Matcher1: LookAheadMatcher<'f, Inner::Arc>,
    Inner::Matcher2: LookAheadMatcher<'f, Inner::Arc>,
    F: Fst<Inner::Arc> + ExpandedFst<Inner::Arc>,
{
    fn lookahead_flags(&self) -> u32 {
        self.flags
    }

    fn lookahead_output(&self) -> bool {
        self.lookahead_type == MatchType::Output
    }

    fn last_lookahead(&self) -> Option<&LookAhead<Inner::Arc>> {
        self.last.as_ref()
    }

    fn look_ahead_label_from(
        &mut self,
        state: <Inner::Arc as Arc>::StateId,
        label: <Inner::Arc as Arc>::Label,
    ) -> bool {
        let output = self.lookahead_output();
        if output {
            self.ahead1.set_state(state);
            self.ahead1.look_ahead_label(label)
        } else {
            self.ahead2.set_state(state);
            self.ahead2.look_ahead_label(label)
        }
    }
}

/// Moves the weight of what lies ahead onto the arc.
///
/// A search over the result then meets that weight sooner and can give up on a
/// path before walking it. What was moved is remembered in the filter state and
/// divided back out on the next arc, so no path ends up weighing more or less
/// than it did.
pub struct PushWeightsComposeFilter<Inner: ComposeFilter> {
    inner: Inner,
    /// What was pushed onto the way in to the current state.
    pushed: FilterWeight<Inner>,
}

impl<Inner> PushWeightsComposeFilter<Inner>
where
    Inner: ComposeFilter + HasLookAhead,
{
    /// Wraps `inner`.
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            pushed: FilterWeight::<Inner>::one(),
        }
    }

    /// The filter underneath.
    pub fn inner(&self) -> &Inner {
        &self.inner
    }
}

impl<Inner> ComposeFilter for PushWeightsComposeFilter<Inner>
where
    Inner: ComposeFilter + HasLookAhead,
    FilterWeight<Inner>: Divide + std::hash::Hash + Eq,
{
    type Arc = Inner::Arc;
    type FilterState = PairFilterState<Inner::FilterState, WeightFilterState<FilterWeight<Inner>>>;
    type Matcher1 = Inner::Matcher1;
    type Matcher2 = Inner::Matcher2;

    fn start(&self) -> Self::FilterState {
        PairFilterState::new(
            self.inner.start(),
            WeightFilterState::new(FilterWeight::<Inner>::one()),
        )
    }

    fn set_state(
        &mut self,
        s1: FilterStateId<Self>,
        s2: FilterStateId<Self>,
        fs: &Self::FilterState,
    ) {
        self.pushed = fs.state2().weight();
        self.inner.set_state(s1, s2, fs.state1());
    }

    fn filter_arc(
        &mut self,
        arc1: &mut Self::Arc,
        arc2: &mut Self::Arc,
    ) -> Option<Self::FilterState> {
        self.look_ahead_filter_arc(arc1, arc2)
    }

    fn filter_final(&self, w1: &mut FilterWeight<Self>, w2: &mut FilterWeight<Self>) {
        self.inner.filter_final(w1, w2);
        if self.inner.lookahead_flags() & LOOKAHEAD_WEIGHT == 0
            || *w1 == FilterWeight::<Inner>::zero()
        {
            return;
        }
        // What was pushed onto the way in has to come back off at the end.
        *w1 = w1.divide(&self.pushed, DivideType::Any);
    }

    fn matcher1(&self) -> &Self::Matcher1 {
        self.inner.matcher1()
    }

    fn matcher1_mut(&mut self) -> &mut Self::Matcher1 {
        self.inner.matcher1_mut()
    }

    fn matcher2(&self) -> &Self::Matcher2 {
        self.inner.matcher2()
    }

    fn matcher2_mut(&mut self) -> &mut Self::Matcher2 {
        self.inner.matcher2_mut()
    }

    fn properties(&self, props: u64) -> u64 {
        self.inner.properties(props) & K_WEIGHT_INVARIANT_PROPERTIES
    }
}

impl<Inner> PushWeightsComposeFilter<Inner>
where
    Inner: ComposeFilter + HasLookAhead,
    FilterWeight<Inner>: Divide + std::hash::Hash + Eq,
{
    fn look_ahead_filter_arc(
        &mut self,
        arc1: &mut <Self as ComposeFilter>::Arc,
        arc2: &mut <Self as ComposeFilter>::Arc,
    ) -> Option<<Self as ComposeFilter>::FilterState> {
        let fs1 = self.inner.filter_arc(arc1, arc2)?;
        let one = FilterWeight::<Inner>::one();
        if self.inner.lookahead_flags() & LOOKAHEAD_WEIGHT == 0 {
            return Some(PairFilterState::new(fs1, WeightFilterState::new(one)));
        }
        let ahead = self
            .inner
            .last_lookahead()
            .map_or(one.clone(), |found| found.weight.clone());
        // A future that weighs zero is no future at all.
        if ahead == FilterWeight::<Inner>::zero() {
            return None;
        }
        let already = self.pushed.clone();
        if ahead == already {
            // Nothing changed, so the arithmetic below is a no-op and the state
            // weight is already quantized.
            return Some(PairFilterState::new(fs1, WeightFilterState::new(already)));
        }
        *arc2 = <Self as ComposeFilter>::Arc::new(
            arc2.ilabel(),
            arc2.olabel(),
            arc2.weight()
                .times(&ahead)
                .divide(&already, DivideType::Any),
            arc2.nextstate(),
        );
        // Quantized so that futures agreeing closely enough are one filter
        // state rather than an unbounded family of them.
        let carried = if ahead == one {
            one
        } else {
            ahead.quantize(DELTA)
        };
        Some(PairFilterState::new(fs1, WeightFilterState::new(carried)))
    }
}

impl<Inner> HasLookAhead for PushWeightsComposeFilter<Inner>
where
    Inner: ComposeFilter + HasLookAhead,
    FilterWeight<Inner>: Divide + std::hash::Hash + Eq,
{
    fn lookahead_flags(&self) -> u32 {
        self.inner.lookahead_flags()
    }

    fn lookahead_output(&self) -> bool {
        self.inner.lookahead_output()
    }

    fn last_lookahead(&self) -> Option<&LookAhead<Self::Arc>> {
        self.inner.last_lookahead()
    }

    fn look_ahead_label_from(
        &mut self,
        state: <Self::Arc as Arc>::StateId,
        label: <Self::Arc as Arc>::Label,
    ) -> bool {
        self.inner.look_ahead_label_from(state, label)
    }
}

/// Moves the one label that must come next onto the arc.
///
/// When the look-ahead finds exactly one way forward, its label is written now
/// rather than when it is reached, so the other side can match it that much
/// earlier. The label is carried in the filter state until the arc that would
/// have written it comes round, and is turned into an epsilon there so it is
/// not written twice.
///
/// SICADA-DIVERGE: upstream wraps the inner filter's matchers in its own
/// [`MultiEpsMatcher`]s, holding non-owning pointers into the filter it also
/// owns. That is a self-referential borrow in Rust; instead the matchers
/// already are multi-epsilon matchers, named by
/// [`MultiEpsLabels`], and this filter adds and clears the pushed label on
/// them. The matchers composition sees are the same objects either way.
pub struct PushLabelsComposeFilter<'f, Inner: ComposeFilter, F> {
    inner: Inner,
    /// The label carried forward, if any.
    carried: FilterLabel<Inner>,
    /// The inner filter state that goes with it.
    fs1: Inner::FilterState,
    /// The FST the look-ahead side matches over, which says how many arcs leave
    /// the state being expanded.
    lookahead_side: &'f F,
    /// How many arcs that is.
    narcs: usize,
    _marker: PhantomData<fn() -> F>,
}

/// A matcher that can be told to treat a label as epsilon.
///
/// [`PushLabelsComposeFilter`] needs it: the label it pushed forward has to
/// match the implicit arc that stands for "the label was already written".
pub trait MultiEpsLabels<L> {
    /// Treats `label` as epsilon as well.
    fn add_multi_eps_label(&mut self, label: L) -> Result<(), OpenFstError>;

    /// Stops treating anything but epsilon itself as epsilon.
    fn clear_multi_eps_labels(&mut self);
}

impl<'f, M, A> MultiEpsLabels<A::Label> for MultiEpsMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    fn add_multi_eps_label(&mut self, label: A::Label) -> Result<(), OpenFstError> {
        MultiEpsMatcher::add_multi_eps_label(self, label)
    }

    fn clear_multi_eps_labels(&mut self) {
        MultiEpsMatcher::clear_multi_eps_labels(self);
    }
}

impl<'f, Inner, F> PushLabelsComposeFilter<'f, Inner, F>
where
    Inner: ComposeFilter + HasLookAhead,
    Inner::Matcher1: MultiEpsLabels<FilterLabel<Inner>> + LookAheadMatcher<'f, Inner::Arc>,
    Inner::Matcher2: MultiEpsLabels<FilterLabel<Inner>> + LookAheadMatcher<'f, Inner::Arc>,
    F: Fst<Inner::Arc> + ExpandedFst<Inner::Arc>,
{
    /// Wraps `inner`.
    ///
    /// `lookahead_side` is the FST the look-ahead matcher matches over: the
    /// first when looking ahead on output labels, the second otherwise.
    pub fn new(inner: Inner, lookahead_side: &'f F) -> Self {
        let fs1 = inner.start();
        Self {
            inner,
            carried: FilterLabel::<Inner>::no_label(),
            fs1,
            lookahead_side,
            narcs: 0,
            _marker: PhantomData,
        }
    }

    /// The filter underneath.
    pub fn inner(&self) -> &Inner {
        &self.inner
    }

    /// The filter state that still owes the carried label.
    fn carried_state(&self) -> <Self as ComposeFilter>::FilterState {
        PairFilterState::new(self.fs1.clone(), LabelFilterState::new(self.carried))
    }

    /// Pays a label carried from an earlier arc.
    ///
    /// `arca` is the arc on the look-ahead side, `arcb` the other.
    fn pay_carried(
        &mut self,
        arca: &mut Inner::Arc,
        arcb: &Inner::Arc,
        output: bool,
    ) -> Option<<Self as ComposeFilter>::FilterState> {
        let no_label = FilterLabel::<Inner>::no_label();
        let epsilon = FilterLabel::<Inner>::epsilon();
        let label_a = if output { arca.olabel() } else { arca.ilabel() };
        // The other side has to be the implicit arc the multi-epsilon matcher
        // hands back, which carries no label at all.
        let label_b = if output { arcb.ilabel() } else { arcb.olabel() };
        if label_b != no_label {
            return None;
        }
        if label_a == self.carried {
            // The label is paid: what was a match of the multi-epsilon label
            // becomes an epsilon, so it is not written twice.
            let (ilabel, olabel) = if output {
                (arca.ilabel(), epsilon)
            } else {
                (epsilon, arca.olabel())
            };
            *arca = Inner::Arc::new(ilabel, olabel, arca.weight().clone(), arca.nextstate());
            return Some(<Self as ComposeFilter>::start(self));
        }
        if label_a == epsilon {
            if self.narcs == 1 {
                // Nowhere else to go, so the epsilon is taken and the label
                // stays owed.
                return Some(self.carried_state());
            }
            // Taking the epsilon is only allowed if the label can still be paid
            // afterwards; otherwise this path leads nowhere.
            let carried = self.carried;
            let next = arca.nextstate();
            let can_still = self.inner.look_ahead_label_from(next, carried);
            return can_still.then(|| self.carried_state());
        }
        // Some other label, which does not pay what is owed.
        None
    }

    /// Writes the one label the look-ahead found, when there is one.
    fn push_label(
        &mut self,
        arca: &mut Inner::Arc,
        arcb: &mut Inner::Arc,
        output: bool,
        fs1: Inner::FilterState,
        found: &LookAhead<Inner::Arc>,
    ) -> <Self as ComposeFilter>::FilterState {
        let no_label = FilterLabel::<Inner>::no_label();
        let epsilon = FilterLabel::<Inner>::epsilon();
        let nothing = || PairFilterState::new(fs1.clone(), LabelFilterState::new(no_label));

        let label_a = if output { arca.olabel() } else { arca.ilabel() };
        let label_b = if output { arcb.olabel() } else { arcb.ilabel() };
        if label_b != epsilon {
            // The other side already writes something here; no room to push.
            return nothing();
        }
        if label_a != epsilon && self.inner.lookahead_flags() & LOOKAHEAD_NON_EPSILON_PREFIX != 0 {
            // The matcher only reports a prefix for a non-epsilon match, so
            // this one is not the kind that may be pushed.
            return nothing();
        }
        let Some(prefix) = found.prefix.as_ref() else {
            return nothing();
        };
        // The arc the look-ahead says must come next is taken now, and the
        // filter then owes the label it would have written.
        let owed = if output {
            prefix.ilabel()
        } else {
            prefix.olabel()
        };
        let (ilabel_a, olabel_a) = if output {
            (arca.ilabel(), owed)
        } else {
            (owed, arca.olabel())
        };
        *arca = Inner::Arc::new(ilabel_a, olabel_a, arca.weight().clone(), arca.nextstate());
        *arcb = Inner::Arc::new(
            prefix.ilabel(),
            prefix.olabel(),
            arcb.weight().times(prefix.weight()),
            prefix.nextstate(),
        );
        PairFilterState::new(fs1, LabelFilterState::new(owed))
    }
}

impl<'f, Inner, F> ComposeFilter for PushLabelsComposeFilter<'f, Inner, F>
where
    Inner: ComposeFilter + HasLookAhead,
    Inner::Matcher1: MultiEpsLabels<FilterLabel<Inner>> + LookAheadMatcher<'f, Inner::Arc>,
    Inner::Matcher2: MultiEpsLabels<FilterLabel<Inner>> + LookAheadMatcher<'f, Inner::Arc>,
    F: Fst<Inner::Arc> + ExpandedFst<Inner::Arc>,
{
    type Arc = Inner::Arc;
    type FilterState = PairFilterState<Inner::FilterState, LabelFilterState<FilterLabel<Inner>>>;
    type Matcher1 = Inner::Matcher1;
    type Matcher2 = Inner::Matcher2;

    fn start(&self) -> Self::FilterState {
        PairFilterState::new(
            self.inner.start(),
            LabelFilterState::new(FilterLabel::<Inner>::no_label()),
        )
    }

    fn set_state(
        &mut self,
        s1: FilterStateId<Self>,
        s2: FilterStateId<Self>,
        fs: &Self::FilterState,
    ) {
        self.carried = fs.state2().label_copied();
        self.fs1 = fs.state1().clone();
        self.inner.set_state(s1, s2, fs.state1());
        if self.inner.lookahead_flags() & LOOKAHEAD_PREFIX == 0 {
            return;
        }
        let here = if self.inner.lookahead_output() {
            s1
        } else {
            s2
        };
        self.narcs = self.lookahead_side.num_arcs(here);
        // The pushed label has to match the implicit arc standing for "already
        // written", which making it a multi-epsilon label arranges.
        self.inner.matcher1_mut().clear_multi_eps_labels();
        self.inner.matcher2_mut().clear_multi_eps_labels();
        if self.carried != FilterLabel::<Inner>::no_label() {
            let carried = self.carried;
            let _ = self.inner.matcher1_mut().add_multi_eps_label(carried);
            let _ = self.inner.matcher2_mut().add_multi_eps_label(carried);
        }
    }

    fn filter_arc(
        &mut self,
        arc1: &mut Self::Arc,
        arc2: &mut Self::Arc,
    ) -> Option<Self::FilterState> {
        self.look_ahead_filter_arc(arc1, arc2)
    }

    fn filter_final(&self, w1: &mut FilterWeight<Self>, w2: &mut FilterWeight<Self>) {
        self.inner.filter_final(w1, w2);
        if self.inner.lookahead_flags() & LOOKAHEAD_PREFIX == 0
            || *w1 == FilterWeight::<Inner>::zero()
        {
            return;
        }
        // A label still owed cannot be left unwritten, so this is not a place
        // the path may stop.
        if self.carried != FilterLabel::<Inner>::no_label() {
            *w1 = FilterWeight::<Inner>::zero();
        }
    }

    fn matcher1(&self) -> &Self::Matcher1 {
        self.inner.matcher1()
    }

    fn matcher1_mut(&mut self) -> &mut Self::Matcher1 {
        self.inner.matcher1_mut()
    }

    fn matcher2(&self) -> &Self::Matcher2 {
        self.inner.matcher2()
    }

    fn matcher2_mut(&mut self) -> &mut Self::Matcher2 {
        self.inner.matcher2_mut()
    }

    fn properties(&self, props: u64) -> u64 {
        let out = self.inner.properties(props);
        if self.inner.lookahead_output() {
            out & K_O_LABEL_INVARIANT_PROPERTIES
        } else {
            out & K_I_LABEL_INVARIANT_PROPERTIES
        }
    }
}

impl<'f, Inner, F> PushLabelsComposeFilter<'f, Inner, F>
where
    Inner: ComposeFilter + HasLookAhead,
    Inner::Matcher1: MultiEpsLabels<FilterLabel<Inner>> + LookAheadMatcher<'f, Inner::Arc>,
    Inner::Matcher2: MultiEpsLabels<FilterLabel<Inner>> + LookAheadMatcher<'f, Inner::Arc>,
    F: Fst<Inner::Arc> + ExpandedFst<Inner::Arc>,
{
    fn look_ahead_filter_arc(
        &mut self,
        arc1: &mut <Self as ComposeFilter>::Arc,
        arc2: &mut <Self as ComposeFilter>::Arc,
    ) -> Option<<Self as ComposeFilter>::FilterState> {
        let no_label = FilterLabel::<Inner>::no_label();
        if self.inner.lookahead_flags() & LOOKAHEAD_PREFIX == 0 {
            let fs1 = self.inner.filter_arc(arc1, arc2)?;
            return Some(PairFilterState::new(fs1, LabelFilterState::new(no_label)));
        }
        let output = self.inner.lookahead_output();
        if self.carried != no_label {
            // A label is owed; this pair either pays it or waits. The inner
            // filter is not consulted, since the pair is one this filter
            // invented.
            let (arca, arcb) = if output {
                (&mut *arc1, &*arc2)
            } else {
                (&mut *arc2, &*arc1)
            };
            // The two borrows are of different arcs, which the compiler cannot
            // see through the conditional, so the read side is copied.
            let arcb = arcb.clone();
            return self.pay_carried(arca, &arcb, output);
        }
        let fs1 = self.inner.filter_arc(arc1, arc2)?;
        let Some(found) = self.inner.last_lookahead().cloned() else {
            return Some(PairFilterState::new(fs1, LabelFilterState::new(no_label)));
        };
        let (arca, arcb) = if output {
            let (a, b) = (arc1, arc2);
            (a, b)
        } else {
            (arc2, arc1)
        };
        Some(self.push_label(arca, arcb, output, fs1, &found))
    }
}

impl<'f, Inner, F> HasLookAhead for PushLabelsComposeFilter<'f, Inner, F>
where
    Inner: ComposeFilter + HasLookAhead,
    Inner::Matcher1: MultiEpsLabels<FilterLabel<Inner>> + LookAheadMatcher<'f, Inner::Arc>,
    Inner::Matcher2: MultiEpsLabels<FilterLabel<Inner>> + LookAheadMatcher<'f, Inner::Arc>,
    F: Fst<Inner::Arc> + ExpandedFst<Inner::Arc>,
{
    fn lookahead_flags(&self) -> u32 {
        self.inner.lookahead_flags()
    }

    fn lookahead_output(&self) -> bool {
        self.inner.lookahead_output()
    }

    fn last_lookahead(&self) -> Option<&LookAhead<Self::Arc>> {
        self.inner.last_lookahead()
    }

    fn look_ahead_label_from(
        &mut self,
        state: <Self::Arc as Arc>::StateId,
        label: <Self::Arc as Arc>::Label,
    ) -> bool {
        self.inner.look_ahead_label_from(state, label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::arcsort::{ILabelCompare, OLabelCompare, arc_sort};
    use crate::algorithms::compose_filter::NullComposeFilter;
    use crate::algorithms::lookahead_matcher::ArcLookAheadMatcher;
    use crate::arc::StdArc;
    use crate::fst::MutableFst;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::matcher::{MultiEpsFlags, SortedMatcher};
    use crate::properties::K_FST_PROPERTIES;
    use crate::weights::float_weight::TropicalWeight;

    /// The look-ahead matcher used throughout: it needs nothing precomputed, so
    /// a test can build any pair of FSTs and ask it about them.
    type Look<'f> =
        ArcLookAheadMatcher<'f, StdVectorFst, SortedMatcher<'f, StdVectorFst, StdArc>, StdArc>;
    type Eps<'f> = MultiEpsMatcher<'f, Look<'f>, StdArc>;

    fn w(x: f32) -> TropicalWeight {
        TropicalWeight(x)
    }

    /// Builds an FST from `(state, ilabel, olabel, weight, nextstate)` arcs,
    /// with the last state final.
    fn build(nstates: usize, arcs: &[(i32, i32, i32, f32, i32)]) -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..nstates {
            fst.add_state();
        }
        fst.set_start(0);
        for &(s, il, ol, weight, next) in arcs {
            fst.add_arc(s, StdArc::new(il, ol, w(weight), next));
        }
        fst.set_final((nstates - 1) as i32, TropicalWeight::one());
        arc_sort(&mut fst, &ILabelCompare);
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    fn matchers<'f>(
        fst1: &'f StdVectorFst,
        fst2: &'f StdVectorFst,
        flags: u32,
    ) -> (Look<'f>, Look<'f>) {
        let m1 = ArcLookAheadMatcher::with_flags(
            fst1,
            SortedMatcher::new(fst1, MatchType::Output).unwrap(),
            flags,
        );
        let m2 = ArcLookAheadMatcher::with_flags(
            fst2,
            SortedMatcher::new(fst2, MatchType::Input).unwrap(),
            flags,
        );
        (m1, m2)
    }

    /// With matcher 1 over output labels and matcher 2 over input labels, the
    /// look-ahead is done on the output side.
    #[test]
    fn the_output_side_is_the_one_that_looks_ahead() {
        let fst1 = build(2, &[(0, 1, 1, 0.0, 1)]);
        let fst2 = build(2, &[(0, 1, 1, 0.0, 1)]);
        let (m1, m2) = matchers(&fst1, &fst2, LOOKAHEAD_NON_EPSILONS);
        assert_eq!(lookahead_match_type(&m1, &m2), MatchType::Output);

        // Two matchers over the same side can look ahead for neither.
        let m1 = ArcLookAheadMatcher::with_flags(
            &fst1,
            SortedMatcher::new(&fst1, MatchType::Input).unwrap(),
            LOOKAHEAD_NON_EPSILONS,
        );
        let m2 = ArcLookAheadMatcher::with_flags(
            &fst2,
            SortedMatcher::new(&fst2, MatchType::Output).unwrap(),
            LOOKAHEAD_NON_EPSILONS,
        );
        assert_eq!(lookahead_match_type(&m1, &m2), MatchType::None);
    }

    /// A filter that cannot look ahead on either side is refused, rather than
    /// built and reported through a property bit later.
    #[test]
    fn a_filter_that_cannot_look_ahead_is_refused() {
        let fst1 = build(2, &[(0, 1, 1, 0.0, 1)]);
        let fst2 = build(2, &[(0, 1, 1, 0.0, 1)]);
        let m1 = ArcLookAheadMatcher::with_flags(
            &fst1,
            SortedMatcher::new(&fst1, MatchType::Input).unwrap(),
            LOOKAHEAD_NON_EPSILONS,
        );
        let m2 = ArcLookAheadMatcher::with_flags(
            &fst2,
            SortedMatcher::new(&fst2, MatchType::Output).unwrap(),
            LOOKAHEAD_NON_EPSILONS,
        );
        let inner = NullComposeFilter::new(m1, m2);
        let Err(err) = LookAheadComposeFilter::new(inner, &fst2) else {
            panic!("a filter that can look ahead on neither side has to be refused")
        };
        assert!(format!("{err}").contains("look-ahead"), "{err}");
    }

    /// The pair is taken only when something can still match past it.
    #[test]
    fn a_pair_with_no_future_is_refused() {
        // 0 -1:1-> 1 -2:2-> 2, so past label 1 only label 2 can be read.
        let fst1 = build(3, &[(0, 1, 1, 0.0, 1), (1, 2, 2, 0.0, 2)]);
        let goes_on = build(3, &[(0, 1, 1, 0.0, 1), (1, 2, 2, 0.0, 2)]);
        let stops = build(2, &[(0, 1, 1, 0.0, 1)]);

        for (fst2, expected) in [(&goes_on, true), (&stops, false)] {
            let (m1, m2) = matchers(&fst1, fst2, LOOKAHEAD_NON_EPSILONS | LOOKAHEAD_EPSILONS);
            let inner = NullComposeFilter::new(m1, m2);
            let mut filter = LookAheadComposeFilter::new(inner, fst2).unwrap();
            let start = filter.start();
            filter.set_state(0, 0, &start);
            let mut arc1 = StdArc::new(1, 1, TropicalWeight::one(), 1);
            let mut arc2 = StdArc::new(1, 1, TropicalWeight::one(), 1);
            assert_eq!(
                filter.filter_arc(&mut arc1, &mut arc2).is_some(),
                expected,
                "the pair leads somewhere: {expected}"
            );
        }
    }

    /// An arc the matcher said it does not look at is left to the inner filter.
    #[test]
    fn an_arc_the_matcher_ignores_is_not_looked_at() {
        let fst1 = build(3, &[(0, 1, 1, 0.0, 1), (1, 2, 2, 0.0, 2)]);
        let fst2 = build(2, &[(0, 1, 1, 0.0, 1)]);
        // Only epsilons are looked at, and this pair carries a label.
        let (m1, m2) = matchers(&fst1, &fst2, LOOKAHEAD_EPSILONS);
        let inner = NullComposeFilter::new(m1, m2);
        let mut filter = LookAheadComposeFilter::new(inner, &fst2).unwrap();
        let start = filter.start();
        filter.set_state(0, 0, &start);
        let mut arc1 = StdArc::new(1, 1, TropicalWeight::one(), 1);
        let mut arc2 = StdArc::new(1, 1, TropicalWeight::one(), 1);
        assert!(
            filter.filter_arc(&mut arc1, &mut arc2).is_some(),
            "with no dead end reported, the pair stands"
        );
        assert!(
            filter.last_lookahead().is_none(),
            "and nothing was looked at, so there is nothing to read back"
        );
    }

    /// The weight of what lies ahead is moved onto the arc, and taken back off
    /// again, so the path still weighs what it did.
    #[test]
    fn the_weight_ahead_is_pushed_and_then_divided_back_out() {
        // 0 -1:1/0-> 1 -2:2/5-> 2 against 0 -1:1/0-> 1 -2:2/3-> 2.
        let fst1 = build(3, &[(0, 1, 1, 0.0, 1), (1, 2, 2, 5.0, 2)]);
        let fst2 = build(3, &[(0, 1, 1, 0.0, 1), (1, 2, 2, 3.0, 2)]);
        // No prefix reporting: a single way forward would be reported as the
        // arc itself, and its weight left off so as not to count it twice.
        let (m1, m2) = matchers(&fst1, &fst2, LOOKAHEAD_NON_EPSILONS | LOOKAHEAD_WEIGHT);
        let inner = NullComposeFilter::new(m1, m2);
        let look = LookAheadComposeFilter::new(inner, &fst2).unwrap();
        let mut filter = PushWeightsComposeFilter::new(look);

        let start = filter.start();
        filter.set_state(0, 0, &start);
        let mut arc1 = StdArc::new(1, 1, TropicalWeight::one(), 1);
        let mut arc2 = StdArc::new(1, 1, TropicalWeight::one(), 1);
        let fs = filter
            .filter_arc(&mut arc1, &mut arc2)
            .expect("the pair leads somewhere");
        assert_eq!(
            arc2.weight(),
            &w(8.0),
            "everything past the pair weighs 5 on one side and 3 on the other"
        );
        assert_eq!(fs.state2().weight(), w(8.0), "and that is what was pushed");

        filter.set_state(1, 1, &fs);
        let mut arc1 = StdArc::new(2, 2, w(5.0), 2);
        let mut arc2 = StdArc::new(2, 2, w(3.0), 2);
        let fs = filter
            .filter_arc(&mut arc1, &mut arc2)
            .expect("the pair leads somewhere");
        assert_eq!(
            arc2.weight(),
            &w(-5.0),
            "what was pushed comes back off, leaving 8 + -5 = 3 = 5 + 3 - 5"
        );
        assert_eq!(fs.state2().weight(), TropicalWeight::one());
    }

    /// A pair the look-ahead refuses never reaches the weight arithmetic.
    #[test]
    fn a_pair_with_no_future_is_refused_before_any_arithmetic() {
        let fst1 = build(3, &[(0, 1, 1, 0.0, 1), (1, 2, 2, 5.0, 2)]);
        // Past label 1 this side reads 9, which the other side never writes.
        let fst2 = build(3, &[(0, 1, 1, 0.0, 1), (1, 9, 9, 0.0, 2)]);
        let (m1, m2) = matchers(&fst1, &fst2, LOOKAHEAD_NON_EPSILONS | LOOKAHEAD_WEIGHT);
        let inner = NullComposeFilter::new(m1, m2);
        let look = LookAheadComposeFilter::new(inner, &fst2).unwrap();
        let mut filter = PushWeightsComposeFilter::new(look);
        let start = filter.start();
        filter.set_state(0, 0, &start);
        let mut arc1 = StdArc::new(1, 1, TropicalWeight::one(), 1);
        let mut arc2 = StdArc::new(1, 1, TropicalWeight::one(), 1);
        assert!(filter.filter_arc(&mut arc1, &mut arc2).is_none());
        assert_eq!(
            arc2.weight(),
            &TropicalWeight::one(),
            "and the arc is left as it was"
        );
    }

    /// The whole stack: the one label that has to come next is written on the
    /// arc that reaches the pair, and the arc that would have written it turns
    /// into an epsilon.
    #[test]
    fn the_one_label_ahead_is_written_early_and_then_paid() {
        // 1 goes in and 5 comes out, but only after an epsilon step.
        let mut fst1 = StdVectorFst::new();
        for _ in 0..3 {
            fst1.add_state();
        }
        fst1.set_start(0);
        fst1.add_arc(0, StdArc::new(1, 0, TropicalWeight::one(), 1));
        fst1.add_arc(1, StdArc::new(0, 5, TropicalWeight::one(), 2));
        fst1.set_final(2, TropicalWeight::one());
        arc_sort(&mut fst1, &OLabelCompare);
        fst1.properties(K_FST_PROPERTIES, true);

        // 5 goes in and 7 comes out.
        let fst2 = build(2, &[(0, 5, 7, 0.0, 1)]);

        let m1 = MultiEpsMatcher::new_with_flags(&fst1, MatchType::Output, MultiEpsFlags::LIST)
            .map(|m: Eps<'_>| m)
            .unwrap();
        let m2 = MultiEpsMatcher::new_with_flags(&fst2, MatchType::Input, MultiEpsFlags::LOOP)
            .map(|m: Eps<'_>| m)
            .unwrap();
        let inner = NullComposeFilter::new(m1, m2);
        let look = LookAheadComposeFilter::new(inner, &fst2).unwrap();
        let mut filter = PushLabelsComposeFilter::new(look, &fst1);

        // Taking fst1's 1:ε against fst2 standing still: the look-ahead finds
        // that 5:7 is the only way on, so it is taken now.
        let start = filter.start();
        filter.set_state(0, 0, &start);
        let mut arc1 = StdArc::new(1, 0, TropicalWeight::one(), 1);
        let mut arc2 = StdArc::new(0, 0, TropicalWeight::one(), 0);
        let fs = filter
            .filter_arc(&mut arc1, &mut arc2)
            .expect("the pair leads somewhere");
        assert_eq!(
            (arc1.ilabel(), arc1.olabel()),
            (1, 5),
            "the 5 fst1 will only write later is written on this arc now"
        );
        assert_eq!(
            (arc2.ilabel(), arc2.olabel(), arc2.nextstate()),
            (5, 7, 1),
            "so fst2 goes forward on the arc that reads it"
        );
        assert_eq!(*fs.state2().label(), 5, "and 5 is what the filter now owes");

        // fst1's ε:5 comes round, and pays it.
        filter.set_state(1, 1, &fs);
        let mut arc1 = StdArc::new(0, 5, TropicalWeight::one(), 2);
        let mut arc2 = StdArc::new(-1, -1, TropicalWeight::one(), 1);
        let fs = filter
            .filter_arc(&mut arc1, &mut arc2)
            .expect("the debt is paid, so the pair is taken");
        assert_eq!(
            (arc1.ilabel(), arc1.olabel()),
            (0, 0),
            "the 5 was already written, so writing it again would be twice"
        );
        assert_eq!(*fs.state2().label(), -1, "and nothing is owed any more");
    }

    /// While a label is owed, an arc that writes something else cannot be
    /// taken, or the debt would be lost.
    #[test]
    fn an_arc_that_pays_something_else_is_refused() {
        let fst1 = build(3, &[(0, 1, 1, 0.0, 1), (1, 2, 2, 0.0, 2)]);
        let fst2 = build(2, &[(0, 5, 7, 0.0, 1)]);
        let m1: Eps<'_> =
            MultiEpsMatcher::new_with_flags(&fst1, MatchType::Output, MultiEpsFlags::LIST).unwrap();
        let m2: Eps<'_> =
            MultiEpsMatcher::new_with_flags(&fst2, MatchType::Input, MultiEpsFlags::LOOP).unwrap();
        let inner = NullComposeFilter::new(m1, m2);
        let look = LookAheadComposeFilter::new(inner, &fst2).unwrap();
        let mut filter = PushLabelsComposeFilter::new(look, &fst1);

        let owing = PairFilterState::new(
            crate::algorithms::filter_state::TrivialFilterState::new(true),
            LabelFilterState::new(5),
        );
        filter.set_state(1, 1, &owing);

        // fst1 writes 2 here, not the 5 that is owed.
        let mut arc1 = StdArc::new(2, 2, TropicalWeight::one(), 2);
        let mut arc2 = StdArc::new(-1, -1, TropicalWeight::one(), 1);
        assert!(filter.filter_arc(&mut arc1, &mut arc2).is_none());

        // Nor can a path stop while it still owes one.
        let mut w1 = TropicalWeight::one();
        let mut w2 = TropicalWeight::one();
        filter.filter_final(&mut w1, &mut w2);
        assert_eq!(w1, TropicalWeight::zero());
    }
}
