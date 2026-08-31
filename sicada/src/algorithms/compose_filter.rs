//! Filters deciding which arc pairs a composition may take.
//!
//! Port of OpenFst's `compose-filter.h`. Composing two FSTs that both have
//! epsilon arcs would otherwise produce one path per interleaving of the two
//! sides' epsilons; a filter picks exactly one, which keeps the result from
//! blowing up and from counting the same path many times over.

use crate::algorithms::filter_state::{STATE_DEFAULT, STATE_EPS1, STATE_EPS2};
use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::fst::Fst;
use crate::properties::{K_I_LABEL_INVARIANT_PROPERTIES, K_O_LABEL_INVARIANT_PROPERTIES};
use crate::weight::Weight;

pub use crate::algorithms::filter_state::{
    CharFilterState, FilterState, IntFilterState, IntegerFilterState, ListFilterState,
    PairFilterState, ShortFilterState, TrivialFilterState, WeightFilterState,
};

/// Composition filters determine which matches are allowed to proceed.
pub trait ComposeFilter {
    type Arc: Arc;
    type FilterState: FilterState;
    type Matcher1;
    type Matcher2;

    fn start(&self) -> Self::FilterState;

    fn set_state(
        &mut self,
        s1: <Self::Arc as Arc>::StateId,
        s2: <Self::Arc as Arc>::StateId,
        fs: &Self::FilterState,
    );

    /// Apply filter at current composition state.
    /// Returns `Some(new_state)` if allowed, or `None` if disallowed (NoState).
    fn filter_arc(
        &mut self,
        arc1: &mut Self::Arc,
        arc2: &mut Self::Arc,
    ) -> Option<Self::FilterState>;

    fn filter_final(
        &self,
        w1: &mut <Self::Arc as Arc>::Weight,
        w2: &mut <Self::Arc as Arc>::Weight,
    );

    fn matcher1(&self) -> &Self::Matcher1;
    fn matcher1_mut(&mut self) -> &mut Self::Matcher1;
    fn matcher2(&self) -> &Self::Matcher2;
    fn matcher2_mut(&mut self) -> &mut Self::Matcher2;

    fn properties(&self, props: u64) -> u64;
}

pub struct NullComposeFilter<A: Arc, M1, M2> {
    pub matcher1: M1,
    pub matcher2: M2,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Arc, M1, M2> NullComposeFilter<A, M1, M2> {
    pub fn new(matcher1: M1, matcher2: M2) -> Self {
        Self {
            matcher1,
            matcher2,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A: Arc, M1, M2> ComposeFilter for NullComposeFilter<A, M1, M2> {
    type Arc = A;
    type FilterState = TrivialFilterState;
    type Matcher1 = M1;
    type Matcher2 = M2;

    #[inline]
    fn start(&self) -> Self::FilterState {
        TrivialFilterState::new(true)
    }

    #[inline]
    fn set_state(&mut self, _s1: A::StateId, _s2: A::StateId, _fs: &Self::FilterState) {}

    #[inline]
    fn filter_arc(&mut self, arc1: &mut A, arc2: &mut A) -> Option<Self::FilterState> {
        if arc1.olabel() == A::Label::no_label() || arc2.ilabel() == A::Label::no_label() {
            None
        } else {
            Some(TrivialFilterState::new(true))
        }
    }

    #[inline]
    fn filter_final(&self, _w1: &mut A::Weight, _w2: &mut A::Weight) {}

    #[inline]
    fn matcher1(&self) -> &M1 {
        &self.matcher1
    }
    #[inline]
    fn matcher1_mut(&mut self) -> &mut M1 {
        &mut self.matcher1
    }
    #[inline]
    fn matcher2(&self) -> &M2 {
        &self.matcher2
    }
    #[inline]
    fn matcher2_mut(&mut self) -> &mut M2 {
        &mut self.matcher2
    }

    #[inline]
    fn properties(&self, props: u64) -> u64 {
        props
    }
}

pub struct TrivialComposeFilter<A: Arc, M1, M2> {
    pub matcher1: M1,
    pub matcher2: M2,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Arc, M1, M2> TrivialComposeFilter<A, M1, M2> {
    pub fn new(matcher1: M1, matcher2: M2) -> Self {
        Self {
            matcher1,
            matcher2,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A: Arc, M1, M2> ComposeFilter for TrivialComposeFilter<A, M1, M2> {
    type Arc = A;
    type FilterState = TrivialFilterState;
    type Matcher1 = M1;
    type Matcher2 = M2;

    #[inline]
    fn start(&self) -> Self::FilterState {
        TrivialFilterState::new(true)
    }

    #[inline]
    fn set_state(&mut self, _s1: A::StateId, _s2: A::StateId, _fs: &Self::FilterState) {}

    #[inline]
    fn filter_arc(&mut self, _arc1: &mut A, _arc2: &mut A) -> Option<Self::FilterState> {
        Some(TrivialFilterState::new(true))
    }

    #[inline]
    fn filter_final(&self, _w1: &mut A::Weight, _w2: &mut A::Weight) {}

    #[inline]
    fn matcher1(&self) -> &M1 {
        &self.matcher1
    }
    #[inline]
    fn matcher1_mut(&mut self) -> &mut M1 {
        &mut self.matcher1
    }
    #[inline]
    fn matcher2(&self) -> &M2 {
        &self.matcher2
    }
    #[inline]
    fn matcher2_mut(&mut self) -> &mut M2 {
        &mut self.matcher2
    }

    #[inline]
    fn properties(&self, props: u64) -> u64 {
        props
    }
}

pub struct SequenceComposeFilter<'a, A: Arc, F1: Fst<A>, M1, M2> {
    pub matcher1: M1,
    pub matcher2: M2,
    fst1: &'a F1,
    s1: A::StateId,
    s2: A::StateId,
    fs: CharFilterState,
    alleps1: bool,
    noeps1: bool,
}

impl<'a, A: Arc, F1: Fst<A>, M1, M2> SequenceComposeFilter<'a, A, F1, M1, M2> {
    pub fn new(fst1: &'a F1, matcher1: M1, matcher2: M2) -> Self {
        Self {
            matcher1,
            matcher2,
            fst1,
            s1: A::StateId::no_state(),
            s2: A::StateId::no_state(),
            fs: CharFilterState::new(STATE_DEFAULT),
            alleps1: false,
            noeps1: false,
        }
    }
}

impl<'a, A: Arc, F1: Fst<A>, M1, M2> ComposeFilter for SequenceComposeFilter<'a, A, F1, M1, M2> {
    type Arc = A;
    type FilterState = CharFilterState;
    type Matcher1 = M1;
    type Matcher2 = M2;

    #[inline]
    fn start(&self) -> Self::FilterState {
        CharFilterState::new(STATE_DEFAULT)
    }

    #[inline]
    fn set_state(&mut self, s1: A::StateId, s2: A::StateId, fs: &Self::FilterState) {
        if self.s1 == s1 && self.s2 == s2 && &self.fs == fs {
            return;
        }
        self.s1 = s1;
        self.s2 = s2;
        self.fs = fs.clone();

        let na1 = self.fst1.num_arcs(s1);
        let ne1 = self.fst1.num_output_epsilons(s1);
        let fin1 = self.fst1.final_weight(s1) != A::Weight::zero();

        self.alleps1 = (na1 == ne1) && !fin1;
        self.noeps1 = ne1 == 0;
    }

    #[inline]
    fn filter_arc(&mut self, arc1: &mut A, arc2: &mut A) -> Option<Self::FilterState> {
        if arc1.olabel() == A::Label::no_label() {
            if self.alleps1 {
                None
            } else if self.noeps1 {
                Some(CharFilterState::new(STATE_DEFAULT))
            } else {
                Some(CharFilterState::new(STATE_EPS1))
            }
        } else if arc2.ilabel() == A::Label::no_label() {
            if self.fs.state_copied() != STATE_DEFAULT {
                None
            } else {
                Some(CharFilterState::new(STATE_DEFAULT))
            }
        } else if arc1.olabel() == A::Label::epsilon() {
            None
        } else {
            Some(CharFilterState::new(STATE_DEFAULT))
        }
    }

    #[inline]
    fn filter_final(&self, _w1: &mut A::Weight, _w2: &mut A::Weight) {}

    #[inline]
    fn matcher1(&self) -> &M1 {
        &self.matcher1
    }
    #[inline]
    fn matcher1_mut(&mut self) -> &mut M1 {
        &mut self.matcher1
    }
    #[inline]
    fn matcher2(&self) -> &M2 {
        &self.matcher2
    }
    #[inline]
    fn matcher2_mut(&mut self) -> &mut M2 {
        &mut self.matcher2
    }

    #[inline]
    fn properties(&self, props: u64) -> u64 {
        props
    }
}

pub struct AltSequenceComposeFilter<'a, A: Arc, F2: Fst<A>, M1, M2> {
    pub matcher1: M1,
    pub matcher2: M2,
    fst2: &'a F2,
    s1: A::StateId,
    s2: A::StateId,
    fs: CharFilterState,
    alleps2: bool,
    noeps2: bool,
}

impl<'a, A: Arc, F2: Fst<A>, M1, M2> AltSequenceComposeFilter<'a, A, F2, M1, M2> {
    pub fn new(fst2: &'a F2, matcher1: M1, matcher2: M2) -> Self {
        Self {
            matcher1,
            matcher2,
            fst2,
            s1: A::StateId::no_state(),
            s2: A::StateId::no_state(),
            fs: CharFilterState::new(STATE_DEFAULT),
            alleps2: false,
            noeps2: false,
        }
    }
}

impl<'a, A: Arc, F2: Fst<A>, M1, M2> ComposeFilter for AltSequenceComposeFilter<'a, A, F2, M1, M2> {
    type Arc = A;
    type FilterState = CharFilterState;
    type Matcher1 = M1;
    type Matcher2 = M2;

    #[inline]
    fn start(&self) -> Self::FilterState {
        CharFilterState::new(STATE_DEFAULT)
    }

    #[inline]
    fn set_state(&mut self, s1: A::StateId, s2: A::StateId, fs: &Self::FilterState) {
        if self.s1 == s1 && self.s2 == s2 && &self.fs == fs {
            return;
        }
        self.s1 = s1;
        self.s2 = s2;
        self.fs = fs.clone();

        let na2 = self.fst2.num_arcs(s2);
        let ne2 = self.fst2.num_input_epsilons(s2);
        let fin2 = self.fst2.final_weight(s2) != A::Weight::zero();

        self.alleps2 = (na2 == ne2) && !fin2;
        self.noeps2 = ne2 == 0;
    }

    #[inline]
    fn filter_arc(&mut self, arc1: &mut A, arc2: &mut A) -> Option<Self::FilterState> {
        if arc2.ilabel() == A::Label::no_label() {
            if self.alleps2 {
                None
            } else if self.noeps2 {
                Some(CharFilterState::new(STATE_DEFAULT))
            } else {
                Some(CharFilterState::new(STATE_EPS1))
            }
        } else if arc1.olabel() == A::Label::no_label() {
            if self.fs.state_copied() == STATE_EPS1 {
                None
            } else {
                Some(CharFilterState::new(STATE_DEFAULT))
            }
        } else if arc1.olabel() == A::Label::epsilon() {
            None
        } else {
            Some(CharFilterState::new(STATE_DEFAULT))
        }
    }

    #[inline]
    fn filter_final(&self, _w1: &mut A::Weight, _w2: &mut A::Weight) {}

    #[inline]
    fn matcher1(&self) -> &M1 {
        &self.matcher1
    }
    #[inline]
    fn matcher1_mut(&mut self) -> &mut M1 {
        &mut self.matcher1
    }
    #[inline]
    fn matcher2(&self) -> &M2 {
        &self.matcher2
    }
    #[inline]
    fn matcher2_mut(&mut self) -> &mut M2 {
        &mut self.matcher2
    }

    #[inline]
    fn properties(&self, props: u64) -> u64 {
        props
    }
}

pub struct MatchComposeFilter<'a, A: Arc, F1: Fst<A>, F2: Fst<A>, M1, M2> {
    pub matcher1: M1,
    pub matcher2: M2,
    fst1: &'a F1,
    fst2: &'a F2,
    s1: A::StateId,
    s2: A::StateId,
    fs: CharFilterState,
    alleps1: bool,
    alleps2: bool,
    noeps1: bool,
    noeps2: bool,
}

impl<'a, A: Arc, F1: Fst<A>, F2: Fst<A>, M1, M2> MatchComposeFilter<'a, A, F1, F2, M1, M2> {
    pub fn new(fst1: &'a F1, fst2: &'a F2, matcher1: M1, matcher2: M2) -> Self {
        Self {
            matcher1,
            matcher2,
            fst1,
            fst2,
            s1: A::StateId::no_state(),
            s2: A::StateId::no_state(),
            fs: CharFilterState::new(STATE_DEFAULT),
            alleps1: false,
            alleps2: false,
            noeps1: false,
            noeps2: false,
        }
    }
}

impl<'a, A: Arc, F1: Fst<A>, F2: Fst<A>, M1, M2> ComposeFilter
    for MatchComposeFilter<'a, A, F1, F2, M1, M2>
{
    type Arc = A;
    type FilterState = CharFilterState;
    type Matcher1 = M1;
    type Matcher2 = M2;

    #[inline]
    fn start(&self) -> Self::FilterState {
        CharFilterState::new(STATE_DEFAULT)
    }

    #[inline]
    fn set_state(&mut self, s1: A::StateId, s2: A::StateId, fs: &Self::FilterState) {
        if self.s1 == s1 && self.s2 == s2 && &self.fs == fs {
            return;
        }
        self.s1 = s1;
        self.s2 = s2;
        self.fs = fs.clone();

        let na1 = self.fst1.num_arcs(s1);
        let ne1 = self.fst1.num_output_epsilons(s1);
        let f1 = self.fst1.final_weight(s1) != A::Weight::zero();
        self.alleps1 = (na1 == ne1) && !f1;
        self.noeps1 = ne1 == 0;

        let na2 = self.fst2.num_arcs(s2);
        let ne2 = self.fst2.num_input_epsilons(s2);
        let f2 = self.fst2.final_weight(s2) != A::Weight::zero();
        self.alleps2 = (na2 == ne2) && !f2;
        self.noeps2 = ne2 == 0;
    }

    #[inline]
    fn filter_arc(&mut self, arc1: &mut A, arc2: &mut A) -> Option<Self::FilterState> {
        let state = self.fs.state_copied();
        if arc2.ilabel() == A::Label::no_label() {
            // Epsilon in FST1
            if state == STATE_DEFAULT {
                if self.noeps2 {
                    Some(CharFilterState::new(STATE_DEFAULT))
                } else if self.alleps2 {
                    None
                } else {
                    Some(CharFilterState::new(STATE_EPS1))
                }
            } else if state == STATE_EPS1 {
                Some(CharFilterState::new(STATE_EPS1))
            } else {
                None
            }
        } else if arc1.olabel() == A::Label::no_label() {
            // Epsilon in FST2
            if state == STATE_DEFAULT {
                if self.noeps1 {
                    Some(CharFilterState::new(STATE_DEFAULT))
                } else if self.alleps1 {
                    None
                } else {
                    Some(CharFilterState::new(STATE_EPS2))
                }
            } else if state == STATE_EPS2 {
                Some(CharFilterState::new(STATE_EPS2))
            } else {
                None
            }
        } else if arc1.olabel() == A::Label::epsilon() {
            // Epsilon in both
            if state == STATE_DEFAULT {
                Some(CharFilterState::new(STATE_DEFAULT))
            } else {
                None
            }
        } else {
            // Both are non-epsilons
            Some(CharFilterState::new(STATE_DEFAULT))
        }
    }

    #[inline]
    fn filter_final(&self, _w1: &mut A::Weight, _w2: &mut A::Weight) {}

    #[inline]
    fn matcher1(&self) -> &M1 {
        &self.matcher1
    }
    #[inline]
    fn matcher1_mut(&mut self) -> &mut M1 {
        &mut self.matcher1
    }
    #[inline]
    fn matcher2(&self) -> &M2 {
        &self.matcher2
    }
    #[inline]
    fn matcher2_mut(&mut self) -> &mut M2 {
        &mut self.matcher2
    }

    #[inline]
    fn properties(&self, props: u64) -> u64 {
        props
    }
}

pub struct NoMatchComposeFilter<A: Arc, M1, M2> {
    pub matcher1: M1,
    pub matcher2: M2,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Arc, M1, M2> NoMatchComposeFilter<A, M1, M2> {
    pub fn new(matcher1: M1, matcher2: M2) -> Self {
        Self {
            matcher1,
            matcher2,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A: Arc, M1, M2> ComposeFilter for NoMatchComposeFilter<A, M1, M2> {
    type Arc = A;
    type FilterState = TrivialFilterState;
    type Matcher1 = M1;
    type Matcher2 = M2;

    #[inline]
    fn start(&self) -> Self::FilterState {
        TrivialFilterState::new(true)
    }

    #[inline]
    fn set_state(&mut self, _s1: A::StateId, _s2: A::StateId, _fs: &Self::FilterState) {}

    #[inline]
    fn filter_arc(&mut self, arc1: &mut A, arc2: &mut A) -> Option<Self::FilterState> {
        if arc1.olabel() != A::Label::epsilon() || arc2.ilabel() != A::Label::epsilon() {
            Some(TrivialFilterState::new(true))
        } else {
            None
        }
    }

    #[inline]
    fn filter_final(&self, _w1: &mut A::Weight, _w2: &mut A::Weight) {}

    #[inline]
    fn matcher1(&self) -> &M1 {
        &self.matcher1
    }
    #[inline]
    fn matcher1_mut(&mut self) -> &mut M1 {
        &mut self.matcher1
    }
    #[inline]
    fn matcher2(&self) -> &M2 {
        &self.matcher2
    }
    #[inline]
    fn matcher2_mut(&mut self) -> &mut M2 {
        &mut self.matcher2
    }

    #[inline]
    fn properties(&self, props: u64) -> u64 {
        props
    }
}

pub struct MultiEpsFilter<F> {
    pub filter: F,
    pub keep_multi_eps: bool,
}

impl<F: ComposeFilter> MultiEpsFilter<F> {
    pub fn new(filter: F, keep_multi_eps: bool) -> Self {
        Self {
            filter,
            keep_multi_eps,
        }
    }
}

impl<F: ComposeFilter> ComposeFilter for MultiEpsFilter<F> {
    type Arc = F::Arc;
    type FilterState = F::FilterState;
    type Matcher1 = F::Matcher1;
    type Matcher2 = F::Matcher2;

    #[inline]
    fn start(&self) -> Self::FilterState {
        self.filter.start()
    }

    #[inline]
    fn set_state(
        &mut self,
        s1: <Self::Arc as Arc>::StateId,
        s2: <Self::Arc as Arc>::StateId,
        fs: &Self::FilterState,
    ) {
        self.filter.set_state(s1, s2, fs);
    }

    #[inline]
    fn filter_arc(
        &mut self,
        arc1: &mut Self::Arc,
        arc2: &mut Self::Arc,
    ) -> Option<Self::FilterState> {
        let fs = self.filter.filter_arc(arc1, arc2);
        if self.keep_multi_eps {
            if arc1.olabel() == <Self::Arc as Arc>::Label::no_label() {
                *arc1 = <Self::Arc as Arc>::new(
                    arc2.ilabel(),
                    arc1.olabel(),
                    arc1.weight().clone(),
                    arc1.nextstate(),
                );
            }
            if arc2.ilabel() == <Self::Arc as Arc>::Label::no_label() {
                *arc2 = <Self::Arc as Arc>::new(
                    arc2.ilabel(),
                    arc1.olabel(),
                    arc2.weight().clone(),
                    arc2.nextstate(),
                );
            }
        }
        fs
    }

    #[inline]
    fn filter_final(
        &self,
        w1: &mut <Self::Arc as Arc>::Weight,
        w2: &mut <Self::Arc as Arc>::Weight,
    ) {
        self.filter.filter_final(w1, w2);
    }

    #[inline]
    fn matcher1(&self) -> &Self::Matcher1 {
        self.filter.matcher1()
    }
    #[inline]
    fn matcher1_mut(&mut self) -> &mut Self::Matcher1 {
        self.filter.matcher1_mut()
    }
    #[inline]
    fn matcher2(&self) -> &Self::Matcher2 {
        self.filter.matcher2()
    }
    #[inline]
    fn matcher2_mut(&mut self) -> &mut Self::Matcher2 {
        self.filter.matcher2_mut()
    }

    #[inline]
    fn properties(&self, props: u64) -> u64 {
        let oprops = self.filter.properties(props);
        oprops & K_I_LABEL_INVARIANT_PROPERTIES & K_O_LABEL_INVARIANT_PROPERTIES
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AtomicRc;
    use crate::arc::StdArc;
    use crate::fst::Fst;
    use crate::symbol_table::SymbolTable;
    use crate::weights::float_weight::TropicalWeight;
    use std::iter::Empty;

    /// Stands in for an input FST: a filter's `set_state` reads exactly these
    /// three quantities, and nothing else about it matters.
    struct StubFst {
        num_arcs: usize,
        num_input_epsilons: usize,
        num_output_epsilons: usize,
        is_final: bool,
    }

    impl Fst<StdArc> for StubFst {
        type StateIter<'a> = Empty<i32>;
        type ArcIter<'a> = Empty<StdArc>;

        fn start(&self) -> Option<i32> {
            Some(0)
        }

        fn final_weight(&self, _state: i32) -> TropicalWeight {
            if self.is_final {
                TropicalWeight::one()
            } else {
                TropicalWeight::zero()
            }
        }

        fn num_arcs(&self, _state: i32) -> usize {
            self.num_arcs
        }

        fn num_input_epsilons(&self, _state: i32) -> usize {
            self.num_input_epsilons
        }

        fn num_output_epsilons(&self, _state: i32) -> usize {
            self.num_output_epsilons
        }

        fn num_states_if_known(&self) -> Option<usize> {
            Some(1)
        }

        fn properties(&self, _mask: u64, _test: bool) -> u64 {
            0
        }

        fn fst_type(&self) -> &str {
            "stub"
        }

        fn input_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
            None
        }

        fn output_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
            None
        }

        fn states<'a>(&'a self) -> Self::StateIter<'a> {
            std::iter::empty()
        }

        fn arcs<'a>(&'a self, _state: i32) -> Self::ArcIter<'a> {
            std::iter::empty()
        }
    }

    /// The labels a filter distinguishes: "this side did not consume a symbol",
    /// epsilon, and an ordinary symbol.
    const LABELS: [i32; 3] = [-1, 0, 5];

    /// The four `(alleps, noeps)` combinations, as the state shapes producing
    /// them: `alleps` is `num_arcs == num_epsilons && !final`, `noeps` is
    /// `num_epsilons == 0`.
    const SHAPES: [(usize, usize); 4] = [(0, 0), (1, 1), (1, 0), (2, 1)];

    fn side1(shape: (usize, usize)) -> StubFst {
        StubFst {
            num_arcs: shape.0,
            num_input_epsilons: 0,
            num_output_epsilons: shape.1,
            is_final: false,
        }
    }

    fn side2(shape: (usize, usize)) -> StubFst {
        StubFst {
            num_arcs: shape.0,
            num_input_epsilons: shape.1,
            num_output_epsilons: 0,
            is_final: false,
        }
    }

    fn arcs(olabel1: i32, ilabel2: i32) -> (StdArc, StdArc) {
        (
            StdArc::new(0, olabel1, TropicalWeight::one(), 1),
            StdArc::new(ilabel2, 0, TropicalWeight::one(), 1),
        )
    }

    fn decide(fs: Option<CharFilterState>) -> char {
        match fs {
            None => '-',
            Some(fs) => (b'0' + fs.state_copied() as u8) as char,
        }
    }

    /// Upstream's `TrivialFilterState::NoState()` *is* `TrivialFilterState(false)`,
    /// so a filter that returns the false state is refusing the arc.
    fn decide_trivial(fs: Option<TrivialFilterState>) -> char {
        match fs {
            Some(fs) if fs.state() => '1',
            _ => '-',
        }
    }

    /// Walks the label and filter-state combinations for one configured filter.
    fn sweep<F: ComposeFilter<Arc = StdArc, FilterState = CharFilterState>>(
        out: &mut String,
        make: impl Fn() -> F,
    ) {
        for fs in [STATE_DEFAULT, STATE_EPS1, STATE_EPS2] {
            for olabel1 in LABELS {
                for ilabel2 in LABELS {
                    let mut filter = make();
                    filter.set_state(0, 0, &CharFilterState::new(fs));
                    let (mut arc1, mut arc2) = arcs(olabel1, ilabel2);
                    out.push(decide(filter.filter_arc(&mut arc1, &mut arc2)));
                }
            }
        }
    }

    /// Every filter's whole decision table, checked against the table the
    /// upstream classes produce.
    ///
    /// The pinned strings come from `tests/oracles/compose-filter-decisions.cc`,
    /// which is a verbatim extraction of the five filters driven over the same
    /// inputs in the same order. These are the tables that decide which of the
    /// many interleavings of two FSTs' epsilons composition keeps, so getting
    /// one cell wrong duplicates paths or drops them.
    #[test]
    fn the_decision_tables_match_openfst() {
        let mut sequence = String::new();
        for shape in SHAPES {
            let fst1 = side1(shape);
            sweep(&mut sequence, || {
                SequenceComposeFilter::<StdArc, _, (), ()>::new(&fst1, (), ())
            });
        }
        assert_eq!(
            sequence,
            "---0--000-------00-------00---0--000-------00-------000000--000000----00000----001110--000111----00111----00"
        );

        let mut altsequence = String::new();
        for shape in SHAPES {
            let fst2 = side2(shape);
            sweep(&mut altsequence, || {
                AltSequenceComposeFilter::<StdArc, _, (), ()>::new(&fst2, (), ())
            });
        }
        assert_eq!(
            altsequence,
            "-00----00-------00-00----00-00----00-------00-00----000000--0000--0--0000000--0001001--1001--1--1001001--100"
        );

        let mut matching = String::new();
        for shape1 in SHAPES {
            for shape2 in SHAPES {
                let fst1 = side1(shape1);
                let fst2 = side2(shape2);
                sweep(&mut matching, || {
                    MatchComposeFilter::<StdArc, _, _, (), ()>::new(&fst1, &fst2, (), ())
                });
            }
        }
        assert_eq!(
            matching,
            "0000000001--1--100-22----00-00-00-001--1--100-22----000000000001--1--100-22----001001001001--1--100-22----000--0000001--1--100-22----00----00-001--1--100-22----000--0000001--1--100-22----001--1001001--1--100-22----000000000001--1--100-22----00-00-00-001--1--100-22----000000000001--1--100-22----001001001001--1--100-22----000220000001--1--100-22----00-22-00-001--1--100-22----000220000001--1--100-22----001221001001--1--100-22----00"
        );

        let stateless = |mut decide: Box<dyn FnMut(i32, i32) -> char>| {
            let mut out = String::new();
            for olabel1 in LABELS {
                for ilabel2 in LABELS {
                    out.push(decide(olabel1, ilabel2));
                }
            }
            out
        };

        let mut nomatch = NoMatchComposeFilter::<StdArc, (), ()>::new((), ());
        assert_eq!(
            stateless(Box::new(|olabel1, ilabel2| {
                let (mut arc1, mut arc2) = arcs(olabel1, ilabel2);
                decide_trivial(nomatch.filter_arc(&mut arc1, &mut arc2))
            })),
            "1111-1111"
        );

        let mut null = NullComposeFilter::<StdArc, (), ()>::new((), ());
        assert_eq!(
            stateless(Box::new(|olabel1, ilabel2| {
                let (mut arc1, mut arc2) = arcs(olabel1, ilabel2);
                decide_trivial(null.filter_arc(&mut arc1, &mut arc2))
            })),
            "----11-11"
        );

        let mut trivial = TrivialComposeFilter::<StdArc, (), ()>::new((), ());
        assert_eq!(
            stateless(Box::new(|olabel1, ilabel2| {
                let (mut arc1, mut arc2) = arcs(olabel1, ilabel2);
                decide_trivial(trivial.filter_arc(&mut arc1, &mut arc2))
            })),
            "111111111"
        );
    }

    /// Every filter starts in the state its own table is indexed by.
    #[test]
    fn every_filter_starts_where_its_table_begins() {
        let fst = side1((1, 0));
        assert_eq!(
            SequenceComposeFilter::<StdArc, _, (), ()>::new(&fst, (), ()).start(),
            CharFilterState::new(STATE_DEFAULT)
        );
        assert_eq!(
            AltSequenceComposeFilter::<StdArc, _, (), ()>::new(&fst, (), ()).start(),
            CharFilterState::new(STATE_DEFAULT)
        );
        assert_eq!(
            MatchComposeFilter::<StdArc, _, _, (), ()>::new(&fst, &fst, (), ()).start(),
            CharFilterState::new(STATE_DEFAULT)
        );
        assert!(
            NullComposeFilter::<StdArc, (), ()>::new((), ())
                .start()
                .state()
        );
        assert!(
            NoMatchComposeFilter::<StdArc, (), ()>::new((), ())
                .start()
                .state()
        );
        assert!(
            TrivialComposeFilter::<StdArc, (), ()>::new((), ())
                .start()
                .state()
        );
    }

    /// `set_state` recomputes only when the composition state actually moved;
    /// the early return keeps a filter from re-reading the FSTs on every arc of
    /// a state.
    #[test]
    fn set_state_is_idempotent() {
        let fst1 = side1((2, 1));
        let mut filter = SequenceComposeFilter::<StdArc, _, (), ()>::new(&fst1, (), ());

        filter.set_state(0, 0, &CharFilterState::new(STATE_DEFAULT));
        let (mut arc1, mut arc2) = arcs(-1, 5);
        let first = filter.filter_arc(&mut arc1, &mut arc2);

        filter.set_state(0, 0, &CharFilterState::new(STATE_DEFAULT));
        let (mut arc1, mut arc2) = arcs(-1, 5);
        assert_eq!(filter.filter_arc(&mut arc1, &mut arc2), first);
    }

    /// With multi-epsilons kept, a side that did not consume a symbol takes the
    /// other side's label rather than being rewritten to epsilon.
    #[test]
    fn multi_eps_copies_the_matched_label_across() {
        let mut filter =
            MultiEpsFilter::new(TrivialComposeFilter::<StdArc, (), ()>::new((), ()), true);

        // Side 1 did not consume: its input label becomes side 2's.
        let mut arc1 = StdArc::new(5, -1, TropicalWeight::one(), 1);
        let mut arc2 = StdArc::new(7, 8, TropicalWeight::one(), 2);
        filter.filter_arc(&mut arc1, &mut arc2);
        assert_eq!(arc1.ilabel(), 7);
        assert_eq!(arc1.olabel(), -1, "the marker itself is left alone");
        assert_eq!(
            (arc2.ilabel(), arc2.olabel()),
            (7, 8),
            "side 2 is untouched"
        );

        // Side 2 did not consume: its output label becomes side 1's.
        let mut arc1 = StdArc::new(5, 9, TropicalWeight::one(), 1);
        let mut arc2 = StdArc::new(-1, 8, TropicalWeight::one(), 2);
        filter.filter_arc(&mut arc1, &mut arc2);
        assert_eq!(arc2.olabel(), 9);
        assert_eq!(arc2.ilabel(), -1, "the marker itself is left alone");
        assert_eq!(
            (arc1.ilabel(), arc1.olabel()),
            (5, 9),
            "side 1 is untouched"
        );
    }

    #[test]
    fn multi_eps_leaves_labels_alone_when_it_is_told_not_to_keep_them() {
        let mut filter =
            MultiEpsFilter::new(TrivialComposeFilter::<StdArc, (), ()>::new((), ()), false);
        let mut arc1 = StdArc::new(5, -1, TropicalWeight::one(), 1);
        let mut arc2 = StdArc::new(7, 8, TropicalWeight::one(), 2);
        filter.filter_arc(&mut arc1, &mut arc2);
        assert_eq!((arc1.ilabel(), arc1.olabel()), (5, -1));
        assert_eq!((arc2.ilabel(), arc2.olabel()), (7, 8));
    }

    /// Wrapping a filter narrows the properties composition may claim: a
    /// multi-epsilon that survives into the output is a label the filter put
    /// there, so nothing label-dependent carries over.
    #[test]
    fn multi_eps_keeps_only_label_invariant_properties() {
        let filter = MultiEpsFilter::new(TrivialComposeFilter::<StdArc, (), ()>::new((), ()), true);
        let all = u64::MAX;
        assert_eq!(
            filter.properties(all),
            K_I_LABEL_INVARIANT_PROPERTIES & K_O_LABEL_INVARIANT_PROPERTIES
        );
    }
}
