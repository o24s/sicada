use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc as StdArc;

use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::error::OpenFstError;
use crate::fst::{ContiguousArcsFst, Fst, MatchType};
use crate::properties::K_ACCEPTOR;
use crate::weight::Weight;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatcherRewriteMode {
    /// Rewrite both sides exactly when the FST is an acceptor.
    Auto = 0,
    /// Always rewrite both sides.
    Always,
    /// Rewrite only the side being matched.
    Never,
}

impl MatcherRewriteMode {
    /// Whether an implicit-label match should rewrite both sides of the arc.
    fn rewrites_both<A: Arc, F: Fst<A>>(self, fst: &F) -> bool {
        match self {
            Self::Auto => (fst.properties(K_ACCEPTOR, true) & K_ACCEPTOR) != 0,
            Self::Always => true,
            Self::Never => false,
        }
    }
}

/// A trait for finding and iterating through requested labels at FST states.
pub trait Matcher<'f, A: Arc>: Clone {
    type Fst: Fst<A>;

    fn new(fst: &'f Self::Fst, match_type: MatchType) -> Result<Self, OpenFstError>
    where
        Self: Sized;

    fn match_type(&self) -> MatchType;
    fn set_state(&mut self, state: A::StateId);
    fn find(&mut self, label: A::Label) -> bool;
    fn done(&self) -> bool;
    fn value(&self) -> A;
    fn next(&mut self);
    fn priority(&mut self, state: A::StateId) -> isize;

    /// Helper method to use the Matcher as a standard Rust Iterator.
    fn iter(&mut self, label: A::Label) -> MatcherIter<'_, 'f, A, Self>
    where
        Self: Sized,
    {
        let has_match = self.find(label);
        MatcherIter {
            matcher: self,
            has_match,
            _marker: PhantomData,
        }
    }
}

/// A zero-cost iterator wrapper over a stateful Matcher.
pub struct MatcherIter<'m, 'f, A, M> {
    matcher: &'m mut M,
    has_match: bool,
    _marker: PhantomData<&'f A>,
}

impl<'m, 'f, A: Arc, M: Matcher<'f, A>> Iterator for MatcherIter<'m, 'f, A, M> {
    type Item = A;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if !self.has_match || self.matcher.done() {
            None
        } else {
            let arc = self.matcher.value();
            self.matcher.next();
            Some(arc)
        }
    }
}

pub struct SortedMatcher<'f, F, A>
where
    A: Arc,
    F: ContiguousArcsFst<A>,
{
    fst: &'f F,
    match_type: MatchType,
    state: Option<A::StateId>,
    arcs: &'f [A],
    pos: usize,
    end: usize,
    loop_arc: A,
    current_loop: bool,
    match_label: A::Label,
    exact_match: bool,
    error: bool,
}

impl<'f, F, A> Clone for SortedMatcher<'f, F, A>
where
    A: Arc,
    F: ContiguousArcsFst<A>,
{
    fn clone(&self) -> Self {
        Self {
            fst: self.fst,
            match_type: self.match_type,
            state: self.state,
            arcs: self.arcs,
            pos: self.pos,
            end: self.end,
            loop_arc: self.loop_arc.clone(),
            current_loop: self.current_loop,
            match_label: self.match_label,
            exact_match: self.exact_match,
            error: self.error,
        }
    }
}

impl<'f, F, A> Matcher<'f, A> for SortedMatcher<'f, F, A>
where
    A: Arc,
    F: ContiguousArcsFst<A>,
{
    type Fst = F;

    fn new(fst: &'f F, match_type: MatchType) -> Result<Self, OpenFstError> {
        if match_type != MatchType::Input && match_type != MatchType::Output {
            return Err(OpenFstError::MatcherInvalidMatchType {
                matcher_name: "SortedMatcher",
                match_type,
            });
        }

        let loop_arc = if match_type == MatchType::Input {
            A::new(
                A::Label::no_label(),
                A::Label::epsilon(),
                A::Weight::one(),
                A::StateId::no_state(),
            )
        } else {
            A::new(
                A::Label::epsilon(),
                A::Label::no_label(),
                A::Weight::one(),
                A::StateId::no_state(),
            )
        };

        Ok(Self {
            fst,
            match_type,
            state: None,
            arcs: &[],
            pos: 0,
            end: 0,
            loop_arc,
            current_loop: false,
            match_label: A::Label::no_label(),
            exact_match: true,
            error: false,
        })
    }

    #[inline(always)]
    fn match_type(&self) -> MatchType {
        self.match_type
    }

    fn set_state(&mut self, state: A::StateId) {
        if self.state == Some(state) {
            return;
        }
        self.state = Some(state);
        self.arcs = self.fst.arcs_slice(state);
        self.loop_arc = A::new(
            self.loop_arc.ilabel(),
            self.loop_arc.olabel(),
            self.loop_arc.weight().clone(),
            state,
        );
    }

    fn find(&mut self, label: A::Label) -> bool {
        self.exact_match = true;
        if self.error {
            self.current_loop = false;
            self.match_label = A::Label::no_label();
            return false;
        }

        self.current_loop = label == A::Label::epsilon();
        self.match_label = if label == A::Label::no_label() {
            A::Label::epsilon()
        } else {
            label
        };

        self.search();
        self.pos < self.end || self.current_loop
    }

    #[inline(always)]
    fn done(&self) -> bool {
        if self.current_loop {
            return false;
        }
        if self.pos >= self.end {
            return true;
        }
        if !self.exact_match {
            return false;
        }
        let l = if self.match_type == MatchType::Input {
            self.arcs[self.pos].ilabel()
        } else {
            self.arcs[self.pos].olabel()
        };
        l != self.match_label
    }

    #[inline(always)]
    fn value(&self) -> A {
        if self.current_loop {
            self.loop_arc.clone()
        } else {
            self.arcs[self.pos].clone()
        }
    }

    #[inline(always)]
    fn next(&mut self) {
        if self.current_loop {
            self.current_loop = false;
        } else {
            self.pos += 1;
        }
    }

    fn priority(&mut self, state: A::StateId) -> isize {
        self.fst.num_arcs(state) as isize
    }
}

impl<'f, F, A> SortedMatcher<'f, F, A>
where
    A: Arc,
    F: ContiguousArcsFst<A>,
{
    fn search(&mut self) {
        let extract_label = if self.match_type == MatchType::Input {
            A::ilabel
        } else {
            A::olabel
        };

        let start = self
            .arcs
            .partition_point(|arc| extract_label(arc) < self.match_label);

        if self.exact_match {
            let end = start
                + self.arcs[start..].partition_point(|arc| extract_label(arc) <= self.match_label);
            self.pos = start;
            self.end = end;
        } else {
            self.pos = start;
            self.end = self.arcs.len();
        }
    }

    pub fn lower_bound(&mut self, label: A::Label) {
        self.exact_match = false;
        self.current_loop = false;
        if self.error {
            self.match_label = A::Label::no_label();
            return;
        }
        self.match_label = label;
        self.search();
    }
}

type ArcSlice<A> = StdArc<[A]>;
type LabelTable<A> = FxHashMap<<A as Arc>::Label, ArcSlice<A>>;
type StateTable<A> = FxHashMap<<A as Arc>::StateId, LabelTable<A>>;

pub struct HashMatcher<'f, F, A>
where
    A: Arc,
    F: Fst<A>,
{
    fst: &'f F,
    match_type: MatchType,
    state: Option<A::StateId>,
    state_table: Rc<RefCell<StateTable<A>>>,
    loop_arc: A,
    current_loop: bool,

    current_matches: Option<ArcSlice<A>>,
    pos: usize,
}

impl<'f, F, A> Clone for HashMatcher<'f, F, A>
where
    A: Arc,
    F: Fst<A>,
{
    fn clone(&self) -> Self {
        Self {
            fst: self.fst,
            match_type: self.match_type,
            state: self.state,
            state_table: Rc::clone(&self.state_table),
            loop_arc: self.loop_arc.clone(),
            current_loop: self.current_loop,
            current_matches: self.current_matches.clone(),
            pos: self.pos,
        }
    }
}

impl<'f, F, A> Matcher<'f, A> for HashMatcher<'f, F, A>
where
    A: Arc,
    F: Fst<A>,
{
    type Fst = F;

    fn new(fst: &'f F, match_type: MatchType) -> Result<Self, OpenFstError> {
        if match_type != MatchType::Input && match_type != MatchType::Output {
            return Err(OpenFstError::MatcherInvalidMatchType {
                matcher_name: "HashMatcher",
                match_type,
            });
        }

        let loop_arc = if match_type == MatchType::Input {
            A::new(
                A::Label::no_label(),
                A::Label::epsilon(),
                A::Weight::one(),
                A::StateId::no_state(),
            )
        } else {
            A::new(
                A::Label::epsilon(),
                A::Label::no_label(),
                A::Weight::one(),
                A::StateId::no_state(),
            )
        };

        Ok(Self {
            fst,
            match_type,
            state: None,
            state_table: Rc::new(RefCell::new(FxHashMap::default())),
            loop_arc,
            current_loop: false,
            current_matches: None,
            pos: 0,
        })
    }

    #[inline(always)]
    fn match_type(&self) -> MatchType {
        self.match_type
    }

    fn set_state(&mut self, state: A::StateId) {
        if self.state == Some(state) {
            return;
        }
        self.state = Some(state);
        self.loop_arc = A::new(
            self.loop_arc.ilabel(),
            self.loop_arc.olabel(),
            self.loop_arc.weight().clone(),
            state,
        );

        let mut table = self.state_table.borrow_mut();
        if let std::collections::hash_map::Entry::Vacant(e) = table.entry(state) {
            let mut label_map: FxHashMap<A::Label, Vec<A>> = FxHashMap::default();
            for arc in self.fst.arcs(state) {
                let label = if self.match_type == MatchType::Input {
                    arc.ilabel()
                } else {
                    arc.olabel()
                };
                label_map.entry(label).or_default().push(arc);
            }

            let mut label_table: LabelTable<A> = FxHashMap::default();
            for (l, arcs) in label_map {
                label_table.insert(l, arcs.into());
            }
            e.insert(label_table);
        }
    }

    fn find(&mut self, label: A::Label) -> bool {
        self.current_loop = label == A::Label::epsilon();
        let match_label = if label == A::Label::no_label() {
            A::Label::epsilon()
        } else {
            label
        };

        let state = self.state.expect("HashMatcher: state not set");
        let table = self.state_table.borrow();
        if let Some(label_table) = table.get(&state)
            && let Some(arcs) = label_table.get(&match_label)
        {
            self.current_matches = Some(StdArc::clone(arcs));
            self.pos = 0;
            return true;
        }

        self.current_matches = None;
        self.pos = 0;
        self.current_loop
    }

    #[inline(always)]
    fn done(&self) -> bool {
        if self.current_loop {
            return false;
        }
        if let Some(arcs) = &self.current_matches {
            self.pos >= arcs.len()
        } else {
            true
        }
    }

    #[inline(always)]
    fn value(&self) -> A {
        if self.current_loop {
            self.loop_arc.clone()
        } else {
            self.current_matches.as_ref().unwrap()[self.pos].clone()
        }
    }

    #[inline(always)]
    fn next(&mut self) {
        if self.current_loop {
            self.current_loop = false;
        } else {
            self.pos += 1;
        }
    }

    fn priority(&mut self, state: A::StateId) -> isize {
        self.fst.num_arcs(state) as isize
    }
}

#[derive(Clone)]
pub struct PhiMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    matcher: M,
    match_type: MatchType,
    phi_label: A::Label,
    phi_loop: bool,
    rewrite_both: bool,
    has_phi: bool,
    phi_match: A::Label,
    state: Option<A::StateId>,
    phi_weight: A::Weight,
    error: bool,
    _marker: std::marker::PhantomData<&'f A>,
}

impl<'f, M, A> PhiMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    pub fn new_with_options(
        fst: &'f M::Fst,
        match_type: MatchType,
        phi_label: A::Label,
        phi_loop: bool,
        rewrite_mode: MatcherRewriteMode,
    ) -> Result<Self, OpenFstError> {
        let matcher = M::new(fst, match_type)?;

        let rewrite_both = match rewrite_mode {
            MatcherRewriteMode::Auto => (fst.properties(K_ACCEPTOR, true) & K_ACCEPTOR) != 0,
            MatcherRewriteMode::Always => true,
            MatcherRewriteMode::Never => false,
        };

        Ok(Self {
            matcher,
            match_type,
            phi_label,
            phi_loop,
            rewrite_both,
            has_phi: false,
            phi_match: A::Label::no_label(),
            state: None,
            phi_weight: A::Weight::one(),
            error: false,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<'f, M, A> Matcher<'f, A> for PhiMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    type Fst = M::Fst;

    fn new(fst: &'f Self::Fst, match_type: MatchType) -> Result<Self, OpenFstError> {
        Self::new_with_options(
            fst,
            match_type,
            A::Label::no_label(),
            true,
            MatcherRewriteMode::Auto,
        )
    }

    #[inline(always)]
    fn match_type(&self) -> MatchType {
        self.match_type
    }

    fn set_state(&mut self, state: A::StateId) {
        if self.state == Some(state) {
            return;
        }
        self.matcher.set_state(state);
        self.state = Some(state);
        self.has_phi = self.phi_label != A::Label::no_label();
    }

    fn find(&mut self, label: A::Label) -> bool {
        if label == self.phi_label
            && self.phi_label != A::Label::no_label()
            && self.phi_label != A::Label::epsilon()
        {
            self.error = true;
            return false;
        }

        let state = self.state.expect("PhiMatcher: state not set");
        self.matcher.set_state(state);
        self.phi_match = A::Label::no_label();
        self.phi_weight = A::Weight::one();

        if self.phi_label == A::Label::epsilon() {
            if label == A::Label::no_label() {
                return false;
            }
            if label == A::Label::epsilon() {
                if !self.matcher.find(A::Label::no_label()) {
                    return self.matcher.find(A::Label::epsilon());
                } else {
                    self.phi_match = A::Label::epsilon();
                    return true;
                }
            }
        }

        if !self.has_phi || label == A::Label::epsilon() || label == A::Label::no_label() {
            return self.matcher.find(label);
        }

        let mut s = state;
        while !self.matcher.find(label) {
            let search_phi = if self.phi_label == A::Label::epsilon() {
                A::Label::no_label() // -1
            } else {
                self.phi_label
            };

            if !self.matcher.find(search_phi) {
                return false;
            }

            let val = self.matcher.value();
            if self.phi_loop && val.nextstate() == s {
                self.phi_match = label;
                return true;
            }

            self.phi_weight = self.phi_weight.times(val.weight());
            s = val.nextstate();
            self.matcher.next();
            if !self.matcher.done() {
                self.error = true;
            }
            self.matcher.set_state(s);
        }
        true
    }

    #[inline(always)]
    fn done(&self) -> bool {
        self.matcher.done()
    }

    fn value(&self) -> A {
        if self.phi_match == A::Label::no_label() && self.phi_weight == A::Weight::one() {
            self.matcher.value()
        } else if self.phi_match == A::Label::epsilon() {
            let state = self.state.unwrap();
            if self.match_type == MatchType::Input {
                A::new(
                    A::Label::no_label(),
                    A::Label::epsilon(),
                    A::Weight::one(),
                    state,
                )
            } else {
                A::new(
                    A::Label::epsilon(),
                    A::Label::no_label(),
                    A::Weight::one(),
                    state,
                )
            }
        } else {
            let arc = self.matcher.value();
            let weight = self.phi_weight.times(arc.weight());
            let rewritten = if self.phi_match == A::Label::no_label() {
                arc
            } else {
                rewrite(
                    &arc,
                    self.phi_label,
                    self.phi_match,
                    self.rewrite_both,
                    self.match_type,
                )
            };
            A::new(
                rewritten.ilabel(),
                rewritten.olabel(),
                weight,
                rewritten.nextstate(),
            )
        }
    }

    #[inline(always)]
    fn next(&mut self) {
        self.matcher.next();
    }

    fn priority(&mut self, state: A::StateId) -> isize {
        if self.phi_label != A::Label::no_label() {
            self.matcher.set_state(state);
            let search_phi = if self.phi_label == A::Label::epsilon() {
                A::Label::no_label() // -1
            } else {
                self.phi_label
            };
            let has_phi = self.matcher.find(search_phi);
            if has_phi {
                REQUIRE_PRIORITY
            } else {
                self.matcher.priority(state)
            }
        } else {
            self.matcher.priority(state)
        }
    }
}

/// The priority a matcher reports when it must be the one composition asks
/// first.
///
/// Port of upstream's `kRequirePriority`. A matcher with an implicit label (φ,
/// ρ or σ) answers for labels that are not written down at the state, so
/// composition cannot fall back to enumerating its arcs; it has to be given the
/// label to look up.
pub const REQUIRE_PRIORITY: isize = -1;

/// Matches any label the state has no transition of its own for.
///
/// ρ is "the rest": an arc labelled ρ stands for every label not otherwise
/// leaving the state, and matching it rewrites the arc to carry the label that
/// was asked for. Port of upstream's `RhoMatcher`.
pub struct RhoMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    matcher: M,
    match_type: MatchType,
    rho_label: A::Label,
    rewrite_both: bool,
    /// The label a ρ arc is standing in for, or `no_label` when the match was
    /// an ordinary one.
    rho_match: A::Label,
    has_rho: bool,
    state: Option<A::StateId>,
    error: bool,
    _marker: std::marker::PhantomData<&'f A>,
}

impl<'f, M, A> Clone for RhoMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    fn clone(&self) -> Self {
        Self {
            matcher: self.matcher.clone(),
            match_type: self.match_type,
            rho_label: self.rho_label,
            rewrite_both: self.rewrite_both,
            rho_match: A::Label::no_label(),
            has_rho: false,
            state: None,
            error: self.error,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'f, M, A> RhoMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    /// Creates a matcher treating `rho_label` as ρ.
    ///
    /// SICADA-DIVERGE: upstream sets an error flag for a bad match type or a ρ
    /// label of epsilon and carries on returning nothing. Both are mistakes in
    /// the caller's code rather than in its data, so they are errors here.
    pub fn new_with_options(
        fst: &'f M::Fst,
        match_type: MatchType,
        rho_label: A::Label,
        rewrite_mode: MatcherRewriteMode,
    ) -> Result<Self, OpenFstError> {
        if match_type == MatchType::Both {
            return Err(OpenFstError::MatcherInvalidMatchType {
                matcher_name: "RhoMatcher",
                match_type,
            });
        }
        if rho_label == A::Label::epsilon() {
            return Err(OpenFstError::MatcherInvalidConfiguration {
                matcher_name: "RhoMatcher",
                reason: "epsilon cannot be used as the rho label",
            });
        }
        let rewrite_both = rewrite_mode.rewrites_both(fst);
        Ok(Self {
            matcher: M::new(fst, match_type)?,
            match_type,
            rho_label,
            rewrite_both,
            rho_match: A::Label::no_label(),
            has_rho: false,
            state: None,
            error: false,
            _marker: std::marker::PhantomData,
        })
    }

    /// Whether the matcher has run into a request it could not honour.
    pub fn error(&self) -> bool {
        self.error
    }
}

impl<'f, M, A> Matcher<'f, A> for RhoMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    type Fst = M::Fst;

    fn new(fst: &'f Self::Fst, match_type: MatchType) -> Result<Self, OpenFstError> {
        Self::new_with_options(
            fst,
            match_type,
            A::Label::no_label(),
            MatcherRewriteMode::Auto,
        )
    }

    #[inline(always)]
    fn match_type(&self) -> MatchType {
        self.match_type
    }

    fn set_state(&mut self, state: A::StateId) {
        if self.state == Some(state) {
            return;
        }
        self.state = Some(state);
        self.matcher.set_state(state);
        self.has_rho = self.rho_label != A::Label::no_label();
    }

    fn find(&mut self, label: A::Label) -> bool {
        if label == self.rho_label && self.rho_label != A::Label::no_label() {
            self.error = true;
            return false;
        }
        if self.matcher.find(label) {
            self.rho_match = A::Label::no_label();
            return true;
        }
        // ρ stands in only for a real label: neither epsilon, which means "no
        // symbol consumed", nor the no-label marker.
        if self.has_rho && label != A::Label::epsilon() && label != A::Label::no_label() {
            self.has_rho = self.matcher.find(self.rho_label);
            if self.has_rho {
                self.rho_match = label;
                return true;
            }
        }
        false
    }

    #[inline(always)]
    fn done(&self) -> bool {
        self.matcher.done()
    }

    fn value(&self) -> A {
        let arc = self.matcher.value();
        if self.rho_match == A::Label::no_label() {
            return arc;
        }
        rewrite(
            &arc,
            self.rho_label,
            self.rho_match,
            self.rewrite_both,
            self.match_type,
        )
    }

    #[inline(always)]
    fn next(&mut self) {
        self.matcher.next();
    }

    fn priority(&mut self, state: A::StateId) -> isize {
        if self.rho_label == A::Label::no_label() {
            return self.matcher.priority(state);
        }
        self.state = Some(state);
        self.matcher.set_state(state);
        self.has_rho = self.matcher.find(self.rho_label);
        if self.has_rho {
            REQUIRE_PRIORITY
        } else {
            self.matcher.priority(state)
        }
    }
}

/// Matches every label.
///
/// σ is "any": an arc labelled σ matches whatever is asked for, in addition to
/// whatever the state matches ordinarily. Port of upstream's `SigmaMatcher`.
pub struct SigmaMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    matcher: M,
    match_type: MatchType,
    sigma_label: A::Label,
    rewrite_both: bool,
    sigma_match: A::Label,
    /// The label this search is for, so that `next` can fall through to σ once
    /// the ordinary matches run out.
    match_label: A::Label,
    has_sigma: bool,
    state: Option<A::StateId>,
    error: bool,
    _marker: std::marker::PhantomData<&'f A>,
}

impl<'f, M, A> Clone for SigmaMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    fn clone(&self) -> Self {
        Self {
            matcher: self.matcher.clone(),
            match_type: self.match_type,
            sigma_label: self.sigma_label,
            rewrite_both: self.rewrite_both,
            sigma_match: A::Label::no_label(),
            match_label: A::Label::no_label(),
            has_sigma: false,
            state: None,
            error: self.error,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'f, M, A> SigmaMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    /// Creates a matcher treating `sigma_label` as σ.
    pub fn new_with_options(
        fst: &'f M::Fst,
        match_type: MatchType,
        sigma_label: A::Label,
        rewrite_mode: MatcherRewriteMode,
    ) -> Result<Self, OpenFstError> {
        if match_type == MatchType::Both {
            return Err(OpenFstError::MatcherInvalidMatchType {
                matcher_name: "SigmaMatcher",
                match_type,
            });
        }
        if sigma_label == A::Label::epsilon() {
            return Err(OpenFstError::MatcherInvalidConfiguration {
                matcher_name: "SigmaMatcher",
                reason: "epsilon cannot be used as the sigma label",
            });
        }
        let rewrite_both = rewrite_mode.rewrites_both(fst);
        Ok(Self {
            matcher: M::new(fst, match_type)?,
            match_type,
            sigma_label,
            rewrite_both,
            sigma_match: A::Label::no_label(),
            match_label: A::Label::no_label(),
            has_sigma: false,
            state: None,
            error: false,
            _marker: std::marker::PhantomData,
        })
    }

    /// Whether the matcher has run into a request it could not honour.
    pub fn error(&self) -> bool {
        self.error
    }
}

impl<'f, M, A> Matcher<'f, A> for SigmaMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    type Fst = M::Fst;

    fn new(fst: &'f Self::Fst, match_type: MatchType) -> Result<Self, OpenFstError> {
        Self::new_with_options(
            fst,
            match_type,
            A::Label::no_label(),
            MatcherRewriteMode::Auto,
        )
    }

    #[inline(always)]
    fn match_type(&self) -> MatchType {
        self.match_type
    }

    fn set_state(&mut self, state: A::StateId) {
        if self.state == Some(state) {
            return;
        }
        self.state = Some(state);
        self.matcher.set_state(state);
        // Whether this state has a σ arc is settled once, here, rather than on
        // every lookup, which is why `next` can rely on `has_sigma`.
        self.has_sigma =
            self.sigma_label != A::Label::no_label() && self.matcher.find(self.sigma_label);
    }

    fn find(&mut self, label: A::Label) -> bool {
        self.match_label = label;
        if label == self.sigma_label && self.sigma_label != A::Label::no_label() {
            self.error = true;
            return false;
        }
        if self.matcher.find(label) {
            self.sigma_match = A::Label::no_label();
            return true;
        }
        if self.has_sigma
            && label != A::Label::epsilon()
            && label != A::Label::no_label()
            && self.matcher.find(self.sigma_label)
        {
            self.sigma_match = label;
            return true;
        }
        false
    }

    #[inline(always)]
    fn done(&self) -> bool {
        self.matcher.done()
    }

    fn value(&self) -> A {
        let arc = self.matcher.value();
        if self.sigma_match == A::Label::no_label() {
            return arc;
        }
        rewrite(
            &arc,
            self.sigma_label,
            self.sigma_match,
            self.rewrite_both,
            self.match_type,
        )
    }

    fn next(&mut self) {
        self.matcher.next();
        // σ matches on top of the ordinary arcs rather than instead of them, so
        // once those are exhausted the search continues into the σ arc.
        if self.matcher.done()
            && self.has_sigma
            && self.sigma_match == A::Label::no_label()
            && self.match_label != A::Label::epsilon()
            && self.match_label != A::Label::no_label()
        {
            self.matcher.find(self.sigma_label);
            self.sigma_match = self.match_label;
        }
    }

    fn priority(&mut self, state: A::StateId) -> isize {
        if self.sigma_label == A::Label::no_label() {
            return self.matcher.priority(state);
        }
        self.set_state(state);
        if self.has_sigma {
            REQUIRE_PRIORITY
        } else {
            self.matcher.priority(state)
        }
    }
}

/// Hides the matches an implicit-label matcher invents.
///
/// Wrapping a φ, ρ or σ matcher in this leaves only the arcs the FST actually
/// has, as an algorithm that wants to see the FST as written requires.
/// Port of upstream's `ExplicitMatcher`.
pub struct ExplicitMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    matcher: M,
    match_type: MatchType,
    _marker: std::marker::PhantomData<&'f A>,
}

impl<'f, M, A> Clone for ExplicitMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    fn clone(&self) -> Self {
        Self {
            matcher: self.matcher.clone(),
            match_type: self.match_type,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'f, M, A> ExplicitMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    /// Wraps an existing matcher.
    pub fn wrap(matcher: M, match_type: MatchType) -> Self {
        Self {
            matcher,
            match_type,
            _marker: std::marker::PhantomData,
        }
    }

    /// Skips past any arc whose matched side is the no-label marker, which is
    /// how an invented match announces itself.
    fn skip_implicit(&mut self) {
        while !self.matcher.done() {
            let arc = self.matcher.value();
            let label = if self.match_type == MatchType::Input {
                arc.ilabel()
            } else {
                arc.olabel()
            };
            if label != A::Label::no_label() {
                return;
            }
            self.matcher.next();
        }
    }
}

impl<'f, M, A> Matcher<'f, A> for ExplicitMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    type Fst = M::Fst;

    fn new(fst: &'f Self::Fst, match_type: MatchType) -> Result<Self, OpenFstError> {
        Ok(Self::wrap(M::new(fst, match_type)?, match_type))
    }

    #[inline(always)]
    fn match_type(&self) -> MatchType {
        self.match_type
    }

    #[inline(always)]
    fn set_state(&mut self, state: A::StateId) {
        self.matcher.set_state(state);
    }

    fn find(&mut self, label: A::Label) -> bool {
        self.matcher.find(label);
        self.skip_implicit();
        !self.done()
    }

    #[inline(always)]
    fn done(&self) -> bool {
        self.matcher.done()
    }

    #[inline(always)]
    fn value(&self) -> A {
        self.matcher.value()
    }

    fn next(&mut self) {
        self.matcher.next();
        self.skip_implicit();
    }

    #[inline(always)]
    fn priority(&mut self, state: A::StateId) -> isize {
        self.matcher.priority(state)
    }
}

bitflags::bitflags! {
    /// What a [`MultiEpsMatcher`] does with the labels it treats as epsilon.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct MultiEpsFlags: u32 {
        /// A request for every non-consuming arc returns the multi-epsilon
        /// arcs as well as the ordinary epsilon ones.
        const LIST = 0x01;
        /// A request for one of the multi-epsilon labels returns an implicit
        /// self-loop rather than looking for an arc.
        const LOOP = 0x02;
    }
}

/// Treats a set of labels as epsilon, on top of epsilon itself.
///
/// Composition uses this to keep several kinds of "consumes no symbol" apart;
/// see [`MultiEpsFilter`](crate::algorithms::compose_filter::MultiEpsFilter),
/// which decides whether they survive into the output. Port of upstream's
/// `MultiEpsMatcher`.
pub struct MultiEpsMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    matcher: M,
    flags: MultiEpsFlags,
    /// Sorted, so that a request for every non-consuming arc walks them in a
    /// fixed order and `Find` is a binary search.
    multi_eps_labels: Vec<A::Label>,
    /// How far through `multi_eps_labels` a "every non-consuming arc" walk has
    /// got, or `None` when it is not walking them.
    multi_eps_index: Option<usize>,
    /// The implicit self-loop, when the current match is one.
    current_loop: bool,
    loop_state: Option<A::StateId>,
    done: bool,
    match_type: MatchType,
    _marker: std::marker::PhantomData<&'f A>,
}

impl<'f, M, A> Clone for MultiEpsMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    fn clone(&self) -> Self {
        Self {
            matcher: self.matcher.clone(),
            flags: self.flags,
            multi_eps_labels: self.multi_eps_labels.clone(),
            multi_eps_index: None,
            current_loop: false,
            loop_state: None,
            done: true,
            match_type: self.match_type,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'f, M, A> MultiEpsMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    /// Creates a matcher with no multi-epsilon labels yet.
    pub fn new_with_flags(
        fst: &'f M::Fst,
        match_type: MatchType,
        flags: MultiEpsFlags,
    ) -> Result<Self, OpenFstError> {
        Ok(Self {
            matcher: M::new(fst, match_type)?,
            flags,
            multi_eps_labels: Vec::new(),
            multi_eps_index: None,
            current_loop: false,
            loop_state: None,
            done: true,
            match_type,
            _marker: std::marker::PhantomData,
        })
    }

    /// Adds a label to be treated as epsilon.
    ///
    /// SICADA-DIVERGE: upstream logs and ignores an attempt to add epsilon
    /// itself, which is already handled and would otherwise be listed twice.
    /// It is an error here, since it can only be a mistake in the caller.
    pub fn add_multi_eps_label(&mut self, label: A::Label) -> Result<(), OpenFstError> {
        if label == A::Label::epsilon() {
            return Err(OpenFstError::MatcherInvalidConfiguration {
                matcher_name: "MultiEpsMatcher",
                reason: "epsilon is already a multi-epsilon label",
            });
        }
        if let Err(at) = self.multi_eps_labels.binary_search(&label) {
            self.multi_eps_labels.insert(at, label);
        }
        Ok(())
    }

    /// Stops treating `label` as epsilon.
    pub fn remove_multi_eps_label(&mut self, label: A::Label) {
        if let Ok(at) = self.multi_eps_labels.binary_search(&label) {
            self.multi_eps_labels.remove(at);
        }
    }

    /// Stops treating anything but epsilon itself as epsilon.
    pub fn clear_multi_eps_labels(&mut self) {
        self.multi_eps_labels.clear();
        self.multi_eps_index = None;
    }

    /// The labels currently treated as epsilon, in order.
    pub fn multi_eps_labels(&self) -> &[A::Label] {
        &self.multi_eps_labels
    }

    /// The matcher underneath.
    pub fn matcher(&self) -> &M {
        &self.matcher
    }

    /// The matcher underneath.
    pub fn matcher_mut(&mut self) -> &mut M {
        &mut self.matcher
    }

    /// Advances `multi_eps_index` to the next label the state actually has an
    /// arc for, and reports whether it found one.
    fn seek_multi_eps(&mut self, mut index: usize) -> bool {
        while index < self.multi_eps_labels.len() {
            if self.matcher.find(self.multi_eps_labels[index]) {
                self.multi_eps_index = Some(index);
                return true;
            }
            index += 1;
        }
        self.multi_eps_index = Some(self.multi_eps_labels.len());
        false
    }
}

impl<'f, M, A> Matcher<'f, A> for MultiEpsMatcher<'f, M, A>
where
    A: Arc,
    M: Matcher<'f, A>,
{
    type Fst = M::Fst;

    fn new(fst: &'f Self::Fst, match_type: MatchType) -> Result<Self, OpenFstError> {
        Self::new_with_flags(fst, match_type, MultiEpsFlags::LIST | MultiEpsFlags::LOOP)
    }

    #[inline(always)]
    fn match_type(&self) -> MatchType {
        self.match_type
    }

    fn set_state(&mut self, state: A::StateId) {
        self.matcher.set_state(state);
        self.loop_state = Some(state);
    }

    fn find(&mut self, label: A::Label) -> bool {
        self.multi_eps_index = Some(self.multi_eps_labels.len());
        self.current_loop = false;
        let found = if label == A::Label::epsilon() {
            self.matcher.find(A::Label::epsilon())
        } else if label == A::Label::no_label() {
            if self.flags.contains(MultiEpsFlags::LIST) {
                // Every arc that consumes nothing: the multi-epsilon ones and
                // then whatever the inner matcher calls non-consuming.
                self.seek_multi_eps(0) || self.matcher.find(A::Label::no_label())
            } else {
                self.matcher.find(A::Label::no_label())
            }
        } else if self.flags.contains(MultiEpsFlags::LOOP)
            && self.multi_eps_labels.binary_search(&label).is_ok()
        {
            self.current_loop = true;
            true
        } else {
            self.matcher.find(label)
        };
        self.done = !found;
        found
    }

    #[inline(always)]
    fn done(&self) -> bool {
        self.done
    }

    fn value(&self) -> A {
        if self.current_loop {
            let state = self
                .loop_state
                .expect("MultiEpsMatcher: state not set before a loop match");
            A::new(
                A::Label::no_label(),
                A::Label::no_label(),
                A::Weight::one(),
                state,
            )
        } else {
            self.matcher.value()
        }
    }

    fn next(&mut self) {
        if self.current_loop {
            self.done = true;
            return;
        }
        self.matcher.next();
        self.done = self.matcher.done();
        if !self.done {
            return;
        }
        // The arcs for one multi-epsilon label ran out; carry on with the next
        // label that has any, and only then with the inner matcher's own idea
        // of a non-consuming arc.
        let Some(index) = self.multi_eps_index else {
            return;
        };
        if index >= self.multi_eps_labels.len() {
            return;
        }
        if self.seek_multi_eps(index + 1) {
            self.done = false;
        } else {
            self.done = !self.matcher.find(A::Label::no_label());
        }
    }

    #[inline(always)]
    fn priority(&mut self, state: A::StateId) -> isize {
        self.matcher.priority(state)
    }
}

/// Rewrites the side or sides an implicit-label match stands in for.
///
/// Shared by ρ and σ, which differ in when they match but not in what they do
/// to the arc once they have.
fn rewrite<A: Arc>(
    arc: &A,
    implicit_label: A::Label,
    matched: A::Label,
    rewrite_both: bool,
    match_type: MatchType,
) -> A {
    let mut ilabel = arc.ilabel();
    let mut olabel = arc.olabel();
    if rewrite_both {
        if ilabel == implicit_label {
            ilabel = matched;
        }
        if olabel == implicit_label {
            olabel = matched;
        }
    } else if match_type == MatchType::Input {
        ilabel = matched;
    } else {
        olabel = matched;
    }
    A::new(ilabel, olabel, arc.weight().clone(), arc.nextstate())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::float_weight::TropicalWeight;
    use crate::fst::{MatchType, MutableFst};
    use crate::properties::K_I_LABEL_SORTED;
    use crate::vector_fst::StdVectorFst;
    use crate::weight::Weight;

    fn build_sorted_test_fst() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Arcs strictly sorted by input label for SortedMatcher
        fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s0, StdArc::new(2, 2, TropicalWeight::one(), s1));
        fst.add_arc(s0, StdArc::new(2, 3, TropicalWeight::one(), s2));
        fst.add_arc(s0, StdArc::new(4, 4, TropicalWeight::one(), s2));

        fst
    }

    #[test]
    fn test_sorted_matcher_exact_match() {
        let fst = build_sorted_test_fst();
        let mut matcher = SortedMatcher::new(&fst, MatchType::Input).unwrap();
        matcher.set_state(0);

        // Matches exactly two arcs
        let arcs: Vec<_> = matcher.iter(2).collect();
        assert_eq!(arcs.len(), 2);
        assert_eq!(arcs[0].olabel(), 2);
        assert_eq!(arcs[1].olabel(), 3);
    }

    #[test]
    fn test_sorted_matcher_no_match() {
        let fst = build_sorted_test_fst();
        let mut matcher = SortedMatcher::new(&fst, MatchType::Input).unwrap();
        matcher.set_state(0);

        assert!(!matcher.find(3));
        assert!(matcher.done());
    }

    #[test]
    fn test_sorted_matcher_lower_bound() {
        let fst = build_sorted_test_fst();
        let mut matcher = SortedMatcher::new(&fst, MatchType::Input).unwrap();
        matcher.set_state(0);

        // Lower bound for 3 should position the matcher at label 4
        matcher.lower_bound(3);
        assert!(!matcher.done());
        assert_eq!(matcher.value().ilabel(), 4);
    }

    #[test]
    fn test_sorted_matcher_epsilon_loop() {
        let fst = build_sorted_test_fst();
        // MATCH_OUTPUT means epsilon (0) input label acts as non-consuming loop
        let mut matcher = SortedMatcher::new(&fst, MatchType::Output).unwrap();
        matcher.set_state(0);

        // In MATCH_OUTPUT, looking up epsilon (0) should yield the implicit loop arc
        assert!(matcher.find(0));
        let val = matcher.value();
        assert_eq!(val.ilabel(), 0);
        assert_eq!(val.olabel(), ArcLabel::no_label()); // kNoLabel
        assert_eq!(val.nextstate(), 0);

        matcher.next();
        // Since we didn't add any explicit 0-output arcs, it should be done after the loop.
        assert!(matcher.done());
    }

    #[test]
    fn test_hash_matcher() {
        let mut fst = StdVectorFst::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        // Intentionally out-of-order labels to prove hashing works
        fst.add_arc(s0, StdArc::new(5, 5, TropicalWeight::one(), s1));
        fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s0, StdArc::new(5, 6, TropicalWeight::one(), s1));

        let mut matcher = HashMatcher::new(&fst, MatchType::Input).unwrap();
        matcher.set_state(0);

        let arcs: Vec<_> = matcher.iter(5).collect();
        assert_eq!(arcs.len(), 2);

        let arcs_1: Vec<_> = matcher.iter(1).collect();
        assert_eq!(arcs_1.len(), 1);

        assert!(!matcher.find(99));
    }

    #[test]
    fn test_phi_matcher_basic_fallback() {
        let mut fst = StdVectorFst::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);

        // s0 has explicit arc for label 1
        // and a phi transition (using label 0 / epsilon) to s1
        // Note: Arcs MUST be sorted by input label for SortedMatcher (0 comes before 1)
        fst.add_arc(s0, StdArc::new(0, 0, TropicalWeight::one(), s1));
        fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight::one(), s2));

        fst.add_arc(s1, StdArc::new(2, 2, TropicalWeight::one(), s2));

        let mut matcher = PhiMatcher::<SortedMatcher<_, _>, _>::new_with_options(
            &fst,
            MatchType::Input,
            0, // phi_label = 0 (epsilon)
            true,
            MatcherRewriteMode::Always,
        )
        .unwrap();

        matcher.set_state(s0);

        // exists directly on s0
        assert!(matcher.find(1));
        assert_eq!(matcher.value().nextstate(), s2);

        // not on s0, should fall back to s1 via phi transition (0)
        assert!(matcher.find(2));
        assert_eq!(matcher.value().nextstate(), s2);
        // Because MatcherRewriteMode::Always is used, both labels are rewritten to the matched label (2)
        assert_eq!(matcher.value().ilabel(), 2);
        assert_eq!(matcher.value().olabel(), 2);

        // neither s0 nor s1 has it
        assert!(!matcher.find(3));
    }

    #[test]
    fn test_phi_matcher_recursive_fallback() {
        let mut fst = StdVectorFst::new();
        let s0 = fst.add_state(); // root
        let s1 = fst.add_state(); // level 1
        let s2 = fst.add_state(); // level 2
        let s3 = fst.add_state(); // target

        fst.set_start(s0);

        // s0 -> s1 -> s2 via phi (0)
        fst.add_arc(s0, StdArc::new(0, 0, TropicalWeight(0.5), s1));
        fst.add_arc(s1, StdArc::new(0, 0, TropicalWeight(0.5), s2));
        // s2 has the target label 5
        fst.add_arc(s2, StdArc::new(5, 5, TropicalWeight(1.0), s3));

        let mut matcher = PhiMatcher::<SortedMatcher<_, _>, _>::new_with_options(
            &fst,
            MatchType::Input,
            0,
            true,
            MatcherRewriteMode::Always,
        )
        .unwrap();

        matcher.set_state(s0);

        assert!(matcher.find(5));
        let val = matcher.value();
        assert_eq!(val.nextstate(), s3);
        // The weight should be the accumulated product along the phi transitions:
        // 0.5 (phi) * 0.5 (phi) * 1.0 (target) = 2.0 (in Tropical semiring addition is +)
        assert_eq!(val.weight().value(), 2.0);
    }

    /// A state whose arcs are labelled 1, 3 and `implicit`, all leading
    /// somewhere distinguishable.
    fn fst_with_implicit_label(implicit: i32) -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..5 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.set_final(4, TropicalWeight::one());
        // SortedMatcher binary-searches the arcs, so they go in sorted order.
        // The implicit label is negative and therefore comes first.
        fst.add_arc(0, StdArc::new(implicit, implicit, TropicalWeight(0.5), 3));
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(0, StdArc::new(3, 3, TropicalWeight::one(), 2));
        fst.set_properties(K_I_LABEL_SORTED, K_I_LABEL_SORTED);
        fst
    }

    /// The arcs a matcher reports for `label`, as (ilabel, olabel, nextstate).
    fn matches<'f, M: Matcher<'f, StdArc>>(
        matcher: &mut M,
        state: i32,
        label: i32,
    ) -> Vec<(i32, i32, i32)> {
        matcher.set_state(state);
        matcher
            .iter(label)
            .map(|a| (a.ilabel(), a.olabel(), a.nextstate()))
            .collect()
    }

    type Sorted<'f> = SortedMatcher<'f, StdVectorFst, StdArc>;

    /// Rho stands in for every label the state does not have an arc for, and
    /// the arc it produces carries the label that was asked for.
    #[test]
    fn a_rho_matcher_answers_for_labels_the_state_lacks() {
        const RHO: i32 = -3;
        let fst = fst_with_implicit_label(RHO);
        let mut matcher = RhoMatcher::<Sorted, StdArc>::new_with_options(
            &fst,
            MatchType::Input,
            RHO,
            MatcherRewriteMode::Always,
        )
        .unwrap();

        // Labels the state has go to their own arcs, unrewritten.
        assert_eq!(matches(&mut matcher, 0, 1), vec![(1, 1, 1)]);
        assert_eq!(matches(&mut matcher, 0, 3), vec![(3, 3, 2)]);
        // Anything else takes the rho arc, rewritten to the label asked for.
        assert_eq!(matches(&mut matcher, 0, 7), vec![(7, 7, 3)]);
        assert_eq!(matches(&mut matcher, 0, 99), vec![(99, 99, 3)]);
        // Rho does not stand in for epsilon: the request goes to the inner
        // matcher, which offers its own implicit epsilon self-loop.
        assert_eq!(matches(&mut matcher, 0, 0), vec![(-1, 0, 0)]);
        // A state with no rho arc answers only for what it has.
        assert_eq!(matches(&mut matcher, 1, 7), vec![]);
    }

    /// Only the matched side is rewritten unless asked for both.
    #[test]
    fn a_rho_matcher_rewrites_the_side_it_was_told_to() {
        const RHO: i32 = -3;
        let fst = fst_with_implicit_label(RHO);
        let mut matcher = RhoMatcher::<Sorted, StdArc>::new_with_options(
            &fst,
            MatchType::Input,
            RHO,
            MatcherRewriteMode::Never,
        )
        .unwrap();
        assert_eq!(matches(&mut matcher, 0, 7), vec![(7, RHO, 3)]);
    }

    /// Sigma matches everything, including labels the state has arcs for, which
    /// come back as well as the ordinary match rather than instead of it.
    #[test]
    fn a_sigma_matcher_answers_for_every_label() {
        const SIGMA: i32 = -3;
        let fst = fst_with_implicit_label(SIGMA);
        let mut matcher = SigmaMatcher::<Sorted, StdArc>::new_with_options(
            &fst,
            MatchType::Input,
            SIGMA,
            MatcherRewriteMode::Always,
        )
        .unwrap();

        assert_eq!(matches(&mut matcher, 0, 1), vec![(1, 1, 1), (1, 1, 3)]);
        assert_eq!(matches(&mut matcher, 0, 7), vec![(7, 7, 3)]);
        // As with rho, epsilon is the inner matcher's business.
        assert_eq!(matches(&mut matcher, 0, 0), vec![(-1, 0, 0)]);
        assert_eq!(matches(&mut matcher, 1, 7), vec![]);
    }

    /// Asking a rho or sigma matcher for the label it treats as implicit is a
    /// mistake in the caller, and it says so rather than answering.
    #[test]
    fn asking_for_the_implicit_label_itself_is_refused() {
        const IMPLICIT: i32 = -3;
        let fst = fst_with_implicit_label(IMPLICIT);

        let mut rho = RhoMatcher::<Sorted, StdArc>::new_with_options(
            &fst,
            MatchType::Input,
            IMPLICIT,
            MatcherRewriteMode::Always,
        )
        .unwrap();
        rho.set_state(0);
        assert!(!rho.find(IMPLICIT));
        assert!(rho.error());

        let mut sigma = SigmaMatcher::<Sorted, StdArc>::new_with_options(
            &fst,
            MatchType::Input,
            IMPLICIT,
            MatcherRewriteMode::Always,
        )
        .unwrap();
        sigma.set_state(0);
        assert!(!sigma.find(IMPLICIT));
        assert!(sigma.error());
    }

    /// Epsilon cannot be the implicit label, or every state would match every
    /// label through it.
    #[test]
    fn epsilon_cannot_be_the_implicit_label() {
        let fst = fst_with_implicit_label(-3);
        assert!(
            RhoMatcher::<Sorted, StdArc>::new_with_options(
                &fst,
                MatchType::Input,
                0,
                MatcherRewriteMode::Auto
            )
            .is_err()
        );
        assert!(
            SigmaMatcher::<Sorted, StdArc>::new_with_options(
                &fst,
                MatchType::Input,
                0,
                MatcherRewriteMode::Auto
            )
            .is_err()
        );
        // Matching both sides at once is not something these can do.
        assert!(
            RhoMatcher::<Sorted, StdArc>::new_with_options(
                &fst,
                MatchType::Both,
                -3,
                MatcherRewriteMode::Auto
            )
            .is_err()
        );
    }

    /// A matcher with an implicit label has to be the one composition asks
    /// first, since it answers for labels that are not written down.
    #[test]
    fn an_implicit_label_demands_priority() {
        const IMPLICIT: i32 = -3;
        let fst = fst_with_implicit_label(IMPLICIT);

        let mut rho = RhoMatcher::<Sorted, StdArc>::new_with_options(
            &fst,
            MatchType::Input,
            IMPLICIT,
            MatcherRewriteMode::Always,
        )
        .unwrap();
        assert_eq!(rho.priority(0), REQUIRE_PRIORITY);
        // State 1 has no implicit arc, so the inner matcher's answer stands.
        assert_ne!(rho.priority(1), REQUIRE_PRIORITY);

        let mut sigma = SigmaMatcher::<Sorted, StdArc>::new_with_options(
            &fst,
            MatchType::Input,
            IMPLICIT,
            MatcherRewriteMode::Always,
        )
        .unwrap();
        assert_eq!(sigma.priority(0), REQUIRE_PRIORITY);
        assert_ne!(sigma.priority(1), REQUIRE_PRIORITY);
    }

    /// Wrapping a rho matcher in an explicit one leaves only the arcs the FST
    /// really has.
    #[test]
    fn an_explicit_matcher_hides_invented_matches() {
        const RHO: i32 = -3;
        let fst = fst_with_implicit_label(RHO);
        let rho = RhoMatcher::<Sorted, StdArc>::new_with_options(
            &fst,
            MatchType::Input,
            RHO,
            // Rewrite only the matched side, so an invented match leaves the
            // other side as the rho label, which marks it as invented.
            MatcherRewriteMode::Never,
        )
        .unwrap();
        let mut explicit = ExplicitMatcher::wrap(rho, MatchType::Input);

        // A real arc still matches.
        assert_eq!(matches(&mut explicit, 0, 1), vec![(1, 1, 1)]);
    }

    /// Labels added as multi-epsilon match as an implicit self-loop, which is
    /// how composition tells "consumed nothing" apart from "consumed epsilon".
    #[test]
    fn a_multi_eps_matcher_loops_on_the_labels_it_was_given() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.set_final(2, TropicalWeight::one());
        fst.add_arc(0, StdArc::new(0, 0, TropicalWeight::one(), 1));
        fst.add_arc(0, StdArc::new(5, 5, TropicalWeight::one(), 2));
        fst.set_properties(K_I_LABEL_SORTED, K_I_LABEL_SORTED);

        let mut matcher = MultiEpsMatcher::<Sorted, StdArc>::new(&fst, MatchType::Input).unwrap();
        matcher.add_multi_eps_label(7).unwrap();
        matcher.add_multi_eps_label(9).unwrap();
        assert_eq!(matcher.multi_eps_labels(), &[7, 9]);

        // A multi-epsilon label loops back to the state it was asked at.
        matcher.set_state(0);
        assert!(matcher.find(7));
        let arc = matcher.value();
        assert_eq!(arc.nextstate(), 0);
        assert_eq!((arc.ilabel(), arc.olabel()), (-1, -1));
        matcher.next();
        assert!(matcher.done(), "the loop is the only match");

        // Ordinary labels are unaffected.
        assert_eq!(matches(&mut matcher, 0, 5), vec![(5, 5, 2)]);
        // The inner matcher's implicit epsilon loop comes first, then the real
        // epsilon arc.
        assert_eq!(matches(&mut matcher, 0, 0), vec![(-1, 0, 0), (0, 0, 1)]);

        matcher.remove_multi_eps_label(7);
        assert_eq!(matcher.multi_eps_labels(), &[9]);
        assert_eq!(matches(&mut matcher, 0, 7), vec![]);
    }

    #[test]
    fn epsilon_cannot_be_added_as_a_multi_eps_label() {
        let fst = fst_with_implicit_label(-3);
        let mut matcher = MultiEpsMatcher::<Sorted, StdArc>::new(&fst, MatchType::Input).unwrap();
        assert!(matcher.add_multi_eps_label(0).is_err());
    }
}
