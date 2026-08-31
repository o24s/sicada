//! The state a composition filter carries between arcs.
//!
//! Port of OpenFst's `filter-state.h`. A filter state is whatever a composition
//! filter needs to remember about the path it took to reach the current pair of
//! states. For the epsilon-sequencing filters that is which side was allowed to
//! consume epsilons.
//!
//! SICADA-DIVERGE: upstream gives every filter state a `NoState()` value
//! standing for "this arc is disallowed", and callers compare against it. Here
//! that is `None` from
//! [`ComposeFilter::filter_arc`](super::compose_filter::ComposeFilter::filter_arc),
//! so the disallowed case cannot be mistaken for a real state and every filter
//! state type is free of a reserved value.

use std::collections::VecDeque;
use std::fmt::Debug;
use std::hash::Hash;

use crate::arc::{ArcLabel, ArcStateId};
use crate::weight::Weight;

/// Number of bits a `usize` has, for the shift-and-xor mixing upstream uses.
const USIZE_BITS: u32 = usize::BITS;

/// The filter state a composition starts in, and returns to once both sides
/// have consumed a real symbol.
pub const STATE_DEFAULT: i8 = 0;
/// Only the first side may consume epsilons from here.
pub const STATE_EPS1: i8 = 1;
/// Only the second side may consume epsilons from here.
pub const STATE_EPS2: i8 = 2;

/// The state of a composition filter.
pub trait FilterState: Clone + PartialEq + Eq + Hash + Debug {
    /// A value no real filter state equals.
    ///
    /// Not a "this arc is disallowed" marker; that is `None` from
    /// [`filter_arc`](super::compose_filter::ComposeFilter::filter_arc). This is
    /// the entry an erasable table fills a freed slot with, which is why it
    /// only has to be distinguishable from anything real.
    fn no_state() -> Self;

    /// A small integer standing for this state.
    ///
    /// Port of upstream's `Hash()`, which is part of the same concept and not
    /// the same thing as [`Hash`]. It has to be a *value*, not a hasher's
    /// output: a composition state table can use it as part of a vector index,
    /// so a well-spread 64-bit hash would be useless there.
    fn hash_value(&self) -> usize;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IntegerFilterState<T> {
    state: T,
}
impl<T: ArcStateId> FilterState for IntegerFilterState<T> {
    #[inline]
    fn no_state() -> Self {
        Self::new(T::no_state())
    }

    #[inline]
    fn hash_value(&self) -> usize {
        self.state.as_usize()
    }
}

impl<T> IntegerFilterState<T> {
    #[inline]
    pub fn new(state: T) -> Self {
        Self { state }
    }
    #[inline]
    pub fn state(&self) -> &T {
        &self.state
    }
    #[inline]
    pub fn state_copied(&self) -> T
    where
        T: Copy,
    {
        self.state
    }
}

/// A filter state that is a label.
///
/// SICADA-DIVERGE: upstream reuses `IntegerFilterState<Label>` for this, which
/// works because in C++ a label and a state id are both plain integers. They
/// are distinct traits here, since a label has [`no_label`](ArcLabel::no_label)
/// and a state id has `no_state`, so the two filter states are distinct types
/// and cannot be crossed by accident.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LabelFilterState<L> {
    label: L,
}

impl<L: ArcLabel> FilterState for LabelFilterState<L> {
    #[inline]
    fn no_state() -> Self {
        Self::new(L::no_label())
    }

    #[inline]
    fn hash_value(&self) -> usize {
        self.label.to_i64().unwrap_or(0) as usize
    }
}

impl<L> LabelFilterState<L> {
    #[inline]
    pub fn new(label: L) -> Self {
        Self { label }
    }
    #[inline]
    pub fn label(&self) -> &L {
        &self.label
    }
    #[inline]
    pub fn label_copied(&self) -> L
    where
        L: Copy,
    {
        self.label
    }
}

pub type CharFilterState = IntegerFilterState<i8>;
pub type ShortFilterState = IntegerFilterState<i16>;
pub type IntFilterState = IntegerFilterState<i32>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WeightFilterState<W> {
    weight: W,
}
impl<W: Weight + Hash + std::cmp::Eq> FilterState for WeightFilterState<W> {
    /// Upstream's is `Zero()`, not a non-member weight: a filter state that is
    /// `Zero` never arises from a real composition, so it does the job.
    #[inline]
    fn no_state() -> Self {
        Self::new(W::zero())
    }

    fn hash_value(&self) -> usize {
        let mut hasher = rustc_hash::FxHasher::default();
        self.weight.hash(&mut hasher);
        std::hash::Hasher::finish(&hasher) as usize
    }
}

impl<W: Weight> WeightFilterState<W> {
    #[inline]
    pub fn new(weight: W) -> Self {
        Self { weight }
    }
    #[inline]
    pub fn weight(&self) -> W {
        self.weight.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ListFilterState<T> {
    list: VecDeque<T>,
}
impl<T: ArcStateId> FilterState for ListFilterState<T> {
    #[inline]
    fn no_state() -> Self {
        Self::new(T::no_state())
    }

    fn hash_value(&self) -> usize {
        self.list
            .iter()
            .fold(0usize, |h, elem| h ^ (h << 1) ^ elem.as_usize())
    }
}

impl<T> ListFilterState<T> {
    #[inline]
    pub fn new(s: T) -> Self {
        let mut list = VecDeque::with_capacity(1);
        list.push_front(s);
        Self { list }
    }
    #[inline]
    pub fn state(&self) -> &VecDeque<T> {
        &self.list
    }
    #[inline]
    pub fn state_mut(&mut self) -> &mut VecDeque<T> {
        &mut self.list
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PairFilterState<FS1, FS2> {
    fs1: FS1,
    fs2: FS2,
}
impl<FS1: FilterState, FS2: FilterState> FilterState for PairFilterState<FS1, FS2> {
    #[inline]
    fn no_state() -> Self {
        Self::new(FS1::no_state(), FS2::no_state())
    }

    fn hash_value(&self) -> usize {
        let h1 = self.fs1.hash_value();
        const LSHIFT: u32 = 5;
        let rshift = USIZE_BITS - LSHIFT;
        (h1 << LSHIFT) ^ (h1 >> rshift) ^ self.fs2.hash_value()
    }
}

impl<FS1: FilterState, FS2: FilterState> PairFilterState<FS1, FS2> {
    #[inline]
    pub fn new(fs1: FS1, fs2: FS2) -> Self {
        Self { fs1, fs2 }
    }
    #[inline]
    pub fn state1(&self) -> &FS1 {
        &self.fs1
    }
    #[inline]
    pub fn state2(&self) -> &FS2 {
        &self.fs2
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TrivialFilterState {
    state: bool,
}
impl FilterState for TrivialFilterState {
    #[inline]
    fn no_state() -> Self {
        Self::new(false)
    }

    #[inline]
    fn hash_value(&self) -> usize {
        0
    }
}

impl TrivialFilterState {
    #[inline]
    pub fn new(state: bool) -> Self {
        Self { state }
    }
    #[inline]
    pub fn state(&self) -> bool {
        self.state
    }
}
