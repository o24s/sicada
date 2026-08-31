//! Caches for FSTs that are produced a state at a time.
//!
//! Port of OpenFst's `expander-cache.h`, together with the `ArcArenaStateStore`
//! half of `arc-arena.h`. Upstream files that class elsewhere, but it satisfies
//! the same contract and belongs beside the caches it competes with.
//!
//! An [`Expander`] knows how to produce one state of an FST that does not exist
//! in memory; a cache remembers the states already produced so a second visit
//! costs a lookup. [`ExpanderFst`](crate::fsts::expander_fst::ExpanderFst) is
//! the two glued together.
//!
//! # A cached state is borrowed, not owned
//!
//! Upstream's caches hand back a `State *` and let the caller mind its lifetime;
//! `NoGcKeepOneExpanderCache` even carries a hand-rolled reference count so it
//! can tell whether freeing a state would strand someone. sicada instead returns
//! a [`StateView`] borrowed from the cache, which the compiler ties to the
//! cache's own lifetime.
//!
//! What makes that possible is a promise every cache here keeps: **a cached
//! state is never removed, replaced, or moved out of the allocation it was first
//! written to.** None of these caches evicts, which is the meaning of "NoGc" in
//! upstream's own name for one of them, so the promise costs nothing. Each
//! `unsafe` block below restates the part of it that block depends on.
//!
//! Its counterpart [`CacheImpl`](crate::cache::CacheImpl) *does* evict, and so
//! hands out `Rc<[A]>` instead: a state that can be dropped under the caller's
//! feet has to be reference counted. The two are not interchangeable.
//!
//! # Expanding through a builder
//!
//! An expander writes into a [`StateBuilder`] rather than into the cached state
//! itself, which is how upstream's `ArcArenaStateStore` already works. The
//! indirection earns three things:
//!
//! - the cache picks its own storage, a boxed slice or a run inside a shared
//!   [`ArcArena`], without the expander knowing which;
//! - an expander that re-enters the cache while expanding cannot interleave its
//!   arcs with the outer expansion's, because nothing reaches the arena until
//!   the expander has returned; and
//! - the arc buffer is reused across expansions, so a run of a million states
//!   grows one buffer rather than a million.

use std::cell::{Cell, UnsafeCell};

use rustc_hash::FxHashMap;

use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::memory::{ArcArena, ArcRun};
use crate::weight::Weight;

/// Produces the states of an FST on demand.
pub trait Expander<A: Arc> {
    /// The initial state, or [`ArcStateId::no_state`] if there is none.
    fn start(&self) -> A::StateId;

    /// The number of states the FST will have once fully expanded.
    fn num_states(&self) -> usize;

    /// Writes the contents of `state_id` into `builder`.
    ///
    /// The builder arrives empty. Calling
    /// [`reserve_arcs`](StateBuilder::reserve_arcs) first lets the arc-arena
    /// store place the whole state in one run.
    fn expand(&self, state_id: A::StateId, builder: &mut StateBuilder<A>);
}

/// Collects one state's contents during expansion.
///
/// Corresponds to the sink upstream's expanders write to: `SimpleVectorCacheState`
/// for most caches, `ArcArenaStateStore::StateBuilder` for the arena store. Here
/// it is one type, so [`Expander::expand`] does not have to be generic over the
/// cache that called it.
pub struct StateBuilder<A: Arc> {
    final_weight: A::Weight,
    niepsilons: u32,
    noepsilons: u32,
    arcs: Vec<A>,
}

impl<A: Arc> Default for StateBuilder<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Arc> StateBuilder<A> {
    /// Creates an empty builder: no arcs, and not final.
    pub fn new() -> Self {
        Self {
            final_weight: A::Weight::zero(),
            niepsilons: 0,
            noepsilons: 0,
            arcs: Vec::new(),
        }
    }

    /// Makes the state final with `weight`, or non-final if it is `Zero`.
    #[inline]
    pub fn set_final(&mut self, weight: A::Weight) {
        self.final_weight = weight;
    }

    /// Reserves room for `n` more arcs.
    #[inline]
    pub fn reserve_arcs(&mut self, n: usize) {
        self.arcs.reserve(n);
    }

    /// Appends an outgoing arc, keeping the epsilon counts up to date.
    #[inline]
    pub fn add_arc(&mut self, arc: A) {
        self.niepsilons += u32::from(arc.ilabel() == A::Label::epsilon());
        self.noepsilons += u32::from(arc.olabel() == A::Label::epsilon());
        self.arcs.push(arc);
    }

    /// The final weight set so far.
    #[inline]
    pub fn final_weight(&self) -> &A::Weight {
        &self.final_weight
    }

    /// The arcs added so far.
    #[inline]
    pub fn arcs(&self) -> &[A] {
        &self.arcs
    }

    /// The number of arcs added so far.
    #[inline]
    pub fn num_arcs(&self) -> usize {
        self.arcs.len()
    }

    /// Empties the builder, keeping the arc buffer's capacity for the next state.
    fn reset(&mut self) {
        self.final_weight = A::Weight::zero();
        self.niepsilons = 0;
        self.noepsilons = 0;
        self.arcs.clear();
    }
}

/// A cached state, borrowed from the cache that holds it.
///
/// Stands in for upstream's `State *`. Every cache resolves its own
/// representation into this one shape, so a reader of
/// [`ExpanderFst`](crate::fsts::expander_fst::ExpanderFst) does not have to know
/// which storage is underneath.
#[derive(Debug)]
pub struct StateView<'a, A: Arc> {
    final_weight: &'a A::Weight,
    niepsilons: u32,
    noepsilons: u32,
    arcs: &'a [A],
}

// Written out rather than derived: the derive would demand `A: Clone + Copy`,
// which a view does not need, since every field is a reference or an integer.
impl<A: Arc> Clone for StateView<'_, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A: Arc> Copy for StateView<'_, A> {}

impl<'a, A: Arc> StateView<'a, A> {
    /// The state's final weight; `Zero` if it is not final.
    #[inline]
    pub fn final_weight(&self) -> &'a A::Weight {
        self.final_weight
    }

    /// The state's outgoing arcs.
    #[inline]
    pub fn arcs(&self) -> &'a [A] {
        self.arcs
    }

    /// The number of outgoing arcs.
    #[inline]
    pub fn num_arcs(&self) -> usize {
        self.arcs.len()
    }

    /// The number of outgoing arcs with an epsilon input label.
    #[inline]
    pub fn num_input_epsilons(&self) -> usize {
        self.niepsilons as usize
    }

    /// The number of outgoing arcs with an epsilon output label.
    #[inline]
    pub fn num_output_epsilons(&self) -> usize {
        self.noepsilons as usize
    }
}

/// Remembers states an [`Expander`] has already produced.
///
/// A cache must be cloneable, and a clone must share nothing with its original
/// so the two can be used from different threads. Upstream states the same
/// requirement in a comment.
pub trait ExpanderCache<A: Arc>: Clone {
    /// Returns `state_id`, expanding it first if this is its first visit.
    fn find_or_expand<E: Expander<A>>(
        &self,
        expander: &E,
        state_id: A::StateId,
    ) -> StateView<'_, A>;

    /// Returns `state_id` only if it has already been expanded.
    ///
    /// Upstream has this on `ArcArenaStateStore` alone; it is the cheap test a
    /// lookahead matcher wants before deciding whether a state is worth forcing.
    fn find(&self, state_id: A::StateId) -> Option<StateView<'_, A>>;
}

/// Runs one expansion through a reusable builder, handing the builder back.
///
/// `finish` converts the builder's contents into whatever the cache stores. It
/// runs after the expander has returned, so a re-entrant expander cannot
/// interleave its writes with this one's.
fn expand_with<A, E, S>(
    scratch: &Cell<StateBuilder<A>>,
    expander: &E,
    state_id: A::StateId,
    finish: impl FnOnce(&StateBuilder<A>) -> S,
) -> S
where
    A: Arc,
    E: Expander<A>,
{
    // Taking the builder leaves an empty one behind, so an expander that
    // re-enters the cache gets a builder of its own instead of appending to
    // this one.
    let mut builder = scratch.take();
    builder.reset();
    expander.expand(state_id, &mut builder);
    let state = finish(&builder);
    builder.reset();
    scratch.set(builder);
    state
}

/// A state whose arcs live in an allocation of their own.
///
/// Upstream calls this `SimpleVectorCacheState` and gives it a `std::vector`,
/// which keeps a capacity the state will never use again. A boxed slice holds
/// the same arcs in one exactly-sized allocation.
#[derive(Debug, Clone)]
pub struct SimpleVectorCacheState<A: Arc> {
    final_weight: A::Weight,
    niepsilons: u32,
    noepsilons: u32,
    arcs: Box<[A]>,
}

impl<A: Arc> SimpleVectorCacheState<A> {
    fn from_builder(builder: &StateBuilder<A>) -> Self {
        Self {
            final_weight: builder.final_weight.clone(),
            niepsilons: builder.niepsilons,
            noepsilons: builder.noepsilons,
            arcs: builder.arcs.as_slice().into(),
        }
    }

    fn view(&self) -> StateView<'_, A> {
        StateView {
            final_weight: &self.final_weight,
            niepsilons: self.niepsilons,
            noepsilons: self.noepsilons,
            arcs: &self.arcs,
        }
    }
}

/// Cache indexed by state ID, for FSTs whose states are numbered densely.
///
/// The default. A slot costs one pointer whether or not the state behind it was
/// ever visited, so a sparse walk over a wide state space is better served by
/// [`HashExpanderCache`].
pub struct VectorExpanderCache<A: Arc> {
    /// Boxed so that a state keeps its address when the vector grows.
    states: UnsafeCell<Vec<Option<Box<SimpleVectorCacheState<A>>>>>,
    scratch: Cell<StateBuilder<A>>,
}

impl<A: Arc> Default for VectorExpanderCache<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Arc> VectorExpanderCache<A> {
    /// Creates an empty cache.
    pub fn new() -> Self {
        Self {
            states: UnsafeCell::new(Vec::new()),
            scratch: Cell::new(StateBuilder::new()),
        }
    }

    #[inline]
    fn get(&self, index: usize) -> Option<&SimpleVectorCacheState<A>> {
        // SAFETY: a shared borrow of the vector, taken and dropped inside this
        // function. The reference handed out points into the boxed state, an
        // allocation the vector only owns and never writes to.
        let states = unsafe { &*self.states.get() };
        states.get(index)?.as_deref()
    }
}

impl<A: Arc> Clone for VectorExpanderCache<A> {
    fn clone(&self) -> Self {
        // SAFETY: `self` is borrowed shared for the length of the clone, and
        // nothing here mutates it.
        let states = unsafe { &*self.states.get() };
        Self {
            states: UnsafeCell::new(states.clone()),
            scratch: Cell::new(StateBuilder::new()),
        }
    }
}

impl<A: Arc> ExpanderCache<A> for VectorExpanderCache<A> {
    fn find_or_expand<E: Expander<A>>(
        &self,
        expander: &E,
        state_id: A::StateId,
    ) -> StateView<'_, A> {
        let index = state_id.as_usize();
        if let Some(state) = self.get(index) {
            return state.view();
        }
        let state = expand_with(
            &self.scratch,
            expander,
            state_id,
            SimpleVectorCacheState::from_builder,
        );
        // SAFETY: the exclusive borrow lives entirely inside this block, which
        // calls nothing that could re-enter the cache. It grows the vector,
        // moving the boxes but not the states they own, so any `StateView`
        // handed out earlier still points at live memory. The occupied test
        // keeps that true: a state already there is never dropped, which an
        // expander that re-entered and filled this slot depends on.
        unsafe {
            let states = &mut *self.states.get();
            if states.len() <= index {
                states.resize_with(index + 1, || None);
            }
            if states[index].is_none() {
                states[index] = Some(Box::new(state));
            }
        }
        self.get(index).expect("the slot was just filled").view()
    }

    #[inline]
    fn find(&self, state_id: A::StateId) -> Option<StateView<'_, A>> {
        Some(self.get(state_id.as_usize())?.view())
    }
}

/// Cache keyed by a hash of the state ID, for FSTs whose reachable states are
/// scattered across a wide range.
pub struct HashExpanderCache<A: Arc> {
    /// Boxed so that a state keeps its address when the table rehashes.
    states: UnsafeCell<FxHashMap<A::StateId, Box<SimpleVectorCacheState<A>>>>,
    scratch: Cell<StateBuilder<A>>,
}

impl<A: Arc> Default for HashExpanderCache<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Arc> HashExpanderCache<A> {
    /// Creates an empty cache.
    pub fn new() -> Self {
        Self {
            states: UnsafeCell::new(FxHashMap::default()),
            scratch: Cell::new(StateBuilder::new()),
        }
    }

    #[inline]
    fn get(&self, state_id: A::StateId) -> Option<&SimpleVectorCacheState<A>> {
        // SAFETY: as in `VectorExpanderCache::get`, a shared borrow of the
        // table that ends here, yielding a reference into a box the table owns
        // but never writes to.
        let states = unsafe { &*self.states.get() };
        states.get(&state_id).map(Box::as_ref)
    }
}

impl<A: Arc> Clone for HashExpanderCache<A> {
    fn clone(&self) -> Self {
        // SAFETY: `self` is only read here.
        let states = unsafe { &*self.states.get() };
        Self {
            states: UnsafeCell::new(states.clone()),
            scratch: Cell::new(StateBuilder::new()),
        }
    }
}

impl<A: Arc> ExpanderCache<A> for HashExpanderCache<A> {
    fn find_or_expand<E: Expander<A>>(
        &self,
        expander: &E,
        state_id: A::StateId,
    ) -> StateView<'_, A> {
        if let Some(state) = self.get(state_id) {
            return state.view();
        }
        let state = expand_with(
            &self.scratch,
            expander,
            state_id,
            SimpleVectorCacheState::from_builder,
        );
        // SAFETY: the exclusive borrow is confined to this block and re-enters
        // nothing. Rehashing moves the boxes, not the states inside them, so
        // views handed out earlier stay valid; `or_insert_with` leaves an entry
        // an expander may have inserted while re-entering exactly where it is.
        unsafe {
            (*self.states.get())
                .entry(state_id)
                .or_insert_with(|| Box::new(state))
        };
        self.get(state_id)
            .expect("the entry was just filled")
            .view()
    }

    #[inline]
    fn find(&self, state_id: A::StateId) -> Option<StateView<'_, A>> {
        Some(self.get(state_id)?.view())
    }
}

/// A state whose arcs live in a run of the store's shared [`ArcArena`].
struct ArenaState<A: Arc> {
    final_weight: A::Weight,
    niepsilons: u32,
    noepsilons: u32,
    run: ArcRun,
}

/// Cache that packs every state's arcs into one shared arena.
///
/// Port of `ArcArenaStateStore` from upstream's `arc-arena.h`. Where
/// [`HashExpanderCache`] gives each state an allocation for its arcs, this one
/// appends them to arena blocks that hold thousands of arcs apiece, so the cost
/// of expanding a state drops to a copy. Neighbouring states end up adjacent in
/// memory as well, which suits a traversal.
///
/// It never reclaims: an [`ArcRun`] stays valid until the store is dropped.
pub struct ArcArenaStateStore<A: Arc> {
    /// Boxed so that a state keeps its address when the table rehashes.
    states: UnsafeCell<FxHashMap<A::StateId, Box<ArenaState<A>>>>,
    arena: UnsafeCell<ArcArena<A>>,
    scratch: Cell<StateBuilder<A>>,
}

/// Arcs per arena block, matching upstream's `ArcArenaStateStore`.
const ARENA_BLOCK_SIZE: usize = 64 * 1024;

impl<A: Arc> Default for ArcArenaStateStore<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Arc> ArcArenaStateStore<A> {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self {
            states: UnsafeCell::new(FxHashMap::default()),
            arena: UnsafeCell::new(ArcArena::with_block_size(ARENA_BLOCK_SIZE)),
            scratch: Cell::new(StateBuilder::new()),
        }
    }

    fn view<'a>(&'a self, state: &'a ArenaState<A>) -> StateView<'a, A> {
        // SAFETY: a shared borrow of the arena that ends here. The slice it
        // yields points into an arena block, and a block is never reallocated,
        // shortened, or freed while the arena lives. Only `ArcArena::clear`
        // would do that, and this store never calls it.
        let arena = unsafe { &*self.arena.get() };
        let arcs = arena.arcs(state.run);
        StateView {
            final_weight: &state.final_weight,
            niepsilons: state.niepsilons,
            noepsilons: state.noepsilons,
            arcs,
        }
    }

    #[inline]
    fn get(&self, state_id: A::StateId) -> Option<&ArenaState<A>> {
        // SAFETY: a shared borrow of the table that ends here, yielding a
        // reference into a box the table owns but never writes to.
        let states = unsafe { &*self.states.get() };
        states.get(&state_id).map(Box::as_ref)
    }
}

impl<A: Arc> Clone for ArcArenaStateStore<A> {
    /// Copies every state into an arena of its own, sharing nothing.
    ///
    /// The runs are laid out afresh in insertion order, so a clone is if
    /// anything better packed than its original.
    fn clone(&self) -> Self {
        let clone = Self::new();
        // SAFETY: `self` is only read, and the borrows of the clone's arena and
        // table are confined to this block.
        unsafe {
            let states = &*self.states.get();
            let source = &*self.arena.get();
            let arena = &mut *clone.arena.get();
            let cloned_states = &mut *clone.states.get();
            cloned_states.reserve(states.len());
            for (&state_id, state) in states {
                let arcs = source.arcs(state.run);
                arena.reserve_arcs(arcs.len());
                for arc in arcs {
                    arena.push_arc(arc.clone());
                }
                cloned_states.insert(
                    state_id,
                    Box::new(ArenaState {
                        final_weight: state.final_weight.clone(),
                        niepsilons: state.niepsilons,
                        noepsilons: state.noepsilons,
                        run: arena.commit_arcs(),
                    }),
                );
            }
        }
        clone
    }
}

impl<A: Arc> ExpanderCache<A> for ArcArenaStateStore<A> {
    fn find_or_expand<E: Expander<A>>(
        &self,
        expander: &E,
        state_id: A::StateId,
    ) -> StateView<'_, A> {
        if let Some(state) = self.get(state_id) {
            return self.view(state);
        }
        let state = expand_with(&self.scratch, expander, state_id, |builder| {
            // SAFETY: the exclusive borrow of the arena is confined to this
            // closure, which runs after the expander has returned and calls
            // nothing that could re-enter. Appending grows the arena with new
            // blocks and never touches the ones already committed, so runs
            // handed out earlier still resolve.
            let arena = unsafe { &mut *self.arena.get() };
            arena.reserve_arcs(builder.num_arcs());
            for arc in builder.arcs() {
                arena.push_arc(arc.clone());
            }
            ArenaState {
                final_weight: builder.final_weight().clone(),
                niepsilons: builder.niepsilons,
                noepsilons: builder.noepsilons,
                run: arena.commit_arcs(),
            }
        });
        // SAFETY: the exclusive borrow is confined to this block. As in
        // `HashExpanderCache`, rehashing moves boxes rather than states, and an
        // entry a re-entrant expander already inserted is left alone, at the
        // cost of the run just committed, which stays unreferenced until the
        // store is dropped.
        unsafe {
            (*self.states.get())
                .entry(state_id)
                .or_insert_with(|| Box::new(state))
        };
        self.view(self.get(state_id).expect("the entry was just filled"))
    }

    #[inline]
    fn find(&self, state_id: A::StateId) -> Option<StateView<'_, A>> {
        Some(self.view(self.get(state_id)?))
    }
}

/// The cache an [`ExpanderFst`](crate::fsts::expander_fst::ExpanderFst) uses
/// unless told otherwise.
pub type DefaultExpanderCache<A> = VectorExpanderCache<A>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::weights::float_weight::TropicalWeight;
    use std::cell::RefCell;

    /// Expands state `n` into `n % 5` arcs, half of them epsilons, and makes
    /// every third state final. Records what it was asked for.
    #[derive(Default)]
    struct Counting {
        expansions: RefCell<Vec<i32>>,
    }

    impl Counting {
        fn count(&self, state: i32) -> usize {
            self.expansions
                .borrow()
                .iter()
                .filter(|&&s| s == state)
                .count()
        }
    }

    impl Expander<StdArc> for Counting {
        fn start(&self) -> i32 {
            0
        }

        fn num_states(&self) -> usize {
            0
        }

        fn expand(&self, state_id: i32, builder: &mut StateBuilder<StdArc>) {
            self.expansions.borrow_mut().push(state_id);
            if state_id % 3 == 0 {
                builder.set_final(TropicalWeight(state_id as f32));
            }
            let narcs = state_id % 5;
            builder.reserve_arcs(narcs as usize);
            for i in 0..narcs {
                let ilabel = if i % 2 == 0 { 0 } else { state_id + i };
                builder.add_arc(StdArc::new(
                    ilabel,
                    state_id + i,
                    TropicalWeight(i as f32),
                    state_id + i,
                ));
            }
        }
    }

    fn expected_final(state: i32) -> TropicalWeight {
        if state % 3 == 0 {
            TropicalWeight(state as f32)
        } else {
            TropicalWeight::zero()
        }
    }

    fn check_state(view: StateView<'_, StdArc>, state: i32) {
        assert_eq!(*view.final_weight(), expected_final(state));
        let narcs = (state % 5) as usize;
        assert_eq!(view.num_arcs(), narcs);
        assert_eq!(view.num_input_epsilons(), narcs.div_ceil(2));
        // No arc gets an epsilon output label: `state + i` is zero only for
        // state 0, which has no arcs.
        assert_eq!(view.num_output_epsilons(), 0);
        for (i, arc) in view.arcs().iter().enumerate() {
            assert_eq!(arc.olabel(), state + i as i32);
            assert_eq!(arc.nextstate(), state + i as i32);
        }
    }

    /// Runs one body against each cache, so a divergence between them shows up
    /// as a failure of the same test rather than of a test only one has.
    macro_rules! for_each_cache {
        ($(#[$meta:meta])* $name:ident, |$cache:ident: $ty:ident| $body:block) => {
            $(#[$meta])*
            mod $name {
                use super::*;

                fn run<$ty: ExpanderCache<StdArc> + Default>() {
                    let $cache = <$ty>::default();
                    $body
                }

                #[test]
                fn vector() {
                    run::<VectorExpanderCache<StdArc>>();
                }

                #[test]
                fn hash() {
                    run::<HashExpanderCache<StdArc>>();
                }

                #[test]
                fn arc_arena() {
                    run::<ArcArenaStateStore<StdArc>>();
                }
            }
        };
    }

    for_each_cache!(a_state_is_expanded_once, |cache: C| {
        let expander = Counting::default();
        for _ in 0..3 {
            for state in 0..20 {
                check_state(cache.find_or_expand(&expander, state), state);
            }
        }
        for state in 0..20 {
            assert_eq!(expander.count(state), 1, "state {state}");
        }
    });

    for_each_cache!(find_does_not_expand, |cache: C| {
        let expander = Counting::default();
        assert!(cache.find(7).is_none());
        check_state(cache.find_or_expand(&expander, 7), 7);
        check_state(cache.find(7).expect("expanded"), 7);
        assert!(cache.find(8).is_none());
        assert_eq!(expander.expansions.borrow().len(), 1);
    });

    for_each_cache!(
        /// The invariant the whole design rests on: a view keeps reading its own
        /// state however much the cache grows underneath it. Both the storage that
        /// holds the states and the one that holds their arcs get reallocated many
        /// times over during this.
        views_survive_later_expansions, |cache: C| {
        let expander = Counting::default();
        let mut views = Vec::new();
        for state in 0..2000 {
            views.push((state, cache.find_or_expand(&expander, state)));
        }
        for (state, view) in views {
            check_state(view, state);
        }
    });

    for_each_cache!(
        /// States expanded out of order still land where they belong: the vector
        /// cache resizes for each, and the gaps must stay empty rather than count
        /// as cached.
        sparse_state_ids_are_cached_where_they_belong, |cache: C| {
        let expander = Counting::default();
        let mut views = Vec::new();
        for state in [900, 3, 5000, 17, 1, 4999] {
            views.push((state, cache.find_or_expand(&expander, state)));
        }
        assert!(cache.find(4).is_none());
        assert!(cache.find(4998).is_none());
        for (state, view) in views {
            check_state(view, state);
        }
        assert_eq!(expander.expansions.borrow().len(), 6);
    });

    for_each_cache!(
        /// Upstream requires that a copy share nothing with its original, so the
        /// two can be handed to different threads.
        a_clone_shares_nothing, |cache: C| {
        let expander = Counting::default();
        for state in 0..50 {
            cache.find_or_expand(&expander, state);
        }
        let clone = cache.clone();

        for state in 0..50 {
            let original = cache.find(state).expect("cached").arcs();
            let copied = clone.find(state).expect("copied").arcs();
            assert_eq!(original, copied);
            if !original.is_empty() {
                assert!(
                    !std::ptr::eq(original.as_ptr(), copied.as_ptr()),
                    "state {state} shares its arcs with the copy"
                );
            }
        }

        // Expanding through the copy leaves the original alone, and the reverse.
        clone.find_or_expand(&expander, 50);
        assert!(cache.find(50).is_none());
        cache.find_or_expand(&expander, 51);
        assert!(clone.find(51).is_none());
        for state in 0..50 {
            check_state(clone.find(state).expect("copied"), state);
        }
    });

    for_each_cache!(
        /// An expander that walks back into the cache while expanding must not lose
        /// the arcs of either state, nor drop a state someone is already holding.
        a_re_entrant_expander_keeps_both_states, |cache: C| {
        struct Nested<'a, C: ExpanderCache<StdArc>> {
            cache: &'a C,
            inner: Counting,
        }

        impl<C: ExpanderCache<StdArc>> Expander<StdArc> for Nested<'_, C> {
            fn start(&self) -> i32 {
                0
            }

            fn num_states(&self) -> usize {
                0
            }

            fn expand(&self, state_id: i32, builder: &mut StateBuilder<StdArc>) {
                // Build part of this state, walk back into the cache for
                // another one, then finish this one, so that the two expansions
                // are interleaved and each must keep its own arcs.
                self.inner.expand(state_id, builder);
                if state_id == 1 {
                    check_state(self.cache.find_or_expand(self, 9), 9);
                    builder.add_arc(StdArc::new(4, 4, TropicalWeight(4.0), 4));
                }
            }
        }

        let expander = Nested {
            cache: &cache,
            inner: Counting::default(),
        };
        let view = expander.cache.find_or_expand(&expander, 1);
        check_state(expander.cache.find(9).expect("cached"), 9);
        // State 1 keeps the arc added after the nested expansion returned,
        // rather than any of state 9's.
        assert_eq!(view.num_arcs(), 2);
        assert_eq!(view.arcs()[0].olabel(), 1);
        assert_eq!(view.arcs()[1].olabel(), 4);
        assert_eq!(view.arcs(), expander.cache.find(1).expect("cached").arcs());
    });

    #[test]
    fn a_builder_counts_epsilons_on_each_side() {
        let mut builder = StateBuilder::<StdArc>::new();
        assert_eq!(*builder.final_weight(), TropicalWeight::zero());
        assert_eq!(builder.num_arcs(), 0);

        builder.set_final(TropicalWeight(2.5));
        builder.add_arc(StdArc::new(0, 0, TropicalWeight::one(), 1));
        builder.add_arc(StdArc::new(0, 7, TropicalWeight::one(), 2));
        builder.add_arc(StdArc::new(7, 0, TropicalWeight::one(), 3));
        builder.add_arc(StdArc::new(7, 7, TropicalWeight::one(), 4));

        assert_eq!(*builder.final_weight(), TropicalWeight(2.5));
        assert_eq!(builder.num_arcs(), 4);
        assert_eq!(builder.niepsilons, 2);
        assert_eq!(builder.noepsilons, 2);

        builder.reset();
        assert_eq!(*builder.final_weight(), TropicalWeight::zero());
        assert_eq!(builder.num_arcs(), 0);
        assert_eq!(builder.niepsilons, 0);
        assert_eq!(builder.noepsilons, 0);
    }

    /// The point of the arena store: one block holds many states' arcs, so a
    /// state costs a copy rather than an allocation.
    #[test]
    fn the_arena_store_packs_states_into_shared_blocks() {
        let cache = ArcArenaStateStore::<StdArc>::new();
        let expander = Counting::default();
        // States 1..5 have 1..4 arcs each, well inside one block.
        let views: Vec<_> = (1..5)
            .map(|state| cache.find_or_expand(&expander, state).arcs())
            .collect();
        let first = views[0].as_ptr();
        for (i, arcs) in views.iter().enumerate() {
            let offset = (0..i).map(|j| views[j].len()).sum::<usize>();
            assert_eq!(
                arcs.as_ptr(),
                // SAFETY: every run so far was committed to the same block, so
                // the pointers are into one allocation and comparable.
                unsafe { first.add(offset) },
                "run {i} is not where the previous one ended"
            );
        }
    }
}
