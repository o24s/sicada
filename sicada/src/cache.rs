//! A cache for FSTs whose states are computed on demand.
//!
//! Port of OpenFst's `cache.h`. Where a delayed FST would otherwise recompute a
//! state every time it is visited, this remembers what it produced, and, when
//! told to, forgets the states it can spare so a traversal of something
//! unbounded does not grow without limit.
//!
//! # Two things upstream has that are not here
//!
//! `CacheImplOptions` carries a raw pointer to a cache store plus an
//! `own_store` flag saying whether to delete it. Passing a store by value says
//! the same thing; there is no second question about who frees it.
//!
//! The choice of *which* state to forget is upstream's `GCCacheStore`, which
//! sweeps states whose `kCacheRecent` bit is clear. sicada uses SIEVE, a
//! clock-like policy with the same one-bit-per-state cost and a better hit rate
//! on the access patterns a traversal produces. Either way the choice is
//! invisible: a state that was forgotten is recomputed, and arcs already handed
//! out are reference counted so that forgetting cannot take them from a caller
//! mid-iteration.

use std::cell::Cell;
use std::mem::size_of;
use std::ops::Deref;
use std::rc::Rc;

use rustc_hash::FxHashMap;

use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::data_structures::{FastCell, GrowableBitSet};
use crate::memory::MemoryPool;
use crate::weight::Weight;

bitflags::bitflags! {
    /// Flags for managing cache status and SIEVE algorithm metadata.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    /// SICADA-DIVERGE: the bit values are not upstream's, and neither is the
    /// set: `EXPANDING_FINAL` and `EXPANDING_ARCS` have no counterpart, and
    /// upstream's `kCacheRecent` is `VISITED` here. None of this reaches a
    /// file: unlike the FST property bits, cache flags live only in memory, so
    /// what they are numbered is nobody's business but this module's.
    pub struct CacheFlags: u8 {
        /// Metadata for this state has been initialized.
        const INIT            = 0x01;
        /// Final weight for this state has been computed and cached.
        const FINAL           = 0x02;
        /// Outgoing arcs for this state have been computed and cached.
        const ARCS            = 0x04;
        /// Access flag used by the SIEVE algorithm (clock-like replacement).
        const VISITED         = 0x08;
        /// State is currently being expanded for its final weight (prevents reentrancy).
        const EXPANDING_FINAL = 0x10;
        /// State is currently being expanded for its outgoing arcs (prevents reentrancy).
        const EXPANDING_ARCS  = 0x20;
    }
}

/// Configuration for the cache eviction policy.
#[derive(Debug, Clone)]
pub struct CacheOptions {
    /// Whether to enable garbage collection.
    pub gc: bool,
    /// Threshold in bytes to trigger garbage collection.
    pub gc_limit: usize,
    /// Target ratio of retained memory after GC (e.g., 0.666 keeps 2/3 of the limit).
    pub gc_fraction: f64,
}

impl Default for CacheOptions {
    fn default() -> Self {
        Self {
            gc: true,
            // Upstream's `fst_default_cache_gc_limit` flag, in `fst.cc`.
            gc_limit: 1 << 20,
            gc_fraction: 0.666,
        }
    }
}

/// Interface for lazy expansion of states in the cache.
/// Implementations provide logic to fetch state data from the source FST.
pub trait CacheExpander<A: Arc> {
    /// Computes the final weight of a state. Returns `None` if it is not a final state.
    fn expand_final(&self, state: A::StateId) -> Option<A::Weight>;
    /// Computes the outgoing arcs from a state.
    fn expand_arcs(&self, state: A::StateId) -> Vec<A>;
}

/// A cached state entry stored in the MemoryPool (Slab).
#[derive(Debug, Clone)]
pub struct CacheState<A: Arc> {
    /// The ID of the state in the original FST.
    pub state_id: A::StateId,
    /// Status flags for expansion and cache replacement logic.
    pub flags: CacheFlags,

    /// Cached final weight.
    pub final_weight: Option<A::Weight>,
    /// Cached outgoing arcs. Shared via Rc to allow zero-copy access.
    pub arcs: Option<Rc<[A]>>,

    /// Number of input epsilon arcs.
    pub niepsilons: usize,
    /// Number of output epsilon arcs.
    pub noepsilons: usize,

    // Doubly linked list pointers for the replacement algorithm.
    next: Option<usize>,
    prev: Option<usize>,
}

impl<A: Arc> CacheState<A> {
    /// Marks the state as recently accessed for the SIEVE algorithm.
    #[inline(always)]
    fn mark_visited(&mut self) {
        self.flags.insert(CacheFlags::VISITED);
    }

    /// Checks if external objects (like iterators) are currently holding a reference to the arcs.
    /// Used by the GC to avoid evicting data that is still in use.
    #[inline(always)]
    fn is_pinned(&self) -> bool {
        // Strong count > 1 means someone other than the CacheStore is holding the Rc.
        self.arcs
            .as_ref()
            .is_some_and(|rc| Rc::strong_count(rc) > 1)
    }

    /// Estimates the total memory footprint of this state entry in bytes.
    fn memory_estimate(&self) -> usize {
        size_of::<Self>() + self.arcs.as_ref().map_or(0, |rc| rc.len() * size_of::<A>())
    }
}

const DENSE_LIMIT: usize = 1_000_000;

/// A tiered index map optimized for O(1) lookup of state IDs.
///
/// In WFST algorithms, state IDs are typically allocated contiguously.
/// This map exploits this property by providing:
///
/// - Hot Path: Direct indexing via a `Vec` for IDs < `DENSE_LIMIT`.
///   This bypasses hashing overhead and maximizes cache locality.
/// - Safety Fallback: An `FxHashMap` for sparse or extremely large IDs.
///   This prevents catastrophic memory allocation (OOM) if IDs are sparse.
struct HybridIndexMap {
    dense: Vec<Option<usize>>,
    sparse: FxHashMap<usize, usize>,
}

impl HybridIndexMap {
    fn new() -> Self {
        Self {
            dense: Vec::new(),
            sparse: FxHashMap::default(),
        }
    }

    #[inline(always)]
    fn get(&self, id: usize) -> Option<usize> {
        if id < DENSE_LIMIT {
            self.dense.get(id).copied().flatten()
        } else {
            self.sparse.get(&id).copied()
        }
    }

    #[inline]
    fn insert(&mut self, id: usize, idx: usize) {
        if id < DENSE_LIMIT {
            if id >= self.dense.len() {
                self.dense.resize(id + 1, None);
            }
            self.dense[id] = Some(idx);
        } else {
            self.sparse.insert(id, idx);
        }
    }

    #[inline]
    fn remove(&mut self, id: usize) {
        if id < DENSE_LIMIT {
            if let Some(entry) = self.dense.get_mut(id) {
                *entry = None;
            }
        } else {
            self.sparse.remove(&id);
        }
    }

    fn clear(&mut self) {
        self.dense.clear();
        self.sparse.clear();
    }
}

/// The internal storage component that manages memory allocation and eviction.
struct CacheStore<A: Arc> {
    pool: MemoryPool<CacheState<A>>,
    map: HybridIndexMap,
    options: CacheOptions,

    current_size_bytes: usize,
    current_gc_limit: usize,

    head: Option<usize>,
    tail: Option<usize>,
    /// Hand pointer for the SIEVE eviction algorithm.
    hand: Option<usize>,
}

impl<A: Arc> CacheStore<A> {
    fn new(mut options: CacheOptions) -> Self {
        options.gc_fraction = if options.gc_fraction.is_nan() {
            0.666
        } else {
            options.gc_fraction.clamp(0.0, 1.0)
        };

        let limit = options.gc_limit;
        Self {
            pool: MemoryPool::with_capacity(1024),
            map: HybridIndexMap::new(),
            options,
            current_size_bytes: 0,
            current_gc_limit: limit,
            head: None,
            tail: None,
            hand: None,
        }
    }

    /// Removes a node from the replacement list and memory pool.
    fn remove_node(&mut self, idx: usize) {
        let state_size = self.pool[idx].memory_estimate();
        self.current_size_bytes = self.current_size_bytes.saturating_sub(state_size);

        let next = self.pool[idx].next;
        let prev = self.pool[idx].prev;

        if let Some(p) = prev {
            self.pool[p].next = next;
        } else {
            self.head = next;
        }
        if let Some(n) = next {
            self.pool[n].prev = prev;
        } else {
            self.tail = prev;
        }

        if self.hand == Some(idx) {
            self.hand = prev.or(self.tail);
        }

        let id_usize = self.pool[idx].state_id.as_usize();
        self.map.remove(id_usize);
        self.pool.remove(idx);
    }

    /// Inserts a node at the head of the linked list.
    fn push_head(&mut self, idx: usize) {
        if let Some(old_head) = self.head {
            self.pool[idx].next = Some(old_head);
            self.pool[idx].prev = None;
            self.pool[old_head].prev = Some(idx);
            self.head = Some(idx);
        } else {
            self.pool[idx].next = None;
            self.pool[idx].prev = None;
            self.head = Some(idx);
            self.tail = Some(idx);
            self.hand = Some(idx);
        }
    }

    /// Executes garbage collection using the SIEVE algorithm.
    ///
    /// The `protect_idx` parameter specifies a node that must not be evicted
    /// (e.g., a node currently being initialized).
    fn gc(&mut self, protect_idx: Option<usize>) {
        if !self.options.gc || self.current_size_bytes <= self.current_gc_limit {
            return;
        }

        let target_size = (self.current_gc_limit as f64 * self.options.gc_fraction) as usize;
        let max_steps = self.pool.len();

        // First pass: Clear VISITED flags and evict unvisited nodes.
        let mut steps = 0;
        let mut current_opt = self.hand.or(self.tail);
        while self.current_size_bytes > target_size && steps < max_steps {
            steps += 1;
            if let Some(current) = current_opt {
                let state = &mut self.pool[current];

                // Protection check based on physical index and pin status.
                let is_protected = Some(current) == protect_idx
                    || state.is_pinned()
                    || state
                        .flags
                        .intersects(CacheFlags::EXPANDING_FINAL | CacheFlags::EXPANDING_ARCS);

                if is_protected {
                    current_opt = state.prev.or(self.tail);
                    continue;
                }

                if state.flags.contains(CacheFlags::VISITED) {
                    state.flags.remove(CacheFlags::VISITED);
                    current_opt = state.prev.or(self.tail);
                } else {
                    let next = state.prev.or(self.tail);
                    self.remove_node(current);
                    current_opt = next;
                }
                self.hand = current_opt;
            } else {
                break;
            }
        }

        // Second pass: If target size still not met, evict even previously VISITED nodes.
        let mut steps = 0;
        let mut current_opt = self.hand.or(self.tail);
        while self.current_size_bytes > target_size && steps < max_steps {
            steps += 1;
            if let Some(current) = current_opt {
                let state = &mut self.pool[current];
                let is_protected = Some(current) == protect_idx
                    || state.is_pinned()
                    || state
                        .flags
                        .intersects(CacheFlags::EXPANDING_FINAL | CacheFlags::EXPANDING_ARCS);

                if is_protected {
                    current_opt = state.prev.or(self.tail);
                    continue;
                }
                let next = state.prev.or(self.tail);
                self.remove_node(current);
                current_opt = next;
                self.hand = current_opt;
            } else {
                break;
            }
        }

        // If target still not reached (e.g. too many pinned nodes), expand the limit to avoid deadlock.
        while self.current_size_bytes > self.current_gc_limit {
            self.current_gc_limit = self.current_gc_limit.saturating_mul(2).max(1024);
        }
    }

    /// Allocates a slot for a state ID if not present, otherwise returns existing index.
    fn allocate_slot_if_needed(&mut self, id: A::StateId) -> usize {
        let id_usize = id.as_usize();

        if let Some(idx) = self.map.get(id_usize) {
            return idx;
        }

        self.gc(None);

        let state = CacheState {
            state_id: id,
            flags: CacheFlags::INIT,
            final_weight: None,
            arcs: None,
            niepsilons: 0,
            noepsilons: 0,
            next: None,
            prev: None,
        };

        self.current_size_bytes += size_of::<CacheState<A>>();

        let idx = self.pool.insert(state);
        self.map.insert(id_usize, idx);
        self.push_head(idx);

        idx
    }
}

/// RAII guard to track expansion state. Ensures flags are cleared even if expansion panics.
struct ExpandingGuard<'a, A: Arc> {
    store: &'a FastCell<CacheStore<A>>,
    idx: usize,
    flag: CacheFlags,
    committed: bool,
}

impl<'a, A: Arc> ExpandingGuard<'a, A> {
    fn new(store: &'a FastCell<CacheStore<A>>, idx: usize, flag: CacheFlags) -> Self {
        store.borrow_mut().pool[idx].flags.insert(flag);
        Self {
            store,
            idx,
            flag,
            committed: false,
        }
    }

    /// Successfully complete the expansion and clear the flag.
    fn commit(mut self) {
        self.store.borrow_mut().pool[self.idx]
            .flags
            .remove(self.flag);
        self.committed = true;
    }
}

impl<'a, A: Arc> Drop for ExpandingGuard<'a, A> {
    fn drop(&mut self) {
        if !self.committed
            && let Some(state) = self.store.borrow_mut().pool.get_mut(self.idx)
        {
            state.flags.remove(self.flag);
        }
    }
}

/// A generic cache implementation for FST states.
pub struct CacheImpl<A: Arc> {
    has_start: Cell<bool>,
    cache_start: Cell<Option<A::StateId>>,
    nknown_states: Cell<usize>,
    min_unexpanded_state_id: Cell<usize>,
    max_expanded_state_id: Cell<Option<usize>>,
    expanded_states: FastCell<GrowableBitSet>,
    store: FastCell<CacheStore<A>>,
}

impl<A: Arc> CacheImpl<A> {
    pub fn new(options: CacheOptions) -> Self {
        Self {
            has_start: Cell::new(false),
            cache_start: Cell::new(None),
            nknown_states: Cell::new(0),
            min_unexpanded_state_id: Cell::new(0),
            max_expanded_state_id: Cell::new(None),
            expanded_states: FastCell::new(GrowableBitSet::new()),
            store: FastCell::new(CacheStore::new(options)),
        }
    }

    #[inline]
    pub fn start(&self) -> Option<A::StateId> {
        self.cache_start.get()
    }

    #[inline]
    pub fn set_start(&self, s: A::StateId) {
        self.cache_start.set(Some(s));
        self.has_start.set(true);
        self.update_nknown_states(s.as_usize() + 1);
    }

    #[inline]
    pub fn has_start(&self) -> bool {
        self.has_start.get()
    }

    #[inline]
    fn update_nknown_states(&self, new_count: usize) {
        if new_count > self.nknown_states.get() {
            self.nknown_states.set(new_count);
        }
    }

    fn set_expanded_state(&self, s: usize) {
        if self.max_expanded_state_id.get().is_none_or(|m| s > m) {
            self.max_expanded_state_id.set(Some(s));
        }
        if s == self.min_unexpanded_state_id.get() {
            self.min_unexpanded_state_id.set(s + 1);
        }

        self.expanded_states.borrow_mut().insert(s);
    }

    /// Retrieves the final weight for a state, computing it via `expander` if not cached.
    pub fn final_weight<E: CacheExpander<A>>(&self, s: A::StateId, expander: &E) -> A::Weight {
        let idx = self.store.borrow_mut().allocate_slot_if_needed(s);

        {
            let mut store = self.store.borrow_mut();
            let state = &mut store.pool[idx];

            if state.flags.contains(CacheFlags::FINAL) {
                state.mark_visited();
                return state.final_weight.clone().unwrap_or_else(A::Weight::zero);
            }
            if state.flags.contains(CacheFlags::EXPANDING_FINAL) {
                panic!(
                    "Recursive cache expansion detected for final_weight on state {}",
                    s.as_usize()
                );
            }
        }

        let guard = ExpandingGuard::new(&self.store, idx, CacheFlags::EXPANDING_FINAL);
        let fw = expander.expand_final(s);
        guard.commit();

        let mut store = self.store.borrow_mut();
        let state = &mut store.pool[idx];
        state.final_weight = fw.clone();
        state.flags.insert(CacheFlags::FINAL);
        state.mark_visited();

        fw.unwrap_or_else(A::Weight::zero)
    }

    /// Internal process to ensure arcs are expanded and return the physical Slab index.
    /// This avoids redundant lookups in arc-related methods.
    fn ensure_arcs<E: CacheExpander<A>>(&self, s: A::StateId, expander: &E) -> usize {
        let idx = self.store.borrow_mut().allocate_slot_if_needed(s);

        {
            let mut store = self.store.borrow_mut();
            let state = &mut store.pool[idx];
            if state.flags.contains(CacheFlags::ARCS) {
                state.mark_visited();
                return idx;
            }
            if state.flags.contains(CacheFlags::EXPANDING_ARCS) {
                panic!(
                    "Recursive cache expansion detected for arcs on state {}",
                    s.as_usize()
                );
            }
        }

        let guard = ExpandingGuard::new(&self.store, idx, CacheFlags::EXPANDING_ARCS);
        let arcs_vec = expander.expand_arcs(s);
        guard.commit();

        let mut nieps = 0;
        let mut noeps = 0;
        let mut max_next = 0;

        for arc in &arcs_vec {
            if arc.ilabel() == A::Label::epsilon() {
                nieps += 1;
            }
            if arc.olabel() == A::Label::epsilon() {
                noeps += 1;
            }
            let next_usize = arc.nextstate().as_usize();
            if next_usize > max_next {
                max_next = next_usize;
            }
        }

        let arcs_rc: Rc<[A]> = arcs_vec.into_boxed_slice().into();
        let arcs_size = arcs_rc.len() * size_of::<A>();

        let mut store = self.store.borrow_mut();
        let state = &mut store.pool[idx];
        state.arcs = Some(arcs_rc);
        state.niepsilons = nieps;
        state.noepsilons = noeps;
        state.flags.insert(CacheFlags::ARCS);
        state.mark_visited();

        store.current_size_bytes += arcs_size;

        // Explicitly protect this recently expanded node from being immediately evicted by GC.
        store.gc(Some(idx));
        drop(store);

        self.update_nknown_states(max_next + 1);
        self.set_expanded_state(s.as_usize());

        idx
    }

    /// Returns a guard that allows zero-copy access to the state's arcs.
    ///
    /// This hands out the underlying arc slice without cloning it, which is what
    /// a traversal on a hot path wants.
    #[inline]
    pub fn arcs_slice<E: CacheExpander<A>>(
        &self,
        s: A::StateId,
        expander: &E,
    ) -> CacheArcsGuard<A> {
        let idx = self.ensure_arcs(s, expander);
        CacheArcsGuard {
            arcs: self.store.borrow().pool[idx].arcs.clone().unwrap(),
        }
    }

    /// Returns an iterator over the state's arcs, for `Fst` trait compatibility.
    ///
    /// Each call to `next()` clones one arc; use `arcs_slice()` for zero-copy
    /// access.
    #[inline]
    pub fn arcs_iter<E: CacheExpander<A>>(&self, s: A::StateId, expander: &E) -> CacheArcIter<A> {
        let idx = self.ensure_arcs(s, expander);
        CacheArcIter {
            arcs: self.store.borrow().pool[idx].arcs.clone().unwrap(),
            pos: 0,
        }
    }

    #[inline]
    pub fn has_arcs(&self, s: A::StateId) -> bool {
        let store = self.store.borrow();
        if let Some(idx) = store.map.get(s.as_usize())
            && let Some(state) = store.pool.get(idx)
        {
            return state.flags.contains(CacheFlags::ARCS);
        }
        false
    }

    #[inline]
    pub fn num_arcs<E: CacheExpander<A>>(&self, s: A::StateId, expander: &E) -> usize {
        let idx = self.ensure_arcs(s, expander);
        self.store.borrow().pool[idx].arcs.as_ref().unwrap().len()
    }

    #[inline]
    pub fn num_input_epsilons<E: CacheExpander<A>>(&self, s: A::StateId, expander: &E) -> usize {
        let idx = self.ensure_arcs(s, expander);
        self.store.borrow().pool[idx].niepsilons
    }

    #[inline]
    pub fn num_output_epsilons<E: CacheExpander<A>>(&self, s: A::StateId, expander: &E) -> usize {
        let idx = self.ensure_arcs(s, expander);
        self.store.borrow().pool[idx].noepsilons
    }

    pub fn clear(&self) {
        self.nknown_states.set(0);
        self.min_unexpanded_state_id.set(0);
        self.max_expanded_state_id.set(None);
        self.has_start.set(false);
        self.cache_start.set(None);
        self.expanded_states.borrow_mut().clear();

        let mut store = self.store.borrow_mut();
        store.map.clear();
        store.pool.clear();
        store.head = None;
        store.tail = None;
        store.hand = None;
        store.current_size_bytes = 0;
        store.current_gc_limit = store.options.gc_limit;
    }

    #[inline]
    pub fn num_known_states(&self) -> usize {
        self.nknown_states.get()
    }
}

/// Smart pointer to a cached arc slice.
/// Prevents the arcs from being evicted by GC while in scope, providing zero-copy access.
pub struct CacheArcsGuard<A: Arc> {
    arcs: Rc<[A]>,
}

impl<A: Arc> Deref for CacheArcsGuard<A> {
    type Target = [A];
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.arcs
    }
}

/// Iterator for cached arcs (Fst trait compatibility).
pub struct CacheArcIter<A: Arc> {
    arcs: Rc<[A]>,
    pos: usize,
}

impl<A: Arc> Iterator for CacheArcIter<A> {
    type Item = A;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.arcs.len() {
            let arc = self.arcs[self.pos].clone();
            self.pos += 1;
            Some(arc)
        } else {
            None
        }
    }
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.arcs.len() - self.pos;
        (remaining, Some(remaining))
    }
}
impl<A: Arc> ExactSizeIterator for CacheArcIter<A> {}
impl<A: Arc> Clone for CacheArcIter<A> {
    fn clone(&self) -> Self {
        Self {
            arcs: Rc::clone(&self.arcs),
            pos: self.pos,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::weight::Weight;

    use super::*;
    use std::panic::AssertUnwindSafe;
    use std::str::FromStr;

    #[derive(Clone, PartialEq, Debug)]
    struct DummyWeight(f32);
    impl Weight for DummyWeight {
        type ReverseWeight = Self;
        fn zero() -> Self {
            DummyWeight(f32::INFINITY)
        }
        fn one() -> Self {
            DummyWeight(0.0)
        }
        fn no_weight() -> Self {
            DummyWeight(f32::NAN)
        }
        fn type_name() -> crate::fst_type::WeightType {
            crate::fst_type::WeightType::new("dummy")
        }
        fn properties() -> u64 {
            0
        }
        fn plus(&self, _: &Self) -> Self {
            unimplemented!()
        }
        fn times(&self, _: &Self) -> Self {
            unimplemented!()
        }
        fn reverse(&self) -> Self::ReverseWeight {
            unimplemented!()
        }
        fn is_member(&self) -> bool {
            true
        }
        fn approx_equal(&self, _: &Self, _: f32) -> bool {
            true
        }
        fn quantize(&self, _: f32) -> Self {
            self.clone()
        }
    }
    impl std::fmt::Display for DummyWeight {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl FromStr for DummyWeight {
        type Err = ();
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            Ok(DummyWeight(s.parse().unwrap_or(0.0)))
        }
    }

    #[derive(Clone, PartialEq, Debug)]
    struct DummyArc {
        ilabel: i32,
        olabel: i32,
        weight: DummyWeight,
        nextstate: usize,
    }
    impl Arc for DummyArc {
        type Weight = DummyWeight;
        type Label = i32;
        type StateId = usize;
        type Reverse = Self;
        fn new(ilabel: i32, olabel: i32, weight: DummyWeight, nextstate: usize) -> Self {
            Self {
                ilabel,
                olabel,
                weight,
                nextstate,
            }
        }
        fn ilabel(&self) -> i32 {
            self.ilabel
        }
        fn olabel(&self) -> i32 {
            self.olabel
        }
        fn weight(&self) -> &DummyWeight {
            &self.weight
        }
        fn nextstate(&self) -> usize {
            self.nextstate
        }
        fn type_name() -> crate::fst_type::ArcType {
            crate::fst_type::ArcType::new_static("dummy")
        }
    }

    struct MockExpander {
        final_calls: FastCell<Vec<usize>>,
        arcs_calls: FastCell<Vec<usize>>,
        heavy_arcs: bool,
    }

    impl CacheExpander<DummyArc> for MockExpander {
        fn expand_final(&self, state: usize) -> Option<DummyWeight> {
            self.final_calls.borrow_mut().push(state);
            Some(DummyWeight(1.0))
        }

        fn expand_arcs(&self, state: usize) -> Vec<DummyArc> {
            self.arcs_calls.borrow_mut().push(state);
            if self.heavy_arcs {
                vec![DummyArc::new(1, 1, DummyWeight(0.5), state + 1); 1000]
            } else {
                vec![DummyArc::new(1, 1, DummyWeight(0.5), state + 1)]
            }
        }
    }

    #[test]
    fn test_byte_based_gc_eviction() {
        let cache = CacheImpl::new(CacheOptions {
            gc: true,
            gc_limit: 1024,
            gc_fraction: 0.5,
        });
        let expander = MockExpander {
            final_calls: FastCell::new(Vec::new()),
            arcs_calls: FastCell::new(Vec::new()),
            heavy_arcs: true,
        };

        let _ = cache.arcs_slice(0, &expander);
        let _ = cache.arcs_slice(1, &expander);

        let store = cache.store.borrow();
        assert!(
            store.current_size_bytes <= store.current_gc_limit,
            "Memory usage must stay within the limit or trigger an emergency limit expansion if exceeded"
        );
    }

    #[test]
    #[should_panic(expected = "Recursive cache expansion detected")]
    fn test_recursive_expansion_prevention() {
        struct EvilExpander<'a> {
            cache: &'a CacheImpl<DummyArc>,
        }
        impl<'a> CacheExpander<DummyArc> for EvilExpander<'a> {
            fn expand_final(&self, state: usize) -> Option<DummyWeight> {
                let _ = self.cache.final_weight(state, self);
                None
            }
            fn expand_arcs(&self, _: usize) -> Vec<DummyArc> {
                vec![]
            }
        }

        let cache = CacheImpl::new(CacheOptions::default());
        let expander = EvilExpander { cache: &cache };
        let _ = cache.final_weight(0, &expander);
    }

    #[test]
    fn test_panic_safety_flags_do_not_leak() {
        struct PanicExpander;
        impl CacheExpander<DummyArc> for PanicExpander {
            fn expand_final(&self, _state: usize) -> Option<DummyWeight> {
                panic!("Simulated crash");
            }
            fn expand_arcs(&self, _: usize) -> Vec<DummyArc> {
                vec![]
            }
        }

        let cache = CacheImpl::new(CacheOptions::default());

        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let expander = PanicExpander;
            cache.final_weight(0, &expander);
        }));
        assert!(result.is_err());

        let store = cache.store.borrow();
        let state = &store.pool[store.map.get(0).unwrap()];
        assert!(
            !state.flags.contains(CacheFlags::EXPANDING_FINAL),
            "Expansion flag should be cleared by the Drop guard even after a panic"
        );
    }

    /// An expander whose answers depend only on the state, so a cache that
    /// forgot a state and recomputed it must come back with the same thing.
    struct Deterministic {
        expansions: FastCell<usize>,
    }

    impl CacheExpander<DummyArc> for Deterministic {
        fn expand_final(&self, state: usize) -> Option<DummyWeight> {
            *self.expansions.borrow_mut() += 1;
            state.is_multiple_of(3).then_some(DummyWeight(state as f32))
        }

        fn expand_arcs(&self, state: usize) -> Vec<DummyArc> {
            *self.expansions.borrow_mut() += 1;
            (0..state % 4)
                .map(|i| {
                    DummyArc::new(
                        (i % 2) as i32,
                        i as i32,
                        DummyWeight(i as f32),
                        state + i + 1,
                    )
                })
                .collect()
        }
    }

    /// Everything a caller can ask a cache about one state.
    fn snapshot(cache: &CacheImpl<DummyArc>, expander: &Deterministic, s: usize) -> String {
        format!(
            "{:?}|{}|{}|{}|{:?}",
            cache.final_weight(s, expander).0,
            cache.num_arcs(s, expander),
            cache.num_input_epsilons(s, expander),
            cache.num_output_epsilons(s, expander),
            cache
                .arcs_iter(s, expander)
                .map(|a| (a.ilabel(), a.olabel(), a.weight().0, a.nextstate()))
                .collect::<Vec<_>>()
        )
    }

    /// Eviction is a memory decision, and it has to be invisible: a cache small
    /// enough to be constantly discarding states must answer exactly as one
    /// that never discards any.
    #[test]
    fn a_cache_that_evicts_answers_the_same_as_one_that_does_not() {
        // A deterministic walk over the states, revisiting freely.
        let mut rng = 0x1234_ABCDu64;
        let mut next = |bound: usize| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as usize) % bound
        };
        let queries: Vec<usize> = (0..400).map(|_| next(40)).collect();

        let unbounded = CacheImpl::<DummyArc>::new(CacheOptions {
            gc: false,
            gc_limit: usize::MAX,
            gc_fraction: 0.666,
        });
        let unbounded_expander = Deterministic {
            expansions: FastCell::new(0),
        };

        // Small enough that only a handful of states fit at a time.
        let evicting = CacheImpl::<DummyArc>::new(CacheOptions {
            gc: true,
            gc_limit: 256,
            gc_fraction: 0.5,
        });
        let evicting_expander = Deterministic {
            expansions: FastCell::new(0),
        };

        for &s in &queries {
            assert_eq!(
                snapshot(&evicting, &evicting_expander, s),
                snapshot(&unbounded, &unbounded_expander, s),
                "state {s}"
            );
        }

        // The point of the exercise: the small cache really did discard states
        // and recompute them, so the agreement above was not vacuous.
        assert!(
            *evicting_expander.expansions.borrow() > *unbounded_expander.expansions.borrow(),
            "the bounded cache never evicted anything"
        );
    }

    /// With garbage collection off, a state is expanded once however often it
    /// is asked for.
    #[test]
    fn without_collection_a_state_is_expanded_once() {
        let cache = CacheImpl::<DummyArc>::new(CacheOptions {
            gc: false,
            gc_limit: usize::MAX,
            gc_fraction: 0.666,
        });
        let expander = Deterministic {
            expansions: FastCell::new(0),
        };

        for _ in 0..5 {
            for s in 0..10 {
                snapshot(&cache, &expander, s);
            }
        }
        // One final-weight expansion and one arc expansion per state.
        assert_eq!(*expander.expansions.borrow(), 20);
    }

    /// Arcs handed out before an eviction stay readable: they are reference
    /// counted precisely so that a caller iterating them cannot have them taken
    /// away. This is the difference from `expander_cache`, which never evicts
    /// and so hands out borrows.
    #[test]
    fn arcs_already_handed_out_survive_an_eviction() {
        let cache = CacheImpl::<DummyArc>::new(CacheOptions {
            gc: true,
            gc_limit: 128,
            gc_fraction: 0.5,
        });
        let expander = Deterministic {
            expansions: FastCell::new(0),
        };

        let held: Vec<_> = (0..4).map(|s| cache.arcs_iter(s, &expander)).collect();
        let expected: Vec<Vec<_>> = (0..4)
            .map(|s| {
                expander
                    .expand_arcs(s)
                    .into_iter()
                    .map(|a| (a.ilabel(), a.nextstate()))
                    .collect()
            })
            .collect();

        // Churn the cache well past its limit.
        for s in 4..60 {
            let _ = cache.arcs_iter(s, &expander);
        }

        for (iter, want) in held.into_iter().zip(expected) {
            let got: Vec<_> = iter.map(|a| (a.ilabel(), a.nextstate())).collect();
            assert_eq!(got, want);
        }
    }
}
