//! An FST whose states are produced on demand.
//!
//! Port of OpenFst's `expander-fst.h`.

use crate::AtomicRc;
use crate::algorithms::test_properties::cached_properties;
use crate::arc::{Arc, ArcStateId};
use crate::expander_cache::{DefaultExpanderCache, Expander, ExpanderCache};
use crate::fst::{ExpandedFst, Fst, PropertyCache};
use crate::fst_type::FstType;
use crate::symbol_table::SymbolTable;

/// An on-the-fly FST: an [`Expander`] that produces one state at a time, plus a
/// cache that remembers the ones already produced.
///
/// SICADA-DIVERGE: upstream takes the symbol tables from the expander, so every
/// expander has to carry a pair whether or not it has anything to put in them.
/// Here they are set on the FST.
pub struct ExpanderFst<A: Arc, E: Expander<A>, C: ExpanderCache<A> = DefaultExpanderCache<A>> {
    expander: E,
    cache: C,
    input_symbols: Option<AtomicRc<SymbolTable>>,
    output_symbols: Option<AtomicRc<SymbolTable>>,
    /// Nothing is known about an FST that does not exist yet, so this starts
    /// empty and fills in as `properties(mask, /*test=*/true)` is asked.
    properties: PropertyCache,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Arc, E: Expander<A>, C: ExpanderCache<A>> ExpanderFst<A, E, C> {
    /// Creates an FST with a default-constructed cache.
    pub fn new(expander: E) -> Self
    where
        C: Default,
    {
        Self {
            expander,
            cache: C::default(),
            input_symbols: None,
            output_symbols: None,
            properties: PropertyCache::new(0),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates an FST with a cache supplied by the caller.
    pub fn new_with_cache(expander: E, cache: C) -> Self {
        Self {
            expander,
            cache,
            input_symbols: None,
            output_symbols: None,
            properties: PropertyCache::new(0),
            _phantom: std::marker::PhantomData,
        }
    }

    /// The expander producing the states.
    pub fn expander(&self) -> &E {
        &self.expander
    }

    /// The cache holding the states produced so far.
    pub fn cache(&self) -> &C {
        &self.cache
    }

    /// Sets the input symbol table.
    pub fn set_input_symbols(&mut self, syms: Option<AtomicRc<SymbolTable>>) {
        self.input_symbols = syms;
    }

    /// Sets the output symbol table.
    pub fn set_output_symbols(&mut self, syms: Option<AtomicRc<SymbolTable>>) {
        self.output_symbols = syms;
    }
}

/// State iterator over an on-the-fly FST.
///
/// Expands each state as it passes, and re-reads the expander's state count
/// each time round. An expander that discovers states while expanding, which is
/// how a lazy composition works, reports a count that grows as it goes, and an
/// iterator that took the count once would stop at whatever it happened to be at
/// the start.
pub struct ExpanderStateIter<'a, A: Arc, E: Expander<A>, C: ExpanderCache<A>> {
    fst: &'a ExpanderFst<A, E, C>,
    state: usize,
    /// The lowest state not yet expanded, so a state is expanded once.
    min_unexpanded: usize,
}

impl<A: Arc, E: Expander<A>, C: ExpanderCache<A>> Iterator for ExpanderStateIter<'_, A, E, C> {
    type Item = A::StateId;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.state >= self.fst.expander.num_states() {
            return None;
        }
        let state = A::StateId::from_usize(self.state);
        if self.state == self.min_unexpanded {
            self.fst.cache.find_or_expand(&self.fst.expander, state);
            self.min_unexpanded = self.state + 1;
        }
        self.state += 1;
        Some(state)
    }
}

/// Arc iterator over one expanded state.
///
/// The arcs stay in the cache and are borrowed from it, so opening an iterator
/// costs nothing beyond the lookup that found the state: there is no per-state
/// allocation and no reference count to touch. Several iterators over different
/// states can be live at once, which a depth-first traversal needs: it holds one
/// per stack frame.
pub type ExpanderArcIter<'a, A> = std::iter::Cloned<std::slice::Iter<'a, A>>;

impl<A: Arc, E: Expander<A>, C: ExpanderCache<A>> Fst<A> for ExpanderFst<A, E, C> {
    type StateIter<'a>
        = ExpanderStateIter<'a, A, E, C>
    where
        Self: 'a;
    type ArcIter<'a>
        = ExpanderArcIter<'a, A>
    where
        Self: 'a;

    #[inline]
    fn start(&self) -> Option<A::StateId> {
        let s = self.expander.start();
        if s == A::StateId::no_state() {
            None
        } else {
            Some(s)
        }
    }

    #[inline]
    fn final_weight(&self, state: A::StateId) -> A::Weight {
        self.cache
            .find_or_expand(&self.expander, state)
            .final_weight()
            .clone()
    }

    #[inline]
    fn num_arcs(&self, state: A::StateId) -> usize {
        self.cache.find_or_expand(&self.expander, state).num_arcs()
    }

    #[inline]
    fn num_input_epsilons(&self, state: A::StateId) -> usize {
        self.cache
            .find_or_expand(&self.expander, state)
            .num_input_epsilons()
    }

    #[inline]
    fn num_output_epsilons(&self, state: A::StateId) -> usize {
        self.cache
            .find_or_expand(&self.expander, state)
            .num_output_epsilons()
    }

    #[inline]
    fn num_states_if_known(&self) -> Option<usize> {
        Some(self.expander.num_states())
    }

    #[inline]
    fn properties(&self, mask: u64, test: bool) -> u64 {
        cached_properties(self, &self.properties, mask, test)
    }

    #[inline]
    fn fst_type(&self) -> &str {
        FstType::EXPANDER.as_str()
    }

    #[inline]
    fn input_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
        self.input_symbols.clone()
    }

    #[inline]
    fn output_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
        self.output_symbols.clone()
    }

    #[inline]
    fn states<'a>(&'a self) -> Self::StateIter<'a> {
        ExpanderStateIter {
            fst: self,
            state: 0,
            min_unexpanded: 0,
        }
    }

    #[inline]
    fn arcs<'a>(&'a self, state: A::StateId) -> Self::ArcIter<'a> {
        self.cache
            .find_or_expand(&self.expander, state)
            .arcs()
            .iter()
            .cloned()
    }
}

impl<A: Arc, E: Expander<A>, C: ExpanderCache<A>> ExpandedFst<A> for ExpanderFst<A, E, C> {
    #[inline]
    fn num_states(&self) -> usize {
        self.expander.num_states()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::expander_cache::{ArcArenaStateStore, HashExpanderCache, StateBuilder};
    use crate::fst::ExpandedFst;
    use crate::properties::{
        K_ACCEPTOR, K_ACYCLIC, K_FST_PROPERTIES, K_NO_EPSILONS, K_NOT_ACCEPTOR, K_O_EPSILONS,
        K_STRING, K_TOP_SORTED, K_WEIGHTED,
    };
    use crate::weight::Weight;
    use crate::weights::float_weight::TropicalWeight;
    use std::cell::{Cell, RefCell};

    /// A chain of `n` states, each with one arc to the next, the last final.
    struct Chain {
        n: usize,
        expansions: RefCell<Vec<i32>>,
    }

    impl Chain {
        fn new(n: usize) -> Self {
            Self {
                n,
                expansions: RefCell::new(Vec::new()),
            }
        }

        fn expansions_of(&self, state: i32) -> usize {
            self.expansions
                .borrow()
                .iter()
                .filter(|&&s| s == state)
                .count()
        }
    }

    impl Expander<StdArc> for Chain {
        fn start(&self) -> i32 {
            0
        }

        fn num_states(&self) -> usize {
            self.n
        }

        fn expand(&self, state_id: i32, builder: &mut StateBuilder<StdArc>) {
            self.expansions.borrow_mut().push(state_id);
            if state_id as usize + 1 < self.n {
                builder.add_arc(StdArc::new(
                    state_id + 1,
                    0,
                    TropicalWeight(state_id as f32),
                    state_id + 1,
                ));
            } else {
                builder.set_final(TropicalWeight(2.5));
            }
        }
    }

    #[test]
    fn a_state_reports_what_the_expander_wrote() {
        let fst = ExpanderFst::<StdArc, _, DefaultExpanderCache<_>>::new(Chain::new(2));

        assert_eq!(fst.start(), Some(0));
        assert_eq!(fst.num_states(), 2);

        assert_eq!(fst.final_weight(0), TropicalWeight::zero());
        assert_eq!(fst.num_arcs(0), 1);
        assert_eq!(fst.num_input_epsilons(0), 0);
        assert_eq!(fst.num_output_epsilons(0), 1);

        let arcs: Vec<StdArc> = fst.arcs(0).collect();
        assert_eq!(arcs.len(), 1);
        assert_eq!(arcs[0].ilabel(), 1);
        assert_eq!(arcs[0].olabel(), 0);
        assert_eq!(arcs[0].nextstate(), 1);

        assert_eq!(fst.final_weight(1).value(), 2.5);
        assert_eq!(fst.num_arcs(1), 0);
    }

    /// Repeated visits are what the cache is for.
    #[test]
    fn a_state_is_expanded_once_however_often_it_is_visited() {
        let fst = ExpanderFst::<StdArc, _, DefaultExpanderCache<_>>::new(Chain::new(4));
        for _ in 0..3 {
            assert_eq!(fst.num_arcs(2), 1);
            let _: Vec<_> = fst.arcs(2).collect();
            let _ = fst.final_weight(2);
        }
        assert_eq!(fst.expander().expansions_of(2), 1);
    }

    /// A depth-first walk holds one arc iterator per stack frame, and expands
    /// further states while they are all still open. Each has to keep reading
    /// the state it was opened on.
    #[test]
    fn iterators_over_earlier_states_survive_later_expansions() {
        fn walk<C: ExpanderCache<StdArc> + Default>() {
            let fst = ExpanderFst::<StdArc, _, C>::new(Chain::new(64));
            let mut open = Vec::new();
            let mut state = 0;
            loop {
                let mut arcs = fst.arcs(state);
                let Some(arc) = arcs.next() else { break };
                open.push((state, arcs, arc));
                state = arc.nextstate();
            }
            assert_eq!(open.len(), 63);
            // Unwind, re-reading each frame's state through the iterator that
            // was opened before any of the later states existed.
            for (state, mut arcs, arc) in open.into_iter().rev() {
                assert_eq!(arc.ilabel(), state + 1);
                assert_eq!(arc.weight().value(), state as f32);
                assert!(arcs.next().is_none());
            }
        }

        walk::<DefaultExpanderCache<StdArc>>();
        walk::<HashExpanderCache<StdArc>>();
        walk::<ArcArenaStateStore<StdArc>>();
    }

    /// An on-the-fly FST discovers states as it expands them. The count grows
    /// while it is being walked, and the walk has to keep up.
    #[test]
    fn a_state_iterator_follows_an_expander_that_grows() {
        /// A chain that admits to one more state each time one is expanded, as
        /// a lazy composition does when it reaches a new pair of states.
        struct Growing {
            known: Cell<usize>,
        }

        impl Expander<StdArc> for Growing {
            fn start(&self) -> i32 {
                0
            }

            fn num_states(&self) -> usize {
                self.known.get()
            }

            fn expand(&self, state_id: i32, builder: &mut StateBuilder<StdArc>) {
                if (state_id as usize) + 1 < 6 {
                    self.known.set(self.known.get().max(state_id as usize + 2));
                    builder.add_arc(StdArc::new(1, 1, TropicalWeight::one(), state_id + 1));
                } else {
                    builder.set_final(TropicalWeight::one());
                }
            }
        }

        let fst = ExpanderFst::<StdArc, _, DefaultExpanderCache<_>>::new(Growing {
            known: Cell::new(1),
        });
        let states: Vec<i32> = fst.states().collect();
        assert_eq!(states, (0..6).collect::<Vec<_>>());
    }

    /// A state is expanded once however many times the walk passes over it.
    #[test]
    fn a_state_iterator_expands_each_state_once() {
        let fst = ExpanderFst::<StdArc, _, DefaultExpanderCache<_>>::new(Chain::new(5));
        let first: Vec<i32> = fst.states().collect();
        let second: Vec<i32> = fst.states().collect();
        assert_eq!(first, (0..5).collect::<Vec<_>>());
        assert_eq!(second, first);
        for state in 0..5 {
            assert_eq!(fst.expander().expansions_of(state), 1, "state {state}");
        }
    }

    /// Nothing is known about an FST that has not been built yet, and claiming
    /// otherwise would mislead every algorithm that gates on a property.
    #[test]
    fn nothing_is_claimed_until_the_fst_is_scanned() {
        let fst = ExpanderFst::<StdArc, _, DefaultExpanderCache<_>>::new(Chain::new(4));
        assert_eq!(fst.properties(K_FST_PROPERTIES, false), 0);

        // Scanning settles them, and the answers are the truth about the
        // chain: a weighted transducer whose arcs all have an epsilon output,
        // laid out as a string.
        let props = fst.properties(K_FST_PROPERTIES, true);
        assert_ne!(props & K_NOT_ACCEPTOR, 0);
        assert_eq!(props & K_ACCEPTOR, 0);
        assert_ne!(props & K_O_EPSILONS, 0);
        assert_ne!(props & K_NO_EPSILONS, 0, "no arc is epsilon on both sides");
        assert_ne!(props & K_TOP_SORTED, 0);
        assert_ne!(props & K_ACYCLIC, 0);
        assert_ne!(props & K_WEIGHTED, 0);
        assert_ne!(props & K_STRING, 0);

        // And are remembered.
        assert_ne!(fst.properties(K_NOT_ACCEPTOR, false) & K_NOT_ACCEPTOR, 0);
    }
}
