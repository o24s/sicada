//! Adding states and arcs to an FST without rebuilding it.
//!
//! Port of OpenFst's `merge-fst.h`.
//!
//! A [`MergeFst`] presents a *primary* FST with a *secondary* one grafted onto
//! it. The secondary is a partial specification rather than a well-formed FST:
//! its arcs already name states of the merged result, not of itself. A state
//! map says which of its states corresponds to which state of the result.
//!
//! See Cyril Allauzen, Michael Riley (2015): "Rapid vocabulary addition to
//! context-dependent decoder graphs", Proceedings of Interspeech 2015,
//! pages 2112–2116. doi: 10.21437/Interspeech.2015-477.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::AtomicRc;
use crate::algorithms::test_properties::cached_properties;
use crate::arc::{Arc, ArcStateId};
use crate::error::OpenFstError;
use crate::fst::{ExpandedFst, Fst, PropertyCache};
use crate::fst_type::FstType;
use crate::properties::{K_EXPANDED, K_FST_PROPERTIES};
use crate::weight::Weight;

/// Which state of the secondary FST a state of the result stands for.
///
/// SICADA-DIVERGE: upstream has three of these (a hash map, a vector, and a
/// fixed one) chosen by a template parameter, and the vector and fixed ones
/// abort the process with `CHECK_EQ` when the map they are handed does not have
/// the shape they need. The map is small next to the FSTs, so one
/// implementation is kept, and the shape it needs is checked into a `Result`.
#[derive(Debug, Clone, Default)]
pub struct MergeStateMap<S> {
    map: FxHashMap<S, S>,
}

impl<S: ArcStateId> MergeStateMap<S> {
    /// Builds a map from merged-state to secondary-state pairs.
    pub fn new(map: FxHashMap<S, S>) -> Self {
        Self { map }
    }

    /// The secondary state `s` stands for, if any.
    #[inline]
    pub fn find(&self, s: S) -> Option<S> {
        self.map.get(&s).copied()
    }

    /// How many states are mapped.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether nothing is mapped.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// A primary FST with a secondary one grafted onto it.
pub struct MergeFst<A: Arc, P: ExpandedFst<A>, S: ExpandedFst<A>> {
    primary: P,
    secondary: S,
    state_map: MergeStateMap<A::StateId>,
    num_states: usize,
    properties: PropertyCache,
    _marker: std::marker::PhantomData<A>,
}

impl<A: Arc, P: ExpandedFst<A>, S: ExpandedFst<A>> MergeFst<A, P, S> {
    /// Grafts `secondary` onto `primary` according to `state_map`.
    ///
    /// Every state of the secondary must be named exactly once, and each must
    /// stand for a distinct state of the result; otherwise two states of the
    /// secondary would share arcs, which is not a merge.
    pub fn new(
        primary: P,
        secondary: S,
        state_map: MergeStateMap<A::StateId>,
    ) -> Result<Self, OpenFstError> {
        if state_map.len() != secondary.num_states() {
            return Err(OpenFstError::InvalidOperation(format!(
                "MergeFst: the state map names {} states but the secondary FST has {}",
                state_map.len(),
                secondary.num_states()
            )));
        }
        let mut sources: FxHashSet<A::StateId> = FxHashSet::default();
        let primary_states = primary.num_states();
        let mut num_states = primary_states + secondary.num_states();
        for (&merged, &source) in &state_map.map {
            if !sources.insert(source) {
                return Err(OpenFstError::InvalidOperation(
                    "MergeFst: two states of the result claim the same secondary state".to_string(),
                ));
            }
            if source.as_usize() >= secondary.num_states() {
                return Err(OpenFstError::InvalidOperation(format!(
                    "MergeFst: the state map names secondary state {} of {}",
                    source.as_usize(),
                    secondary.num_states()
                )));
            }
            // A state that is both a primary state and a secondary one is one
            // state of the result, not two.
            if merged.as_usize() < primary_states {
                num_states -= 1;
            }
        }

        let properties = primary.properties(K_FST_PROPERTIES, false)
            & secondary.properties(K_FST_PROPERTIES, false);
        Ok(Self {
            primary,
            secondary,
            state_map,
            num_states,
            properties: PropertyCache::new(properties | K_EXPANDED),
            _marker: std::marker::PhantomData,
        })
    }

    /// The FST being added to.
    pub fn primary(&self) -> &P {
        &self.primary
    }

    /// The partial FST being added.
    pub fn secondary(&self) -> &S {
        &self.secondary
    }

    /// Whether `state` is one the primary FST already had.
    #[inline]
    fn in_primary(&self, state: A::StateId) -> bool {
        state.as_usize() < self.primary.num_states()
    }
}

/// The arcs of a merged state: the primary's, then the secondary's.
pub struct MergeArcIter<'a, A: Arc + 'a, P: Fst<A> + 'a, S: Fst<A> + 'a> {
    primary: Option<P::ArcIter<'a>>,
    secondary: Option<S::ArcIter<'a>>,
}

impl<'a, A: Arc + 'a, P: Fst<A> + 'a, S: Fst<A> + 'a> Clone for MergeArcIter<'a, A, P, S> {
    fn clone(&self) -> Self {
        Self {
            primary: self.primary.clone(),
            secondary: self.secondary.clone(),
        }
    }
}

impl<'a, A: Arc + 'a, P: Fst<A> + 'a, S: Fst<A> + 'a> Iterator for MergeArcIter<'a, A, P, S> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        if let Some(iter) = &mut self.primary {
            if let Some(arc) = iter.next() {
                return Some(arc);
            }
            self.primary = None;
        }
        self.secondary.as_mut().and_then(Iterator::next)
    }
}

impl<A: Arc, P: ExpandedFst<A>, S: ExpandedFst<A>> Fst<A> for MergeFst<A, P, S> {
    type StateIter<'a>
        = std::iter::Map<std::ops::Range<usize>, fn(usize) -> A::StateId>
    where
        Self: 'a;
    type ArcIter<'a>
        = MergeArcIter<'a, A, P, S>
    where
        Self: 'a;

    #[inline]
    fn start(&self) -> Option<A::StateId> {
        self.primary.start()
    }

    fn final_weight(&self, state: A::StateId) -> A::Weight {
        let mut weight = if self.in_primary(state) {
            self.primary.final_weight(state)
        } else {
            A::Weight::zero()
        };
        if let Some(source) = self.state_map.find(state) {
            // SICADA-BUGFIX: upstream multiplies the two final weights, and a
            // state that only the secondary makes final therefore starts from
            // `Zero`, which multiplication annihilates. Every state
            // the merge adds comes out non-final, and so does every primary
            // state the secondary was meant to make final, which leaves the
            // type unable to do the thing it was written for: adding
            // vocabulary, whose words have to end somewhere.
            //
            // A merged state's arcs are the union of the two sets, so its ways
            // of accepting are too, and in a semiring alternatives combine with
            // Plus. That gives the right answer in all four cases.
            weight = weight.plus(&self.secondary.final_weight(source));
        }
        weight
    }

    fn num_arcs(&self, state: A::StateId) -> usize {
        let primary = if self.in_primary(state) {
            self.primary.num_arcs(state)
        } else {
            0
        };
        primary
            + self
                .state_map
                .find(state)
                .map_or(0, |source| self.secondary.num_arcs(source))
    }

    fn num_input_epsilons(&self, state: A::StateId) -> usize {
        let primary = if self.in_primary(state) {
            self.primary.num_input_epsilons(state)
        } else {
            0
        };
        primary
            + self
                .state_map
                .find(state)
                .map_or(0, |source| self.secondary.num_input_epsilons(source))
    }

    fn num_output_epsilons(&self, state: A::StateId) -> usize {
        let primary = if self.in_primary(state) {
            self.primary.num_output_epsilons(state)
        } else {
            0
        };
        primary
            + self
                .state_map
                .find(state)
                .map_or(0, |source| self.secondary.num_output_epsilons(source))
    }

    #[inline]
    fn num_states_if_known(&self) -> Option<usize> {
        Some(self.num_states)
    }

    fn properties(&self, mask: u64, test: bool) -> u64 {
        cached_properties(self, &self.properties, mask, test)
    }

    #[inline]
    fn fst_type(&self) -> &str {
        FstType::MERGE.as_str()
    }

    fn input_symbols(&self) -> Option<AtomicRc<crate::symbol_table::SymbolTable>> {
        self.primary
            .input_symbols()
            .or_else(|| self.secondary.input_symbols())
    }

    fn output_symbols(&self) -> Option<AtomicRc<crate::symbol_table::SymbolTable>> {
        self.primary
            .output_symbols()
            .or_else(|| self.secondary.output_symbols())
    }

    fn states<'a>(&'a self) -> Self::StateIter<'a> {
        (0..self.num_states).map(A::StateId::from_usize as fn(usize) -> A::StateId)
    }

    fn arcs<'a>(&'a self, state: A::StateId) -> Self::ArcIter<'a> {
        MergeArcIter {
            primary: self.in_primary(state).then(|| self.primary.arcs(state)),
            secondary: self
                .state_map
                .find(state)
                .map(|source| self.secondary.arcs(source)),
        }
    }
}

impl<A: Arc, P: ExpandedFst<A>, S: ExpandedFst<A>> ExpandedFst<A> for MergeFst<A, P, S> {
    #[inline]
    fn num_states(&self) -> usize {
        self.num_states
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::fst::MutableFst;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::weights::float_weight::TropicalWeight;

    /// 0 → 1 → 2, with 2 final.
    fn primary() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(1.0), 1));
        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight(2.0), 2));
        fst.set_final(2, TropicalWeight(3.0));
        fst
    }

    fn map(pairs: &[(i32, i32)]) -> MergeStateMap<i32> {
        MergeStateMap::new(pairs.iter().copied().collect())
    }

    /// A secondary FST that adds an arc to an existing state and one new state.
    #[test]
    fn a_merged_state_has_both_sets_of_arcs() {
        let mut secondary = StdVectorFst::new();
        secondary.add_state();
        secondary.add_state();
        // Its arcs already name states of the *result*: state 3 is the new one.
        secondary.add_arc(0, StdArc::new(9, 9, TropicalWeight(4.0), 3));
        secondary.set_final(1, TropicalWeight(5.0));

        // Secondary state 0 lands on primary state 1; secondary state 1 becomes
        // the new state 3.
        let merged = MergeFst::new(primary(), secondary, map(&[(1, 0), (3, 1)])).unwrap();

        assert_eq!(merged.num_states(), 4, "one merged, one added");
        assert_eq!(merged.start(), Some(0));

        // State 1 has the primary's arc and the secondary's.
        let arcs: Vec<_> = merged.arcs(1).collect();
        assert_eq!(arcs.len(), 2);
        assert_eq!(merged.num_arcs(1), 2);
        assert_eq!((arcs[0].ilabel(), arcs[0].nextstate()), (2, 2));
        assert_eq!((arcs[1].ilabel(), arcs[1].nextstate()), (9, 3));

        // The new state carries the secondary's final weight.
        assert_eq!(merged.final_weight(3), TropicalWeight(5.0));
        assert_eq!(merged.num_arcs(3), 0);
    }

    /// A state the map does not name reads through from the primary unchanged.
    #[test]
    fn an_unmapped_state_reads_through() {
        let mut secondary = StdVectorFst::new();
        secondary.add_state();
        secondary.add_arc(0, StdArc::new(9, 9, TropicalWeight::one(), 0));

        let base = primary();
        let merged = MergeFst::new(base.clone(), secondary, map(&[(0, 0)])).unwrap();

        for s in 1..3 {
            assert_eq!(
                merged.arcs(s).collect::<Vec<_>>(),
                base.arcs(s).collect::<Vec<_>>(),
                "state {s}"
            );
            assert_eq!(merged.final_weight(s), base.final_weight(s));
        }
    }

    /// A merged state's arcs are the union of the two sets, so its ways of
    /// accepting are too, and in a semiring alternatives combine with Plus.
    ///
    /// Upstream multiplies, which makes `Zero` an annihilator and leaves every
    /// state the merge adds non-final.
    #[test]
    fn final_weights_combine_as_alternatives_at_a_merged_state() {
        let mut secondary = StdVectorFst::new();
        secondary.add_state();
        secondary.set_final(0, TropicalWeight(10.0));

        // Secondary state 0 lands on primary state 2, which is final with 3.
        let merged = MergeFst::new(primary(), secondary, map(&[(2, 0)])).unwrap();
        // Tropical plus is min: the cheaper of the two ways to accept.
        assert_eq!(merged.final_weight(2), TropicalWeight(3.0));
    }

    /// The case upstream cannot express at all: a primary state that is not
    /// final, made final by the secondary.
    #[test]
    fn the_secondary_can_make_a_primary_state_final() {
        let mut secondary = StdVectorFst::new();
        secondary.add_state();
        secondary.set_final(0, TropicalWeight(7.0));

        // Primary state 1 is not final.
        assert_eq!(primary().final_weight(1), TropicalWeight::zero());
        let merged = MergeFst::new(primary(), secondary, map(&[(1, 0)])).unwrap();
        assert_eq!(merged.final_weight(1), TropicalWeight(7.0));
    }

    #[test]
    fn a_state_map_that_does_not_cover_the_secondary_is_refused() {
        let mut secondary = StdVectorFst::new();
        secondary.add_state();
        secondary.add_state();
        assert!(MergeFst::new(primary(), secondary.clone(), map(&[(1, 0)])).is_err());

        // Two result states claiming the same secondary state is not a merge.
        assert!(MergeFst::new(primary(), secondary.clone(), map(&[(1, 0), (3, 0)])).is_err());

        // A secondary state that does not exist.
        assert!(MergeFst::new(primary(), secondary, map(&[(1, 0), (3, 7)])).is_err());
    }

    /// Grafting nothing leaves the primary as it was.
    #[test]
    fn merging_an_empty_secondary_changes_nothing() {
        let base = primary();
        let merged =
            MergeFst::new(base.clone(), StdVectorFst::new(), MergeStateMap::default()).unwrap();

        assert_eq!(merged.num_states(), base.num_states());
        for s in 0..base.num_states() as i32 {
            assert_eq!(
                merged.arcs(s).collect::<Vec<_>>(),
                base.arcs(s).collect::<Vec<_>>()
            );
            assert_eq!(merged.final_weight(s), base.final_weight(s));
        }
    }
}
