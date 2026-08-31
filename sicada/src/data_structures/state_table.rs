//! The mapping between state tuples and state IDs.
//!
//! Port of OpenFst's `state-table.h`. Composition builds its states out of
//! tuples, one state from each input plus a filter state, and needs to go both
//! ways: tuple to ID when following an arc, ID back to tuple when expanding
//! one. That is what [`bi_table`](super::bi_table) provides; this file is the
//! composition-shaped use of it.
//!
//! Upstream's five general-purpose state tables (`HashStateTable`,
//! `CompactHashStateTable`, `VectorStateTable`, `VectorHashStateTable` and
//! `ErasableStateTable`) are subclasses that rename two methods of the
//! corresponding bi-table and add nothing else, so here they are aliases.

use std::hash::Hash;

use crate::algorithms::filter_state::FilterState;
use crate::arc::{Arc, ArcStateId};
use crate::data_structures::bi_table::{
    BiTableId, CompactHashBiTable, ErasableBiTable, Fingerprint, HashBiTable, VectorBiTable,
    VectorHashBiTable,
};
use crate::fst::Fst;
use crate::properties::{
    K_I_DETERMINISTIC, K_NO_I_EPSILONS, K_NO_O_EPSILONS, K_O_DETERMINISTIC, K_STRING,
};

/// A state table keyed by a hash of the tuple.
///
/// Upstream's `HashStateTable`, which subclasses `HashBiTable` to rename
/// `FindId` to `FindState` and `FindEntry` to `Tuple`.
pub type HashStateTable<I, T> = HashBiTable<I, T>;

/// A state table keyed by a hash of the tuple, storing each tuple once.
///
/// Upstream's `CompactHashStateTable`.
pub type CompactHashStateTable<I, T> = CompactHashBiTable<I, T>;

/// A state table indexed by a fingerprint of the tuple.
///
/// Upstream's `VectorStateTable`.
pub type VectorStateTable<I, T, FP> = VectorBiTable<I, T, FP>;

/// A state table indexed by fingerprint for the tuples a selector picks, and
/// hashed for the rest.
///
/// Upstream's `VectorHashStateTable`.
pub type VectorHashStateTable<I, T, S, FP> = VectorHashBiTable<I, T, S, FP>;

/// A state table whose entries can be erased.
///
/// Upstream's `ErasableStateTable`.
pub type ErasableStateTable<I, T> = ErasableBiTable<I, T>;

/// The composition state: one state from each input, plus the filter's own.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DefaultComposeStateTuple<S, FS> {
    state_pair: (S, S),
    fs: FS,
}

impl<S: ArcStateId, FS> DefaultComposeStateTuple<S, FS> {
    #[inline(always)]
    pub fn new(s1: S, s2: S, fs: FS) -> Self {
        Self {
            state_pair: (s1, s2),
            fs,
        }
    }

    #[inline(always)]
    pub fn state_id1(&self) -> S {
        self.state_pair.0
    }

    #[inline(always)]
    pub fn state_id2(&self) -> S {
        self.state_pair.1
    }

    #[inline(always)]
    pub fn get_filter_state(&self) -> &FS {
        &self.fs
    }

    #[inline(always)]
    pub fn state_pair(&self) -> (S, S) {
        self.state_pair
    }

    /// A small integer standing for this tuple's filter state.
    ///
    /// SICADA-OPT: this used to ask `dyn Any` which filter state type it had
    /// and pick a formula per answer, which is a chain of type-identity
    /// comparisons on composition's hottest path, and which fell back without
    /// warning to a full hash for any type not on the list, giving a number far
    /// too large to be the vector index `ComposeFingerprint` uses it as. It is a
    /// method on [`FilterState`] instead, matching upstream's `Hash()`, which
    /// is part of the same interface.
    #[inline]
    pub fn hash_value(&self) -> usize
    where
        FS: FilterState,
    {
        self.fs.hash_value()
    }

    /// A tuple no composition produces, for a table slot that holds nothing.
    #[inline]
    pub fn no_state() -> Self
    where
        FS: FilterState,
    {
        Self::new(S::no_state(), S::no_state(), FS::no_state())
    }
}

pub trait ComposeStateTable<A: Arc, FS: FilterState> {
    type StateTuple: Clone + Hash + Eq;

    fn find_state(&mut self, tuple: &Self::StateTuple) -> A::StateId;
    fn tuple(&self, s: A::StateId) -> &Self::StateTuple;
    fn size(&self) -> usize;
    fn error(&self) -> bool {
        false
    }
}

pub struct GenericComposeStateTable<A: Arc, FS: FilterState>
where
    A::StateId: BiTableId,
{
    table: CompactHashBiTable<A::StateId, DefaultComposeStateTuple<A::StateId, FS>>,
}

impl<A: Arc, FS: FilterState> GenericComposeStateTable<A, FS>
where
    A::StateId: BiTableId,
{
    pub fn new<F1: Fst<A>, F2: Fst<A>>(_fst1: &F1, _fst2: &F2) -> Self {
        Self {
            table: CompactHashBiTable::new(0),
        }
    }

    pub fn new_with_size<F1: Fst<A>, F2: Fst<A>>(
        _fst1: &F1,
        _fst2: &F2,
        table_size: usize,
    ) -> Self {
        Self {
            table: CompactHashBiTable::new(table_size),
        }
    }
}

impl<A: Arc, FS: FilterState> ComposeStateTable<A, FS> for GenericComposeStateTable<A, FS>
where
    A::StateId: BiTableId,
{
    type StateTuple = DefaultComposeStateTuple<A::StateId, FS>;

    #[inline]
    fn find_state(&mut self, tuple: &Self::StateTuple) -> A::StateId {
        self.table.find_id(tuple, true).unwrap()
    }

    #[inline]
    fn tuple(&self, s: A::StateId) -> &Self::StateTuple {
        self.table.find_entry(s).unwrap()
    }

    #[inline]
    fn size(&self) -> usize {
        self.table.size()
    }
}

/// Numbers a composition tuple by treating it as a mixed-radix digit string:
/// the first state, then the second scaled past it, then the filter state
/// scaled past both.
///
/// Injective as long as the two inputs really have the state counts it was
/// given, and the filter state's [`hash_value`](FilterState::hash_value) is
/// small, which is why a table built on it is only for the case where that
/// product is manageable.
#[derive(Clone, Debug)]
pub struct ComposeFingerprint {
    mult1: usize,
    mult2: usize,
}

impl ComposeFingerprint {
    /// For inputs with `nstates1` and `nstates2` states.
    pub fn new(nstates1: usize, nstates2: usize) -> Self {
        Self {
            mult1: nstates1,
            mult2: nstates1.saturating_mul(nstates2),
        }
    }
}

impl<S: ArcStateId, FS: FilterState> Fingerprint<DefaultComposeStateTuple<S, FS>>
    for ComposeFingerprint
{
    #[inline]
    fn fingerprint(&self, tuple: &DefaultComposeStateTuple<S, FS>) -> usize {
        tuple
            .state_id1()
            .as_usize()
            .wrapping_add(tuple.state_id2().as_usize().wrapping_mul(self.mult1))
            .wrapping_add(tuple.hash_value().wrapping_mul(self.mult2))
    }
}

/// Numbers a tuple by its first state, for when that alone determines it.
#[derive(Clone, Copy, Debug, Default)]
pub struct ComposeState1Fingerprint;

impl<S: ArcStateId, FS> Fingerprint<DefaultComposeStateTuple<S, FS>> for ComposeState1Fingerprint {
    #[inline]
    fn fingerprint(&self, tuple: &DefaultComposeStateTuple<S, FS>) -> usize {
        tuple.state_id1().as_usize()
    }
}

/// Numbers a tuple by its second state, for when that alone determines it.
#[derive(Clone, Copy, Debug, Default)]
pub struct ComposeState2Fingerprint;

impl<S: ArcStateId, FS> Fingerprint<DefaultComposeStateTuple<S, FS>> for ComposeState2Fingerprint {
    #[inline]
    fn fingerprint(&self, tuple: &DefaultComposeStateTuple<S, FS>) -> usize {
        tuple.state_id2().as_usize()
    }
}

pub struct ProductComposeStateTable<A: Arc, FS: FilterState>
where
    A::StateId: BiTableId,
{
    table:
        VectorStateTable<A::StateId, DefaultComposeStateTuple<A::StateId, FS>, ComposeFingerprint>,
}

impl<A: Arc, FS: FilterState> ProductComposeStateTable<A, FS>
where
    A::StateId: BiTableId,
{
    /// For composing `fst1` with `fst2`. Both are counted, which walks them if
    /// they do not already know their size.
    pub fn new<F1: Fst<A>, F2: Fst<A>>(fst1: &F1, fst2: &F2, table_size: usize) -> Self {
        Self {
            table: VectorStateTable::new(
                ComposeFingerprint::new(fst1.count_states(), fst2.count_states()),
                table_size,
            ),
        }
    }
}

impl<A: Arc, FS: FilterState> ComposeStateTable<A, FS> for ProductComposeStateTable<A, FS>
where
    A::StateId: BiTableId,
{
    type StateTuple = DefaultComposeStateTuple<A::StateId, FS>;

    #[inline]
    fn find_state(&mut self, tuple: &Self::StateTuple) -> A::StateId {
        self.table.find_id(tuple, true).unwrap()
    }

    #[inline]
    fn tuple(&self, s: A::StateId) -> &Self::StateTuple {
        self.table.find_entry(s).unwrap()
    }

    #[inline]
    fn size(&self) -> usize {
        self.table.size()
    }
}

pub struct StringDetComposeStateTable<A: Arc, FS: FilterState>
where
    A::StateId: BiTableId,
{
    table: VectorStateTable<
        A::StateId,
        DefaultComposeStateTuple<A::StateId, FS>,
        ComposeState1Fingerprint,
    >,
    error: bool,
}

impl<A: Arc, FS: FilterState> StringDetComposeStateTable<A, FS>
where
    A::StateId: BiTableId,
{
    pub fn new<F1: Fst<A>, F2: Fst<A>>(fst1: &F1, fst2: &F2) -> Self {
        let mut error = false;
        let props2 = K_I_DETERMINISTIC | K_NO_I_EPSILONS;

        if fst1.properties(K_STRING, true) != K_STRING {
            log::error!("StringDetComposeStateTable: 1st FST is not a string");
            error = true;
        } else if fst2.properties(props2, true) != props2 {
            log::error!(
                "StringDetComposeStateTable: 2nd FST is not deterministic and epsilon-free"
            );
            error = true;
        }

        Self {
            table: VectorStateTable::new(ComposeState1Fingerprint, 0),
            error,
        }
    }
}

impl<A: Arc, FS: FilterState> ComposeStateTable<A, FS> for StringDetComposeStateTable<A, FS>
where
    A::StateId: BiTableId,
{
    type StateTuple = DefaultComposeStateTuple<A::StateId, FS>;

    #[inline]
    fn find_state(&mut self, tuple: &Self::StateTuple) -> A::StateId {
        self.table.find_id(tuple, true).unwrap()
    }

    #[inline]
    fn tuple(&self, s: A::StateId) -> &Self::StateTuple {
        self.table.find_entry(s).unwrap()
    }

    #[inline]
    fn size(&self) -> usize {
        self.table.size()
    }

    #[inline]
    fn error(&self) -> bool {
        self.error
    }
}

pub struct DetStringComposeStateTable<A: Arc, FS: FilterState>
where
    A::StateId: BiTableId,
{
    table: VectorStateTable<
        A::StateId,
        DefaultComposeStateTuple<A::StateId, FS>,
        ComposeState2Fingerprint,
    >,
    error: bool,
}

impl<A: Arc, FS: FilterState> DetStringComposeStateTable<A, FS>
where
    A::StateId: BiTableId,
{
    pub fn new<F1: Fst<A>, F2: Fst<A>>(fst1: &F1, fst2: &F2) -> Self {
        let mut error = false;
        let props1 = K_O_DETERMINISTIC | K_NO_O_EPSILONS;

        if fst1.properties(props1, true) != props1 {
            log::error!(
                "DetStringComposeStateTable: 1st FST is not input-deterministic and epsilon-free"
            );
            error = true;
        } else if fst2.properties(K_STRING, true) != K_STRING {
            log::error!("DetStringComposeStateTable: 2nd FST is not a string");
            error = true;
        }

        Self {
            table: VectorStateTable::new(ComposeState2Fingerprint, 0),
            error,
        }
    }
}

impl<A: Arc, FS: FilterState> ComposeStateTable<A, FS> for DetStringComposeStateTable<A, FS>
where
    A::StateId: BiTableId,
{
    type StateTuple = DefaultComposeStateTuple<A::StateId, FS>;

    #[inline]
    fn find_state(&mut self, tuple: &Self::StateTuple) -> A::StateId {
        self.table.find_id(tuple, true).unwrap()
    }

    #[inline]
    fn tuple(&self, s: A::StateId) -> &Self::StateTuple {
        self.table.find_entry(s).unwrap()
    }

    #[inline]
    fn size(&self) -> usize {
        self.table.size()
    }

    #[inline]
    fn error(&self) -> bool {
        self.error
    }
}

pub struct ErasableComposeStateTable<A: Arc, FS: FilterState>
where
    A::StateId: BiTableId,
{
    table: ErasableBiTable<A::StateId, DefaultComposeStateTuple<A::StateId, FS>>,
}

impl<A: Arc, FS: FilterState> ErasableComposeStateTable<A, FS>
where
    A::StateId: BiTableId,
{
    pub fn new<F1: Fst<A>, F2: Fst<A>>(_fst1: &F1, _fst2: &F2) -> Self {
        Self {
            table: ErasableStateTable::new(DefaultComposeStateTuple::no_state()),
        }
    }

    #[inline]
    pub fn erase(&mut self, s: A::StateId) {
        self.table.erase(s);
    }
}

impl<A: Arc, FS: FilterState> ComposeStateTable<A, FS> for ErasableComposeStateTable<A, FS>
where
    A::StateId: BiTableId,
{
    type StateTuple = DefaultComposeStateTuple<A::StateId, FS>;

    #[inline]
    fn find_state(&mut self, tuple: &Self::StateTuple) -> A::StateId {
        self.table.find_id(tuple, true).unwrap()
    }

    #[inline]
    fn tuple(&self, s: A::StateId) -> &Self::StateTuple {
        self.table.find_entry(s).unwrap()
    }

    #[inline]
    fn size(&self) -> usize {
        self.table.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AtomicRc;
    use crate::algorithms::filter_state::{CharFilterState, TrivialFilterState};
    use crate::arc::StdArc;
    use crate::float_weight::TropicalWeight;
    use crate::symbol_table::SymbolTable;
    use crate::weight::Weight as _;
    use std::iter::Empty;

    struct DummyFst {
        props: u64,
    }

    impl Fst<StdArc> for DummyFst {
        type StateIter<'a> = Empty<i32>;
        type ArcIter<'a> = Empty<StdArc>;
        fn start(&self) -> Option<i32> {
            Some(0)
        }
        fn final_weight(&self, _state: i32) -> TropicalWeight {
            TropicalWeight::zero()
        }
        fn num_arcs(&self, _state: i32) -> usize {
            0
        }
        fn num_input_epsilons(&self, _state: i32) -> usize {
            0
        }
        fn num_output_epsilons(&self, _state: i32) -> usize {
            0
        }
        fn num_states_if_known(&self) -> Option<usize> {
            Some(5)
        }
        fn properties(&self, mask: u64, _test: bool) -> u64 {
            self.props & mask
        }
        fn fst_type(&self) -> &str {
            "dummy"
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

    #[test]
    fn test_generic_compose_state_table() {
        let fst = DummyFst { props: 0 };
        let mut table = GenericComposeStateTable::<StdArc, TrivialFilterState>::new(&fst, &fst);

        let tuple1 = DefaultComposeStateTuple::new(0, 1, TrivialFilterState::new(true));
        let tuple2 = DefaultComposeStateTuple::new(0, 2, TrivialFilterState::new(true));

        let id1 = table.find_state(&tuple1);
        let id2 = table.find_state(&tuple2);

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(table.size(), 2);

        assert_eq!(table.find_state(&tuple1), 0);
        assert_eq!(table.tuple(0), &tuple1);
    }

    #[test]
    fn test_product_compose_state_table() {
        let fst = DummyFst { props: 0 };
        let mut table = ProductComposeStateTable::<StdArc, TrivialFilterState>::new(&fst, &fst, 0);

        let tuple = DefaultComposeStateTuple::new(1, 2, TrivialFilterState::new(true));
        let id = table.find_state(&tuple);

        assert_eq!(id, 0);
        assert_eq!(table.tuple(0), &tuple);
    }

    #[test]
    fn test_string_det_compose_state_table_error() {
        let fst1 = DummyFst { props: 0 };
        let fst2 = DummyFst {
            props: K_I_DETERMINISTIC | K_NO_I_EPSILONS,
        };

        let table = StringDetComposeStateTable::<StdArc, TrivialFilterState>::new(&fst1, &fst2);
        assert!(table.error());
    }

    #[test]
    fn test_erasable_compose_state_table() {
        let fst = DummyFst { props: 0 };
        let mut table = ErasableComposeStateTable::<StdArc, TrivialFilterState>::new(&fst, &fst);

        let tuple1 = DefaultComposeStateTuple::new(0, 1, TrivialFilterState::new(true));
        let id1 = table.find_state(&tuple1);
        assert_eq!(id1, 0);

        table.erase(id1);
        let id1_new = table.find_state(&tuple1);
        assert_ne!(id1, id1_new);
    }

    type Tuple = DefaultComposeStateTuple<i32, CharFilterState>;

    fn tuple(s1: i32, s2: i32, fs: i8) -> Tuple {
        DefaultComposeStateTuple::new(s1, s2, CharFilterState::new(fs))
    }

    /// Every tuple a composition could reach, over a 4x4 product with three
    /// filter states.
    fn all_tuples() -> Vec<Tuple> {
        let mut out = Vec::new();
        for s1 in 0..4 {
            for s2 in 0..4 {
                for fs in 0..3 {
                    out.push(tuple(s1, s2, fs));
                }
            }
        }
        out
    }

    /// What a state table is: a bijection. The same tuple always gets the same
    /// ID, different tuples never share one, and every ID leads back to the
    /// tuple it was made for. Composition builds its states on this, and two
    /// tuples sharing an ID would merge two states of the output.
    fn assert_bijective<T: ComposeStateTable<StdArc, CharFilterState, StateTuple = Tuple>>(
        table: &mut T,
        tuples: &[Tuple],
    ) {
        let ids: Vec<i32> = tuples.iter().map(|t| table.find_state(t)).collect();

        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), tuples.len(), "two tuples share an ID");
        assert_eq!(table.size(), tuples.len());

        for (t, &id) in tuples.iter().zip(&ids) {
            assert_eq!(table.tuple(id), t, "ID {id} leads back to another tuple");
            assert_eq!(table.find_state(t), id, "looking up again moved the ID");
        }
        assert_eq!(table.size(), tuples.len(), "looking up again added entries");
    }

    #[test]
    fn a_generic_table_is_a_bijection() {
        let fst = DummyFst { props: 0 };
        let mut table = GenericComposeStateTable::<StdArc, CharFilterState>::new(&fst, &fst);
        assert_bijective(&mut table, &all_tuples());
    }

    #[test]
    fn a_product_table_is_a_bijection() {
        let fst = DummyFst { props: 0 };
        let mut table = ProductComposeStateTable::<StdArc, CharFilterState>::new(&fst, &fst, 0);
        assert_bijective(&mut table, &all_tuples());
    }

    #[test]
    fn an_erasable_table_is_a_bijection() {
        let fst = DummyFst { props: 0 };
        let mut table = ErasableComposeStateTable::<StdArc, CharFilterState>::new(&fst, &fst);
        assert_bijective(&mut table, &all_tuples());
    }

    /// The vector-backed tables key on one side alone, so they are bijections
    /// only over tuples that differ on that side, which is the condition their
    /// constructors check the inputs for.
    #[test]
    fn the_one_sided_tables_are_bijections_over_the_side_they_key_on() {
        let string = DummyFst {
            props: K_STRING | K_I_DETERMINISTIC | K_NO_I_EPSILONS,
        };
        let det = DummyFst {
            props: K_I_DETERMINISTIC
                | K_NO_I_EPSILONS
                | K_O_DETERMINISTIC
                | K_NO_O_EPSILONS
                | K_STRING,
        };

        let by_first: Vec<Tuple> = (0..4).map(|s1| tuple(s1, 0, 0)).collect();
        let mut table = StringDetComposeStateTable::<StdArc, CharFilterState>::new(&string, &det);
        assert!(!table.error());
        assert_bijective(&mut table, &by_first);

        let by_second: Vec<Tuple> = (0..4).map(|s2| tuple(0, s2, 0)).collect();
        let mut table = DetStringComposeStateTable::<StdArc, CharFilterState>::new(&det, &string);
        assert!(!table.error());
        assert_bijective(&mut table, &by_second);
    }

    /// Erasing frees the ID for reuse; the table's own doc says a caller must
    /// either never come back to that tuple or not mind a new ID.
    #[test]
    fn erasing_lets_a_table_forget_a_state() {
        let fst = DummyFst { props: 0 };
        let mut table = ErasableComposeStateTable::<StdArc, CharFilterState>::new(&fst, &fst);

        let first = table.find_state(&tuple(0, 0, 0));
        let second = table.find_state(&tuple(1, 1, 0));
        assert_eq!(table.size(), 2);

        table.erase(first);
        // The other entry is untouched.
        assert_eq!(table.tuple(second), &tuple(1, 1, 0));
        assert_eq!(table.find_state(&tuple(1, 1, 0)), second);
    }

    /// The fingerprint has to be injective, or the vector table it indexes
    /// would put two tuples in one slot.
    #[test]
    fn the_compose_fingerprint_separates_every_tuple_of_the_product() {
        let fingerprint = ComposeFingerprint::new(4, 4);
        let mut seen: Vec<usize> = all_tuples()
            .iter()
            .map(|t| fingerprint.fingerprint(t))
            .collect();
        let count = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), count, "two tuples share a fingerprint");
    }

    #[test]
    fn the_one_sided_fingerprints_read_the_side_they_name() {
        assert_eq!(ComposeState1Fingerprint.fingerprint(&tuple(3, 7, 2)), 3);
        assert_eq!(ComposeState2Fingerprint.fingerprint(&tuple(3, 7, 2)), 7);
    }

    /// A tuple's filter-state number comes from the filter state itself, which
    /// is the purpose of upstream's `FilterState::Hash()`.
    #[test]
    fn a_tuples_filter_state_number_is_the_filter_states_own() {
        assert_eq!(tuple(0, 0, 2).hash_value(), 2);
        let trivial: DefaultComposeStateTuple<i32, TrivialFilterState> =
            DefaultComposeStateTuple::new(0, 0, TrivialFilterState::new(true));
        assert_eq!(
            trivial.hash_value(),
            0,
            "a trivial filter state adds nothing"
        );
    }
}
