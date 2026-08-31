//! Editing an FST without copying it.
//!
//! Port of OpenFst's `edit-fst.h`. An [`EditFst`] wraps an FST that cannot be
//! changed and records the changes beside it, copying a state into its own
//! storage only when that state is first touched. For a large FST and a handful
//! of edits, that is the difference between copying everything and copying
//! almost nothing.

use std::io::{Read, Seek, Write};

use rustc_hash::FxHashMap;

use crate::AtomicRc;
use crate::algorithms::test_properties::cached_properties;
use crate::arc::{Arc, ArcStateId};
use crate::error::OpenFstError;
use crate::fst::{
    ExpandedFst, Fst, FstReadOptions, FstWriteOptions, MutableFst, PropertyCache, read_fst_header,
    write_fst_header,
};
use crate::fst_header::FstHeader;
use crate::fst_type::FstType;
#[cfg(feature = "fst-types")]
use crate::fsts::any_fst::AnyFst;
use crate::fsts::vector_fst::VectorFst;
use crate::properties::*;
use crate::symbol_table::SymbolTable;
use crate::utils::io::{FstScalar, read_scalar, write_scalar};
use crate::weight::WeightIo;

/// An FST that records edits to another rather than changing it.
///
/// SICADA-DIVERGE: upstream refuses `DeleteStates(span)`, the one `MutableFst`
/// method it cannot support, reporting an error at run time. Here it is not
/// on the type at all, because [`MutableFst`] does not need to be implemented
/// in full to be useful. Deleting states renumbers the ones that remain, which
/// is precisely what an FST that maps its own numbering onto a wrapped one
/// cannot survive.
pub struct EditFst<A: Arc, F: ExpandedFst<A>> {
    wrapped: F,
    /// The states that have been copied out and edited, plus any new ones.
    edits: VectorFst<A>,
    /// Where a state of the wrapped FST lives once it has been edited.
    external_to_internal: FxHashMap<A::StateId, A::StateId>,
    /// A final weight changed on a state that is otherwise untouched.
    ///
    /// Kept apart from `edits` so that setting a final weight does not force
    /// the state's arcs to be copied, which for a state with many arcs is the
    /// whole cost the type exists to avoid.
    edited_final_weights: FxHashMap<A::StateId, A::Weight>,
    num_new_states: usize,
    start: Option<A::StateId>,
    properties: PropertyCache,
    input_symbols: Option<AtomicRc<SymbolTable>>,
    output_symbols: Option<AtomicRc<SymbolTable>>,
}

impl<A: Arc, F: ExpandedFst<A>> EditFst<A, F> {
    /// Wraps `fst`, which is left untouched.
    pub fn new(fst: F) -> Self {
        let properties = fst.properties(K_FST_PROPERTIES, false);
        let start = fst.start();
        let input_symbols = fst.input_symbols();
        let output_symbols = fst.output_symbols();
        Self {
            wrapped: fst,
            edits: VectorFst::new(),
            external_to_internal: FxHashMap::default(),
            edited_final_weights: FxHashMap::default(),
            num_new_states: 0,
            start,
            properties: PropertyCache::new(properties | K_MUTABLE),
            input_symbols,
            output_symbols,
        }
    }

    /// The FST being edited, as it was.
    pub fn wrapped(&self) -> &F {
        &self.wrapped
    }

    /// How many states have been copied out or newly added.
    pub fn num_edited_states(&self) -> usize {
        self.external_to_internal.len()
    }

    /// Where `state` lives now, copying it out of the wrapped FST if this is
    /// the first time it has been edited.
    fn make_editable(&mut self, state: A::StateId) -> A::StateId {
        if let Some(&internal) = self.external_to_internal.get(&state) {
            return internal;
        }
        let internal = self.edits.add_state();
        self.external_to_internal.insert(state, internal);
        for arc in self.wrapped.arcs(state) {
            self.edits.add_arc(internal, arc);
        }
        // A final weight already changed for this state wins over the wrapped
        // one, and moves into the copy along with the arcs.
        match self.edited_final_weights.remove(&state) {
            Some(weight) => self.edits.set_final(internal, weight),
            None => self
                .edits
                .set_final(internal, self.wrapped.final_weight(state)),
        }
        internal
    }

    /// Where `state` lives, without copying anything.
    #[inline]
    fn internal(&self, state: A::StateId) -> Option<A::StateId> {
        self.external_to_internal.get(&state).copied()
    }
}

/// Arcs of a state, from whichever of the two FSTs holds it.
pub enum EditArcIter<'a, A: Arc + 'a, F: Fst<A> + 'a> {
    /// The state has not been edited, so its arcs are the wrapped FST's.
    Wrapped(F::ArcIter<'a>),
    /// The state has been copied out.
    Edited(<VectorFst<A> as Fst<A>>::ArcIter<'a>),
}

impl<'a, A: Arc + 'a, F: Fst<A> + 'a> Clone for EditArcIter<'a, A, F> {
    fn clone(&self) -> Self {
        match self {
            Self::Wrapped(iter) => Self::Wrapped(iter.clone()),
            Self::Edited(iter) => Self::Edited(iter.clone()),
        }
    }
}

impl<'a, A: Arc + 'a, F: Fst<A> + 'a> Iterator for EditArcIter<'a, A, F> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        match self {
            Self::Wrapped(iter) => iter.next(),
            Self::Edited(iter) => iter.next(),
        }
    }
}

impl<A: Arc, F: ExpandedFst<A>> Fst<A> for EditFst<A, F> {
    type StateIter<'a>
        = std::iter::Map<std::ops::Range<usize>, fn(usize) -> A::StateId>
    where
        Self: 'a;
    type ArcIter<'a>
        = EditArcIter<'a, A, F>
    where
        Self: 'a;

    #[inline]
    fn start(&self) -> Option<A::StateId> {
        self.start
    }

    fn final_weight(&self, state: A::StateId) -> A::Weight {
        if let Some(weight) = self.edited_final_weights.get(&state) {
            return weight.clone();
        }
        match self.internal(state) {
            Some(internal) => self.edits.final_weight(internal),
            None => self.wrapped.final_weight(state),
        }
    }

    fn num_arcs(&self, state: A::StateId) -> usize {
        match self.internal(state) {
            Some(internal) => self.edits.num_arcs(internal),
            None => self.wrapped.num_arcs(state),
        }
    }

    fn num_input_epsilons(&self, state: A::StateId) -> usize {
        match self.internal(state) {
            Some(internal) => self.edits.num_input_epsilons(internal),
            None => self.wrapped.num_input_epsilons(state),
        }
    }

    fn num_output_epsilons(&self, state: A::StateId) -> usize {
        match self.internal(state) {
            Some(internal) => self.edits.num_output_epsilons(internal),
            None => self.wrapped.num_output_epsilons(state),
        }
    }

    #[inline]
    fn num_states_if_known(&self) -> Option<usize> {
        Some(self.num_states())
    }

    fn properties(&self, mask: u64, test: bool) -> u64 {
        cached_properties(self, &self.properties, mask, test)
    }

    #[inline]
    fn fst_type(&self) -> &str {
        FstType::EDIT.as_str()
    }

    fn input_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
        self.input_symbols.clone()
    }

    fn output_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
        self.output_symbols.clone()
    }

    fn states<'a>(&'a self) -> Self::StateIter<'a> {
        (0..self.num_states()).map(A::StateId::from_usize as fn(usize) -> A::StateId)
    }

    fn arcs<'a>(&'a self, state: A::StateId) -> Self::ArcIter<'a> {
        match self.internal(state) {
            Some(internal) => EditArcIter::Edited(self.edits.arcs(internal)),
            None => EditArcIter::Wrapped(self.wrapped.arcs(state)),
        }
    }
}

impl<A: Arc, F: ExpandedFst<A>> ExpandedFst<A> for EditFst<A, F> {
    #[inline]
    fn num_states(&self) -> usize {
        self.wrapped.num_states() + self.num_new_states
    }
}

impl<A: Arc, F: ExpandedFst<A>> EditFst<A, F> {
    /// Sets the initial state.
    pub fn set_start(&mut self, state: A::StateId) {
        self.start = (state != A::StateId::no_state()).then_some(state);
        self.properties.modify(set_start_properties);
    }

    /// Sets a state's final weight, without copying its arcs.
    pub fn set_final(&mut self, state: A::StateId, weight: A::Weight) {
        let old = self.final_weight(state);
        match self.internal(state) {
            Some(internal) => self.edits.set_final(internal, weight.clone()),
            None => {
                self.edited_final_weights.insert(state, weight.clone());
            }
        }
        self.properties
            .modify(|props| set_final_properties(props, &old, &weight));
    }

    /// Adds a state, numbered after every state the wrapped FST has.
    pub fn add_state(&mut self) -> A::StateId {
        let external = A::StateId::from_usize(self.num_states());
        let internal = self.edits.add_state();
        self.external_to_internal.insert(external, internal);
        self.num_new_states += 1;
        self.properties.modify(add_state_properties);
        external
    }

    /// Adds an arc, copying the state out of the wrapped FST if it has not
    /// been edited before.
    pub fn add_arc(&mut self, state: A::StateId, arc: A) {
        let internal = self.make_editable(state);
        let prev = self.edits.arcs(internal).last();
        self.properties
            .modify(|props| add_arc_properties(props, state, &arc, prev.as_ref()));
        self.edits.add_arc(internal, arc);
    }

    /// Removes every arc leaving a state.
    pub fn delete_arcs(&mut self, state: A::StateId) {
        let internal = self.make_editable(state);
        self.edits.delete_arcs(internal);
        self.properties.modify(delete_arcs_properties);
    }

    /// Removes the last `n` arcs leaving a state.
    pub fn delete_arcs_n(&mut self, state: A::StateId, n: usize) {
        let internal = self.make_editable(state);
        self.edits.delete_arcs_n(internal, n);
        self.properties.modify(delete_arcs_properties);
    }

    /// Rewrites every arc leaving a state.
    pub fn mutate_arcs<M>(&mut self, state: A::StateId, mutator: M)
    where
        M: FnMut(&mut A),
    {
        let internal = self.make_editable(state);
        self.edits.mutate_arcs(internal, mutator);
        // The edited FST recomputed its own bits; take them for this state's
        // worth of arcs, which is all that changed.
        let edited = self.edits.properties(K_FST_PROPERTIES, false);
        self.properties.modify(|props| props & edited);
    }

    /// Sets the property bits directly.
    pub fn set_properties(&mut self, props: u64, mask: u64) {
        self.properties
            .modify(|current| (current & (!mask | K_ERROR)) | (props & mask));
    }
}

/// The file version this format writes, matching upstream's `kFileVersion`.
const FILE_VERSION: i32 = 2;
/// The oldest version that can be read, matching upstream's `kMinFileVersion`.
const MIN_FILE_VERSION: i32 = 2;

#[cfg(feature = "fst-types")]
impl<'f, A: Arc + 'static> EditFst<A, AnyFst<'f, A>>
where
    A::Label: FstScalar,
    A::StateId: FstScalar,
    A::Weight: Copy + WeightIo,
{
    /// Writes the FST: its own header, then the wrapped FST, then the edits.
    ///
    /// SICADA-DIVERGE: upstream writes the two edit maps in whatever order its
    /// hash tables happen to iterate in, so writing the same FST twice can
    /// produce different bytes. Sorting by state costs one pass over the states
    /// that have been edited, a set that is small by construction since that is
    /// what the type exists for, and makes the output reproducible. What the
    /// reader does is unchanged either way.
    pub fn write<W: Write>(
        &self,
        writer: &mut W,
        opts: &FstWriteOptions,
    ) -> Result<(), OpenFstError> {
        let header = FstHeader {
            fst_type: FstType::EDIT.as_str().to_string(),
            arc_type: A::type_name().as_str().to_string(),
            version: FILE_VERSION,
            flags: 0,
            properties: self.properties(K_FST_PROPERTIES, false),
            start: self.start.map_or(-1, |s| s.as_usize() as i64),
            num_states: self.num_states() as i64,
            num_arcs: -1,
        };
        // The symbol tables belong to the wrapped FST, which writes its own
        // header; carrying a second copy here would let the two disagree.
        let header_opts = FstWriteOptions {
            write_isymbols: false,
            write_osymbols: false,
            ..opts.clone()
        };
        write_fst_header(writer, &header_opts, &header, None, None)?;

        // Both contained FSTs are written with their own headers, since
        // reading them back has to work without knowing what they are.
        let contained = FstWriteOptions {
            write_header: true,
            ..opts.clone()
        };
        self.wrapped.write(writer, &contained)?;
        self.edits.write(writer, &contained)?;

        let mut mapped: Vec<(A::StateId, A::StateId)> = self
            .external_to_internal
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect();
        mapped.sort_unstable_by_key(|(external, _)| external.as_usize());
        write_scalar(writer, mapped.len() as i64)?;
        for (external, internal) in mapped {
            write_scalar(writer, external)?;
            write_scalar(writer, internal)?;
        }

        let mut finals: Vec<(A::StateId, A::Weight)> = self
            .edited_final_weights
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect();
        finals.sort_unstable_by_key(|(state, _)| state.as_usize());
        write_scalar(writer, finals.len() as i64)?;
        for (state, weight) in finals {
            write_scalar(writer, state)?;
            weight.write(writer)?;
        }

        write_scalar(writer, A::StateId::from_usize(self.num_new_states))?;
        Ok(())
    }

    /// Reads an FST written by [`write`](Self::write).
    ///
    /// SICADA-DIVERGE: upstream reads the wrapped FST through its dynamic
    /// registry and then `down_cast`s the result to whichever type the
    /// `WrappedFstT` template parameter names. That is an unchecked cast in a
    /// release build, so an `EditFst<Arc, VectorFst<Arc>>` reading a file whose
    /// wrapped FST is a `ConstFst` gets a `VectorFst*` pointing at a `ConstFst`.
    /// The wrapped FST here is an [`AnyFst`], which expresses what the cast was
    /// reaching for, and no cast is involved.
    ///
    /// SICADA-DIVERGE: upstream writes no symbol tables in the `edit` header
    /// and does not take them from the wrapped FST when reading either, so a
    /// round trip loses them. They are taken from the wrapped FST here.
    pub fn read<R: Read + Seek>(
        reader: &mut R,
        opts: &FstReadOptions,
    ) -> Result<Self, OpenFstError> {
        let read = read_fst_header::<A, _>(reader, opts, FstType::EDIT.as_str(), MIN_FILE_VERSION)?;
        let header = read.header;

        // Each contained FST carries its own header, so the outer one must not
        // be offered to it.
        let contained = FstReadOptions {
            header: None,
            ..opts.clone()
        };
        let wrapped = AnyFst::read(reader, &contained)?;
        // The edit store's arcs point into the *outer* numbering, not its own,
        // so the arc-range check a standalone vector FST gets would reject it.
        // That is a property of upstream's format: the same vector FST layout
        // is reused for a fragment that is not self-contained.
        let edits = VectorFst::read(
            reader,
            &FstReadOptions {
                verify: false,
                ..contained.clone()
            },
        )?;

        let mut fst = Self::new(wrapped);
        fst.edits = edits;

        let count: i64 = read_scalar(reader)?;
        for _ in 0..count.max(0) {
            let external: A::StateId = read_scalar(reader)?;
            let internal: A::StateId = read_scalar(reader)?;
            fst.external_to_internal.insert(external, internal);
        }
        let count: i64 = read_scalar(reader)?;
        for _ in 0..count.max(0) {
            let state: A::StateId = read_scalar(reader)?;
            let weight = A::Weight::read(reader)?;
            fst.edited_final_weights.insert(state, weight);
        }
        let num_new_states: A::StateId = read_scalar(reader)?;
        fst.num_new_states = num_new_states.as_usize();

        fst.start = (header.start >= 0).then(|| A::StateId::from_usize(header.start as usize));
        fst.properties = PropertyCache::new(header.properties | K_MUTABLE);
        Ok(fst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::weight::Weight;
    use crate::weights::float_weight::TropicalWeight;

    /// A chain of `n` states with one arc apiece, the last final.
    fn chain(n: usize) -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..n {
            fst.add_state();
        }
        fst.set_start(0);
        for s in 0..n - 1 {
            fst.add_arc(
                s as i32,
                StdArc::new(
                    s as i32 + 1,
                    s as i32 + 1,
                    TropicalWeight(s as f32),
                    s as i32 + 1,
                ),
            );
        }
        fst.set_final(n as i32 - 1, TropicalWeight(9.0));
        fst
    }

    #[test]
    fn an_unedited_fst_reads_exactly_as_the_one_it_wraps() {
        let wrapped = chain(5);
        let edit = EditFst::new(wrapped.clone());

        assert_eq!(edit.num_states(), wrapped.num_states());
        assert_eq!(edit.start(), wrapped.start());
        for s in 0..wrapped.num_states() as i32 {
            assert_eq!(edit.final_weight(s), wrapped.final_weight(s));
            assert_eq!(edit.num_arcs(s), wrapped.num_arcs(s));
            assert_eq!(
                edit.arcs(s).collect::<Vec<_>>(),
                wrapped.arcs(s).collect::<Vec<_>>()
            );
        }
        assert_eq!(edit.num_edited_states(), 0, "nothing was copied");
    }

    /// The point of the type: a change to one state copies that state and
    /// nothing else.
    #[test]
    fn editing_one_state_copies_only_that_state() {
        let mut edit = EditFst::new(chain(100));
        edit.add_arc(7, StdArc::new(1, 1, TropicalWeight::one(), 9));

        assert_eq!(edit.num_edited_states(), 1);
        assert_eq!(edit.num_arcs(7), 2, "the original arc came along");
        assert_eq!(edit.num_arcs(8), 1, "its neighbour was left alone");
        // The wrapped FST is untouched.
        assert_eq!(edit.wrapped().num_arcs(7), 1);
    }

    /// Changing only a final weight does not copy the state's arcs, which for
    /// a state with many of them is the cost the type exists to avoid.
    #[test]
    fn setting_a_final_weight_does_not_copy_the_arcs() {
        let mut edit = EditFst::new(chain(10));
        edit.set_final(3, TropicalWeight(4.0));

        assert_eq!(edit.final_weight(3), TropicalWeight(4.0));
        assert_eq!(edit.num_edited_states(), 0, "no state was copied out");
        assert_eq!(edit.arcs(3).count(), 1, "its arcs still read through");
    }

    /// And if the arcs are edited afterwards, the final weight already set
    /// comes along rather than being overwritten by the wrapped one.
    #[test]
    fn a_final_weight_set_first_survives_a_later_arc_edit() {
        let mut edit = EditFst::new(chain(10));
        edit.set_final(3, TropicalWeight(4.0));
        edit.add_arc(3, StdArc::new(1, 1, TropicalWeight::one(), 5));

        assert_eq!(edit.final_weight(3), TropicalWeight(4.0));
        assert_eq!(edit.num_arcs(3), 2);
    }

    /// New states are numbered after every state the wrapped FST has, and can
    /// be reached from the states it already had.
    #[test]
    fn new_states_are_numbered_after_the_wrapped_ones() {
        let mut edit = EditFst::new(chain(4));
        let added = edit.add_state();
        assert_eq!(added, 4);
        assert_eq!(edit.num_states(), 5);

        edit.set_final(added, TropicalWeight(1.0));
        edit.add_arc(0, StdArc::new(9, 9, TropicalWeight::one(), added));

        assert_eq!(edit.final_weight(added), TropicalWeight(1.0));
        let arcs: Vec<_> = edit.arcs(0).collect();
        assert_eq!(arcs.len(), 2);
        assert_eq!(arcs[1].nextstate(), added);
    }

    #[test]
    fn deleting_arcs_leaves_the_wrapped_fst_alone() {
        let mut edit = EditFst::new(chain(5));
        edit.delete_arcs(2);
        assert_eq!(edit.num_arcs(2), 0);
        assert_eq!(edit.wrapped().num_arcs(2), 1);
        // Every other state still reads through.
        assert_eq!(edit.num_arcs(1), 1);
    }

    #[test]
    fn rewriting_arcs_goes_through_the_copy() {
        let mut edit = EditFst::new(chain(5));
        edit.mutate_arcs(1, |arc| {
            *arc = StdArc::new(99, 99, *arc.weight(), arc.nextstate());
        });
        assert_eq!(edit.arcs(1).next().unwrap().ilabel(), 99);
        assert_eq!(edit.wrapped().arcs(1).next().unwrap().ilabel(), 2);
    }

    /// The start state can be moved, including onto a state that was added.
    #[test]
    fn the_start_state_can_be_moved() {
        let mut edit = EditFst::new(chain(3));
        assert_eq!(edit.start(), Some(0));
        let added = edit.add_state();
        edit.set_start(added);
        assert_eq!(edit.start(), Some(added));
        edit.set_start(<i32 as ArcStateId>::no_state());
        assert_eq!(edit.start(), None);
    }
}
