//! An FST that carries the index its matcher needs.
//!
//! Port of OpenFst's `matcher-fst.h`. Building a
//! [`LabelReachable`](crate::algorithms::label_reachable::LabelReachable) index
//! costs a pass over the whole FST, and composition against the same FST is
//! usually done many times, so the index is built once, saved beside the FST,
//! and read back with it. That is all this type is: an FST, plus one index per
//! side, plus the file format that keeps them together.
//!
//! Upstream calls the attached object an "add-on" and the pairing
//! `AddOnImpl<FST, AddOnPair<Data, Data>>`; see [`AddOnImpl`] for the bytes.

use std::io::{Read, Seek, Write};
use std::marker::PhantomData;
use std::sync::Arc as StdArc;

use crate::AtomicRc;
use crate::add_on::{AddOn, AddOnImpl, AddOnPair};
use crate::algorithms::accumulator::WeightAccumulator;
use crate::algorithms::label_reachable::{Index, LabelReachableData};
use crate::algorithms::lookahead_matcher::{
    DEFAULT_LABEL_LOOKAHEAD_FLAGS, INPUT_LOOKAHEAD_MATCHER, LabelLookAheadMatcher,
    OUTPUT_LOOKAHEAD_MATCHER,
};
use crate::arc::{Arc, ArcLabel};
use crate::data_structures::interval_set::{IntInterval, IntervalSet};
use crate::error::OpenFstError;
use crate::fst::{ExpandedFst, Fst, FstReadOptions, FstWriteOptions, MatchType};
use crate::fst_type::FstType;
use crate::fsts::any_fst::{AnyArcIter, AnyFst, AnyStateIter};
use crate::symbol_table::SymbolTable;
use crate::utils::io::{FstScalar, read_scalar, write_scalar};
use crate::weight::WeightIo;

/// The flags upstream gives its `ilabel_lookahead` FST's matcher.
///
/// Note it does *not* include `kLookAheadKeepRelabelData`: upstream relabels
/// the contained FST to the index numbering once and throws the label map
/// away. See [`ReachableAddOn`] for why sicada keeps it.
pub const ILABEL_LOOKAHEAD_FLAGS: u32 = INPUT_LOOKAHEAD_MATCHER
    | crate::algorithms::lookahead_matcher::LOOKAHEAD_WEIGHT
    | crate::algorithms::lookahead_matcher::LOOKAHEAD_PREFIX
    | crate::algorithms::lookahead_matcher::LOOKAHEAD_EPSILONS
    | crate::algorithms::lookahead_matcher::LOOKAHEAD_NON_EPSILON_PREFIX;

/// The flags upstream gives its `olabel_lookahead` FST's matcher.
pub const OLABEL_LOOKAHEAD_FLAGS: u32 = OUTPUT_LOOKAHEAD_MATCHER
    | crate::algorithms::lookahead_matcher::LOOKAHEAD_WEIGHT
    | crate::algorithms::lookahead_matcher::LOOKAHEAD_PREFIX
    | crate::algorithms::lookahead_matcher::LOOKAHEAD_EPSILONS
    | crate::algorithms::lookahead_matcher::LOOKAHEAD_NON_EPSILON_PREFIX;

/// A [`LabelReachableData`] on its way to or from a file.
///
/// The index is held in memory with `i64` numbers whatever the arc's label type
/// is, but written with the label's own width, which is the width upstream's
/// `LabelReachableData<Label>` writes and so what its files hold. `L` is that
/// label type; it says nothing about the value at run time, only how it is
/// spelled on the wire.
pub struct ReachableAddOn<L> {
    data: StdArc<LabelReachableData>,
    _marker: PhantomData<fn() -> L>,
}

impl<L> Clone for ReachableAddOn<L> {
    fn clone(&self) -> Self {
        Self {
            data: StdArc::clone(&self.data),
            _marker: PhantomData,
        }
    }
}

impl<L> ReachableAddOn<L> {
    /// Wraps an index for saving.
    pub fn new(data: StdArc<LabelReachableData>) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }

    /// The index.
    pub fn data(&self) -> &StdArc<LabelReachableData> {
        &self.data
    }
}

/// Turns an in-memory index number into the label-width number a file holds.
fn to_wire<L: ArcLabel>(index: Index) -> Result<L, OpenFstError> {
    L::from_i64(index).ok_or_else(|| {
        OpenFstError::InvalidOperation(format!(
            "MatcherFst: index {index} does not fit the arc's label type"
        ))
    })
}

/// And back again.
fn from_wire<L: ArcLabel>(value: L) -> Result<Index, OpenFstError> {
    value.to_i64().ok_or_else(|| {
        OpenFstError::InvalidOperation("MatcherFst: a label on file does not fit an i64".into())
    })
}

impl<L: ArcLabel + FstScalar> AddOn for ReachableAddOn<L> {
    fn read<R: Read>(reader: &mut R, _opts: &FstReadOptions) -> Result<Self, OpenFstError> {
        let reach_input: bool = read_scalar(reader)?;
        let keep_relabel_data: bool = read_scalar(reader)?;
        if !keep_relabel_data {
            // Upstream drops the label map and relabels the contained FST to
            // the index numbering instead. sicada keeps the map and looks a
            // label up per question, so a file written without it names labels
            // this index can no longer speak about; saying so beats handing
            // back a matcher that answers "no" to everything.
            return Err(OpenFstError::InvalidOperation(
                "MatcherFst: the file was written without its label map (upstream's \
                 kLookAheadKeepRelabelData was off), so its labels are already renumbered"
                    .into(),
            ));
        }
        let count: i64 = read_scalar(reader)?;
        let mut label2index = hashbrown::HashMap::with_capacity(count.max(0) as usize);
        for _ in 0..count.max(0) {
            let label: L = read_scalar(reader)?;
            let index: L = read_scalar(reader)?;
            label2index.insert(from_wire(label)?, from_wire(index)?);
        }
        let final_index: L = read_scalar(reader)?;
        let count: i64 = read_scalar(reader)?;
        let mut interval_sets = Vec::with_capacity(count.max(0) as usize);
        for _ in 0..count.max(0) {
            let n: i64 = read_scalar(reader)?;
            let mut set = IntervalSet::with_capacity(n.max(0) as usize);
            for _ in 0..n.max(0) {
                let begin: L = read_scalar(reader)?;
                let end: L = read_scalar(reader)?;
                set.push(IntInterval::new(from_wire(begin)?, from_wire(end)?));
            }
            // SICADA-DIVERGE: upstream reads the member count from the file and
            // takes it on trust. It is a function of the intervals, so it is
            // recomputed; a file whose count disagreed with its intervals would
            // otherwise make every "how many labels" answer wrong.
            let _stored_count: L = read_scalar(reader)?;
            set.normalize();
            interval_sets.push(set);
        }
        Ok(Self::new(StdArc::new(LabelReachableData::from_parts(
            reach_input,
            label2index,
            from_wire(final_index)?,
            interval_sets,
        ))))
    }

    fn write<W: Write>(&self, writer: &mut W, _opts: &FstWriteOptions) -> Result<(), OpenFstError> {
        write_scalar(writer, self.data.reach_input())?;
        // Always: see `read`.
        write_scalar(writer, true)?;
        write_scalar(writer, self.data.num_labels() as i64)?;
        // Sorted, so that writing the same index twice produces the same bytes;
        // upstream writes in whatever order its hash table iterates in.
        let mut pairs: Vec<(i64, Index)> = self.data.label_indices().collect();
        pairs.sort_unstable();
        for (label, index) in pairs {
            write_scalar(writer, to_wire::<L>(label)?)?;
            write_scalar(writer, to_wire::<L>(index)?)?;
        }
        write_scalar(writer, to_wire::<L>(self.data.final_index())?)?;
        let sets = self.data.interval_sets();
        write_scalar(writer, sets.len() as i64)?;
        for set in sets {
            write_scalar(writer, set.intervals().len() as i64)?;
            for interval in set.intervals() {
                write_scalar(writer, to_wire::<L>(interval.begin)?)?;
                write_scalar(writer, to_wire::<L>(interval.end)?)?;
            }
            write_scalar(writer, to_wire::<L>(set.count() as Index)?)?;
        }
        Ok(())
    }
}

/// The pair of indices a matcher FST carries: one for each side.
pub type ReachablePair<L> = AddOnPair<ReachableAddOn<L>, ReachableAddOn<L>>;

/// An FST with a look-ahead index saved beside it.
///
/// SICADA-DIVERGE: upstream's `MatcherFst` takes the matcher type, the type
/// name and an initializer functor as template parameters, so
/// `StdILabelLookAheadFst` is a distinct C++ type from `StdOLabelLookAheadFst`.
/// Which side is indexed is a property of the data, not of the code, so it is
/// a field here and the two are the same type; [`ilabel_lookahead`] and
/// [`olabel_lookahead`] build them.
pub struct MatcherFst<'f, A: Arc + 'static>
where
    A::Weight: Copy,
{
    impl_: AddOnImpl<'f, A, ReachablePair<A::Label>>,
}

impl<'f, A> MatcherFst<'f, A>
where
    A: Arc + 'static,
    A::Weight: Copy,
{
    /// The FST underneath.
    pub fn fst(&self) -> &AnyFst<'f, A> {
        self.impl_.fst()
    }

    /// The name this FST goes by in a header.
    pub fn fst_type_name(&self) -> FstType {
        self.impl_.fst_type()
    }

    /// The index for one side, if it was built.
    ///
    /// [`MatchType::Input`] asks for the index over input labels, as a matcher
    /// looking labels up on the input side requires.
    pub fn data(&self, match_type: MatchType) -> Option<StdArc<LabelReachableData>> {
        let pair = self.impl_.add_on()?;
        let half = match match_type {
            MatchType::Input => pair.first(),
            _ => pair.second(),
        };
        half.map(|half| StdArc::clone(half.data()))
    }

    /// A matcher over this FST, answering from the saved index.
    ///
    /// `inner` is the matcher that does the ordinary label lookup; the index
    /// only answers the look-ahead questions.
    pub fn matcher<M, Acc>(
        &self,
        inner: M,
        match_type: MatchType,
        accumulator: Acc,
    ) -> Result<LabelLookAheadMatcher<A, M, Acc>, OpenFstError>
    where
        Acc: WeightAccumulator<A>,
    {
        let Some(data) = self.data(match_type) else {
            return Err(OpenFstError::InvalidOperation(format!(
                "MatcherFst: no index was saved for {match_type:?} labels"
            )));
        };
        let flags = if data.reach_input() {
            DEFAULT_LABEL_LOOKAHEAD_FLAGS | INPUT_LOOKAHEAD_MATCHER
        } else {
            DEFAULT_LABEL_LOOKAHEAD_FLAGS | OUTPUT_LOOKAHEAD_MATCHER
        };
        Ok(LabelLookAheadMatcher::from_data(
            data,
            inner,
            flags,
            accumulator,
        ))
    }
}

impl<'f, A> MatcherFst<'f, A>
where
    A: Arc + 'static,
    A::Weight: Copy,
{
    /// Attaches indices already built.
    pub fn new(
        fst: AnyFst<'f, A>,
        fst_type: FstType,
        input: Option<StdArc<LabelReachableData>>,
        output: Option<StdArc<LabelReachableData>>,
    ) -> Self {
        let pair = AddOnPair::new(
            input.map(|data| StdArc::new(ReachableAddOn::new(data))),
            output.map(|data| StdArc::new(ReachableAddOn::new(data))),
        );
        Self {
            impl_: AddOnImpl::new(fst, fst_type, Some(StdArc::new(pair))),
        }
    }
}

/// Builds the `arc_lookahead` FST: no index at all.
///
/// Its matcher,
/// [`ArcLookAheadMatcher`](crate::algorithms::lookahead_matcher::ArcLookAheadMatcher),
/// answers by walking the other state's arcs, so there is nothing to save.
/// The type exists so that a pipeline written against a look-ahead FST can be
/// pointed at an ordinary one.
pub fn arc_lookahead<A>(fst: AnyFst<'_, A>) -> MatcherFst<'_, A>
where
    A: Arc + 'static,
    A::Weight: Copy,
{
    MatcherFst::new(fst, FstType::ARC_LOOKAHEAD, None, None)
}

/// Builds the `ilabel_lookahead` FST: an index over input labels.
pub fn ilabel_lookahead<'f, A, F, Acc>(
    fst: AnyFst<'f, A>,
    source: &F,
    accumulator: Acc,
) -> Result<MatcherFst<'f, A>, OpenFstError>
where
    A: Arc + 'static,
    A::Weight: Copy,
    F: Fst<A> + ExpandedFst<A>,
    Acc: WeightAccumulator<A>,
{
    let reachable = crate::algorithms::label_reachable::LabelReachable::<A, Acc>::with_accumulator(
        source,
        true,
        accumulator,
    )?;
    let data = StdArc::clone(reachable.data());
    Ok(MatcherFst::new(
        fst,
        FstType::ILABEL_LOOKAHEAD,
        Some(data),
        None,
    ))
}

/// Builds the `olabel_lookahead` FST: an index over output labels.
pub fn olabel_lookahead<'f, A, F, Acc>(
    fst: AnyFst<'f, A>,
    source: &F,
    accumulator: Acc,
) -> Result<MatcherFst<'f, A>, OpenFstError>
where
    A: Arc + 'static,
    A::Weight: Copy,
    F: Fst<A> + ExpandedFst<A>,
    Acc: WeightAccumulator<A>,
{
    let reachable = crate::algorithms::label_reachable::LabelReachable::<A, Acc>::with_accumulator(
        source,
        false,
        accumulator,
    )?;
    let data = StdArc::clone(reachable.data());
    Ok(MatcherFst::new(
        fst,
        FstType::OLABEL_LOOKAHEAD,
        None,
        Some(data),
    ))
}

impl<'f, A> MatcherFst<'f, A>
where
    A: Arc + 'static,
    A::Label: FstScalar,
    A::StateId: FstScalar,
    A::Weight: Copy + WeightIo,
{
    /// Writes the FST and its indices.
    pub fn write<W: Write>(
        &self,
        writer: &mut W,
        opts: &FstWriteOptions,
    ) -> Result<(), OpenFstError> {
        self.impl_.write(writer, opts)
    }
}

impl<A> MatcherFst<'static, A>
where
    A: Arc + 'static,
    A::Label: FstScalar,
    A::StateId: FstScalar,
    A::Weight: Copy + WeightIo,
{
    /// Reads what [`write`](Self::write) wrote.
    pub fn read<R: Read + Seek>(
        reader: &mut R,
        opts: &FstReadOptions,
        fst_type: FstType,
    ) -> Result<Self, OpenFstError> {
        Ok(Self {
            impl_: AddOnImpl::read(reader, opts, fst_type)?,
        })
    }
}

// Everything about the FST itself is the contained FST's answer: attaching an
// index changes what a matcher can do, not what states and arcs are there.
impl<'f, A> Fst<A> for MatcherFst<'f, A>
where
    A: Arc + 'static,
    A::Weight: Copy,
{
    type StateIter<'a>
        = AnyStateIter<'f, 'a, A>
    where
        Self: 'a;
    type ArcIter<'a>
        = AnyArcIter<'f, 'a, A>
    where
        Self: 'a;

    fn start(&self) -> Option<A::StateId> {
        self.fst().start()
    }

    fn final_weight(&self, state: A::StateId) -> A::Weight {
        self.fst().final_weight(state)
    }

    fn states(&self) -> Self::StateIter<'_> {
        self.fst().states()
    }

    fn arcs(&self, state: A::StateId) -> Self::ArcIter<'_> {
        self.fst().arcs(state)
    }

    fn num_arcs(&self, state: A::StateId) -> usize {
        self.fst().num_arcs(state)
    }

    fn num_states_if_known(&self) -> Option<usize> {
        self.fst().num_states_if_known()
    }

    fn num_input_epsilons(&self, state: A::StateId) -> usize {
        self.fst().num_input_epsilons(state)
    }

    fn num_output_epsilons(&self, state: A::StateId) -> usize {
        self.fst().num_output_epsilons(state)
    }

    fn properties(&self, mask: u64, test: bool) -> u64 {
        self.fst().properties(mask, test)
    }

    fn fst_type(&self) -> &str {
        self.impl_.fst_type().as_str()
    }

    fn input_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
        self.fst().input_symbols()
    }

    fn output_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
        self.fst().output_symbols()
    }
}

impl<'f, A> ExpandedFst<A> for MatcherFst<'f, A>
where
    A: Arc + 'static,
    A::Weight: Copy,
{
    fn num_states(&self) -> usize {
        self.fst().num_states()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::accumulator::DefaultAccumulator;
    use crate::algorithms::arcsort::{ILabelCompare, arc_sort};
    use crate::algorithms::lookahead_matcher::LookAheadMatcher;
    use crate::arc::StdArc;
    use crate::fst::MutableFst;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::matcher::{Matcher, SortedMatcher};
    use crate::properties::K_FST_PROPERTIES;
    use crate::weight::Weight;
    use crate::weights::float_weight::TropicalWeight;
    use std::io::Cursor;

    /// 0 -1-> 1 -2-> 2, so from state 0 only label 1 can be read next.
    fn source() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight::one(), 2));
        fst.set_final(2, TropicalWeight::one());
        arc_sort(&mut fst, &ILabelCompare);
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    fn indexed() -> MatcherFst<'static, StdArc> {
        let fst = source();
        ilabel_lookahead(AnyFst::Vector(Box::new(source())), &fst, DefaultAccumulator).unwrap()
    }

    /// The FST it wraps is the FST it answers as.
    #[test]
    fn it_is_the_fst_it_wraps() {
        let matcher_fst = indexed();
        let plain = source();
        assert_eq!(matcher_fst.start(), plain.start());
        assert_eq!(matcher_fst.num_states(), plain.num_states());
        assert_eq!(matcher_fst.num_arcs(0), 1);
        assert_eq!(matcher_fst.fst_type(), "ilabel_lookahead");
    }

    /// The index that was built is the index a matcher gets back.
    #[test]
    fn the_saved_index_answers_the_lookahead() {
        let fst = source();
        let matcher_fst = indexed();
        let inner = SortedMatcher::new(&fst, MatchType::Input).unwrap();
        let mut matcher = matcher_fst
            .matcher(inner, MatchType::Input, DefaultAccumulator)
            .unwrap();

        matcher.set_state(0);
        assert!(matcher.look_ahead_label(1), "1 is what comes next");
        assert!(!matcher.look_ahead_label(2), "2 comes after that");
        matcher.set_state(1);
        assert!(matcher.look_ahead_label(2));
    }

    /// Only the side that was indexed can be asked.
    #[test]
    fn the_side_that_was_not_indexed_is_not_there() {
        let matcher_fst = indexed();
        assert!(matcher_fst.data(MatchType::Input).is_some());
        assert!(matcher_fst.data(MatchType::Output).is_none());
    }

    /// The index survives a round trip through a stream, answering the same
    /// questions afterwards.
    #[test]
    fn the_index_round_trips_through_a_stream() {
        let matcher_fst = indexed();
        let mut bytes = Vec::new();
        matcher_fst
            .write(&mut bytes, &FstWriteOptions::default())
            .unwrap();

        let mut cursor = Cursor::new(bytes.clone());
        let read = MatcherFst::<StdArc>::read(
            &mut cursor,
            &FstReadOptions::default(),
            FstType::ILABEL_LOOKAHEAD,
        )
        .unwrap();

        assert_eq!(read.num_states(), matcher_fst.num_states());
        assert_eq!(read.start(), matcher_fst.start());
        let before = matcher_fst.data(MatchType::Input).unwrap();
        let after = read.data(MatchType::Input).unwrap();
        assert_eq!(after.reach_input(), before.reach_input());
        assert_eq!(after.final_index(), before.final_index());
        assert_eq!(after.num_labels(), before.num_labels());
        assert_eq!(after.interval_sets(), before.interval_sets());

        let fst = source();
        let inner = SortedMatcher::new(&fst, MatchType::Input).unwrap();
        let mut matcher = read
            .matcher(inner, MatchType::Input, DefaultAccumulator)
            .unwrap();
        matcher.set_state(0);
        assert!(matcher.look_ahead_label(1));
        assert!(!matcher.look_ahead_label(2));

        // And writing what was read gives the same bytes back.
        let mut again = Vec::new();
        read.write(&mut again, &FstWriteOptions::default()).unwrap();
        assert_eq!(again, bytes);
    }

    /// An arc-look-ahead FST saves no index, and says so.
    #[test]
    fn the_arc_lookahead_fst_saves_nothing() {
        let matcher_fst = arc_lookahead(AnyFst::Vector(Box::new(source())));
        assert_eq!(matcher_fst.fst_type(), "arc_lookahead");
        assert!(matcher_fst.data(MatchType::Input).is_none());
        assert!(matcher_fst.data(MatchType::Output).is_none());

        let mut bytes = Vec::new();
        matcher_fst
            .write(&mut bytes, &FstWriteOptions::default())
            .unwrap();
        let mut cursor = Cursor::new(bytes);
        let read = MatcherFst::<StdArc>::read(
            &mut cursor,
            &FstReadOptions::default(),
            FstType::ARC_LOOKAHEAD,
        )
        .unwrap();
        assert_eq!(read.num_states(), matcher_fst.num_states());
        assert!(read.data(MatchType::Input).is_none());

        // Asking for a matcher that would need one says what is missing.
        let fst = source();
        let inner = SortedMatcher::new(&fst, MatchType::Input).unwrap();
        let Err(err) = read.matcher(inner, MatchType::Input, DefaultAccumulator) else {
            panic!("there is no index to answer from")
        };
        assert!(format!("{err}").contains("no index"), "{err}");
    }

    /// A file claiming to be something else is refused rather than misread.
    #[test]
    fn a_file_of_another_type_is_refused() {
        let matcher_fst = indexed();
        let mut bytes = Vec::new();
        matcher_fst
            .write(&mut bytes, &FstWriteOptions::default())
            .unwrap();
        let mut cursor = Cursor::new(bytes);
        assert!(
            MatcherFst::<StdArc>::read(
                &mut cursor,
                &FstReadOptions::default(),
                FstType::OLABEL_LOOKAHEAD,
            )
            .is_err()
        );
    }

    /// The output-side index is over output labels, which is a different
    /// question from the input-side one.
    #[test]
    fn the_output_index_indexes_output_labels() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        // Input 1 comes with output 7.
        fst.add_arc(0, StdArc::new(1, 7, TropicalWeight::one(), 1));
        fst.add_arc(1, StdArc::new(2, 8, TropicalWeight::one(), 2));
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let matcher_fst = olabel_lookahead(
            AnyFst::Vector(Box::new(fst.clone())),
            &fst,
            DefaultAccumulator,
        )
        .unwrap();
        assert_eq!(matcher_fst.fst_type(), "olabel_lookahead");
        assert!(matcher_fst.data(MatchType::Output).is_some());

        let inner = SortedMatcher::new(&fst, MatchType::Output).unwrap();
        let mut matcher = matcher_fst
            .matcher(inner, MatchType::Output, DefaultAccumulator)
            .unwrap();
        matcher.set_state(0);
        assert!(
            matcher.look_ahead_label(7),
            "7 is the output that comes next"
        );
        assert!(!matcher.look_ahead_label(1), "1 is an input label");
    }
}
