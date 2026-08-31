//! The acoustic model's output, seen as an FST.
//!
//! A neural acoustic model hands back a `T × V` matrix: for each of `T` frames,
//! a score for each of `V` symbols. Decoding is the composition of that matrix
//! with a decoding graph, so the matrix has to be an FST first.
//!
//! As an FST it is a chain of `T + 1` states, with `V` arcs from frame `t` to
//! frame `t + 1`, one per symbol, weighted by that symbol's score in that
//! frame. It is an acceptor: the symbol is both input and output label.
//!
//! Nothing is materialised. [`DenseFst`] borrows the matrix and computes each
//! arc as it is asked for, so composing against it costs no copy of the
//! acoustic scores. This is k2's `DenseFsaVec` without the batch dimension:
//! there is one utterance here, because the point of this crate is decoding one
//! stream on a CPU rather than many on a GPU.

use std::marker::PhantomData;

use sicada::AtomicRc;
use sicada::arc::{Arc, ArcLabel, ArcStateId};
use sicada::error::OpenFstError;
use sicada::fst::{ExpandedFst, Fst};
use sicada::properties::{
    K_ACCEPTOR, K_ACCESSIBLE, K_ACYCLIC, K_CO_ACCESSIBLE, K_EPSILONS, K_EXPANDED,
    K_I_DETERMINISTIC, K_I_EPSILONS, K_I_LABEL_SORTED, K_INITIAL_ACYCLIC, K_NO_EPSILONS,
    K_NO_I_EPSILONS, K_NO_O_EPSILONS, K_NOT_STRING, K_O_DETERMINISTIC, K_O_EPSILONS,
    K_O_LABEL_SORTED, K_STRING, K_TOP_SORTED, K_UNWEIGHTED, K_UNWEIGHTED_CYCLES, K_WEIGHTED,
};
use sicada::symbol_table::SymbolTable;
use sicada::weight::Weight;

/// A weight that can be built from one acoustic score.
///
/// The scores are *costs*, that is, negative log probabilities where smaller is
/// better, which is already what both the tropical and the log semiring mean by
/// a weight's value. Keeping the trait rather than hard-wiring `TropicalWeight`
/// lets the same decoder run Viterbi (tropical) and forward-sum (log) without a
/// second implementation.
pub trait FromScore: Weight {
    /// The weight for a score of `cost` nats.
    fn from_cost(cost: f32) -> Self;

    /// The score back out, for a decoder that wants to compare in `f32`.
    fn to_cost(&self) -> f32;
}

macro_rules! from_score {
    ($weight:ty, $float:ty) => {
        impl FromScore for $weight {
            #[inline(always)]
            fn from_cost(cost: f32) -> Self {
                Self(cost as $float)
            }
            #[inline(always)]
            fn to_cost(&self) -> f32 {
                self.0 as f32
            }
        }
    };
}

from_score!(sicada::weights::float_weight::TropicalWeight, f32);
from_score!(sicada::weights::float_weight::TropicalWeight64, f64);
from_score!(sicada::weights::float_weight::LogWeight, f32);
from_score!(sicada::weights::float_weight::Log64Weight, f64);

/// A `T × V` matrix of acoustic scores, read as an FST.
///
/// Row-major: the score of symbol `c` in frame `t` is `scores[t * V + c]`.
#[derive(Debug, Clone)]
pub struct DenseFst<'a, A: Arc> {
    scores: &'a [f32],
    num_frames: usize,
    num_symbols: usize,
    /// Column `c` becomes label `c + label_offset`.
    ///
    /// The default is 1, not 0. An acoustic model's column 0 is an ordinary
    /// symbol, usually CTC's blank, but label 0 is epsilon to every FST
    /// algorithm in sicada, so leaving the columns where they are would turn
    /// the whole first row into epsilon arcs that consume no frame. The
    /// decoding graph is expected to carry the same offset.
    label_offset: i64,
    props: u64,
    _arc: PhantomData<A>,
}

impl<'a, A: Arc> DenseFst<'a, A>
where
    A::Weight: FromScore,
{
    /// Reads `scores` as `num_frames × num_symbols`, with column `c` as label
    /// `c + 1`.
    pub fn new(
        scores: &'a [f32],
        num_frames: usize,
        num_symbols: usize,
    ) -> Result<Self, OpenFstError> {
        Self::with_label_offset(scores, num_frames, num_symbols, 1)
    }

    /// As [`new`](Self::new), with the columns placed at a chosen label.
    ///
    /// An offset of 0 puts column 0 on epsilon; that is allowed, because a
    /// graph may genuinely want it, but the properties then say so.
    pub fn with_label_offset(
        scores: &'a [f32],
        num_frames: usize,
        num_symbols: usize,
        label_offset: i64,
    ) -> Result<Self, OpenFstError> {
        let expected = num_frames.checked_mul(num_symbols).ok_or_else(|| {
            OpenFstError::InvalidOperation(format!(
                "DenseFst: {num_frames} frames of {num_symbols} symbols overflows"
            ))
        })?;
        if scores.len() != expected {
            return Err(OpenFstError::InvalidOperation(format!(
                "DenseFst: {} scores for {num_frames} frames of {num_symbols} symbols, expected {expected}",
                scores.len()
            )));
        }
        if num_symbols == 0 && num_frames > 0 {
            return Err(OpenFstError::InvalidOperation(
                "DenseFst: frames with no symbols leave the last frame unreachable".into(),
            ));
        }
        if label_offset < 0 {
            return Err(OpenFstError::InvalidOperation(
                "DenseFst: a negative label offset would put a column on kNoLabel".into(),
            ));
        }
        let last = label_offset + num_symbols as i64 - 1;
        if num_symbols > 0 && A::Label::from_i64(last).is_none() {
            return Err(OpenFstError::InvalidOperation(format!(
                "DenseFst: label {last} does not fit the arc's label type"
            )));
        }

        let props = Self::compute_properties(scores, num_frames, num_symbols, label_offset);
        Ok(Self {
            scores,
            num_frames,
            num_symbols,
            label_offset,
            props,
            _arc: PhantomData,
        })
    }

    /// The number of frames, which is one less than the number of states.
    #[inline(always)]
    pub fn num_frames(&self) -> usize {
        self.num_frames
    }

    /// The number of symbols the acoustic model scores.
    #[inline(always)]
    pub fn num_symbols(&self) -> usize {
        self.num_symbols
    }

    /// The label column 0 was placed on.
    #[inline(always)]
    pub fn label_offset(&self) -> i64 {
        self.label_offset
    }

    /// The scores of one frame, indexed by column.
    ///
    /// A decoder wants exactly this: it walks the *graph*'s arcs and looks up
    /// each one's label here, rather than iterating this FST's `V` arcs and
    /// discarding the ones the graph has no use for.
    #[inline(always)]
    pub fn frame(&self, t: usize) -> &'a [f32] {
        &self.scores[t * self.num_symbols..(t + 1) * self.num_symbols]
    }

    /// The column a label falls in, or `None` if it names no column.
    #[inline(always)]
    pub fn column_of(&self, label: A::Label) -> Option<usize> {
        let index = label.to_i64()? - self.label_offset;
        (index >= 0 && (index as u64) < self.num_symbols as u64).then_some(index as usize)
    }

    fn compute_properties(
        scores: &[f32],
        num_frames: usize,
        num_symbols: usize,
        label_offset: i64,
    ) -> u64 {
        // Every claim below is structural: one arc per symbol per frame, in
        // column order, from frame t to frame t + 1 only.
        let mut props = K_EXPANDED
            | K_ACCEPTOR
            | K_I_DETERMINISTIC
            | K_O_DETERMINISTIC
            | K_I_LABEL_SORTED
            | K_O_LABEL_SORTED
            | K_ACYCLIC
            | K_INITIAL_ACYCLIC
            | K_TOP_SORTED
            | K_ACCESSIBLE
            | K_CO_ACCESSIBLE
            // Vacuously: there are no cycles to weight.
            | K_UNWEIGHTED_CYCLES;

        // Column 0 sits on epsilon only if it was asked to.
        if label_offset == 0 && num_symbols > 0 && num_frames > 0 {
            props |= K_EPSILONS | K_I_EPSILONS | K_O_EPSILONS;
        } else {
            props |= K_NO_EPSILONS | K_NO_I_EPSILONS | K_NO_O_EPSILONS;
        }

        // A string is an FST with exactly one path, which this is when each
        // frame offers a single symbol, or when there are no frames at all.
        if num_symbols <= 1 || num_frames == 0 {
            props |= K_STRING;
        } else {
            props |= K_NOT_STRING;
        }

        // The one claim that is about the numbers rather than the shape. It
        // costs one pass over the matrix, once, at construction.
        if scores.iter().all(|&score| score == 0.0) {
            props |= K_UNWEIGHTED;
        } else {
            props |= K_WEIGHTED;
        }
        props
    }
}

/// The arcs of one frame: one per column, in column order.
#[derive(Debug, Clone)]
pub struct DenseArcIter<'a, A: Arc> {
    row: &'a [f32],
    column: usize,
    label_offset: i64,
    nextstate: A::StateId,
    _arc: PhantomData<A>,
}

impl<A: Arc> Iterator for DenseArcIter<'_, A>
where
    A::Weight: FromScore,
{
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        let &score = self.row.get(self.column)?;
        // Checked once in the constructor, for the largest column.
        let label = A::Label::from_i64(self.label_offset + self.column as i64)?;
        self.column += 1;
        Some(A::new(
            label,
            label,
            A::Weight::from_cost(score),
            self.nextstate,
        ))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.row.len() - self.column;
        (left, Some(left))
    }
}

impl<A: Arc> ExactSizeIterator for DenseArcIter<'_, A> where A::Weight: FromScore {}

impl<A: Arc> Fst<A> for DenseFst<'_, A>
where
    A::Weight: FromScore,
{
    type StateIter<'s>
        = DenseStateIter<A>
    where
        Self: 's;
    type ArcIter<'s>
        = DenseArcIter<'s, A>
    where
        Self: 's;

    #[inline]
    fn start(&self) -> Option<A::StateId> {
        Some(A::StateId::from_usize(0))
    }

    #[inline]
    fn final_weight(&self, state: A::StateId) -> A::Weight {
        if state.as_usize() == self.num_frames {
            A::Weight::one()
        } else {
            A::Weight::zero()
        }
    }

    #[inline]
    fn num_arcs(&self, state: A::StateId) -> usize {
        if state.as_usize() < self.num_frames {
            self.num_symbols
        } else {
            0
        }
    }

    #[inline]
    fn num_input_epsilons(&self, state: A::StateId) -> usize {
        usize::from(self.label_offset == 0 && self.num_arcs(state) > 0)
    }

    #[inline]
    fn num_output_epsilons(&self, state: A::StateId) -> usize {
        self.num_input_epsilons(state)
    }

    #[inline]
    fn num_states_if_known(&self) -> Option<usize> {
        Some(self.num_frames + 1)
    }

    #[inline]
    fn properties(&self, mask: u64, _test: bool) -> u64 {
        // Everything was settled at construction, so there is nothing `test`
        // could compute that is not already here.
        self.props & mask
    }

    #[inline]
    fn fst_type(&self) -> &str {
        "dense"
    }

    fn input_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
        None
    }

    fn output_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
        None
    }

    #[inline]
    fn states<'s>(&'s self) -> Self::StateIter<'s> {
        DenseStateIter {
            next: 0,
            end: self.num_frames + 1,
            _arc: PhantomData,
        }
    }

    #[inline]
    fn arcs<'s>(&'s self, state: A::StateId) -> Self::ArcIter<'s> {
        let t = state.as_usize();
        let row: &[f32] = if t < self.num_frames {
            self.frame(t)
        } else {
            &[]
        };
        DenseArcIter {
            row,
            column: 0,
            label_offset: self.label_offset,
            nextstate: A::StateId::from_usize(t + 1),
            _arc: PhantomData,
        }
    }
}

impl<A: Arc> ExpandedFst<A> for DenseFst<'_, A>
where
    A::Weight: FromScore,
{
    #[inline]
    fn num_states(&self) -> usize {
        self.num_frames + 1
    }
}

/// The states of a [`DenseFst`], which are the frame boundaries.
#[derive(Debug, Clone)]
pub struct DenseStateIter<A: Arc> {
    next: usize,
    end: usize,
    _arc: PhantomData<A>,
}

impl<A: Arc> Iterator for DenseStateIter<A> {
    type Item = A::StateId;

    #[inline]
    fn next(&mut self) -> Option<A::StateId> {
        (self.next < self.end).then(|| {
            let state = A::StateId::from_usize(self.next);
            self.next += 1;
            state
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.end - self.next;
        (left, Some(left))
    }
}

impl<A: Arc> ExactSizeIterator for DenseStateIter<A> {}

#[cfg(test)]
mod tests {
    use super::*;
    use sicada::arc::StdArc;
    use sicada::fst::MutableFst;
    use sicada::fsts::vector_fst::{StdVectorFst, VectorFst};
    use sicada::properties::K_FST_PROPERTIES;
    use sicada::weights::float_weight::TropicalWeight;

    /// Two frames over three symbols.
    const SCORES: [f32; 6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

    fn dense() -> DenseFst<'static, StdArc> {
        DenseFst::new(&SCORES, 2, 3).expect("a dense FST")
    }

    /// The same thing built by hand, to compare a computed FST against a
    /// stored one.
    fn materialised() -> StdVectorFst {
        let mut fst = VectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.set_final(2, TropicalWeight::one());
        for t in 0..2 {
            for c in 0..3 {
                let label = c as i32 + 1;
                fst.add_arc(
                    t,
                    StdArc::new(
                        label,
                        label,
                        TropicalWeight(SCORES[t as usize * 3 + c]),
                        t + 1,
                    ),
                );
            }
        }
        fst
    }

    #[test]
    fn it_reads_the_matrix_as_a_chain_of_frames() {
        let dense = dense();
        let expected = materialised();

        assert_eq!(dense.start(), expected.start());
        assert_eq!(dense.num_states(), 3);
        assert_eq!(
            dense.states().collect::<Vec<_>>(),
            expected.states().collect::<Vec<_>>()
        );
        for state in expected.states() {
            assert_eq!(dense.num_arcs(state), expected.num_arcs(state));
            assert_eq!(
                dense.arcs(state).collect::<Vec<_>>(),
                expected.arcs(state).collect::<Vec<_>>(),
                "state {state}"
            );
            assert_eq!(dense.final_weight(state), expected.final_weight(state));
        }
    }

    /// The properties are claims other algorithms act on, so they are compared
    /// against what sicada computes for the same FST stored.
    #[test]
    fn its_properties_are_the_ones_the_same_fst_stored_has() {
        let dense = dense();
        let expected = materialised();
        let computed = expected.properties(K_FST_PROPERTIES, true);

        // The stored FST is also mutable, which this is not.
        let shared = K_FST_PROPERTIES & !sicada::properties::K_MUTABLE;
        assert_eq!(
            dense.properties(shared, true),
            computed & shared,
            "dense {:#x} vs vector {:#x}",
            dense.properties(shared, true),
            computed & shared
        );
    }

    #[test]
    fn one_symbol_per_frame_is_a_string() {
        let scores = [0.5, 0.25];
        let dense = DenseFst::<StdArc>::new(&scores, 2, 1).expect("a dense FST");
        assert_ne!(dense.properties(K_STRING, true), 0);
        assert_eq!(dense.properties(K_NOT_STRING, true), 0);
    }

    #[test]
    fn no_frames_is_the_empty_string() {
        let dense = DenseFst::<StdArc>::new(&[], 0, 5).expect("a dense FST");
        assert_eq!(dense.num_states(), 1);
        assert_eq!(dense.final_weight(0), TropicalWeight::one());
        assert_eq!(dense.arcs(0).count(), 0);
    }

    #[test]
    fn the_columns_move_off_epsilon_by_default() {
        let dense = dense();
        assert_ne!(dense.properties(K_NO_EPSILONS, true), 0);
        assert!(dense.arcs(0).all(|arc| arc.ilabel() != 0));

        let on_epsilon = DenseFst::<StdArc>::with_label_offset(&SCORES, 2, 3, 0).unwrap();
        assert_ne!(on_epsilon.properties(K_EPSILONS, true), 0);
        assert_eq!(on_epsilon.num_input_epsilons(0), 1);
    }

    #[test]
    fn a_matrix_of_the_wrong_size_is_refused() {
        let err = DenseFst::<StdArc>::new(&SCORES, 2, 4).unwrap_err();
        assert!(format!("{err}").contains("expected 8"), "{err}");
    }

    #[test]
    fn a_frame_lookup_is_the_same_score_the_arc_carries() {
        let dense = dense();
        for t in 0..dense.num_frames() {
            for arc in dense.arcs(t as i32) {
                let column = dense.column_of(arc.ilabel()).expect("a column");
                assert_eq!(dense.frame(t)[column], arc.weight().to_cost());
            }
        }
        assert_eq!(dense.column_of(0), None, "epsilon names no column");
        assert_eq!(dense.column_of(4), None, "past the last column");
    }
}
