//! The decoding graph for a CTC model.
//!
//! A CTC model emits one symbol per frame from an alphabet that includes a
//! *blank*, and the transcript is recovered by collapsing runs of the same
//! symbol and then deleting the blanks, so `_ a a _ a b` reads as `a a b`.
//! Note where the blank matters: it is what separates the two `a`s. Without it
//! the run would collapse to one.
//!
//! As an FST that rule is a graph of `V + 1` states, one per symbol the model
//! could have emitted last, plus the state meaning "the last thing was a
//! blank". [`ctc_topo`] builds it. Composing it with the acoustic matrix and
//! taking the best path is CTC decoding; composing the two of them with a
//! lexicon and a language model instead is the rest of a recogniser, and this
//! is the piece it starts from.
//!
//! This is k2's `k2.ctc_topo(max_token, modified=False)`. The *modified*
//! topology, which lets a frame be skipped, is not here.

use sicada::arc::{Arc, ArcLabel, ArcStateId};
use sicada::error::OpenFstError;
use sicada::fst::{Fst, MutableFst};
use sicada::fsts::vector_fst::VectorFst;
use sicada::properties::K_FST_PROPERTIES;
use sicada::weight::Weight;

/// Builds the CTC topology for a model with `num_symbols` columns, blank first.
///
/// `label_offset` says where column 0 sits, and has to be the one the
/// [`DenseFst`](crate::dense::DenseFst) was built with. It is 1 by default,
/// because label 0 is epsilon to every FST algorithm and a blank is not one.
///
/// Input labels are the model's columns, offset. Output labels are the same
/// symbols, so subtracting `label_offset` from an answer's labels gives columns
/// back; blank and repeats emit nothing.
///
/// # Errors
///
/// A model with no symbols, or an alphabet that does not fit the arc's label
/// type.
pub fn ctc_topo<A: Arc>(num_symbols: usize, label_offset: i64) -> Result<VectorFst<A>, OpenFstError>
where
    A::Weight: Weight,
{
    if num_symbols < 2 {
        return Err(OpenFstError::InvalidOperation(format!(
            "ctc_topo: {num_symbols} symbols is not enough for a blank and something else"
        )));
    }
    let label_of = |column: usize| -> Result<A::Label, OpenFstError> {
        A::Label::from_i64(label_offset + column as i64).ok_or_else(|| {
            OpenFstError::InvalidOperation(format!(
                "ctc_topo: label {} does not fit the arc's label type",
                label_offset + column as i64
            ))
        })
    };
    if label_offset < 1 {
        return Err(OpenFstError::InvalidOperation(
            "ctc_topo: column 0 would be epsilon, which consumes no frame".into(),
        ));
    }

    // State 0 means "the last frame was a blank, or there has been none"; state
    // `t` means "the last frame emitted symbol `t`". That is all the history
    // the collapsing rule needs.
    let mut fst: VectorFst<A> = VectorFst::new();
    fst.reserve_states(num_symbols);
    for _ in 0..num_symbols {
        fst.add_state();
    }
    fst.set_start(A::StateId::from_usize(0));

    let blank = label_of(0)?;
    for last in 0..num_symbols {
        let from = A::StateId::from_usize(last);
        // Every state is final: the audio may end wherever it ends.
        fst.set_final(from, A::Weight::one());

        // A blank says nothing and resets what may repeat.
        fst.add_arc(
            from,
            A::new(
                blank,
                A::Label::epsilon(),
                A::Weight::one(),
                A::StateId::from_usize(0),
            ),
        );

        for symbol in 1..num_symbols {
            let label = label_of(symbol)?;
            // Emitting the same symbol again with no blank between is the same
            // symbol held longer, so it says nothing more.
            let says = if symbol == last {
                A::Label::epsilon()
            } else {
                label
            };
            fst.add_arc(
                from,
                A::new(
                    label,
                    says,
                    A::Weight::one(),
                    A::StateId::from_usize(symbol),
                ),
            );
        }
    }

    fst.properties(K_FST_PROPERTIES, true);
    Ok(fst)
}

/// The CTC collapsing rule applied to a sequence of columns: runs of the same
/// symbol become one, then the blanks go.
///
/// Column 0 is the blank. [`ctc_topo`] encodes this same rule; here it is
/// written out directly, which is useful for checking an answer and for a caller
/// who has a greedy argmax rather than a lattice.
pub fn collapse(columns: &[usize]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(columns.len());
    let mut previous = usize::MAX;
    for &column in columns {
        if column != previous && column != 0 {
            out.push(column);
        }
        previous = column;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use sicada::arc::StdArc;
    use sicada::fst::ExpandedFst;
    use sicada::fsts::vector_fst::StdVectorFst;
    use sicada::properties::{K_I_DETERMINISTIC, K_NO_I_EPSILONS};

    use crate::compact::{DeterminizeLatticeOptions, determinize_lattice};
    use crate::dense::DenseFst;
    use crate::frontier::DecodeOptions;
    use crate::lattice::{LatticeDecodeOptions, lattice_decode};
    use crate::nbest::n_best;
    use crate::viterbi::viterbi_decode;

    /// Blank plus three symbols.
    const SYMBOLS: usize = 4;

    fn topo() -> StdVectorFst {
        ctc_topo(SYMBOLS, 1).expect("a topology")
    }

    /// Scores that make one column certain in every frame.
    fn certain(columns: &[usize]) -> Vec<f32> {
        let mut scores = vec![10.0; columns.len() * SYMBOLS];
        for (frame, &column) in columns.iter().enumerate() {
            scores[frame * SYMBOLS + column] = 0.0;
        }
        scores
    }

    /// A decoded answer's labels back as the model's columns.
    fn columns_of(labels: &[i32]) -> Vec<usize> {
        labels.iter().map(|label| (label - 1) as usize).collect()
    }

    #[test]
    fn it_is_deterministic_on_the_frames_it_reads() {
        let fst = topo();
        assert_eq!(fst.num_states(), SYMBOLS);
        let props = fst.properties(K_I_DETERMINISTIC | K_NO_I_EPSILONS, true);
        assert_ne!(props & K_I_DETERMINISTIC, 0, "two arcs read the same frame");
        assert_ne!(props & K_NO_I_EPSILONS, 0, "an arc reads no frame");
        for state in fst.states() {
            assert_eq!(fst.num_arcs(state), SYMBOLS, "one arc per column");
        }
    }

    /// The rule the topology exists to encode, checked against the rule written
    /// out directly, over every alignment of up to five frames, so there is
    /// nothing left to have missed.
    #[test]
    fn it_collapses_exactly_as_the_rule_says() {
        let graph = topo();
        for length in 1..=5usize {
            let mut columns = vec![0usize; length];
            loop {
                let scores = certain(&columns);
                let dense = DenseFst::<StdArc>::new(&scores, length, SYMBOLS).unwrap();
                let decoded = viterbi_decode(&graph, &dense, &DecodeOptions::exhaustive())
                    .unwrap()
                    .expect("a path");
                assert_eq!(
                    columns_of(&decoded.labels),
                    collapse(&columns),
                    "for the alignment {columns:?}"
                );

                // Odometer over every alignment of this length.
                let mut place = 0;
                loop {
                    if place == length {
                        break;
                    }
                    columns[place] += 1;
                    if columns[place] < SYMBOLS {
                        break;
                    }
                    columns[place] = 0;
                    place += 1;
                }
                if place == length {
                    break;
                }
            }
        }
    }

    /// The case that distinguishes CTC from plain collapsing: a blank between
    /// two of the same symbol keeps them apart.
    #[test]
    fn a_blank_is_what_lets_a_symbol_repeat() {
        let graph = topo();

        let held = certain(&[1, 1, 1]);
        let dense = DenseFst::<StdArc>::new(&held, 3, SYMBOLS).unwrap();
        let decoded = viterbi_decode(&graph, &dense, &DecodeOptions::exhaustive())
            .unwrap()
            .unwrap();
        assert_eq!(columns_of(&decoded.labels), vec![1], "one long symbol");

        let separated = certain(&[1, 0, 1]);
        let dense = DenseFst::<StdArc>::new(&separated, 3, SYMBOLS).unwrap();
        let decoded = viterbi_decode(&graph, &dense, &DecodeOptions::exhaustive())
            .unwrap()
            .unwrap();
        assert_eq!(columns_of(&decoded.labels), vec![1, 1], "two of them");
    }

    /// End to end: acoustic scores in, a lattice out, the alignments collapsed,
    /// and the answers read back in order.
    #[test]
    fn the_whole_pipeline_agrees_with_the_rule() {
        let graph = topo();
        // Frames 0 and 2 are sure of symbol 1. Frame 1 is torn between holding
        // it, giving one long symbol, and a blank, which would make them two.
        // It costs 1.0 more to blank, so holding wins and "1 1" is the
        // runner-up.
        let scores = [
            9.0, 0.0, 9.0, 9.0, //
            1.0, 0.0, 9.0, 9.0, //
            9.0, 0.0, 9.0, 9.0,
        ];
        let dense = DenseFst::<StdArc>::new(&scores, 3, SYMBOLS).unwrap();
        let lattice = lattice_decode(&graph, &dense, &LatticeDecodeOptions::exhaustive())
            .unwrap()
            .expect("a lattice");
        let compact = determinize_lattice(&lattice, &DeterminizeLatticeOptions::default()).unwrap();
        let best = n_best(&compact, 2).unwrap();
        assert_eq!(best.len(), 2);

        assert_eq!(columns_of(&best[0].words), vec![1], "one long symbol");
        assert_eq!(columns_of(&best[1].words), vec![1, 1], "two of them");
        assert!((best[0].cost() - 0.0).abs() < 1e-6, "{}", best[0].cost());
        assert!((best[1].cost() - 1.0).abs() < 1e-6, "{}", best[1].cost());

        // And each answer carries the frames that produced it.
        assert_eq!(best[0].alignment().len(), 3, "one label per frame");
        assert_eq!(columns_of(best[0].alignment()), vec![1, 1, 1]);
        assert_eq!(columns_of(best[1].alignment()), vec![1, 0, 1]);
    }

    #[test]
    fn an_alphabet_with_nothing_in_it_is_refused() {
        assert!(ctc_topo::<StdArc>(1, 1).is_err());
        assert!(ctc_topo::<StdArc>(4, 0).is_err(), "column 0 on epsilon");
    }

    #[test]
    fn collapsing_is_runs_first_then_blanks() {
        assert_eq!(collapse(&[0, 1, 1, 0, 1, 2]), vec![1, 1, 2]);
        assert_eq!(collapse(&[]), Vec::<usize>::new());
        assert_eq!(collapse(&[0, 0, 0]), Vec::<usize>::new());
        assert_eq!(collapse(&[2, 2, 2]), vec![2]);
    }
}
