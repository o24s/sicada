//! Reading answers out of a compact lattice.
//!
//! *n*-best is only meaningful once the alignments are collapsed. Over a raw
//! lattice the *n* best *paths* are usually *n* ways of cutting up the same
//! sentence; over a compact one, which is deterministic on words, every path is
//! a different word sequence, so the *n* best paths are *n* different answers.
//! Nothing here does anything clever for that reason: it is
//! [`shortest_path`] over an
//! FST that has already been put in the right shape.
//!
//! What *is* here is the rescoring knob the whole compact-lattice semiring
//! exists for: [`scale`] rescales the acoustic half against the graph half
//! without touching the alignments, and the answers move accordingly.

use sicada::algorithms::shortest_path::{ShortestPathOptions, shortest_path};
use sicada::arc::{Arc, ArcLabel, ArcStateId, ArcTpl};
use sicada::error::OpenFstError;
use sicada::fst::{ExpandedFst, Fst, MutableFst};
use sicada::fsts::vector_fst::VectorFst;
use sicada::weight::Weight;

use crate::compact_lattice_weight::CompactLatticeWeight;
use crate::lattice_weight::LatticeWeight;

/// One answer read off a compact lattice.
#[derive(Debug, Clone, PartialEq)]
pub struct Hypothesis<L: ArcLabel> {
    /// The words, epsilons removed.
    pub words: Vec<L>,
    /// The cost, still in its two halves, with the frames each word spanned.
    pub weight: CompactLatticeWeight<L>,
}

impl<L: ArcLabel> Hypothesis<L> {
    /// The frames this hypothesis used, in order.
    #[inline]
    pub fn alignment(&self) -> &[L] {
        self.weight.alignment()
    }

    /// The one number this hypothesis comes down to.
    #[inline]
    pub fn cost(&self) -> f32 {
        self.weight.weight().total()
    }
}

/// The `n` best word sequences, cheapest first.
///
/// Fewer than `n` come back when the lattice has fewer paths. Two answers are
/// distinct word sequences as long as `lattice` is deterministic on words,
/// as [`determinize_lattice`](crate::compact::determinize_lattice) makes it;
/// over a lattice that is not, this returns the `n` best *paths* and
/// they may repeat themselves.
pub fn n_best<L, S>(
    lattice: &VectorFst<ArcTpl<CompactLatticeWeight<L>, L, S>>,
    n: usize,
) -> Result<Vec<Hypothesis<L>>, OpenFstError>
where
    L: ArcLabel,
    S: ArcStateId,
{
    if n == 0 || lattice.start().is_none() {
        return Ok(Vec::new());
    }

    let mut best: VectorFst<ArcTpl<CompactLatticeWeight<L>, L, S>> = VectorFst::new();
    shortest_path(
        lattice,
        &mut best,
        &ShortestPathOptions {
            nshortest: n,
            ..ShortestPathOptions::default()
        },
    )?;

    let mut found = enumerate(&best);
    // `shortest_path` returns the paths as one FST, in no particular order.
    found.sort_by(|a, b| a.cost().total_cmp(&b.cost()));
    found.truncate(n);
    Ok(found)
}

/// Every path of an acyclic FST, as a hypothesis.
///
/// The result of `shortest_path` is a tree of at most `n` paths, so walking it
/// exhaustively is bounded by what was asked for.
fn enumerate<L, S>(fst: &VectorFst<ArcTpl<CompactLatticeWeight<L>, L, S>>) -> Vec<Hypothesis<L>>
where
    L: ArcLabel,
    S: ArcStateId,
{
    let mut found = Vec::new();
    let Some(start) = fst.start() else {
        return found;
    };
    let zero = CompactLatticeWeight::<L>::zero();
    let mut stack = vec![(start, Vec::new(), CompactLatticeWeight::<L>::one())];
    while let Some((state, words, weight)) = stack.pop() {
        let final_weight = fst.final_weight(state);
        if final_weight.is_member() && final_weight != zero {
            found.push(Hypothesis {
                words: words.clone(),
                weight: weight.times(&final_weight),
            });
        }
        for arc in fst.arcs(state) {
            let mut next = words.clone();
            if arc.olabel() != L::epsilon() {
                next.push(arc.olabel());
            }
            stack.push((arc.nextstate(), next, weight.times(arc.weight())));
        }
    }
    found
}

/// Rescales the two halves of every weight.
///
/// `acoustic` multiplies the acoustic cost and `graph` the graph cost. This is
/// what a compact lattice is *for*: the first pass decoded under one balance
/// between the acoustic model and the language model, and a second pass can ask
/// what the answer would have been under another, without decoding again.
///
/// Note that the halves are scaled where they sit, so the result is still a
/// compact lattice and still says which frames each word spanned. Only what
/// counts as *best* moves.
pub fn scale<L, S>(
    lattice: &mut VectorFst<ArcTpl<CompactLatticeWeight<L>, L, S>>,
    acoustic: f32,
    graph: f32,
) where
    L: ArcLabel,
    S: ArcStateId,
{
    let rescale = |weight: &CompactLatticeWeight<L>| {
        CompactLatticeWeight::new(
            LatticeWeight::new(
                graph * weight.weight().graph,
                acoustic * weight.weight().acoustic,
            ),
            weight.alignment().iter().copied().collect(),
        )
    };

    for state in 0..lattice.num_states() {
        let state = S::from_usize(state);
        let final_weight = lattice.final_weight(state);
        if final_weight.is_member() && final_weight != CompactLatticeWeight::zero() {
            lattice.set_final(state, rescale(&final_weight));
        }
        for arc in lattice.arcs_mut(state) {
            arc.weight = rescale(&arc.weight);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sicada::arc::StdArc;
    use sicada::fsts::vector_fst::StdVectorFst;
    use sicada::properties::K_FST_PROPERTIES;
    use sicada::weights::float_weight::TropicalWeight;

    use crate::compact::{DeterminizeLatticeOptions, determinize_lattice};
    use crate::dense::DenseFst;
    use crate::lattice::{LatticeDecodeOptions, lattice_decode};

    type Compact = VectorFst<ArcTpl<CompactLatticeWeight<i32>, i32, i32>>;

    /// Three symbols, each its own word, any sequence allowed.
    fn graph() -> StdVectorFst {
        let mut fst: StdVectorFst = VectorFst::new();
        fst.add_state();
        fst.set_start(0);
        fst.set_final(0, TropicalWeight::one());
        for label in 1..=3 {
            fst.add_arc(0, StdArc::new(label, label * 10, TropicalWeight::one(), 0));
        }
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    fn compact_of(scores: &[f32], frames: usize) -> Compact {
        let dense = DenseFst::<StdArc>::new(scores, frames, 3).unwrap();
        let lattice = lattice_decode(&graph(), &dense, &LatticeDecodeOptions::exhaustive())
            .unwrap()
            .expect("a lattice");
        determinize_lattice(&lattice, &DeterminizeLatticeOptions::default()).unwrap()
    }

    /// Two frames, so nine word sequences, whose costs are the two frames'
    /// scores added.
    const SCORES: [f32; 6] = [
        0.0, 1.0, 2.0, //
        0.0, 0.5, 3.0,
    ];

    #[test]
    fn it_returns_distinct_word_sequences_cheapest_first() {
        let compact = compact_of(&SCORES, 2);
        let best = n_best(&compact, 4).expect("four answers");

        assert_eq!(best.len(), 4);
        let words: Vec<&[i32]> = best.iter().map(|h| h.words.as_slice()).collect();
        assert_eq!(
            words,
            vec![
                &[10, 10][..], // 0.0 + 0.0
                &[10, 20][..], // 0.0 + 0.5
                &[20, 10][..], // 1.0 + 0.0
                &[20, 20][..], // 1.0 + 0.5
            ]
        );
        for pair in best.windows(2) {
            assert!(pair[0].cost() <= pair[1].cost(), "not sorted");
        }
        assert!((best[0].cost() - 0.0).abs() < 1e-6);
        assert!((best[3].cost() - 1.5).abs() < 1e-6);
    }

    /// The distinctness is the whole reason to determinize first, so it is
    /// asserted rather than assumed.
    #[test]
    fn no_two_answers_say_the_same_thing() {
        let compact = compact_of(&SCORES, 2);
        let best = n_best(&compact, 9).expect("nine answers");
        assert_eq!(best.len(), 9, "three symbols over two frames");

        let mut seen: Vec<&[i32]> = best.iter().map(|h| h.words.as_slice()).collect();
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert_eq!(seen.len(), before, "an answer was repeated");
    }

    #[test]
    fn asking_for_more_than_there_are_returns_what_there_is() {
        let compact = compact_of(&SCORES, 2);
        assert_eq!(n_best(&compact, 100).unwrap().len(), 9);
        assert!(n_best(&compact, 0).unwrap().is_empty());
    }

    #[test]
    fn every_answer_carries_the_frames_it_used() {
        let compact = compact_of(&SCORES, 2);
        for hypothesis in n_best(&compact, 9).unwrap() {
            assert_eq!(
                hypothesis.alignment().len(),
                2,
                "two frames were decoded: {hypothesis:?}"
            );
            // Word 10n came from label n, so the alignment says the words back.
            let from_alignment: Vec<i32> = hypothesis
                .alignment()
                .iter()
                .map(|label| label * 10)
                .collect();
            assert_eq!(from_alignment, hypothesis.words);
        }
    }

    /// What keeping the halves apart is for: the same lattice, a different
    /// balance between the models, and a different answer, with no decoding.
    #[test]
    fn rescaling_the_acoustic_half_changes_which_answer_wins() {
        // Word 10 is cheap acoustically and dear in the graph; word 20 is the
        // other way round. Under equal weight, 10 wins by a hair.
        let mut fst: StdVectorFst = VectorFst::new();
        fst.add_state();
        fst.set_start(0);
        fst.set_final(0, TropicalWeight::one());
        fst.add_arc(0, StdArc::new(1, 10, TropicalWeight(1.0), 0));
        fst.add_arc(0, StdArc::new(2, 20, TropicalWeight(0.0), 0));
        fst.properties(K_FST_PROPERTIES, true);

        // One frame: symbol 1 costs 0.0, symbol 2 costs 1.2.
        let scores = [0.0, 1.2, 9.0];
        let dense = DenseFst::<StdArc>::new(&scores, 1, 3).unwrap();
        let lattice = lattice_decode(&fst, &dense, &LatticeDecodeOptions::exhaustive())
            .unwrap()
            .unwrap();
        let compact = determinize_lattice(&lattice, &DeterminizeLatticeOptions::default()).unwrap();

        // graph 1.0 + acoustic 0.0 = 1.0 beats graph 0.0 + acoustic 1.2.
        assert_eq!(n_best(&compact, 1).unwrap()[0].words, vec![10]);

        // Halve what the acoustic model has to say and the graph decides.
        let mut quieter = compact.clone();
        scale(&mut quieter, 0.5, 1.0);
        assert_eq!(n_best(&quieter, 1).unwrap()[0].words, vec![20]);

        // The alignments are untouched by the rescaling.
        assert_eq!(n_best(&quieter, 1).unwrap()[0].alignment(), &[2]);
    }

    #[test]
    fn scaling_leaves_the_alignments_alone() {
        let mut compact = compact_of(&SCORES, 2);
        let before: Vec<Vec<i32>> = n_best(&compact, 9)
            .unwrap()
            .iter()
            .map(|h| h.alignment().to_vec())
            .collect();
        scale(&mut compact, 3.0, 2.0);
        let after: Vec<Vec<i32>> = n_best(&compact, 9)
            .unwrap()
            .iter()
            .map(|h| h.alignment().to_vec())
            .collect();
        assert_eq!(before.len(), after.len());
        // Scaling is monotone here, so the order is unchanged as well.
        assert_eq!(before, after);
    }
}
