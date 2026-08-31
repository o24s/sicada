//! Collapsing a lattice's alignments, so each word sequence appears once.
//!
//! The lattice a decoder produces has one arc per frame. A word that took eight
//! frames to say is eight arcs, and every way of placing its boundaries is a
//! separate path, so the same sentence comes back dozens of times, differing
//! only in where the words were cut. *n*-best over that is not *n* sentences,
//! and rescoring it does the same work over and over.
//!
//! A *compact* lattice has one arc per word, with the frames it spanned moved
//! into the weight ([`CompactLatticeWeight`]). Once they are in the weight,
//! ordinary determinization over words does the collapsing: two arcs for the
//! same word merge, and ⊕ keeps the better alignment instead of both.
//!
//! Upstream says the same thing, and says why gallic will not do
//! (`fstext/determinize-lattice.h`):
//!
//! > We determinize this using acceptor determinization with epsilon removal.
//! > […] `CompactLatticeWeightTpl` has a special kind of semiring where we
//! > always take the string corresponding to the best cost […] and discard the
//! > other. […] We couldn't use the Gallic weight for this, or it would die as
//! > soon as it detected that the input FST was non-functional.
//!
//! A lattice is exactly that non-functional transducer: one word sequence, many
//! alignments.
//!
//! What the algorithm needs beyond the semiring is the right *common divisor*.
//! Determinization normalises each subset by dividing out what its members
//! share, and for this weight that is the better cost together with the longest
//! common **prefix** of the alignments, rather than ⊕, whose alignment belongs
//! to one member and divides none of the others. That is the one piece
//! [`CompactLatticeCommonDivisor`] supplies; the rest is sicada's own
//! [`determinize_fsa`], which already takes a divisor because OpenFst's does.
//!
//! Kaldi writes its own determinizer instead, to prune as it goes and to keep
//! the alignments in a shared trie rather than copying them. Neither is here.
//! What is here is the outer half of the same idea: [`determinize_lattice_pruned`]
//! narrows the lattice and tries again when determinization runs away, which is
//! what Kaldi's wrapper does around its own. That much matters, because a bare
//! CTC topology constrains nothing, so a thousand-frame lattice has
//! astronomically many symbol sequences inside any generous beam, and
//! determinizing it whole does not finish.

use sicada::algorithms::connect::connect;
use sicada::algorithms::determinize::{CommonDivisor, determinize_fsa};
use sicada::algorithms::prune::{PruneOptions, prune as prune_fst};
use sicada::algorithms::rmepsilon::rm_epsilon;
use sicada::arc::{Arc, ArcLabel, ArcStateId, ArcTpl};
use sicada::error::OpenFstError;
use sicada::fst::{ExpandedFst, Fst, MutableFst};
use sicada::fsts::vector_fst::VectorFst;
use sicada::weight::Weight;

use crate::compact_lattice_weight::{Alignment, CompactLatticeArc, CompactLatticeWeight};
use crate::lattice_weight::LatticeWeight;

/// A lattice with one arc per word, for a decoding graph over `A`.
pub type CompactLattice<A> = VectorFst<CompactLatticeArc<A>>;

/// What [`determinize_lattice`] may be asked to do differently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeterminizeLatticeOptions {
    /// How closely two subsets' weights must agree to count as the same subset.
    pub delta: f32,
    /// A cap on the states built, or `None` for no cap.
    ///
    /// Determinizing a lattice can produce far more states than it consumed,
    /// and a decoder that is otherwise bounded should not become unbounded
    /// here. Reaching the cap is reported rather than truncated: half a lattice
    /// looks exactly like a whole one to everything downstream.
    pub max_states: Option<usize>,
}

impl Default for DeterminizeLatticeOptions {
    fn default() -> Self {
        Self {
            delta: 1.0 / 32.0,
            max_states: Some(1 << 20),
        }
    }
}

/// What the members of a determinized subset share.
///
/// The better cost, and the longest common prefix of the alignments. ⊕ will not
/// do: its alignment is one member's whole sequence, which does not divide any
/// of the others, and determinization would find every subset unnormalisable.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompactLatticeCommonDivisor;

impl<L: ArcLabel> CommonDivisor<CompactLatticeWeight<L>> for CompactLatticeCommonDivisor {
    fn divisor(
        &self,
        w1: &CompactLatticeWeight<L>,
        w2: &CompactLatticeWeight<L>,
    ) -> CompactLatticeWeight<L> {
        // Zero contributes nothing, so the other one stands whole. This is the
        // same convention `LabelCommonDivisor` follows, and it lets the divisor
        // be folded over a subset starting from zero.
        let zero = CompactLatticeWeight::zero();
        match (w1 == &zero, w2 == &zero) {
            (true, true) => return zero,
            (true, false) => return w2.clone(),
            (false, true) => return w1.clone(),
            (false, false) => {}
        }
        let shared = w1
            .alignment()
            .iter()
            .zip(w2.alignment())
            .take_while(|(a, b)| a == b)
            .map(|(a, _)| *a)
            .collect();
        CompactLatticeWeight::new(w1.weight().plus(w2.weight()), shared)
    }
}

/// Moves each arc's input label into its weight, leaving the words on the arcs.
///
/// The result has the same shape as `lattice`; it is [`determinize_lattice`]
/// that collapses it. An input epsilon contributes nothing to the alignment,
/// having consumed no frame.
///
/// Written over the label and state-id types rather than over the graph's arc:
/// a lattice mentions that arc only behind its associated types, which do not
/// determine it, so a caller would have had to name it.
pub fn to_compact<L, S>(
    lattice: &VectorFst<ArcTpl<LatticeWeight, L, S>>,
) -> VectorFst<ArcTpl<CompactLatticeWeight<L>, L, S>>
where
    L: ArcLabel,
    S: ArcStateId,
{
    let mut compact: VectorFst<ArcTpl<CompactLatticeWeight<L>, L, S>> = VectorFst::new();
    compact.reserve_states(lattice.num_states());
    for _ in 0..lattice.num_states() {
        compact.add_state();
    }
    if let Some(start) = lattice.start() {
        compact.set_start(start);
    }
    compact.set_input_symbols(lattice.output_symbols());
    compact.set_output_symbols(lattice.output_symbols());

    for state in lattice.states() {
        let final_weight = lattice.final_weight(state);
        if final_weight.is_member() && final_weight != LatticeWeight::zero() {
            compact.set_final(state, CompactLatticeWeight::from_weight(final_weight));
        }
        for arc in lattice.arcs(state) {
            let mut alignment = Alignment::new();
            if arc.ilabel() != L::epsilon() {
                alignment.push(arc.ilabel());
            }
            compact.add_arc(
                state,
                ArcTpl::new(
                    // An acceptor over words: what determinization merges on.
                    arc.olabel(),
                    arc.olabel(),
                    CompactLatticeWeight::new(*arc.weight(), alignment),
                    arc.nextstate(),
                ),
            );
        }
    }
    compact
}

/// Rewrites `lattice` so that each word sequence appears once, with its best
/// alignment.
///
/// # Errors
///
/// Reaching `opts.max_states` is an error, not a truncation.
pub fn determinize_lattice<L, S>(
    lattice: &VectorFst<ArcTpl<LatticeWeight, L, S>>,
    opts: &DeterminizeLatticeOptions,
) -> Result<VectorFst<ArcTpl<CompactLatticeWeight<L>, L, S>>, OpenFstError>
where
    L: ArcLabel,
    S: ArcStateId,
{
    let mut compact = to_compact(lattice);

    // A word-epsilon arc is one the determinization cannot merge on, so it has
    // to go first. Removing it is also what carries its frames onto whatever
    // word comes next, which is where they belong.
    rm_epsilon(&mut compact, true)?;

    let mut determinized: VectorFst<ArcTpl<CompactLatticeWeight<L>, L, S>> = VectorFst::new();
    determinize_fsa(
        &compact,
        &mut determinized,
        &CompactLatticeCommonDivisor,
        opts.delta,
        opts.max_states,
    )?;
    connect(&mut determinized);
    Ok(determinized)
}

/// What [`determinize_lattice_pruned`] may be asked to do differently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrunedDeterminizeOptions {
    /// The beam to try first: paths worse than the best by more than this are
    /// dropped before determinizing.
    pub beam: f32,
    /// What to multiply the beam by when an attempt runs away. Kaldi's wrapper
    /// halves it.
    pub beam_ratio: f32,
    /// How many times to narrow and try again.
    pub max_retries: usize,
    /// Passed through to [`determinize_lattice`].
    pub determinize: DeterminizeLatticeOptions,
}

impl Default for PrunedDeterminizeOptions {
    fn default() -> Self {
        Self {
            beam: 8.0,
            beam_ratio: 0.5,
            max_retries: 6,
            determinize: DeterminizeLatticeOptions::default(),
        }
    }
}

/// A collapsed lattice, and the beam it took to get one.
#[derive(Debug, Clone)]
pub struct PrunedLattice<L: ArcLabel, S: ArcStateId> {
    /// The lattice.
    pub lattice: VectorFst<ArcTpl<CompactLatticeWeight<L>, L, S>>,
    /// The beam actually used, which is `opts.beam` unless it had to narrow.
    ///
    /// Worth looking at: a lattice narrowed to a quarter of what was asked for
    /// still holds the best path, but it holds fewer alternatives than the
    /// caller planned to rescore.
    pub beam: f32,
    /// How many attempts ran away before one finished.
    pub narrowed: usize,
}

/// As [`determinize_lattice`], narrowing the lattice and trying again when an
/// attempt runs past `max_states`.
///
/// Determinization can produce far more states than it consumes, and how many is
/// not knowable in advance, since it depends on how many distinct symbol
/// sequences the beam admits and the lattice does not say. The answer is
/// therefore to try, and to ask for less if it runs away. Upstream's wrapper
/// does the same around its own determinizer.
///
/// # Errors
///
/// The last attempt's error, once the retries are used up.
pub fn determinize_lattice_pruned<L, S>(
    lattice: &VectorFst<ArcTpl<LatticeWeight, L, S>>,
    opts: &PrunedDeterminizeOptions,
) -> Result<PrunedLattice<L, S>, OpenFstError>
where
    L: ArcLabel,
    S: ArcStateId,
{
    let mut beam = opts.beam;
    let mut last: Option<OpenFstError> = None;

    for narrowed in 0..=opts.max_retries {
        let mut narrowed_lattice = lattice.clone();
        if beam.is_finite() {
            prune_fst(
                &mut narrowed_lattice,
                &PruneOptions::threshold(LatticeWeight::new(beam, 0.0)),
            )?;
        }
        // Every path is gone, which no narrower beam will undo.
        if narrowed_lattice.start().is_none() {
            return Err(OpenFstError::InvalidOperation(format!(
                "determinize_lattice_pruned: a beam of {beam} left no path at all"
            )));
        }

        match determinize_lattice(&narrowed_lattice, &opts.determinize) {
            Ok(lattice) => {
                return Ok(PrunedLattice {
                    lattice,
                    beam,
                    narrowed,
                });
            }
            // Any failure is retried: a narrower beam can only make the
            // determinization smaller, so there is nothing to gain by telling
            // one kind of failure from another here.
            Err(error) => {
                last = Some(error);
                beam *= opts.beam_ratio;
            }
        }
    }

    Err(last.unwrap_or_else(|| {
        OpenFstError::InvalidOperation("determinize_lattice_pruned: no attempts were made".into())
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;
    use sicada::arc::StdArc;
    use sicada::fsts::vector_fst::StdVectorFst;
    use sicada::properties::{K_ACYCLIC, K_FST_PROPERTIES, K_I_DETERMINISTIC};
    use sicada::weights::float_weight::TropicalWeight;

    use crate::dense::DenseFst;
    use crate::lattice::{Lattice, LatticeDecodeOptions, lattice_decode};

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn cost(&mut self) -> f32 {
            self.below(256) as f32 / 16.0
        }
    }

    /// Every word sequence the FST accepts, with the best cost for each.
    ///
    /// Only usable on an acyclic FST, as the lattices below are: their states
    /// are `(frame, graph state)` and no arc goes back a frame.
    fn word_sequences<W, F>(fst: &F) -> FxHashMap<Vec<i32>, f32>
    where
        W: Weight,
        F: Fst<ArcTpl<W, i32, i32>>,
        W: TotalCost,
    {
        let mut found: FxHashMap<Vec<i32>, f32> = FxHashMap::default();
        let Some(start) = fst.start() else {
            return found;
        };
        let mut stack = vec![(start, Vec::<i32>::new(), 0.0f32)];
        while let Some((state, words, cost)) = stack.pop() {
            let final_weight = fst.final_weight(state);
            if final_weight.is_member() && final_weight != W::zero() {
                let total = cost + final_weight.total_cost();
                found
                    .entry(words.clone())
                    .and_modify(|best| *best = best.min(total))
                    .or_insert(total);
            }
            for arc in fst.arcs(state) {
                let mut next = words.clone();
                if arc.olabel() != 0 {
                    next.push(arc.olabel());
                }
                stack.push((arc.nextstate(), next, cost + arc.weight().total_cost()));
            }
        }
        found
    }

    /// The one number a path's weight comes down to, whichever of the two
    /// lattice semirings it is in.
    trait TotalCost {
        fn total_cost(&self) -> f32;
    }

    impl TotalCost for LatticeWeight {
        fn total_cost(&self) -> f32 {
            self.total()
        }
    }

    impl TotalCost for CompactLatticeWeight<i32> {
        fn total_cost(&self) -> f32 {
            self.weight().total()
        }
    }

    /// A graph with no input epsilons, so the lattice comes out acyclic and can
    /// be enumerated. Output epsilons are plentiful, so the alignments
    /// collapse.
    fn random_graph(rng: &mut Rng, symbols: usize) -> StdVectorFst {
        let states = 1 + rng.below(4);
        let mut graph: StdVectorFst = VectorFst::new();
        for _ in 0..states {
            graph.add_state();
        }
        graph.set_start(0);
        for from in 0..states as i32 {
            for _ in 0..1 + rng.below(3) {
                let ilabel = 1 + rng.below(symbols) as i32;
                // Half the arcs say no word at all, which is how one word comes
                // to span several frames.
                let olabel = if rng.below(2) == 0 {
                    0
                } else {
                    10 * (1 + rng.below(2) as i32)
                };
                let to = rng.below(states) as i32;
                graph.add_arc(
                    from,
                    StdArc::new(ilabel, olabel, TropicalWeight(rng.cost()), to),
                );
            }
            if rng.below(2) == 0 {
                graph.set_final(from, TropicalWeight(rng.cost()));
            }
        }
        graph.properties(K_FST_PROPERTIES, true);
        graph
    }

    fn decode(
        graph: &StdVectorFst,
        scores: &[f32],
        frames: usize,
        symbols: usize,
    ) -> Option<Lattice<StdArc>> {
        let dense = DenseFst::<StdArc>::new(scores, frames, symbols).unwrap();
        lattice_decode(graph, &dense, &LatticeDecodeOptions::exhaustive()).unwrap()
    }

    /// The statement that matters: determinizing changes which alignments are
    /// kept, and nothing else. Every word sequence the lattice had is still
    /// there, at the same cost, and no new one appeared.
    #[test]
    fn it_keeps_every_word_sequence_at_the_same_cost() {
        let symbols = 3;
        let mut rng = Rng(0x00D1_5EA5_E1A1_2345);
        let mut compared = 0;

        for round in 0..150 {
            let graph = random_graph(&mut rng, symbols);
            let frames = 1 + rng.below(4);
            let scores: Vec<f32> = (0..frames * symbols).map(|_| rng.cost()).collect();
            let Some(lattice) = decode(&graph, &scores, frames, symbols) else {
                continue;
            };

            let before = word_sequences(&lattice);
            let compact = determinize_lattice(&lattice, &DeterminizeLatticeOptions::default())
                .expect("a determinization");
            let after = word_sequences(&compact);

            assert_eq!(
                before.len(),
                after.len(),
                "round {round}: {} sequences became {}",
                before.len(),
                after.len()
            );
            for (words, cost) in &before {
                let found = after
                    .get(words)
                    .unwrap_or_else(|| panic!("round {round}: {words:?} went missing"));
                assert!(
                    (found - cost).abs() < 1e-3,
                    "round {round}: {words:?} cost {found}, was {cost}"
                );
            }
            compared += 1;
        }

        assert!(compared > 80, "only {compared} rounds produced a lattice");
    }

    /// Every path's `(cost, alignment)`, folded per word sequence by the
    /// semiring's own ⊕.
    ///
    /// That fold *is* the specification: ⊕ keeps the better cost's alignment
    /// whole, and breaks a tie on the shorter one. Writing the oracle as the
    /// fold rather than as "take the minimum cost" makes it check the alignment
    /// half too, ties included.
    fn best_per_sequence<W, F>(fst: &F) -> FxHashMap<Vec<i32>, CompactLatticeWeight<i32>>
    where
        W: Weight + AsCompact,
        F: Fst<ArcTpl<W, i32, i32>>,
    {
        let mut found: FxHashMap<Vec<i32>, CompactLatticeWeight<i32>> = FxHashMap::default();
        let Some(start) = fst.start() else {
            return found;
        };
        let one = CompactLatticeWeight::<i32>::one();
        let mut stack = vec![(start, Vec::<i32>::new(), one)];
        while let Some((state, words, weight)) = stack.pop() {
            let final_weight = fst.final_weight(state);
            if final_weight.is_member() && final_weight != W::zero() {
                let whole = weight.times(&final_weight.as_compact());
                found
                    .entry(words.clone())
                    .and_modify(|best| *best = best.plus(&whole))
                    .or_insert(whole);
            }
            for arc in fst.arcs(state) {
                let mut next = words.clone();
                if arc.olabel() != 0 {
                    next.push(arc.olabel());
                }
                stack.push((
                    arc.nextstate(),
                    next,
                    weight.times(&arc.weight().as_compact()),
                ));
            }
        }
        found
    }

    /// A path's weight as the compact semiring sees it.
    ///
    /// Only the compact weight implements it: the enumeration runs over
    /// [`to_compact`]'s output, which is the raw lattice with each arc's frame
    /// already moved into its weight and nothing else changed.
    trait AsCompact {
        fn as_compact(&self) -> CompactLatticeWeight<i32>;
    }

    impl AsCompact for CompactLatticeWeight<i32> {
        fn as_compact(&self) -> Self {
            self.clone()
        }
    }

    /// The alignment half is the point, so it is checked and not only the cost:
    /// each word sequence must come back with the *best-scoring* alignment the
    /// lattice had for it, chosen by the same ⊕ the semiring defines.
    #[test]
    fn it_keeps_the_best_alignment_for_each_word_sequence() {
        let symbols = 3;
        let mut rng = Rng(0x00AB_CDEF_0123_4567);
        let mut compared = 0;

        for round in 0..150 {
            let graph = random_graph(&mut rng, symbols);
            let frames = 1 + rng.below(4);
            let scores: Vec<f32> = (0..frames * symbols).map(|_| rng.cost()).collect();
            let Some(lattice) = decode(&graph, &scores, frames, symbols) else {
                continue;
            };

            // `to_compact` moves each arc's frame into its weight without
            // changing anything else, so enumerating *that* gives the same
            // paths as the raw lattice with their alignments attached.
            let expected = best_per_sequence(&to_compact(&lattice));
            let compact = determinize_lattice(&lattice, &DeterminizeLatticeOptions::default())
                .expect("a determinization");
            let found = best_per_sequence(&compact);

            assert_eq!(expected.len(), found.len(), "round {round}");
            for (words, want) in &expected {
                let got = found
                    .get(words)
                    .unwrap_or_else(|| panic!("round {round}: {words:?} went missing"));
                assert_eq!(
                    got.alignment(),
                    want.alignment(),
                    "round {round}: {words:?} kept the wrong alignment"
                );
                assert!(
                    (got.weight().total() - want.weight().total()).abs() < 1e-3,
                    "round {round}: {words:?} cost {got} vs {want}"
                );
            }
            compared += 1;
        }

        assert!(compared > 80, "only {compared} rounds produced a lattice");
    }

    /// What "compact" buys: one arc per word, and one path per word sequence.
    #[test]
    fn each_word_sequence_is_a_single_path() {
        let symbols = 3;
        let mut rng = Rng(0x0FED_CBA9_8765_4321);
        let mut collapsed = 0;

        for round in 0..150 {
            let graph = random_graph(&mut rng, symbols);
            let frames = 1 + rng.below(4);
            let scores: Vec<f32> = (0..frames * symbols).map(|_| rng.cost()).collect();
            let Some(lattice) = decode(&graph, &scores, frames, symbols) else {
                continue;
            };

            let compact = determinize_lattice(&lattice, &DeterminizeLatticeOptions::default())
                .expect("a determinization");
            if compact.start().is_none() {
                continue;
            }

            let props = compact.properties(K_I_DETERMINISTIC | K_ACYCLIC, true);
            assert_ne!(
                props & K_I_DETERMINISTIC,
                0,
                "round {round}: not deterministic, so a word sequence has two paths"
            );

            // The raw lattice usually had several paths per sequence; the
            // compact one has exactly as many paths as sequences.
            let paths_before = count_paths(&lattice);
            let sequences = word_sequences(&compact).len();
            let paths_after = count_paths(&compact);
            assert_eq!(paths_after, sequences, "round {round}");
            if paths_before > paths_after {
                collapsed += 1;
            }
        }

        assert!(collapsed > 40, "nothing collapsed in {collapsed} rounds");
    }

    fn count_paths<W, F>(fst: &F) -> usize
    where
        W: Weight,
        F: Fst<ArcTpl<W, i32, i32>>,
    {
        let Some(start) = fst.start() else {
            return 0;
        };
        let mut stack = vec![start];
        let mut paths = 0;
        while let Some(state) = stack.pop() {
            let final_weight = fst.final_weight(state);
            if final_weight.is_member() && final_weight != W::zero() {
                paths += 1;
            }
            for arc in fst.arcs(state) {
                stack.push(arc.nextstate());
            }
        }
        paths
    }

    /// The frames a word spanned have to survive. A second pass rescores them,
    /// and an alignment is read from them.
    #[test]
    fn the_alignment_travels_with_the_word() {
        // Three frames, one symbol each, all mapping to the same word 10.
        let mut graph: StdVectorFst = VectorFst::new();
        graph.add_state();
        graph.set_start(0);
        graph.set_final(0, TropicalWeight::one());
        // Label 1 says the word, labels 2 and 3 continue it silently.
        graph.add_arc(0, StdArc::new(1, 10, TropicalWeight::one(), 0));
        graph.add_arc(0, StdArc::new(2, 0, TropicalWeight::one(), 0));
        graph.add_arc(0, StdArc::new(3, 0, TropicalWeight::one(), 0));
        graph.properties(K_FST_PROPERTIES, true);

        // Frame 0 wants label 1, frames 1 and 2 want labels 2 and 3.
        let scores = [
            0.0, 9.0, 9.0, //
            9.0, 0.0, 9.0, //
            9.0, 9.0, 0.0,
        ];
        let lattice = decode(&graph, &scores, 3, 3).expect("a lattice");
        let compact = determinize_lattice(&lattice, &DeterminizeLatticeOptions::default()).unwrap();

        // One word leaves the start: the two silent frames are gone as arcs,
        // their frames folded into the weights around them.
        let start = compact.start().expect("a start");
        let arcs: Vec<_> = compact.arcs(start).collect();
        assert_eq!(arcs.len(), 1, "one word, one arc");
        assert_eq!(arcs[0].olabel(), 10);

        // The frames after the last word land on the final weight, which is
        // Kaldi's arrangement too, so the path's alignment is the ⊗ of the
        // arc's and the final state's, in that order.
        let mut best: VectorFst<ArcTpl<CompactLatticeWeight<i32>, i32, i32>> = VectorFst::new();
        sicada::algorithms::shortest_path::shortest_path(
            &compact,
            &mut best,
            &sicada::algorithms::shortest_path::ShortestPathOptions::default(),
        )
        .expect("a best path");
        let (words, weight) =
            sicada::string::string_fst_to_output_labels(&best).expect("a single path");

        assert_eq!(words, vec![10], "one word was said");
        assert_eq!(
            weight.alignment(),
            &[1, 2, 3],
            "the three frames the word spanned"
        );
        assert!(weight.weight().total().abs() < 1e-6, "{weight}");
    }

    /// A compact lattice is an FST like any other, so it writes and reads like
    /// one, and the header it writes carries Kaldi's names, which is why the
    /// weight's `type_name` was matched to upstream.
    #[test]
    fn it_writes_and_reads_back_as_an_fst() {
        use sicada::fst::{FstReadOptions, FstWriteOptions};
        use std::io::Write as _;

        let scores = [0.0, 1.0, 2.0, 0.5, 0.25, 3.0];
        let mut rng = Rng(0x0011_2233_4455_6677);
        let graph = random_graph(&mut rng, 3);
        let Some(lattice) = decode(&graph, &scores, 2, 3) else {
            return;
        };
        let compact = determinize_lattice(&lattice, &DeterminizeLatticeOptions::default()).unwrap();
        if compact.start().is_none() {
            return;
        }

        assert_eq!(
            <ArcTpl<CompactLatticeWeight<i32>, i32, i32> as Arc>::type_name().as_str(),
            "compactlattice44",
            "the name the header records"
        );

        let mut bytes = Vec::new();
        compact
            .write(&mut bytes, &FstWriteOptions::default())
            .expect("written");

        let directory = tempfile::tempdir().expect("a directory");
        let path = directory.path().join("lattice.fst");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();

        let mut file = std::fs::File::open(&path).unwrap();
        let read: VectorFst<ArcTpl<CompactLatticeWeight<i32>, i32, i32>> =
            VectorFst::read(&mut file, &FstReadOptions::default()).expect("read back");

        assert_eq!(read.num_states(), compact.num_states());
        assert_eq!(read.start(), compact.start());
        for state in compact.states() {
            assert_eq!(read.final_weight(state), compact.final_weight(state));
            assert_eq!(
                read.arcs(state).collect::<Vec<_>>(),
                compact.arcs(state).collect::<Vec<_>>(),
                "state {state}"
            );
        }
    }

    /// The whole point of narrowing: a lattice determinization that runs away
    /// finishes anyway, at a smaller beam, and says so.
    #[test]
    fn it_narrows_the_beam_rather_than_giving_up() {
        let symbols = 3;
        let mut rng = Rng(0x0777_8888_9999_AAAA);
        let graph = random_graph(&mut rng, symbols);
        let frames = 4;
        let scores: Vec<f32> = (0..frames * symbols).map(|_| rng.cost()).collect();
        let Some(lattice) = decode(&graph, &scores, frames, symbols) else {
            return;
        };

        // A cap of one state cannot be met at any beam, so every retry runs
        // away and the last error comes back rather than a truncated lattice.
        let impossible = determinize_lattice_pruned(
            &lattice,
            &PrunedDeterminizeOptions {
                determinize: DeterminizeLatticeOptions {
                    max_states: Some(1),
                    ..DeterminizeLatticeOptions::default()
                },
                max_retries: 2,
                ..PrunedDeterminizeOptions::default()
            },
        );
        assert!(impossible.is_err());

        // With a workable cap the first attempt succeeds and nothing narrows.
        let fine = determinize_lattice_pruned(&lattice, &PrunedDeterminizeOptions::default())
            .expect("a lattice");
        assert_eq!(fine.narrowed, 0);
        assert_eq!(fine.beam, PrunedDeterminizeOptions::default().beam);
    }

    /// Narrowing keeps the best path, which makes it a safe answer to running
    /// away rather than a wrong one.
    #[test]
    fn narrowing_never_loses_the_best_path() {
        let symbols = 3;
        let mut rng = Rng(0x00BB_CCDD_EEFF_0011);
        let mut compared = 0;

        for round in 0..100 {
            let graph = random_graph(&mut rng, symbols);
            let frames = 1 + rng.below(4);
            let scores: Vec<f32> = (0..frames * symbols).map(|_| rng.cost()).collect();
            let Some(lattice) = decode(&graph, &scores, frames, symbols) else {
                continue;
            };

            let whole = determinize_lattice(&lattice, &DeterminizeLatticeOptions::default())
                .expect("a determinization");
            let best = best_per_sequence(&whole)
                .into_values()
                .map(|weight| weight.weight().total())
                .fold(f32::INFINITY, f32::min);

            for beam in [8.0f32, 2.0, 0.5] {
                let narrowed = determinize_lattice_pruned(
                    &lattice,
                    &PrunedDeterminizeOptions {
                        beam,
                        ..PrunedDeterminizeOptions::default()
                    },
                )
                .expect("a lattice");
                let after = best_per_sequence(&narrowed.lattice)
                    .into_values()
                    .map(|weight| weight.weight().total())
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    (after - best).abs() < 1e-3,
                    "round {round} at beam {beam}: best became {after}, was {best}"
                );
            }
            compared += 1;
        }

        assert!(compared > 50, "only {compared} rounds produced a lattice");
    }

    #[test]
    fn a_cap_on_the_states_is_reported_rather_than_truncating() {
        let symbols = 3;
        let mut rng = Rng(0x0123_4567_89AB_CDEF);
        let graph = random_graph(&mut rng, symbols);
        let frames = 4;
        let scores: Vec<f32> = (0..frames * symbols).map(|_| rng.cost()).collect();
        let Some(lattice) = decode(&graph, &scores, frames, symbols) else {
            return;
        };

        let err = determinize_lattice(
            &lattice,
            &DeterminizeLatticeOptions {
                max_states: Some(1),
                ..DeterminizeLatticeOptions::default()
            },
        );
        assert!(err.is_err(), "a cap of one state should not be reachable");
    }
}
