//! The decoder, measured against the composition it exists to avoid.
//!
//! There is no third-party decoder in this comparison yet. k2's is a GPU one,
//! and putting it here honestly means reporting two regimes rather than one
//! number. What *can* be measured today is the thing
//! the decoder is for: `graph ∘ dense` then shortest path gives the same
//! answer, and building that composition is exactly the work a frame-
//! synchronous search skips.
//!
//! Both sides are given the same graph, the same scores and the same beam
//! (none), and the harness checks they agree before timing either.

use sicada::algorithms::arcsort::{ILabelCompare, arc_sort};
use sicada::algorithms::compose::compose;
use sicada::algorithms::shortest_path::{ShortestPathOptions, shortest_path};
use sicada::arc::StdArc;
use sicada::fst::{ExpandedFst, Fst};
use sicada::fsts::vector_fst::{StdVectorFst, VectorFst};
use sicada::string::string_fst_to_output_labels;
use sicada_decode::DecodeOptions;
use sicada_decode::compact::{
    DeterminizeLatticeOptions, PrunedDeterminizeOptions, determinize_lattice,
    determinize_lattice_pruned,
};
use sicada_decode::ctc::ctc_topo;
use sicada_decode::dense::DenseFst;
use sicada_decode::lattice::{LatticeDecodeOptions, lattice_decode};
use sicada_decode::viterbi::viterbi_decode;

/// A CTC topology over `symbols` columns, arc-sorted so that composition can
/// use it as its right-hand side.
pub fn graph(symbols: usize) -> StdVectorFst {
    let mut fst = ctc_topo::<StdArc>(symbols, 1).expect("a topology");
    arc_sort(&mut fst, &ILabelCompare);
    fst
}

/// Acoustic scores that look like a model's: one column favoured per frame,
/// the rest well behind it.
///
/// The favoured column's cost is *not* zero, and no two frames' costs are the
/// same. That matters. A generator that put an exact 0.0 in every frame gave
/// every frame a free choice, so the whole thing was one enormous tie: the
/// frame-synchronous search and the composition both found a zero-cost path,
/// and they were different zero-cost paths. The harness read that as a
/// disagreement and dropped the comparison, correctly, since a benchmark whose
/// answer is not unique cannot check that two implementations agree.
pub fn scores(frames: usize, symbols: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut scores = vec![0.0f32; frames * symbols];
    for frame in 0..frames {
        for column in 0..symbols {
            // Well above whatever the favoured column gets, so the beam has
            // something to narrow onto.
            scores[frame * symbols + column] = 2.0 + (next() % 2048) as f32 / 128.0;
        }
        let favoured = (next() % symbols as u64) as usize;
        scores[frame * symbols + favoured] = (next() % 4096) as f32 / 4096.0;
    }
    scores
}

fn checksum(labels: &[i32], cost: f32) -> u64 {
    let mut sum = (cost * 64.0).round() as i64 as u64;
    for (index, &label) in labels.iter().enumerate() {
        sum = sum
            .wrapping_mul(1_000_003)
            .wrapping_add((label as u64).wrapping_add(index as u64));
    }
    sum
}

/// Frame-synchronous search, no beam.
pub fn viterbi(graph: &StdVectorFst, scores: &[f32], frames: usize, symbols: usize) -> u64 {
    let dense = DenseFst::<StdArc>::new(scores, frames, symbols).expect("a dense FST");
    let decoded = viterbi_decode(graph, &dense, &DecodeOptions::exhaustive())
        .expect("a decode")
        .expect("a path");
    checksum(&decoded.labels, decoded.weight.0)
}

/// Frame-synchronous search under a beam, which is how it would actually run.
pub fn viterbi_pruned(
    graph: &StdVectorFst,
    scores: &[f32],
    frames: usize,
    symbols: usize,
    beam: f32,
) -> u64 {
    let dense = DenseFst::<StdArc>::new(scores, frames, symbols).expect("a dense FST");
    let decoded = viterbi_decode(
        graph,
        &dense,
        &DecodeOptions {
            beam,
            ..DecodeOptions::default()
        },
    )
    .expect("a decode")
    .expect("a path");
    checksum(&decoded.labels, decoded.weight.0)
}

/// The same answer by building the whole composition and taking its best path.
pub fn via_composition(graph: &StdVectorFst, scores: &[f32], frames: usize, symbols: usize) -> u64 {
    let dense = DenseFst::<StdArc>::new(scores, frames, symbols).expect("a dense FST");
    let mut composed: StdVectorFst = VectorFst::new();
    compose(&dense, graph, &mut composed).expect("a composition");
    let mut best: StdVectorFst = VectorFst::new();
    shortest_path(&composed, &mut best, &ShortestPathOptions::default()).expect("a best path");
    let (labels, weight) = string_fst_to_output_labels(&best).expect("a single path");
    let labels: Vec<i32> = labels.into_iter().filter(|&label| label != 0).collect();
    checksum(&labels, weight.0)
}

/// How large the composition the decoder skipped would have been.
pub fn composition_size(
    graph: &StdVectorFst,
    scores: &[f32],
    frames: usize,
    symbols: usize,
) -> (usize, usize) {
    let dense = DenseFst::<StdArc>::new(scores, frames, symbols).expect("a dense FST");
    let mut composed: StdVectorFst = VectorFst::new();
    compose(&dense, graph, &mut composed).expect("a composition");
    (composed.num_states(), composed.count_arcs())
}

/// Decoding to a lattice, which keeps the alternatives.
pub fn lattice(
    graph: &StdVectorFst,
    scores: &[f32],
    frames: usize,
    symbols: usize,
    beam: f32,
) -> u64 {
    let dense = DenseFst::<StdArc>::new(scores, frames, symbols).expect("a dense FST");
    let lattice = lattice_decode(
        graph,
        &dense,
        &LatticeDecodeOptions {
            search: DecodeOptions {
                beam,
                ..DecodeOptions::default()
            },
            lattice_beam: beam,
        },
    )
    .expect("a decode")
    .expect("a lattice");
    (lattice.num_states() as u64).wrapping_mul(1_000_003) ^ lattice.count_arcs() as u64
}

/// Collapsing under a beam that narrows itself when determinization runs away.
///
/// The row this feeds is the one that finishes on a long utterance; the plain
/// one above does not.
pub fn lattice_collapsed_pruned(
    graph: &StdVectorFst,
    scores: &[f32],
    frames: usize,
    symbols: usize,
    beam: f32,
) -> u64 {
    let dense = DenseFst::<StdArc>::new(scores, frames, symbols).expect("a dense FST");
    let lattice = lattice_decode(
        graph,
        &dense,
        &LatticeDecodeOptions {
            search: DecodeOptions {
                beam,
                ..DecodeOptions::default()
            },
            lattice_beam: beam,
        },
    )
    .expect("a decode")
    .expect("a lattice");
    let pruned = determinize_lattice_pruned(&lattice, &PrunedDeterminizeOptions::default())
        .expect("collapsed");
    (pruned.lattice.num_states() as u64).wrapping_mul(1_000_003)
        ^ pruned.lattice.count_arcs() as u64
}

/// The beam a narrowing collapse settled on, and how many attempts it took.
pub fn collapse_narrowing(
    graph: &StdVectorFst,
    scores: &[f32],
    frames: usize,
    symbols: usize,
    beam: f32,
) -> (f32, usize, usize) {
    let dense = DenseFst::<StdArc>::new(scores, frames, symbols).expect("a dense FST");
    let lattice = lattice_decode(
        graph,
        &dense,
        &LatticeDecodeOptions {
            search: DecodeOptions {
                beam,
                ..DecodeOptions::default()
            },
            lattice_beam: beam,
        },
    )
    .expect("a decode")
    .expect("a lattice");
    let pruned = determinize_lattice_pruned(&lattice, &PrunedDeterminizeOptions::default())
        .expect("collapsed");
    (pruned.beam, pruned.narrowed, pruned.lattice.num_states())
}

/// The same, then collapsing the alignments.
pub fn lattice_determinized(
    graph: &StdVectorFst,
    scores: &[f32],
    frames: usize,
    symbols: usize,
    beam: f32,
) -> u64 {
    let dense = DenseFst::<StdArc>::new(scores, frames, symbols).expect("a dense FST");
    let lattice = lattice_decode(
        graph,
        &dense,
        &LatticeDecodeOptions {
            search: DecodeOptions {
                beam,
                ..DecodeOptions::default()
            },
            lattice_beam: beam,
        },
    )
    .expect("a decode")
    .expect("a lattice");
    let compact =
        determinize_lattice(&lattice, &DeterminizeLatticeOptions::default()).expect("collapsed");
    (compact.num_states() as u64).wrapping_mul(1_000_003) ^ compact.count_arcs() as u64
}
