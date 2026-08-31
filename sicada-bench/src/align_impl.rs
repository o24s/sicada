//! Forced alignment, measured against the general decoder on the same chain.
//!
//! There is no third-party aligner in this comparison. k2's is a pruned search
//! in a GPU library that is not installed here; the k2 numbers quoted against
//! these rows were measured elsewhere, on another machine, and are labelled as
//! such rather than being run alongside.
//!
//! What *can* be raced here is the thing the exact aligner exists to avoid: the
//! same chain, handed to [`viterbi_decode`] as an ordinary FST, computes the
//! same answer through a hash-map frontier and an ever-growing link arena. Both
//! sides get the same reference, the same scores and no beam, and the harness
//! checks they agree on the cost before timing either.

use sicada::arc::StdArc;
use sicada::fsts::vector_fst::StdVectorFst;
use sicada_decode::DecodeOptions;
use sicada_decode::align::{AlignChain, align};
use sicada_decode::dense::DenseFst;
use sicada_decode::occupancy::occupancy;
use sicada_decode::viterbi::viterbi_decode;

/// A reference, and the scores of an utterance that says it.
///
/// The scores are built along a path through the chain rather than at random,
/// so the alignment has one clear answer. A matrix of noise would take the same
/// time, since the band fixes the work regardless, but the two implementations
/// would be timed finding different answers and the harness could not check them
/// against each other.
///
/// One frame in seven is silence, so the alignment has gaps to place rather
/// than a phone in every frame.
pub fn utterance(
    frames: usize,
    num_phones: usize,
    symbols: usize,
    seed: u64,
) -> (AlignChain, Vec<f32>) {
    assert!(symbols > 1 && frames >= 3 * num_phones.max(1));
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let phones: Vec<u32> = (0..num_phones)
        .map(|_| 1 + (next() % (symbols as u64 - 1)) as u32)
        .collect();

    let mut scores = vec![0.0f32; frames * symbols];
    for frame in 0..frames {
        // The phone this frame belongs to, walking the reference in step with
        // the audio.
        let position = if num_phones == 0 {
            0
        } else {
            frame * num_phones / frames
        };
        let favoured = if num_phones == 0 || next() % 7 == 0 {
            0
        } else {
            phones[position] as usize
        };
        let row = &mut scores[frame * symbols..(frame + 1) * symbols];
        for (column, score) in row.iter_mut().enumerate() {
            // Jittered, so that no two paths land on the same total and the
            // best alignment is unique.
            let jitter = (next() % 4096) as f32 / 4096.0;
            *score = if column == favoured {
                0.1 + jitter * 0.1
            } else {
                4.0 + jitter
            };
        }
    }
    // Skipping allowed but expensive, which is the setting the reference
    // material runs at, and it is the arm that costs the aligner work.
    (
        AlignChain::new(phones).with_uniform_skip_cost(6.0).unwrap(),
        scores,
    )
}

/// The cost, quantised, so two implementations can be checked against each
/// other without their tie-breaking having to match.
fn checksum(cost: f32) -> u64 {
    (cost as f64 * 64.0).round() as u64
}

/// The exact aligner: a banded plane of `f32` and a two-bit traceback.
pub fn exact(chain: &AlignChain, scores: &[f32], frames: usize, symbols: usize) -> u64 {
    let dense = DenseFst::<StdArc>::new(scores, frames, symbols).expect("a dense FST");
    let alignment = align(chain, &dense).expect("an alignment").expect("a path");
    checksum(alignment.cost())
}

/// The same chain as an ordinary FST, through the general decoder.
///
/// No beam, so it computes the same answer rather than an approximation of it.
pub fn by_decoder(chain: &AlignChain, scores: &[f32], frames: usize, symbols: usize) -> u64 {
    let dense = DenseFst::<StdArc>::new(scores, frames, symbols).expect("a dense FST");
    let fst: StdVectorFst = chain.to_fst(1).expect("a chain FST");
    let decoded = viterbi_decode(&fst, &dense, &DecodeOptions::exhaustive())
        .expect("a decode")
        .expect("a path");
    checksum(decoded.weight.0)
}

/// The log-semiring pass over the same chain, which is a different answer and
/// so gets its own row rather than a contender.
pub fn soft(chain: &AlignChain, scores: &[f32], frames: usize, symbols: usize) -> u64 {
    let dense = DenseFst::<StdArc>::new(scores, frames, symbols).expect("a dense FST");
    let spread = occupancy(chain, &dense)
        .expect("an occupancy")
        .expect("a path");
    checksum(spread.cost())
}
