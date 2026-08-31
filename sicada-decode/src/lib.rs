//! Decoding an acoustic model's output with sicada.
//!
//! This crate is the inference half of a pair: train on a GPU with k2, then run
//! inference on a CPU here. That is the regime where a GPU decoder has the least
//! to offer, since a single stream has no batch to fill and the time axis is
//! serial either way, and correspondingly the regime where an FST library that
//! is fast on a CPU earns its keep.
//!
//! It sits outside `sicada` proper because `sicada` is a port of OpenFst's
//! library and none of this belongs to it; the pieces here come from the
//! Kaldi/k2 side.
//!
//! - [`dense`] reads the acoustic model's `T × V` score matrix as an FST, so
//!   that composing a decoding graph against it is an ordinary composition.
//! - [`viterbi`] walks that composition one frame at a time without building
//!   it, which is how a decoder works.
//! - [`lattice`] does the same but keeps the alternatives, so a second pass has
//!   something to rescore. [`lattice_weight`] is the semiring its arcs carry,
//!   which holds the graph cost and the acoustic cost apart.
//! - [`compact`] collapses the alignments, so each word sequence appears once
//!   with the best one. [`compact_lattice_weight`] is the semiring that makes
//!   that possible: a cost with the frames it spanned attached.
//! - [`nbest`] reads the answers back out, and rescales the two halves against
//!   each other without decoding again.
//! - [`ctc`] builds the graph side for a CTC model, which is where a decoder
//!   with no language model starts.
//! - [`align`](mod@align) covers the other half of the same model's use. When the
//!   transcript is already known the graph is a single chain, and the only
//!   question is which frames each phone occupies. That case is small enough to
//!   solve exactly, so it has no beam. [`occupancy`](mod@occupancy) walks the same chain in the
//!   log semiring, for the soft answer that a single path cannot give.
//! - [`trellis`] is the solver those two are built on, and the piece to use
//!   when the chain is not the shape you want. Supply the transitions into a cell
//!   (how many there are, what they cost, what they mean) and the band, the
//!   packed traceback and the forward-backward come with it.
//!
//! # Decoding a CTC model
//!
//! The whole pipeline, from a score matrix to the answers. The scores here are
//! made up; a real one comes out of the acoustic model as negative log
//! probabilities, `T × V` row-major, with the blank in column 0.
//!
//! ```
//! use sicada::arc::StdArc;
//! use sicada_decode::{
//!     DecodeOptions, DenseFst, LatticeDecodeOptions, PrunedDeterminizeOptions, ctc_topo,
//!     determinize_lattice_pruned, lattice_decode, n_best, viterbi_decode,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Blank plus three tokens, four frames. Column 0 is the blank.
//! let (frames, symbols) = (4, 4);
//! let scores = vec![
//!     9.0, 0.0, 9.0, 9.0, // token 1
//!     9.0, 0.0, 9.0, 9.0, // token 1 again, held rather than repeated
//!     0.0, 9.0, 9.0, 9.0, // blank
//!     9.0, 0.0, 9.0, 9.0, // token 1, and the blank makes it a second one
//! ];
//!
//! let graph = ctc_topo::<StdArc>(symbols, 1)?;
//! let dense = DenseFst::<StdArc>::new(&scores, frames, symbols)?;
//!
//! // The transcript, and nothing else.
//! let best = viterbi_decode(&graph, &dense, &DecodeOptions::default())?
//!     .expect("the beam kept a path");
//! // Labels are columns offset by one, which is where `ctc_topo` put them.
//! let columns: Vec<i32> = best.labels.iter().map(|label| label - 1).collect();
//! assert_eq!(columns, vec![1, 1]);
//!
//! // Or the alternatives too, for a second pass to rescore.
//! let lattice = lattice_decode(&graph, &dense, &LatticeDecodeOptions::default())?
//!     .expect("the beam kept a path");
//! let compact = determinize_lattice_pruned(&lattice, &PrunedDeterminizeOptions::default())?;
//! for answer in n_best(&compact.lattice, 3)? {
//!     let columns: Vec<i32> = answer.words.iter().map(|label| label - 1).collect();
//!     // Each answer knows which frames produced it.
//!     assert_eq!(answer.alignment().len(), frames);
//!     let _ = (columns, answer.cost());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Aligning a transcript that is already known
//!
//! Here the answer is given and only the timing is wanted, so there is no
//! topology and no lattice: a single chain, solved exactly.
//!
//! ```
//! use sicada::arc::StdArc;
//! use sicada_decode::{AlignChain, DenseFst, align, occupancy};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Six frames over a blank and three phones. Column 0 is the blank.
//! let (frames, symbols) = (6, 4);
//! let scores = vec![
//!     9.0, 0.0, 9.0, 9.0, // phone 1
//!     9.0, 0.0, 9.0, 9.0, // still phone 1
//!     0.0, 9.0, 9.0, 9.0, // silence
//!     9.0, 9.0, 0.0, 9.0, // phone 2
//!     0.0, 9.0, 9.0, 9.0, // silence
//!     0.0, 9.0, 9.0, 9.0, // silence
//! ];
//!
//! // The reference: phones as *columns*, in order. Labels do not come into it.
//! let chain = AlignChain::new(vec![1, 2]);
//! let dense = DenseFst::<StdArc>::new(&scores, frames, symbols)?;
//!
//! let alignment = align(&chain, &dense)?.expect("the reference fits");
//! // Each phone gets the frames that sound it, and the blank frames belong to
//! // nobody, so the last phone does not swallow the silence after it.
//! assert_eq!(alignment.spans(), vec![Some(0..2), Some(3..4)]);
//! assert!(alignment.skipped().is_empty());
//!
//! // The mean per-frame cost is the warning that the reference is not what
//! // was said. Here it is low, because the reference is correct.
//! assert!(alignment.mean_acoustic_cost(&chain, &dense) < 0.1);
//!
//! // The same chain in the log semiring, when the soft answer is wanted.
//! let spread = occupancy(&chain, &dense)?.expect("the reference fits");
//! assert!((spread.expected_durations()[0] - 2.0).abs() < 0.01);
//! # Ok(())
//! # }
//! ```
//!
//! The references are Kaldi (`decoder/lattice-faster-decoder.*`,
//! `fstext/lattice-weight.h`) and k2.

pub mod align;
pub mod compact;
pub mod compact_lattice_weight;
pub mod ctc;
pub mod dense;
mod frontier;
pub mod lattice;
pub mod lattice_weight;
pub mod nbest;
pub mod occupancy;
pub mod trellis;
pub mod viterbi;

pub use align::{AlignChain, Alignment, ChainTrellis, align};
pub use compact::{
    CompactLattice, DeterminizeLatticeOptions, PrunedDeterminizeOptions, PrunedLattice,
    determinize_lattice, determinize_lattice_pruned, to_compact,
};
pub use compact_lattice_weight::{CompactLatticeArc, CompactLatticeWeight};
pub use ctc::{collapse, ctc_topo};
pub use dense::{DenseFst, FromScore};
pub use frontier::DecodeOptions;
pub use lattice::{Lattice, LatticeDecodeOptions, lattice_decode};
pub use lattice_weight::{LatticeArc, LatticeWeight, LatticeWeight64};
pub use nbest::{Hypothesis, n_best, scale};
pub use occupancy::{Occupancy, occupancy};
pub use trellis::{Path, ReversibleTrellis, Step, Transition, Trellis, best_path, posteriors};
pub use viterbi::{Decoded, viterbi_decode};
