//! A weighted finite-state transducer library, file-compatible with OpenFst.
//!
//! An FST is a graph whose arcs carry an input label, an output label and a
//! weight, so it holds a relation between strings together with a cost for each
//! pairing. Composing two of them applies one after the other, and searching one
//! answers what the cheapest pairing is. Speech recognition, morphology and text
//! normalisation are the usual reasons to want that.
//!
//! ```
//! use sicada::prelude::*;
//!
//! let mut fst = StdVectorFst::new();
//! let start = fst.add_state();
//! let middle = fst.add_state();
//! let end = fst.add_state();
//! fst.set_start(start);
//! fst.set_final(end, TropicalWeight::one());
//!
//! // Two ways from the start to the end, one cheaper than the other.
//! fst.add_arc(start, StdArc::new(1, 1, TropicalWeight(0.5), middle));
//! fst.add_arc(middle, StdArc::new(2, 2, TropicalWeight(0.5), end));
//! fst.add_arc(start, StdArc::new(1, 1, TropicalWeight(3.0), end));
//!
//! let mut best = StdVectorFst::new();
//! shortest_path(&fst, &mut best, &ShortestPathOptions::default())?;
//! assert_eq!(best.num_states(), 3);
//! # Ok::<(), sicada::error::OpenFstError>(())
//! ```
//!
//! # How it is put together
//!
//! Everything is generic over an [`Arc`](arc::Arc), which fixes the weight and
//! the integer types the labels and state ids use, rather than over the weight
//! alone. [`StdArc`](arc::StdArc) is the usual one: tropical weights with
//! 32-bit labels.
//!
//! - [`fst`] holds the trait hierarchy. [`Fst`](fst::Fst) is what an algorithm
//!   reads, [`ExpandedFst`](fst::ExpandedFst) adds a state count in constant
//!   time, and [`MutableFst`](fst::MutableFst) adds building.
//! - [`fsts`] holds the implementations: [`VectorFst`](fsts::vector_fst::VectorFst)
//!   to build one, [`ConstFst`](fsts::const_fst::ConstFst) to map a file,
//!   [`CompactFst`](fsts::compact_fst::CompactFst) to hold one compressed, and
//!   [`ExpanderFst`](fsts::expander_fst::ExpanderFst) to produce states as they
//!   are asked for.
//! - [`weight`] is the semiring trait and [`weights`] the semirings themselves,
//!   from the tropical and log weights up to the product, string, lexicographic
//!   and expectation ones.
//! - [`algorithms`] holds the operations, each stating in its bounds which
//!   semiring properties it needs.
//! - [`properties`] carries the bitmask OpenFst files record, and on top of it
//!   [`Verified`], which checks a property once and then holds it in the type.
//! - [`prelude`] is all of the above that a caller normally names, in one `use`.
//!
//! # Reading and writing OpenFst's files
//!
//! The binary format is upstream's, so an FST written by OpenFst can be read
//! here and one written here can be read there. `read` and `write` on each FST
//! type take the byte stream; [`AnyFst`](fsts::any_fst::AnyFst) reads one whose
//! type is only known from its header.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

pub mod add_on;
pub mod algorithms;
pub mod arc;
pub mod arc_filter;
pub mod cache;
pub mod data_structures;
pub mod error;
pub mod expander_cache;
pub mod fst;
pub mod fst_header;
pub mod fst_type;
pub mod fsts;
pub mod macros;
pub mod matcher;
pub mod memory;
pub mod prelude;
pub mod properties;
pub mod queue;
pub mod string;
pub mod symbol_table;
pub mod symbol_table_ops;
pub mod utils;
pub mod weight;
pub mod weights;

pub use properties::{
    Acceptor, Acyclic, DetEpsFreeAcceptor, StringFst, UnweightedDetEpsFreeAcceptor, Verified,
    VerifyExt,
};

pub type AtomicRc<T> = std::sync::Arc<T>;

pub use fsts::*;
pub use weights::*;
