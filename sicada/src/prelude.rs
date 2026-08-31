//! Everything a user of the library normally wants, in one `use`.
//!
//! Port of OpenFst's `fstlib.h`, which is 88 `#include` lines and nothing else:
//! a convenience header that pulls the public API in so that a program does not
//! have to name each piece.
//!
//! ```
//! use sicada::prelude::*;
//!
//! let mut fst = StdVectorFst::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! fst.set_start(s0);
//! fst.add_arc(s0, StdArc::new(1, 2, TropicalWeight(0.5), s1));
//! fst.set_final(s1, TropicalWeight::one());
//!
//! let mut reversed = StdVectorFst::new();
//! reverse(&fst, &mut reversed, true);
//! assert_eq!(reversed.num_states(), 3);
//! ```
//!
//! SICADA-DIVERGE: upstream's header brings in every declaration in the
//! library, because in C++ that is the only granularity a header has. This is a
//! curated set: the traits, the concrete types, and the entry point of each
//! algorithm. The pieces an algorithm is *built* from (matchers, compose
//! filters, queues, accumulators, arc mappers, state mappers, visitors, cache
//! stores) stay in their own modules: a caller reaches for them only when
//! writing a new algorithm, and importing them all here would put dozens of
//! names like `Match` and `Sequence` into every scope that wanted `compose`.

// The trait hierarchy, and what it is generic over.
pub use crate::arc::{Arc, ArcLabel, ArcStateId, ArcTpl, StdArc};
pub use crate::error::{OpenFstError, ParseError};
pub use crate::fst::{
    ContiguousArcsFst, ExpandedFst, Fst, FstReadOptions, FstWriteOptions, MatchType, MutableFst,
};
pub use crate::fst_header::FstHeader;
pub use crate::fst_type::{ArcType, FstType, WeightType};
pub use crate::symbol_table::SymbolTable;
pub use crate::weight::{
    CommutativeWeight, Divide, DivideType, IdempotentWeight, LeftSemiring, Minus, PathWeight,
    RightSemiring, Weight, WeightIo,
};

// The typestate layer over the property bits.
pub use crate::properties::{
    Acceptor, Acyclic, DetEpsFreeAcceptor, FstProperty, StringFst, UnweightedDetEpsFreeAcceptor,
    Verified, VerifyExt,
};

// The concrete FSTs, and the modules the rest of each one lives in.
pub use crate::fsts::compact_fst::{
    ArcCompactor, CompactAcceptorFst, CompactFst, CompactStringFst, CompactUnweightedAcceptorFst,
    CompactUnweightedFst, CompactWeightedStringFst,
};
pub use crate::fsts::complement_fst::ComplementFst;
pub use crate::fsts::const_fst::ConstFst;
pub use crate::fsts::edit_fst::EditFst;
pub use crate::fsts::expander_fst::ExpanderFst;
pub use crate::fsts::vector_fst::{Log64VectorFst, StdVectorFst, VectorFst};
pub use crate::fsts::*;
#[cfg(feature = "fst-types")]
pub use crate::fsts::{any_fst::AnyFst, matcher_fst::MatcherFst};

// The weights, and the modules the rest of each one lives in.
pub use crate::weights::float_weight::{Log64Weight, LogWeight, MinMaxWeight, TropicalWeight};
pub use crate::weights::lexicographic_weight::LexicographicWeight;
pub use crate::weights::product_weight::ProductWeight;
pub use crate::weights::string_weight::{StringLeft, StringRight, StringWeight};
pub use crate::weights::*;

// Reading and writing text and strings.
pub use crate::string::{
    StringCompiler, StringPrinter, TokenType, compile_labels, labels_to_string, string_to_labels,
};

/// Every algorithm's entry point.
///
/// The options types travel with them, since an algorithm that takes options
/// cannot be called without naming its own.
pub mod algorithms {
    pub use crate::algorithms::arc_map::{arc_map, arc_map_to};
    pub use crate::algorithms::arcsort::{ILabelCompare, OLabelCompare, arc_sort};
    pub use crate::algorithms::closure::{ClosureType, closure};
    pub use crate::algorithms::compose::{compose, compose_with};
    pub use crate::algorithms::concat::{concat, concat_onto};
    pub use crate::algorithms::connect::{condense, connect};
    pub use crate::algorithms::determinize::{DeterminizeOptions, DeterminizeType, determinize};
    pub use crate::algorithms::difference::difference;
    pub use crate::algorithms::disambiguate::{DisambiguateOptions, disambiguate};
    pub use crate::algorithms::encode::{EncodeMapper, EncodeType, decode, encode};
    pub use crate::algorithms::epsnormalize::{EpsNormalizeType, eps_normalize};
    pub use crate::algorithms::equal::{equal, equal_with};
    pub use crate::algorithms::equivalent::equivalent;
    pub use crate::algorithms::factor_weight::{FactorIterator, factor_weight};
    pub use crate::algorithms::intersect::intersect;
    pub use crate::algorithms::invert::{invert, invert_to};
    pub use crate::algorithms::isomorphic::{isomorphic, isomorphic_with};
    pub use crate::algorithms::minimize::minimize;
    pub use crate::algorithms::project::{project, project_to};
    pub use crate::algorithms::prune::{PruneOptions, prune, prune_to};
    pub use crate::algorithms::push::{
        ReweightType, push_to_final, push_to_initial, push_weights, push_weights_to_final,
        push_weights_to_initial, remove_weight, total_weight,
    };
    pub use crate::algorithms::randequivalent::{rand_equivalent, rand_equivalent_default};
    pub use crate::algorithms::randgen::{
        LogProbArcSelector, RandGenOptions, UniformArcSelector, rand_gen,
    };
    pub use crate::algorithms::rational::RationalFst;
    pub use crate::algorithms::relabel::{relabel, relabel_tables};
    pub use crate::algorithms::replace::{ReplaceLabelType, ReplaceOptions, replace};
    pub use crate::algorithms::replace_util::ReplaceUtil;
    pub use crate::algorithms::reverse::reverse;
    pub use crate::algorithms::reweight::{reweight_to_final, reweight_to_initial};
    pub use crate::algorithms::rmepsilon::{RmEpsilonOptions, rm_epsilon, rm_epsilon_with};
    pub use crate::algorithms::rmfinalepsilon::rm_final_epsilon;
    pub use crate::algorithms::shortest_distance::{
        ShortestDistanceOptions, shortest_distance, shortest_distance_forward,
        shortest_distance_reverse, shortest_distance_with,
    };
    pub use crate::algorithms::shortest_path::{ShortestPathOptions, shortest_path};
    pub use crate::algorithms::state_map::{state_map, state_map_to};
    pub use crate::algorithms::state_sort::state_sort;
    pub use crate::algorithms::synchronize::synchronize;
    pub use crate::algorithms::test_properties::{check_properties, test_properties};
    pub use crate::algorithms::topsort::top_sort;
    pub use crate::algorithms::union::{union, union_all};
    pub use crate::algorithms::verify::verify;
}

pub use algorithms::*;
