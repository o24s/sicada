//! The algorithms take their type parameters from their arguments.
//!
//! Nothing here asserts a value. It is a compile-time claim: an algorithm's
//! type parameters all appear in an argument, so a call site never has to name
//! them. If a signature grows a parameter the compiler cannot pick, this file
//! stops compiling, which is the only way to notice.
//!
//! The parameters that would otherwise need naming are the ones OpenFst passes
//! as extra template arguments: the reverse arc a backwards walk uses
//! (`ShortestPath<Arc, RevArc>`) and the gallic type a factorization goes
//! through. The first is `Arc::Reverse`, an associated type derived from the
//! arc; the second is a unit value passed as an argument.

use sicada::algorithms::compose::compose;
use sicada::algorithms::determinize::{DeterminizeOptions, determinize};
use sicada::algorithms::epsnormalize::{EpsNormalizeType, eps_normalize};
use sicada::algorithms::factor_weight::{
    FactorIterator as _, FactorWeightOptions, GallicFactor, factor_weight,
};
use sicada::algorithms::minimize::minimize;
use sicada::algorithms::push::push_weights_to_initial;
use sicada::algorithms::shortest_path::{ShortestPathOptions, shortest_path};
use sicada::arc::{GallicArc, StdArc};
use sicada::fsts::vector_fst::VectorFst;
use sicada::weights::string_weight::GallicRight;

#[test]
fn no_call_site_names_a_type() {
    // Only the first binding says what it holds; the rest follow from it.
    let fst: VectorFst<StdArc> = VectorFst::new();
    let other = VectorFst::new();
    let mut out = VectorFst::new();

    compose(&fst, &other, &mut out).unwrap();
    determinize(&fst, &mut out, &DeterminizeOptions::default()).unwrap();
    minimize(&mut out, 1e-6, false).unwrap();

    // Backwards: upstream takes the reverse arc as a second template argument.
    shortest_path(&fst, &mut out, &ShortestPathOptions::default()).unwrap();
    push_weights_to_initial(&mut out, 1e-6, false).unwrap();

    // Through the gallic semiring: upstream takes the gallic type and the
    // factor iterator as template arguments; both are values here.
    let gfst: VectorFst<GallicArc<StdArc, GallicRight>> = VectorFst::new();
    let mut gout = VectorFst::new();
    factor_weight(
        &gfst,
        &mut gout,
        GallicFactor::new,
        &FactorWeightOptions::default(),
    );
    let mut eout = VectorFst::new();
    eps_normalize(&fst, &mut eout, EpsNormalizeType::Input, GallicRight).unwrap();
}
