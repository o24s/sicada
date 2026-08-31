//! Deciding whether two FSTs agree, by sampling.
//!
//! Port of OpenFst's `randequivalent.h`. Deciding equivalence outright needs
//! two deterministic acceptors ([`equivalent`](super::equivalent::equivalent));
//! this asks a weaker question of anything: draw a path from one of the two,
//! and check that both give that path's input the same total weight. Enough
//! draws that agree is evidence rather than proof: a difference the sampling
//! never visits is a difference it never sees.

use crate::algorithms::arcsort::{ILabelCompare, OLabelCompare, arc_sort};
use crate::algorithms::compose::compose;
use crate::algorithms::connect::connect;
use crate::algorithms::project::{ProjectType, project};
use crate::algorithms::randgen::{ArcSelector, RandGenOptions, Rng, rand_gen};
use crate::algorithms::shortest_distance::{SHORTEST_DELTA, shortest_distance};
use crate::arc::Arc;
use crate::data_structures::bi_table::BiTableId;
use crate::error::OpenFstError;
use crate::fst::{ExpandedFst, Fst, MutableFst};
use crate::fsts::vector_fst::VectorFst;
use crate::properties::{K_CYCLIC, K_FST_PROPERTIES};
use crate::symbol_table::compat_symbols_rc;
use crate::weight::{IDEMPOTENT, Weight};
use crate::weights::float_weight::Log64Weight;

/// Whether `npath` drawn paths get the same weight from both FSTs.
///
/// Each draw comes from one of the two at random, so a path only one of them
/// has is as likely to be tried as one they share.
///
/// SICADA-DIVERGE: upstream reports a refusal, such as incompatible symbol
/// tables or an FST already in error, by returning `false` and setting a `bool*`
/// that defaults to null, so a caller cannot tell "not equivalent" from "could
/// not tell". It is an error here.
pub fn rand_equivalent<A, F1, F2, S>(
    fst1: &F1,
    fst2: &F2,
    npath: usize,
    rng: &mut Rng,
    opts: &RandGenOptions<S>,
    delta: f32,
) -> Result<bool, OpenFstError>
where
    A: Arc,
    A::StateId: BiTableId,
    A::Weight: From<Log64Weight>,
    F1: Fst<A> + ExpandedFst<A>,
    F2: Fst<A> + ExpandedFst<A>,
    S: ArcSelector<A>,
    <A::Weight as Weight>::ReverseWeight: Weight<ReverseWeight = A::Weight>,
{
    if !compat_symbols_rc(fst1.input_symbols(), fst2.input_symbols())
        || !compat_symbols_rc(fst1.output_symbols(), fst2.output_symbols())
    {
        return Err(OpenFstError::SymbolTable(
            "RandEquivalent: the two FSTs' symbol tables do not agree".into(),
        ));
    }

    let mut left: VectorFst<A> = copy_of(fst1);
    let mut right: VectorFst<A> = copy_of(fst2);
    connect(&mut left);
    connect(&mut right);
    arc_sort(&mut left, &ILabelCompare);
    arc_sort(&mut right, &ILabelCompare);

    let idempotent = A::Weight::properties() & IDEMPOTENT != 0;

    for _ in 0..npath {
        // Drawing from one or the other, so that a path only one of them has is
        // as likely to be tried.
        let mut path: VectorFst<A> = VectorFst::new();
        if rng.below(2) == 0 {
            rand_gen(&left, &mut path, rng, opts)?;
        } else {
            rand_gen(&right, &mut path, rng, opts)?;
        }

        // The drawn path, read as its input side alone and its output side
        // alone: composing one FST between the two keeps exactly what that FST
        // does with this path.
        let mut inputs: VectorFst<A> = copy_of(&path);
        let mut outputs: VectorFst<A> = copy_of(&path);
        project(&mut inputs, ProjectType::Input)?;
        project(&mut outputs, ProjectType::Output)?;

        let mut sums = [A::Weight::zero(), A::Weight::zero()];
        let mut give_up = false;
        for (index, fst) in [&left, &right].into_iter().enumerate() {
            let mut through: VectorFst<A> = VectorFst::new();
            compose(&inputs, fst, &mut through)?;
            arc_sort(&mut through, &OLabelCompare);
            let mut both: VectorFst<A> = VectorFst::new();
            compose(&through, &outputs, &mut both)?;
            // Over a semiring where a cycle does not settle, the sum of the
            // paths through one is not a number to compare.
            if !idempotent && both.properties(K_CYCLIC, true) & K_CYCLIC != 0 {
                give_up = true;
                break;
            }
            sums[index] = shortest_distance(&both, delta)?;
        }
        if give_up {
            continue;
        }
        if !sums[0].approx_equal(&sums[1], delta) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// As [`rand_equivalent`], with the defaults upstream uses.
pub fn rand_equivalent_default<A, F1, F2, S>(
    fst1: &F1,
    fst2: &F2,
    npath: usize,
    rng: &mut Rng,
    selector: S,
) -> Result<bool, OpenFstError>
where
    A: Arc,
    A::StateId: BiTableId,
    A::Weight: From<Log64Weight>,
    F1: Fst<A> + ExpandedFst<A>,
    F2: Fst<A> + ExpandedFst<A>,
    S: ArcSelector<A>,
    <A::Weight as Weight>::ReverseWeight: Weight<ReverseWeight = A::Weight>,
{
    rand_equivalent(
        fst1,
        fst2,
        npath,
        rng,
        &RandGenOptions::new(selector),
        SHORTEST_DELTA,
    )
}

/// A `VectorFst` holding what `fst` holds.
fn copy_of<A, F>(fst: &F) -> VectorFst<A>
where
    A: Arc,
    F: Fst<A> + ExpandedFst<A>,
{
    let mut out = VectorFst::new();
    out.add_states(fst.num_states());
    out.set_input_symbols(fst.input_symbols());
    out.set_output_symbols(fst.output_symbols());
    if let Some(start) = fst.start() {
        out.set_start(start);
    }
    for state in fst.states() {
        out.set_final(state, fst.final_weight(state));
        for arc in fst.arcs(state) {
            out.add_arc(state, arc);
        }
    }
    out.set_properties(fst.properties(K_FST_PROPERTIES, false), K_FST_PROPERTIES);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::randgen::UniformArcSelector;
    use crate::arc::StdArc;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::K_FST_PROPERTIES as _KFP;
    use crate::weights::float_weight::TropicalWeight;

    /// A deterministic acceptor over the given (string, weight) pairs.
    fn acceptor(strings: &[(&[i32], f32)]) -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        let start = fst.add_state();
        fst.set_start(start);
        for (labels, weight) in strings {
            let mut state = start;
            for label in *labels {
                let existing = fst.arcs(state).find(|arc| arc.ilabel() == *label);
                state = match existing {
                    Some(arc) => arc.nextstate(),
                    None => {
                        let next = fst.add_state();
                        fst.add_arc(
                            state,
                            StdArc::new(*label, *label, TropicalWeight::one(), next),
                        );
                        next
                    }
                };
            }
            fst.set_final(state, TropicalWeight(*weight));
        }
        fst.properties(_KFP, true);
        fst
    }

    fn agrees(fst1: &StdVectorFst, fst2: &StdVectorFst, npath: usize, seed: u64) -> bool {
        let mut rng = Rng::new(seed);
        rand_equivalent_default(fst1, fst2, npath, &mut rng, UniformArcSelector).unwrap()
    }

    #[test]
    fn an_fst_agrees_with_itself() {
        let fst = acceptor(&[(&[1, 2], 1.0), (&[3], 2.0)]);
        assert!(agrees(&fst, &fst, 50, 1));
    }

    /// Two FSTs built differently but saying the same thing agree.
    #[test]
    fn the_same_language_written_differently_agrees() {
        let a = acceptor(&[(&[1, 2], 1.0), (&[3], 2.0)]);
        let b = acceptor(&[(&[3], 2.0), (&[1, 2], 1.0)]);
        assert!(agrees(&a, &b, 50, 2));
    }

    /// A difference in the weights is found.
    #[test]
    fn a_difference_in_the_weights_is_found() {
        let a = acceptor(&[(&[1], 1.0)]);
        let b = acceptor(&[(&[1], 2.0)]);
        assert!(!agrees(&a, &b, 20, 3));
    }

    /// A string one has and the other does not is found, given enough draws.
    #[test]
    fn a_string_only_one_of_them_has_is_found() {
        let a = acceptor(&[(&[1], 0.0), (&[2], 0.0)]);
        let b = acceptor(&[(&[1], 0.0)]);
        // The draw has to land on the string only `a` has, which it will over
        // enough tries.
        assert!(
            (0..8).any(|seed| !agrees(&a, &b, 40, seed)),
            "no seed found the difference"
        );
    }

    /// Sampling proves nothing when nothing is sampled.
    #[test]
    fn no_draws_means_no_evidence() {
        let a = acceptor(&[(&[1], 1.0)]);
        let b = acceptor(&[(&[1], 2.0)]);
        assert!(
            agrees(&a, &b, 0, 4),
            "with no draws there is nothing to see"
        );
    }

    /// Symbol tables that disagree mean the question cannot be asked.
    #[test]
    fn symbol_tables_that_disagree_are_refused() {
        use crate::AtomicRc;
        use crate::symbol_table::SymbolTable;

        let mut ours = SymbolTable::new("ours");
        ours.add_symbol("a", 1);
        let mut theirs = SymbolTable::new("theirs");
        theirs.add_symbol("b", 1);

        let mut a = acceptor(&[(&[1], 0.0)]);
        a.set_input_symbols(Some(AtomicRc::new(ours)));
        let mut b = acceptor(&[(&[1], 0.0)]);
        b.set_input_symbols(Some(AtomicRc::new(theirs)));

        let mut rng = Rng::new(5);
        assert!(rand_equivalent_default(&a, &b, 10, &mut rng, UniformArcSelector).is_err());
    }
}
