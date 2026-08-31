//! Moving a transducer's epsilons to one end of each path.
//!
//! Port of OpenFst's `epsnormalize.h`. A transducer is *input
//! epsilon-normalized* when, along every path, no input label follows an input
//! epsilon, so the epsilons on that side have all been pushed to the end. That is
//! a canonical form: two transducers that transduce the same thing normalize to
//! the same shape, so they can be compared.
//!
//! > Mohri, M. 2002. Generic epsilon-removal and input epsilon-normalization
//! > algorithms for weighted transducers. *International Journal of Foundations
//! > of Computer Science* 13(1): 129-143.
//!
//! The work is done by moving one side into a gallic weight and removing
//! epsilons there: an arc whose input is epsilon then has nothing left to
//! consume, so removing it is exactly what pushes the other side's labels
//! forward.

use crate::algorithms::arc_map::{FromGallicMapper, ToGallicMapper, arc_map_to};
use crate::algorithms::connect::connect;
use crate::algorithms::factor_weight::{
    FactorIterator, FactorWeightOptions, GallicFactor, factor_weight,
};
use crate::algorithms::invert::invert;
use crate::algorithms::rmepsilon::rm_epsilon;
use crate::arc::{Arc, GallicArc};
use crate::error::OpenFstError;
use crate::fst::{ExpandedFst, Fst, MutableFst};
use crate::fsts::vector_fst::VectorFst;
use crate::properties::K_FST_PROPERTIES;
use crate::weight::{IdempotentWeight, Weight};
use crate::weights::string_weight::{GallicTypeMarker, GallicWeight};

/// Which side to push the epsilons off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EpsNormalizeType {
    /// No input label follows an input epsilon.
    #[default]
    Input,
    /// No output label follows an output epsilon.
    Output,
}

/// Rewrites `ifst` into `ofst` with its epsilons pushed to one end of every
/// path, moving the labels through the gallic type `gallic` names.
///
/// The type is handed over as a value: `GallicLeft`, `GallicRight`,
/// `GallicRestrict` or `GallicMin`. They are unit structs, so it costs nothing;
/// it exists so that the choice is inferred from an argument instead of being
/// spelled as `eps_normalize::<StdArc, GallicRight, _, _>(..)`.
///
/// SICADA-DIVERGE: upstream defaults the choice, through a two-argument
/// overload that picks `GALLIC`, the *union* gallic type, whose ⊕ keeps both
/// outputs and so can never fail. sicada models that type as
/// `GeneralGallicWeight` and the mappers here are not written against it yet
/// (see `check_representable`), so the default upstream offers cannot be
/// reproduced. Defaulting to a *different* one would be a silent change of
/// behaviour, since every other gallic type refuses two differing outputs where
/// upstream's does not, so the choice stays with the caller. When
/// `GeneralGallicWeight` is wired up, a defaulted entry point matching upstream
/// can be added.
pub fn eps_normalize<A, G, F1, F2>(
    ifst: &F1,
    ofst: &mut F2,
    normalize_type: EpsNormalizeType,
    _gallic: G,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: std::hash::Hash + Eq,
    G: GallicTypeMarker,
    F1: Fst<A> + ExpandedFst<A>,
    F2: MutableFst<A> + ExpandedFst<A>,
    GallicWeight<A::Label, A::Weight, G>: Weight + IdempotentWeight + std::hash::Hash + Eq,
{
    // Normalizing the output side is normalizing the input side of the
    // inverse, so only one direction has to be written.
    let mut source: VectorFst<A> = VectorFst::new();
    copy_into(ifst, &mut source);
    if normalize_type == EpsNormalizeType::Output {
        invert(&mut source)?;
    }
    // The symbol table of the side that is *not* being normalized survives; the
    // other side's labels move into the weights and back, and the table that
    // described them no longer describes anything.
    let symbols = match normalize_type {
        EpsNormalizeType::Input => ifst.output_symbols(),
        EpsNormalizeType::Output => ifst.input_symbols(),
    };

    let mut gfst: VectorFst<GallicArc<A, G>> = VectorFst::new();
    arc_map_to(&source, &mut gfst, &mut ToGallicMapper::<G>::new())?;

    // With the output side inside the weight, an input-epsilon arc consumes
    // nothing at all, so removing it carries the output labels forward onto
    // whatever comes next.
    //
    // Connecting is done afterwards rather than by `rm_epsilon`, so that a
    // weight that left the semiring can be seen before the states carrying it
    // are dropped as unreachable.
    rm_epsilon(&mut gfst, false)?;
    check_representable::<A, G>(&gfst)?;
    connect(&mut gfst);

    // A weight may now hold several labels, which one arc cannot carry.
    let mut factored: VectorFst<GallicArc<A, G>> = VectorFst::new();
    factor_weight(
        &gfst,
        &mut factored,
        GallicFactor::new,
        &FactorWeightOptions::<A::Label>::default(),
    );

    let mut mapper = FromGallicMapper::<A::Label, G>::new();
    arc_map_to(&factored, ofst, &mut mapper)?;
    if mapper.error() {
        return Err(OpenFstError::InvalidOperation(
            "EpsNormalize: a weight came out that no single arc can carry".into(),
        ));
    }
    ofst.set_output_symbols(symbols);
    if normalize_type == EpsNormalizeType::Output {
        invert(ofst)?;
    }
    Ok(())
}

/// Reports a weight that left the semiring while the epsilons were removed.
///
/// Two epsilon paths between the same pair of states are combined with ⊕. Over
/// a gallic type whose string half is restricted, ⊕ of two *different* label
/// sequences is not a weight at all, which is exactly the case where the
/// transducer maps one input to two outputs.
///
/// SICADA-DIVERGE: upstream's default is the union gallic type, whose ⊕ keeps
/// both outputs, so it has nothing to refuse. sicada models that type as
/// `GeneralGallicWeight` rather than as one more `GallicWeight` marker, and the
/// mappers here are written against the latter, so it cannot be plugged in
/// yet. Rather than hand back the empty FST the non-member weights collapse
/// into, the situation is named.
fn check_representable<A, G>(gfst: &VectorFst<GallicArc<A, G>>) -> Result<(), OpenFstError>
where
    A: Arc,
    G: GallicTypeMarker,
    GallicWeight<A::Label, A::Weight, G>: Weight,
{
    for state in gfst.states() {
        if !gfst.final_weight(state).is_member() {
            return Err(not_functional(state));
        }
        for arc in gfst.arcs(state) {
            if !arc.weight().is_member() {
                return Err(not_functional(state));
            }
        }
    }
    Ok(())
}

fn not_functional<S: std::fmt::Debug>(state: S) -> OpenFstError {
    OpenFstError::InvalidOperation(format!(
        "EpsNormalize: at state {state:?}, two epsilon paths give different outputs, which the \
         gallic type chosen cannot combine. That takes the union gallic weight, which is not \
         wired up yet; a functional transducer does not run into it."
    ))
}

/// Copies an FST state for state.
fn copy_into<A, F1, F2>(ifst: &F1, ofst: &mut F2)
where
    A: Arc,
    F1: Fst<A> + ExpandedFst<A>,
    F2: MutableFst<A>,
{
    ofst.delete_all_states();
    ofst.add_states(ifst.num_states());
    ofst.set_input_symbols(ifst.input_symbols());
    ofst.set_output_symbols(ifst.output_symbols());
    if let Some(start) = ifst.start() {
        ofst.set_start(start);
    }
    for state in ifst.states() {
        ofst.set_final(state, ifst.final_weight(state));
        for arc in ifst.arcs(state) {
            ofst.add_arc(state, arc);
        }
    }
    ofst.set_properties(ifst.properties(K_FST_PROPERTIES, false), K_FST_PROPERTIES);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{Rng, random_acyclic_fst, string_weights, visible_paths};
    use crate::arc::StdArc;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::K_FST_PROPERTIES as _KP;
    use crate::weights::float_weight::TropicalWeight;
    use crate::weights::string_weight::GallicRestrict;

    fn normalized(fst: &StdVectorFst, which: EpsNormalizeType) -> StdVectorFst {
        let mut out = StdVectorFst::new();
        eps_normalize(fst, &mut out, which, GallicRestrict).unwrap();
        out
    }

    /// What the FST transduces, epsilons already invisible.
    fn language(fst: &StdVectorFst) -> Vec<(Vec<i32>, Vec<i32>, String)> {
        string_weights(visible_paths(fst, 16))
    }

    /// Whether no label follows an epsilon on the named side, along any path.
    fn is_normalized(fst: &StdVectorFst, which: EpsNormalizeType) -> bool {
        // Walk every path, watching the side in question.
        let mut stack = vec![(fst.start(), false, 0usize)];
        while let Some((state, seen_epsilon, depth)) = stack.pop() {
            let Some(state) = state else { continue };
            if depth > 16 {
                continue;
            }
            for arc in fst.arcs(state) {
                let label = match which {
                    EpsNormalizeType::Input => arc.ilabel(),
                    EpsNormalizeType::Output => arc.olabel(),
                };
                if label == 0 {
                    stack.push((Some(arc.nextstate()), true, depth + 1));
                } else {
                    if seen_epsilon {
                        return false;
                    }
                    stack.push((Some(arc.nextstate()), false, depth + 1));
                }
            }
        }
        true
    }

    /// An input epsilon followed by an input label is moved.
    #[test]
    fn an_input_epsilon_is_pushed_past_what_follows_it() {
        // 0 -eps:7-> 1 -2:8-> 2, final. The output 7 has to move onto the arc
        // that reads 2, so that the epsilon comes last.
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(0, 7, TropicalWeight::one(), 1));
        fst.add_arc(1, StdArc::new(2, 8, TropicalWeight::one(), 2));
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        assert!(!is_normalized(&fst, EpsNormalizeType::Input));
        let out = normalized(&fst, EpsNormalizeType::Input);
        assert!(is_normalized(&out, EpsNormalizeType::Input));
        assert_eq!(language(&out), language(&fst));
    }

    /// The output side can be normalized instead.
    #[test]
    fn the_output_side_can_be_normalized() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(7, 0, TropicalWeight::one(), 1));
        fst.add_arc(1, StdArc::new(8, 2, TropicalWeight::one(), 2));
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        assert!(!is_normalized(&fst, EpsNormalizeType::Output));
        let out = normalized(&fst, EpsNormalizeType::Output);
        assert!(is_normalized(&out, EpsNormalizeType::Output));
        assert_eq!(language(&out), language(&fst));
    }

    /// An FST that is already normalized keeps saying what it said.
    #[test]
    fn an_already_normalized_fst_is_unchanged_in_what_it_says() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 7, TropicalWeight(1.0), 1));
        fst.add_arc(1, StdArc::new(0, 8, TropicalWeight(2.0), 2));
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        assert!(is_normalized(&fst, EpsNormalizeType::Input));
        let out = normalized(&fst, EpsNormalizeType::Input);
        assert!(is_normalized(&out, EpsNormalizeType::Input));
        assert_eq!(language(&out), language(&fst));
    }

    /// Whatever the transducer, normalizing leaves it normalized and saying the
    /// same thing.
    #[test]
    fn normalizing_keeps_the_transduction_and_normalizes() {
        let mut rng = Rng::new(0x00E7_5000_u64);
        let mut checked = 0;
        let mut refused = 0;
        for round in 0..100 {
            let mut fst = random_acyclic_fst(&mut rng, 5);
            // Make it a transducer with some input epsilons.
            let states: Vec<i32> = fst.states().collect();
            for state in states {
                fst.mutate_arcs(state, |arc| {
                    let ilabel = if arc.ilabel() % 3 == 1 {
                        0
                    } else {
                        arc.ilabel()
                    };
                    *arc = StdArc::new(ilabel, arc.ilabel() + 10, *arc.weight(), arc.nextstate());
                });
            }
            fst.properties(K_FST_PROPERTIES, true);
            if language(&fst).is_empty() {
                continue;
            }
            checked += 1;

            let mut out = StdVectorFst::new();
            match eps_normalize(&fst, &mut out, EpsNormalizeType::Input, GallicRestrict) {
                Ok(()) => {}
                // A non-functional transducer needs the union gallic weight,
                // which is not wired up; those rounds are refused by name.
                Err(err) => {
                    assert!(format!("{err}").contains("different outputs"), "{err}");
                    refused += 1;
                    continue;
                }
            }
            assert!(
                is_normalized(&out, EpsNormalizeType::Input),
                "round {round}"
            );
            assert_eq!(language(&out), language(&fst), "round {round}");
        }
        assert!(
            checked > 30,
            "only {checked} FSTs said anything ({refused} refused)"
        );
    }

    /// An empty FST normalizes to an empty one.
    #[test]
    fn an_empty_fst_normalizes_to_nothing() {
        let out = normalized(&StdVectorFst::new(), EpsNormalizeType::Input);
        assert_eq!(out.num_states(), 0);
    }

    /// A transducer that maps one input to two outputs cannot be normalized
    /// through a restricted gallic weight, and says so rather than coming back
    /// empty.
    #[test]
    fn a_non_functional_transducer_is_refused_by_name() {
        let mut fst = StdVectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.set_final(0, TropicalWeight::one());
        fst.add_arc(0, StdArc::new(0, 11, TropicalWeight(2.0), 1));
        fst.set_final(1, TropicalWeight::one());
        fst.properties(_KP, true);
        // The empty input maps to both "" and "11".
        assert_eq!(language(&fst).len(), 2);

        let mut out = StdVectorFst::new();
        let err =
            eps_normalize(&fst, &mut out, EpsNormalizeType::Input, GallicRestrict).unwrap_err();
        assert!(format!("{err}").contains("different outputs"), "{err}");
    }
}
