//! The concatenation of two FSTs: what one accepts followed by what the other
//! does.
//!
//! Port of OpenFst's `concat.h`.

use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::error::OpenFstError;
use crate::fst::{ExpandedFst, Fst, MutableFst};
use crate::properties::{K_ERROR, K_FST_PROPERTIES, concat_properties};
use crate::symbol_table::compat_symbols_rc;
use crate::weight::Weight;

/// Appends `fst2` to `fst1`.
///
/// SICADA-DIVERGE: as with [`union`](super::union::union), incompatible symbol
/// tables are an error rather than a flag on the result.
pub fn concat<A, F1, F2>(fst1: &mut F1, fst2: &F2) -> Result<(), OpenFstError>
where
    A: Arc,
    F1: MutableFst<A> + ExpandedFst<A>,
    F2: Fst<A>,
{
    if !compat_symbols_rc(fst1.input_symbols(), fst2.input_symbols())
        || !compat_symbols_rc(fst1.output_symbols(), fst2.output_symbols())
    {
        fst1.set_properties(K_ERROR, K_ERROR);
        return Err(OpenFstError::SymbolTable(
            "Concat: the two FSTs' symbol tables do not agree".into(),
        ));
    }

    let props1 = fst1.properties(K_FST_PROPERTIES, false);
    let props2 = fst2.properties(K_FST_PROPERTIES, false);
    let Some(_) = fst1.start() else {
        // The first FST accepts nothing, so nothing followed by anything is
        // still nothing.
        if props2 & K_ERROR != 0 {
            fst1.set_properties(K_ERROR, K_ERROR);
        }
        return Ok(());
    };

    let numstates1 = fst1.num_states();
    if let Some(numstates2) = fst2.num_states_if_known() {
        fst1.reserve_states(numstates1 + numstates2);
    }
    for state2 in fst2.states() {
        let state1 = fst1.add_state();
        fst1.set_final(state1, fst2.final_weight(state2));
        for arc in fst2.arcs(state2) {
            let shifted = A::StateId::from_usize(arc.nextstate().as_usize() + numstates1);
            fst1.add_arc(
                state1,
                A::new(arc.ilabel(), arc.olabel(), arc.weight().clone(), shifted),
            );
        }
    }

    // Where the first FST used to end, it now carries on into the second, and
    // its final weight moves onto the arc that does so.
    let epsilon = A::Label::epsilon();
    let zero = A::Weight::zero();
    let start2 = fst2.start();
    for index in 0..numstates1 {
        let state = A::StateId::from_usize(index);
        let weight = fst1.final_weight(state);
        if weight == zero {
            continue;
        }
        fst1.set_final(state, zero.clone());
        if let Some(start2) = start2 {
            let shifted = A::StateId::from_usize(start2.as_usize() + numstates1);
            fst1.add_arc(state, A::new(epsilon, epsilon, weight, shifted));
        }
    }
    if start2.is_some() {
        fst1.set_properties(concat_properties(props1, props2, false), K_FST_PROPERTIES);
    }
    Ok(())
}

/// Prepends `fst1` to `fst2`.
///
/// Kept apart from [`concat`](fn@concat) because which FST is the mutable one decides
/// which one's states have to be renumbered, and renumbering the shorter of the
/// two is the point of having both.
pub fn concat_onto<A, F1, F2>(fst1: &F1, fst2: &mut F2) -> Result<(), OpenFstError>
where
    A: Arc,
    F1: Fst<A>,
    F2: MutableFst<A> + ExpandedFst<A>,
{
    if !compat_symbols_rc(fst1.input_symbols(), fst2.input_symbols())
        || !compat_symbols_rc(fst1.output_symbols(), fst2.output_symbols())
    {
        fst2.set_properties(K_ERROR, K_ERROR);
        return Err(OpenFstError::SymbolTable(
            "Concat: the two FSTs' symbol tables do not agree".into(),
        ));
    }

    let props1 = fst1.properties(K_FST_PROPERTIES, false);
    let props2 = fst2.properties(K_FST_PROPERTIES, false);
    let Some(start2) = fst2.start() else {
        if props1 & K_ERROR != 0 {
            fst2.set_properties(K_ERROR, K_ERROR);
        }
        return Ok(());
    };

    let numstates2 = fst2.num_states();
    if let Some(numstates1) = fst1.num_states_if_known() {
        fst2.reserve_states(numstates2 + numstates1);
    }
    let epsilon = A::Label::epsilon();
    let zero = A::Weight::zero();
    for state1 in fst1.states() {
        let state = fst2.add_state();
        let weight = fst1.final_weight(state1);
        if weight != zero {
            fst2.add_arc(state, A::new(epsilon, epsilon, weight, start2));
        }
        for arc in fst1.arcs(state1) {
            let shifted = A::StateId::from_usize(arc.nextstate().as_usize() + numstates2);
            fst2.add_arc(
                state,
                A::new(arc.ilabel(), arc.olabel(), arc.weight().clone(), shifted),
            );
        }
    }

    match fst1.start() {
        Some(start1) => {
            fst2.set_start(A::StateId::from_usize(start1.as_usize() + numstates2));
            fst2.set_properties(concat_properties(props1, props2, false), K_FST_PROPERTIES);
        }
        None => {
            // The first FST accepts nothing, so neither does the result; a
            // start state with no way out says exactly that.
            let dead = fst2.add_state();
            fst2.set_start(dead);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{Rng, random_acyclic_fst, visible_paths};
    use crate::arc::StdArc;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::K_FST_PROPERTIES;
    use crate::weights::float_weight::TropicalWeight;

    fn chain(labels: &[i32], weight: f32) -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        let mut state = fst.add_state();
        fst.set_start(state);
        for label in labels {
            let next = fst.add_state();
            fst.add_arc(
                state,
                StdArc::new(*label, *label, TropicalWeight::one(), next),
            );
            state = next;
        }
        fst.set_final(state, TropicalWeight(weight));
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    /// The string/weight pairs an FST defines, keeping only the lightest way to
    /// each string.
    fn language(fst: &StdVectorFst) -> Vec<(Vec<i32>, String)> {
        let mut best: std::collections::BTreeMap<Vec<i32>, f32> = std::collections::BTreeMap::new();
        for (ilabels, _, weight) in visible_paths(fst, 16) {
            best.entry(ilabels)
                .and_modify(|at| *at = at.min(weight.value()))
                .or_insert(weight.value());
        }
        best.into_iter()
            .map(|(s, w)| (s, format!("{w:.4}")))
            .collect()
    }

    /// Everything the first accepts, followed by everything the second does,
    /// with the weights multiplied.
    #[test]
    fn concatenation_joins_the_two_languages() {
        let mut fst = chain(&[1, 2], 1.0);
        concat(&mut fst, &chain(&[3], 2.0)).unwrap();
        assert_eq!(language(&fst), vec![(vec![1, 2, 3], "3.0000".to_string())]);
    }

    /// Prepending gives the same language as appending the other way round.
    #[test]
    fn prepending_and_appending_agree() {
        let mut appended = chain(&[1, 2], 1.0);
        concat(&mut appended, &chain(&[3, 4], 2.0)).unwrap();

        let mut prepended = chain(&[3, 4], 2.0);
        concat_onto(&chain(&[1, 2], 1.0), &mut prepended).unwrap();

        assert_eq!(language(&prepended), language(&appended));
    }

    /// Concatenating with something that accepts nothing accepts nothing.
    #[test]
    fn concatenating_with_nothing_accepts_nothing() {
        let mut fst = chain(&[1], 0.0);
        concat(&mut fst, &StdVectorFst::new()).unwrap();
        assert!(language(&fst).is_empty(), "{:?}", language(&fst));

        let mut fst = StdVectorFst::new();
        concat_onto(&chain(&[1], 0.0), &mut fst).unwrap();
        assert!(language(&fst).is_empty(), "{:?}", language(&fst));

        let mut fst = chain(&[1], 0.0);
        concat_onto(&StdVectorFst::new(), &mut fst).unwrap();
        assert!(language(&fst).is_empty(), "{:?}", language(&fst));
    }

    /// The concatenation of two languages is every first string followed by
    /// every second, over random FSTs.
    #[test]
    fn the_concatenation_is_every_pairing_of_the_two() {
        let mut rng = Rng::new(0x00CA_7CA7_u64);
        for round in 0..200 {
            let fst1 = random_acyclic_fst(&mut rng, 4);
            let fst2 = random_acyclic_fst(&mut rng, 4);

            let mut want: std::collections::BTreeMap<Vec<i32>, f32> =
                std::collections::BTreeMap::new();
            for (left, lweight) in language(&fst1) {
                for (right, rweight) in language(&fst2) {
                    let mut joined = left.clone();
                    joined.extend(right);
                    let weight: f32 =
                        lweight.parse::<f32>().unwrap() + rweight.parse::<f32>().unwrap();
                    want.entry(joined)
                        .and_modify(|at| *at = at.min(weight))
                        .or_insert(weight);
                }
            }
            let want: Vec<(Vec<i32>, String)> = want
                .into_iter()
                .map(|(s, w)| (s, format!("{w:.4}")))
                .collect();

            let mut got = fst1.clone();
            concat(&mut got, &fst2).unwrap();
            assert_eq!(language(&got), want, "round {round}");

            let mut onto = fst2.clone();
            concat_onto(&fst1, &mut onto).unwrap();
            assert_eq!(language(&onto), want, "round {round}, prepended");
        }
    }
}
