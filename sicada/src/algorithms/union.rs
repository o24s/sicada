//! The union of two FSTs: what either one accepts.
//!
//! Port of OpenFst's `union.h`.

use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::error::OpenFstError;
use crate::fst::{ExpandedFst, Fst, MutableFst};
use crate::properties::{
    K_COPY_PROPERTIES, K_ERROR, K_FST_PROPERTIES, K_INITIAL_ACYCLIC, union_properties,
};
use crate::symbol_table::compat_symbols_rc;
use crate::weight::Weight;

/// Adds everything `fst2` accepts to `fst1`.
///
/// SICADA-DIVERGE: upstream reports incompatible symbol tables by setting
/// `kError` on the first FST and returning, leaving the caller to notice. It is
/// an error here: the two FSTs disagree about what their labels mean, and no
/// union of them says anything.
pub fn union<A, F1, F2>(fst1: &mut F1, fst2: &F2) -> Result<(), OpenFstError>
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
            "Union: the two FSTs' symbol tables do not agree".into(),
        ));
    }

    let numstates1 = fst1.num_states();
    let initial_acyclic1 = fst1.properties(K_INITIAL_ACYCLIC, false) & K_INITIAL_ACYCLIC != 0;
    let props1 = fst1.properties(K_FST_PROPERTIES, false);
    let props2 = fst2.properties(K_FST_PROPERTIES, false);

    let Some(start2) = fst2.start() else {
        // The second FST accepts nothing, so the union is the first.
        if props2 & K_ERROR != 0 {
            fst1.set_properties(K_ERROR, K_ERROR);
        }
        return Ok(());
    };
    if let Some(numstates2) = fst2.num_states_if_known() {
        fst1.reserve_states(numstates1 + numstates2 + usize::from(!initial_acyclic1));
    }

    // The second FST is copied in with its states shifted past the first's.
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

    let shifted_start2 = A::StateId::from_usize(start2.as_usize() + numstates1);
    let Some(start1) = fst1.start() else {
        // The first FST accepted nothing, so the union is the second.
        fst1.set_start(shifted_start2);
        fst1.set_properties(props2, K_COPY_PROPERTIES);
        return Ok(());
    };

    let epsilon = A::Label::epsilon();
    if initial_acyclic1 {
        // Nothing comes back to the start state, so a second way out of it can
        // be added without changing what the first FST accepts.
        fst1.add_arc(
            start1,
            A::new(epsilon, epsilon, A::Weight::one(), shifted_start2),
        );
    } else {
        // Something re-enters the start state, so an arc added there would be
        // reachable from inside the first FST. A new start state is needed.
        let nstart = fst1.add_state();
        fst1.set_start(nstart);
        fst1.add_arc(nstart, A::new(epsilon, epsilon, A::Weight::one(), start1));
        fst1.add_arc(
            nstart,
            A::new(epsilon, epsilon, A::Weight::one(), shifted_start2),
        );
    }
    fst1.set_properties(union_properties(props1, props2, false), K_FST_PROPERTIES);
    Ok(())
}

/// Adds everything each of `fsts2` accepts to `fst1`.
pub fn union_all<A, F1, F2>(fst1: &mut F1, fsts2: &[&F2]) -> Result<(), OpenFstError>
where
    A: Arc,
    F1: MutableFst<A> + ExpandedFst<A>,
    F2: Fst<A>,
{
    // One extra in case the first FST has a cycle through its start state.
    let total: usize = fsts2.iter().map(|fst| fst.count_states()).sum();
    fst1.reserve_states(1 + fst1.num_states() + total);
    for fst2 in fsts2 {
        union(fst1, *fst2)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AtomicRc;
    use crate::algorithms::test_support::{Rng, random_acyclic_fst, visible_paths};
    use crate::arc::StdArc;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::K_FST_PROPERTIES;
    use crate::symbol_table::SymbolTable;
    use crate::weights::float_weight::TropicalWeight;

    /// A linear acceptor over `labels`, final at `weight`.
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

    /// The string/weight pairs an FST defines, with epsilons dropped.
    fn language(fst: &StdVectorFst) -> Vec<(Vec<i32>, String)> {
        let mut out: Vec<(Vec<i32>, String)> = visible_paths(fst, 16)
            .into_iter()
            .map(|(ilabels, _, weight)| (ilabels, format!("{:.4}", weight.value())))
            .collect();
        out.sort();
        out
    }

    #[test]
    fn the_union_accepts_what_either_accepts() {
        let mut fst = chain(&[1, 2], 1.0);
        union(&mut fst, &chain(&[3], 2.0)).unwrap();
        assert_eq!(
            language(&fst),
            vec![
                (vec![1, 2], "1.0000".to_string()),
                (vec![3], "2.0000".to_string())
            ]
        );
    }

    /// Unioning with an FST that accepts nothing changes nothing.
    #[test]
    fn a_union_with_nothing_is_the_original() {
        let before = language(&chain(&[1, 2], 1.0));
        let mut fst = chain(&[1, 2], 1.0);
        union(&mut fst, &StdVectorFst::new()).unwrap();
        assert_eq!(language(&fst), before);
    }

    /// Unioning into an FST that accepts nothing gives the other one.
    #[test]
    fn a_union_into_nothing_is_the_other() {
        let mut fst = StdVectorFst::new();
        let other = chain(&[4, 5], 3.0);
        union(&mut fst, &other).unwrap();
        assert_eq!(language(&fst), language(&other));
    }

    /// A cycle back into the start state means an arc added there would be
    /// reachable from inside, so a new start state is needed.
    #[test]
    fn a_cycle_through_the_start_state_forces_a_new_one() {
        let mut cyclic = StdVectorFst::new();
        for _ in 0..2 {
            cyclic.add_state();
        }
        cyclic.set_start(0);
        cyclic.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        cyclic.add_arc(1, StdArc::new(2, 2, TropicalWeight::one(), 0));
        cyclic.set_final(1, TropicalWeight::one());
        cyclic.properties(K_FST_PROPERTIES, true);

        let before = cyclic.start();
        let mut fst = cyclic.clone();
        union(&mut fst, &chain(&[9], 0.0)).unwrap();
        assert_ne!(fst.start(), before, "the start state was replaced");

        // And 9 is not something the cyclic part can reach.
        let strings: Vec<Vec<i32>> = language(&fst).into_iter().map(|(s, _)| s).collect();
        assert!(strings.contains(&vec![9]));
        assert!(!strings.iter().any(|s| s.len() > 1 && s.contains(&9)));
    }

    /// Union is what set union is: over random FSTs, the result's language is
    /// the union of the two.
    #[test]
    fn the_union_is_the_union_of_the_two_languages() {
        let mut rng = Rng::new(0x0011_0110_u64);
        for round in 0..200 {
            let fst1 = random_acyclic_fst(&mut rng, 5);
            let fst2 = random_acyclic_fst(&mut rng, 5);
            let mut want: Vec<(Vec<i32>, String)> = language(&fst1);
            want.extend(language(&fst2));
            // Over the tropical semiring a string in both keeps the lighter.
            want.sort();
            let mut merged: std::collections::BTreeMap<Vec<i32>, f32> =
                std::collections::BTreeMap::new();
            for (string, weight) in want {
                let weight: f32 = weight.parse().unwrap();
                merged
                    .entry(string)
                    .and_modify(|best| *best = best.min(weight))
                    .or_insert(weight);
            }
            let want: Vec<(Vec<i32>, String)> = merged
                .into_iter()
                .map(|(s, w)| (s, format!("{w:.4}")))
                .collect();

            let mut got = fst1.clone();
            union(&mut got, &fst2).unwrap();
            let mut merged: std::collections::BTreeMap<Vec<i32>, f32> =
                std::collections::BTreeMap::new();
            for (string, weight) in language(&got) {
                let weight: f32 = weight.parse().unwrap();
                merged
                    .entry(string)
                    .and_modify(|best| *best = best.min(weight))
                    .or_insert(weight);
            }
            let got: Vec<(Vec<i32>, String)> = merged
                .into_iter()
                .map(|(s, w)| (s, format!("{w:.4}")))
                .collect();

            assert_eq!(got, want, "round {round}");
        }
    }

    /// Two FSTs whose symbol tables disagree do not agree on what their labels
    /// mean, so there is no union of them.
    #[test]
    fn symbol_tables_that_disagree_are_refused() {
        let mut ours = SymbolTable::new("ours");
        ours.add_symbol("a", 1);
        let mut theirs = SymbolTable::new("theirs");
        theirs.add_symbol("b", 1);

        let mut fst1 = chain(&[1], 0.0);
        fst1.set_input_symbols(Some(AtomicRc::new(ours)));
        let mut fst2 = chain(&[1], 0.0);
        fst2.set_input_symbols(Some(AtomicRc::new(theirs)));

        let err = union(&mut fst1, &fst2).unwrap_err();
        assert!(format!("{err}").contains("symbol tables"), "{err}");
    }

    #[test]
    fn unioning_several_at_once_is_unioning_each() {
        let parts = [chain(&[1], 0.0), chain(&[2], 1.0), chain(&[3], 2.0)];
        let refs: Vec<&StdVectorFst> = parts.iter().collect();

        let mut all = StdVectorFst::new();
        union_all(&mut all, &refs).unwrap();

        let mut one_by_one = StdVectorFst::new();
        for part in &parts {
            union(&mut one_by_one, part).unwrap();
        }
        assert_eq!(language(&all), language(&one_by_one));
        assert_eq!(language(&all).len(), 3);
    }
}
