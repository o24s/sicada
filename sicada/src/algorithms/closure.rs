//! Repeating what an FST accepts, any number of times.
//!
//! Port of OpenFst's `closure.h`.

use crate::arc::{Arc, ArcLabel};
use crate::fst::{ExpandedFst, MutableFst};
use crate::properties::{K_FST_PROPERTIES, closure_properties};
use crate::weight::Weight;

/// Whether the empty string is one of the repetitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosureType {
    /// Zero times or more, so the empty string is accepted.
    Star,
    /// One time or more.
    Plus,
}

/// Lets `fst` repeat what it accepts.
///
/// SICADA-DIVERGE: upstream sends the arc it adds at each final state to
/// `fst->Start()` without checking that there is one. An FST with final states
/// but no start state, which nothing forbids and which a half-built FST has,
/// gets arcs pointing at `kNoStateId`. Nothing is added here when there is
/// nowhere to go back to.
pub fn closure<A, F>(fst: &mut F, closure_type: ClosureType)
where
    A: Arc,
    F: MutableFst<A> + ExpandedFst<A>,
{
    let props = fst.properties(K_FST_PROPERTIES, false);
    let start = fst.start();
    let epsilon = A::Label::epsilon();
    let zero = A::Weight::zero();

    // Every way of finishing becomes a way of going round again, carrying the
    // weight of having finished.
    if let Some(start) = start {
        let states: Vec<A::StateId> = fst.states().collect();
        for state in states {
            let weight = fst.final_weight(state);
            if weight != zero {
                fst.add_arc(state, A::new(epsilon, epsilon, weight, start));
            }
        }
    }

    if closure_type == ClosureType::Star {
        // A new start state that is also final: the empty string, plus a way
        // into the FST proper.
        fst.reserve_states(fst.num_states() + 1);
        let nstart = fst.add_state();
        fst.set_start(nstart);
        fst.set_final(nstart, A::Weight::one());
        if let Some(start) = start {
            fst.add_arc(nstart, A::new(epsilon, epsilon, A::Weight::one(), start));
        }
    }

    fst.set_properties(
        closure_properties(props, closure_type == ClosureType::Star, false),
        K_FST_PROPERTIES,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::fst::Fst as _;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::K_FST_PROPERTIES;
    use crate::weight::Weight;
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

    /// The strings an FST accepts, up to a length, with their weights.
    fn language(fst: &StdVectorFst, max_len: usize) -> Vec<(Vec<i32>, String)> {
        use crate::algorithms::test_support::visible_paths;
        let mut best: std::collections::BTreeMap<Vec<i32>, f32> = std::collections::BTreeMap::new();
        for (ilabels, _, weight) in visible_paths(fst, max_len) {
            best.entry(ilabels)
                .and_modify(|at| *at = at.min(weight.value()))
                .or_insert(weight.value());
        }
        best.into_iter()
            .map(|(s, w)| (s, format!("{w:.4}")))
            .collect()
    }

    /// Star accepts the empty string and any number of repetitions, each
    /// costing what one repetition costs.
    ///
    /// The bound is on arcs, not on labels, and going round again takes an
    /// epsilon arc as well as the label, so it has to be generous enough for
    /// the repetitions being looked for.
    #[test]
    fn star_accepts_the_empty_string_and_any_number_of_repetitions() {
        let mut fst = chain(&[1], 2.0);
        closure(&mut fst, ClosureType::Star);
        let language = language(&fst, 9);

        for (repetitions, weight) in [(0, 0.0), (1, 2.0), (2, 4.0), (3, 6.0)] {
            assert!(
                language.contains(&(vec![1; repetitions], format!("{weight:.4}"))),
                "{repetitions} repetitions missing from {language:?}"
            );
        }
        // And nothing else: every string is a run of 1s costing 2 each.
        for (string, weight) in &language {
            assert!(string.iter().all(|label| *label == 1), "{string:?}");
            assert_eq!(*weight, format!("{:.4}", 2.0 * string.len() as f32));
        }
    }

    /// Plus is the same without the empty string.
    #[test]
    fn plus_needs_at_least_one_repetition() {
        let mut fst = chain(&[1], 2.0);
        closure(&mut fst, ClosureType::Plus);
        let strings: Vec<Vec<i32>> = language(&fst, 6).into_iter().map(|(s, _)| s).collect();
        assert!(!strings.contains(&vec![]), "{strings:?}");
        assert!(strings.contains(&vec![1]));
        assert!(strings.contains(&vec![1, 1]));
    }

    /// The star of nothing is the empty string, and the plus of nothing is
    /// nothing.
    #[test]
    fn the_closure_of_nothing() {
        let mut star = StdVectorFst::new();
        closure(&mut star, ClosureType::Star);
        assert_eq!(language(&star, 4), vec![(vec![], "0.0000".to_string())]);

        let mut plus = StdVectorFst::new();
        closure(&mut plus, ClosureType::Plus);
        assert!(language(&plus, 4).is_empty());
    }

    /// An FST with final states but no start state has nowhere to go back to,
    /// so no arc is added. Upstream sends one to `kNoStateId`.
    #[test]
    fn a_final_state_with_no_start_state_gets_no_arc_back() {
        let mut fst = StdVectorFst::new();
        let state = fst.add_state();
        fst.set_final(state, TropicalWeight::one());
        assert_eq!(fst.start(), None);

        closure(&mut fst, ClosureType::Plus);
        for state in fst.states() {
            for arc in fst.arcs(state) {
                assert!(
                    (arc.nextstate() as usize) < fst.num_states(),
                    "arc to state {} of {}",
                    arc.nextstate(),
                    fst.num_states()
                );
            }
        }
    }

    /// Repeating a language that already had several strings gives every
    /// sequence of them.
    #[test]
    fn star_gives_every_sequence_of_what_was_accepted() {
        let mut fst = StdVectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(0, StdArc::new(2, 2, TropicalWeight::one(), 1));
        fst.set_final(1, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        closure(&mut fst, ClosureType::Star);
        let strings: Vec<Vec<i32>> = language(&fst, 6).into_iter().map(|(s, _)| s).collect();
        for want in [vec![], vec![1], vec![2], vec![1, 2], vec![2, 1], vec![1, 1]] {
            assert!(strings.contains(&want), "{want:?} missing from {strings:?}");
        }
    }
}
