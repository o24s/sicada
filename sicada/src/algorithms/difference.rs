//! What one acceptor accepts and another does not.
//!
//! Port of OpenFst's `difference.h`. The difference is the intersection with
//! the complement: `A - B = A ∩ B'`.
//!
//! The complement of an unweighted deterministic acceptor is that acceptor with
//! a state added standing for "somewhere B could not go", which every label it
//! has no arc for leads to. Composition reaches that state through a ρ label,
//! which stands for "any label the state has no arc for", and that is why it
//! takes a matcher that understands ρ to read it.

use crate::algorithms::compose::compose_with;
use crate::algorithms::compose_filter::SequenceComposeFilter;
use crate::arc::Arc;
use crate::data_structures::bi_table::BiTableId;
use crate::error::OpenFstError;
use crate::fst::{ExpandedFst, Fst, MatchType, MutableFst};
use crate::fsts::complement_fst::{ComplementFst, RhoLabel};
use crate::matcher::{HashMatcher, Matcher, RhoMatcher, SortedMatcher};
use crate::properties::{K_ACCEPTOR, K_I_DETERMINISTIC, K_NO_EPSILONS, K_UNWEIGHTED};

/// The acceptor accepting what `fst1` accepts and `fst2` does not.
///
/// `fst2` has to be an unweighted epsilon-free deterministic acceptor, since
/// only such an FST can be complemented.
pub fn difference<A, F1, F2, FO>(fst1: &F1, fst2: &F2, ofst: &mut FO) -> Result<(), OpenFstError>
where
    A: Arc,
    A::StateId: BiTableId,
    A::Label: RhoLabel,
    F1: Fst<A> + ExpandedFst<A>,
    F2: Fst<A> + ExpandedFst<A> + Clone,
    FO: MutableFst<A> + ExpandedFst<A>,
{
    if fst1.properties(K_ACCEPTOR, true) & K_ACCEPTOR == 0 {
        return Err(OpenFstError::InvalidOperation(
            "Difference: the 1st argument is not an acceptor".into(),
        ));
    }
    let required = K_ACCEPTOR | K_I_DETERMINISTIC | K_NO_EPSILONS | K_UNWEIGHTED;
    if fst2.properties(required, true) != required {
        return Err(OpenFstError::InvalidOperation(
            "Difference: the 2nd argument is not an unweighted epsilon-free deterministic \
             acceptor, so it cannot be complemented"
                .into(),
        ));
    }

    // The matcher over the first FST searches by output label, which takes it
    // sorted that way.
    let left = crate::algorithms::compose::sorted_copy(fst1, true);
    let complement = ComplementFst::new(fst2.clone())?;
    // The complement's "anything else" arcs carry ρ, so the matcher over it has
    // to know that ρ stands for whatever label is asked for.
    let matcher1 = SortedMatcher::new(&left, MatchType::Output)?;
    let matcher2 = RhoMatcher::<HashMatcher<ComplementFst<A, F2>, A>, A>::new_with_options(
        &complement,
        MatchType::Input,
        <A::Label as RhoLabel>::rho_label(),
        crate::matcher::MatcherRewriteMode::Always,
    )?;
    let mut filter = SequenceComposeFilter::new(&left, matcher1, matcher2);
    compose_with(&left, &complement, ofst, &mut filter)?;
    // As upstream's `Difference`, whose `ComposeOptions` connect by default:
    // the states the result cannot reach or cannot finish from are the pairs
    // composition had to build to find out they lead nowhere.
    crate::algorithms::connect::connect(ofst);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{Rng, string_weights, visible_paths};
    use crate::arc::StdArc;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::K_FST_PROPERTIES;
    use crate::weight::Weight;
    use crate::weights::float_weight::TropicalWeight;

    /// A deterministic acceptor over the given strings.
    fn acceptor(strings: &[&[i32]]) -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        let start = fst.add_state();
        fst.set_start(start);
        for labels in strings {
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
            fst.set_final(state, TropicalWeight::one());
        }
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    fn strings(fst: &StdVectorFst) -> Vec<Vec<i32>> {
        let mut out: Vec<Vec<i32>> = string_weights(visible_paths(fst, 16))
            .into_iter()
            .map(|(ilabels, _, _)| ilabels)
            .collect();
        out.sort();
        out
    }

    fn minus(fst1: &StdVectorFst, fst2: &StdVectorFst) -> StdVectorFst {
        let mut out = StdVectorFst::new();
        difference(fst1, fst2, &mut out).unwrap();
        out
    }

    #[test]
    fn the_difference_is_what_the_first_has_and_the_second_does_not() {
        let a = acceptor(&[&[1, 2], &[3], &[4, 5]]);
        let b = acceptor(&[&[3], &[9]]);
        assert_eq!(strings(&minus(&a, &b)), vec![vec![1, 2], vec![4, 5]]);
    }

    /// Taking away nothing leaves everything.
    #[test]
    fn taking_away_nothing_leaves_everything() {
        let a = acceptor(&[&[1, 2], &[3]]);
        let mut nothing = StdVectorFst::new();
        nothing.add_state();
        nothing.set_start(0);
        nothing.properties(K_FST_PROPERTIES, true);
        assert_eq!(strings(&minus(&a, &nothing)), strings(&a));
    }

    /// Taking a language away from itself leaves nothing.
    #[test]
    fn taking_a_language_from_itself_leaves_nothing() {
        let a = acceptor(&[&[1, 2], &[3]]);
        assert!(strings(&minus(&a, &a)).is_empty());
    }

    /// A prefix of a removed string is still there if the first accepts it.
    #[test]
    fn only_the_strings_themselves_are_removed() {
        let a = acceptor(&[&[1], &[1, 2]]);
        let b = acceptor(&[&[1, 2]]);
        assert_eq!(strings(&minus(&a, &b)), vec![vec![1]]);
    }

    /// The second argument has to be something that can be complemented.
    #[test]
    fn a_second_argument_that_cannot_be_complemented_is_refused() {
        let a = acceptor(&[&[1]]);

        let mut weighted = acceptor(&[&[1]]);
        weighted.set_final(1, TropicalWeight(2.0));
        weighted.properties(K_FST_PROPERTIES, true);
        let mut out = StdVectorFst::new();
        let err = difference(&a, &weighted, &mut out).unwrap_err();
        assert!(format!("{err}").contains("complemented"), "{err}");

        let mut nondeterministic = StdVectorFst::new();
        for _ in 0..3 {
            nondeterministic.add_state();
        }
        nondeterministic.set_start(0);
        nondeterministic.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        nondeterministic.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 2));
        nondeterministic.set_final(1, TropicalWeight::one());
        nondeterministic.properties(K_FST_PROPERTIES, true);
        assert!(difference(&a, &nondeterministic, &mut out).is_err());
    }

    /// The difference holds exactly the strings the first has that the second
    /// does not, over random acceptors.
    #[test]
    fn the_difference_is_the_set_difference() {
        let mut rng = Rng::new(0x00D1_FF01_u64);
        let mut checked = 0;
        for round in 0..100 {
            let make = |rng: &mut Rng| {
                let count = 1 + rng.below(5);
                let strings: Vec<Vec<i32>> = (0..count)
                    .map(|_| {
                        let len = 1 + rng.below(3);
                        (0..len).map(|_| 1 + rng.below(3) as i32).collect()
                    })
                    .collect();
                let refs: Vec<&[i32]> = strings.iter().map(|s| s.as_slice()).collect();
                acceptor(&refs)
            };
            let a = make(&mut rng);
            let b = make(&mut rng);

            let sa = strings(&a);
            let sb = strings(&b);
            let want: Vec<Vec<i32>> = sa.iter().filter(|s| !sb.contains(s)).cloned().collect();
            if !want.is_empty() {
                checked += 1;
            }
            assert_eq!(strings(&minus(&a, &b)), want, "round {round}");
        }
        assert!(checked > 20, "only {checked} differences had anything");
    }
}
