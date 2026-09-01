//! Running one transducer's output into another's input.
//!
//! Port of OpenFst's `compose.h`. A state of the result is a state of each
//! input together with a *filter state*; an arc is a pair of arcs, one from
//! each, whose meeting labels agree. The filter stops the same path being found
//! several times when epsilons are involved: without it, an epsilon
//! on one side and an epsilon on the other can be taken in either order, and the
//! result counts the path twice.
//!
//! > Mohri, M., Pereira, F. and Riley, M. 1996. Weighted automata in text and
//! > speech processing. In *Proc. ECAI-96 Workshop*.
//!
//! Which side is matched against is not fixed: at each state the matcher that
//! says it has less work to do is the one iterated over, so composing a small
//! FST with a large one costs what the small one costs.

use crate::algorithms::accumulator::DefaultAccumulator;
use crate::algorithms::arcsort::{ILabelCompare, OLabelCompare, arc_sort};
use crate::algorithms::compose_filter::ComposeFilter;
use crate::algorithms::compose_filter::SequenceComposeFilter;
use crate::algorithms::label_reachable::{LabelReachable, LabelReachableData};
use crate::algorithms::lookahead_filter::LookAheadComposeFilter;
use crate::algorithms::lookahead_matcher::{
    DEFAULT_LABEL_LOOKAHEAD_FLAGS, LabelLookAheadMatcher, OUTPUT_LOOKAHEAD_MATCHER,
    TrivialLookAheadMatcher,
};
use crate::arc::{Arc, ArcLabel};
use crate::data_structures::bi_table::BiTableId;
use crate::data_structures::bit_set::GrowableBitSet;
use crate::data_structures::state_table::{
    ComposeStateTable, DefaultComposeStateTuple, GenericComposeStateTable,
};
use crate::error::OpenFstError;
use crate::fst::ExpandedFst;
use crate::fst::MatchType;
use crate::fst::{Fst, MutableFst};
use crate::fsts::vector_fst::VectorFst;
use crate::matcher::SortedMatcher;
use crate::matcher::{Matcher, REQUIRE_PRIORITY};
use crate::properties::{K_FST_PROPERTIES, K_NO_I_EPSILONS, compose_properties};
use crate::properties::{K_I_LABEL_SORTED, K_O_LABEL_SORTED};
use crate::weight::Weight;

/// Which side each matcher works on, and which side is iterated over.
///
/// Composition matches the *output* of the first against the *input* of the
/// second, so a matcher over the first has to look up output labels and one
/// over the second input labels.
fn match_types<'f, A, M1, M2>(matcher1: &M1, matcher2: &M2) -> Result<MatchType, OpenFstError>
where
    A: Arc,
    M1: Matcher<'f, A>,
    M2: Matcher<'f, A>,
{
    let can_output =
        matcher1.match_type() == MatchType::Output || matcher1.match_type() == MatchType::Both;
    let can_input =
        matcher2.match_type() == MatchType::Input || matcher2.match_type() == MatchType::Both;
    match (can_output, can_input) {
        (true, true) => Ok(MatchType::Both),
        (true, false) => Ok(MatchType::Output),
        (false, true) => Ok(MatchType::Input),
        (false, false) => Err(OpenFstError::InvalidOperation(
            "Compose: neither matcher can match the side composition needs".into(),
        )),
    }
}

/// Composes `fst1` and `fst2` into `ofst`, through `filter`.
///
/// `filter` carries the two matchers, one over each input, since which matches
/// are allowed and how they are found are the same decision.
///
/// SICADA-DIVERGE: upstream's `ComposeFst` is a delayed FST, and `Compose`
/// materializes it into a `MutableFst`. Building the result directly is the
/// same work without a cache in between; the delayed form is still outstanding.
pub fn compose_with<'f, A, F1, F2, FO, Filter>(
    fst1: &'f F1,
    fst2: &'f F2,
    ofst: &mut FO,
    filter: &mut Filter,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::StateId: BiTableId,
    F1: Fst<A>,
    F2: Fst<A>,
    FO: MutableFst<A>,
    Filter: ComposeFilter<Arc = A>,
    Filter::Matcher1: Matcher<'f, A>,
    Filter::Matcher2: Matcher<'f, A>,
{
    ofst.delete_all_states();
    ofst.set_input_symbols(fst1.input_symbols());
    ofst.set_output_symbols(fst2.output_symbols());

    let match_type = match_types(filter.matcher1(), filter.matcher2())?;

    let props1 = fst1.properties(K_FST_PROPERTIES, false);
    let props2 = fst2.properties(K_FST_PROPERTIES, false);
    let props = filter.properties(compose_properties(props1, props2));

    let (Some(start1), Some(start2)) = (fst1.start(), fst2.start()) else {
        ofst.set_properties(props, K_FST_PROPERTIES);
        return Ok(());
    };

    let mut states: GenericComposeStateTable<A, Filter::FilterState> =
        GenericComposeStateTable::new(fst1, fst2);
    let start = states.find_state(&DefaultComposeStateTuple::new(
        start1,
        start2,
        filter.start(),
    ));
    ofst.add_state();
    ofst.set_start(start);

    let mut pending: Vec<A::StateId> = vec![start];
    let mut done = GrowableBitSet::new();
    done.insert(start.as_usize());
    let zero = A::Weight::zero();
    let epsilon = A::Label::epsilon();
    let no_label = A::Label::no_label();
    // Collected before being added, since finding a destination state may add
    // states and the arcs are written to one state at a time.
    let mut arcs: Vec<A> = Vec::new();

    while let Some(state) = pending.pop() {
        let tuple = states.tuple(state).clone();
        let (s1, s2) = (tuple.state_id1(), tuple.state_id2());

        // The final weight is what both say, once the filter has had its say
        // about whether finishing here is allowed at all.
        let mut final1 = fst1.final_weight(s1);
        let mut final2 = fst2.final_weight(s2);
        if final1 != zero && final2 != zero {
            filter.set_state(s1, s2, tuple.get_filter_state());
            filter.filter_final(&mut final1, &mut final2);
            let weight = final1.times(&final2);
            if weight != zero {
                ofst.set_final(state, weight);
            }
        }

        filter.set_state(s1, s2, tuple.get_filter_state());
        // Matching is done on whichever side has less to do, so the arcs of the
        // busier one are never walked.
        let match_input = match match_type {
            MatchType::Input => true,
            MatchType::Output => false,
            _ => {
                let priority1 = filter.matcher1_mut().priority(s1);
                let priority2 = filter.matcher2_mut().priority(s2);
                if priority1 == REQUIRE_PRIORITY && priority2 == REQUIRE_PRIORITY {
                    return Err(OpenFstError::InvalidOperation(
                        "Compose: both sides require the match to be made on them".into(),
                    ));
                }
                if priority1 == REQUIRE_PRIORITY {
                    false
                } else if priority2 == REQUIRE_PRIORITY {
                    true
                } else {
                    priority1 <= priority2
                }
            }
        };

        arcs.clear();
        // Matching on one side means walking the *other* side's arcs and
        // asking the matcher for each. The extra arc at the front stands for
        // "take nothing here", which is how an epsilon on the walked side is
        // followed on its own: it stays where it is on the matched side.
        if match_input {
            // Matching on the second FST, so the first FST's arcs are walked.
            filter.matcher2_mut().set_state(s2);
            let loop_arc = A::new(epsilon, no_label, A::Weight::one(), s1);
            let walked: Vec<A> = std::iter::once(loop_arc).chain(fst1.arcs(s1)).collect();
            for arc in walked {
                match_arc(filter, &mut states, &mut arcs, &arc, true);
            }
        } else {
            // Matching on the first FST, so the second FST's arcs are walked.
            filter.matcher1_mut().set_state(s1);
            let loop_arc = A::new(no_label, epsilon, A::Weight::one(), s2);
            let walked: Vec<A> = std::iter::once(loop_arc).chain(fst2.arcs(s2)).collect();
            for arc in walked {
                match_arc(filter, &mut states, &mut arcs, &arc, false);
            }
        }

        // The state table may have grown; the result needs a state for each.
        while ofst.num_states() < states.size() {
            ofst.add_state();
        }
        for arc in arcs.drain(..) {
            let next = arc.nextstate();
            if done.insert(next.as_usize()) {
                pending.push(next);
            }
            ofst.add_arc(state, arc);
        }
    }

    ofst.set_properties(props, K_FST_PROPERTIES);
    Ok(())
}

/// A `VectorFst` holding what `fst` holds, sorted the way a matcher needs.
///
/// Sorting lets the matcher find a label by binary search rather than by
/// scanning; an FST already sorted that way is copied but not re-sorted.
/// Public because [`compose_lookahead_indexed`] takes its arguments already
/// sorted, and this is how a caller gets them that way.
pub fn sorted_copy<A, F>(fst: &F, by_output: bool) -> VectorFst<A>
where
    A: Arc,
    F: Fst<A> + ExpandedFst<A>,
{
    let mut copy: VectorFst<A> = VectorFst::new();
    copy.add_states(fst.num_states());
    copy.set_input_symbols(fst.input_symbols());
    copy.set_output_symbols(fst.output_symbols());
    if let Some(start) = fst.start() {
        copy.set_start(start);
    }
    for state in fst.states() {
        copy.set_final(state, fst.final_weight(state));
        for arc in fst.arcs(state) {
            copy.add_arc(state, arc);
        }
    }
    copy.set_properties(fst.properties(K_FST_PROPERTIES, false), K_FST_PROPERTIES);
    let wanted = if by_output {
        K_O_LABEL_SORTED
    } else {
        K_I_LABEL_SORTED
    };
    if copy.properties(wanted, true) & wanted == 0 {
        if by_output {
            arc_sort(&mut copy, &OLabelCompare);
        } else {
            arc_sort(&mut copy, &ILabelCompare);
        }
    }
    copy
}

/// Composes `fst1` and `fst2`.
///
/// The first FST's output side is matched against the second's input side, so
/// the result reads what the first reads and writes what the second writes. The
/// sequence filter is used, which is upstream's default: it settles the order
/// epsilons are taken in so that a path is found once rather than once per
/// order.
///
/// SICADA-DIVERGE: the two are copied, and sorted if they are not already, so
/// that the matchers can search them. Upstream leaves that to the caller:
/// `ComposeFst` reads the sort bits and decides which side it can match on, and
/// composing two FSTs sorted the wrong way produces a result missing most of its
/// paths, with nothing reported. The copy is `O(V + E)`, the same order as
/// composition itself; [`compose_with`] takes matchers directly and copies
/// nothing.
pub fn compose<A, F1, F2, FO>(fst1: &F1, fst2: &F2, ofst: &mut FO) -> Result<(), OpenFstError>
where
    A: Arc,
    A::StateId: BiTableId,
    F1: Fst<A> + ExpandedFst<A>,
    F2: Fst<A> + ExpandedFst<A>,
    FO: MutableFst<A> + ExpandedFst<A>,
{
    compose_options(fst1, fst2, ofst, &ComposeOptions::default())
}

/// What [`compose`] may be asked to do differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComposeOptions {
    /// Whether to drop the states the result cannot reach or cannot finish
    /// from.
    ///
    /// Composition builds a state for every pair it can reach, and most pairs
    /// turn out to lead nowhere; on by default, as upstream's is.
    pub connect: bool,
}

impl Default for ComposeOptions {
    fn default() -> Self {
        Self { connect: true }
    }
}

/// As [`compose`], saying whether to connect the result.
pub fn compose_options<A, F1, F2, FO>(
    fst1: &F1,
    fst2: &F2,
    ofst: &mut FO,
    opts: &ComposeOptions,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::StateId: BiTableId,
    F1: Fst<A> + ExpandedFst<A>,
    F2: Fst<A> + ExpandedFst<A>,
    FO: MutableFst<A> + ExpandedFst<A>,
{
    let left = sorted_copy(fst1, true);
    let right = sorted_copy(fst2, false);
    let matcher1 = SortedMatcher::new(&left, MatchType::Output)?;
    let matcher2 = SortedMatcher::new(&right, MatchType::Input)?;
    let mut filter = SequenceComposeFilter::new(&left, matcher1, matcher2);
    compose_with(&left, &right, ofst, &mut filter)?;
    if opts.connect {
        crate::algorithms::connect::connect(ofst);
    }
    Ok(())
}

/// Composes with a look-ahead matcher over the first FST's output side.
///
/// Plain [`compose`] builds a state for every pair of states it can reach and
/// finds out only afterwards that most of them are dead ends. A look-ahead
/// matcher is asked first: from where the first FST would be, can anything
/// still match what the second FST has? An index over the first FST's output
/// labels answers that in the time it takes to walk the second state's arcs,
/// and the pairs that lead nowhere are never built.
///
/// The index costs one pass over `fst1` to build, which is why this is a
/// separate entry point rather than what [`compose`] always does: it pays when
/// the composition is large or when the same `fst1` is composed against many
/// second arguments, and not otherwise. Saving the index beside the FST, so
/// that the pass happens once ever, is what
/// [`MatcherFst`](crate::fsts::matcher_fst::MatcherFst) is for.
///
/// # `fst2` must have no input epsilons
///
/// The index answers "which label can `fst1` read next"; an epsilon on the
/// other side is not a label, and
/// [`LabelReachable`]
/// answers `false` for it, as upstream's `Reach` does explicitly. So a state
/// of `fst2` whose way on is its own epsilon looks like a dead end, and the
/// pair is refused although the composition could have gone through it.
///
/// Measured on 90000 state pairs of two 300-state acceptors: with input
/// epsilons in `fst2`, 2686 pairs are refused that should not be; with them
/// removed, none. It is a precondition, not a tuning knob, so it is checked.
///
/// SICADA-DIVERGE: upstream leaves this to the caller, and its look-ahead
/// composition simply produces fewer paths than it should if the second
/// argument has input epsilons. The pipelines that use look-ahead composition
/// remove them first ([`rm_epsilon`](super::rmepsilon::rm_epsilon)) in any
/// case.
pub fn compose_lookahead<A, F1, F2, FO>(
    fst1: &F1,
    fst2: &F2,
    ofst: &mut FO,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::StateId: BiTableId,
    F1: Fst<A> + ExpandedFst<A>,
    F2: Fst<A> + ExpandedFst<A>,
    FO: MutableFst<A> + ExpandedFst<A>,
{
    let required = K_NO_I_EPSILONS;
    if fst2.properties(required, true) & required != required {
        return Err(OpenFstError::InvalidOperation(
            "ComposeLookAhead: the 2nd argument has input epsilons, which the look-ahead index \
             cannot see past; remove them first"
                .into(),
        ));
    }
    let left = sorted_copy(fst1, true);
    let right = sorted_copy(fst2, false);
    compose_lookahead_sorted(&left, &right, ofst)
}

/// The index [`compose_lookahead`] builds, so that it can be built once and
/// used against many second arguments.
///
/// One pass over `fst1`, which is why [`compose_lookahead`] is a loss on a
/// single small composition and a win on a lexicon reused all day. `fst1` must
/// be sorted on its output labels, which [`sorted_copy`] does.
pub fn lookahead_index<A, F>(fst1: &F) -> Result<std::sync::Arc<LabelReachableData>, OpenFstError>
where
    A: Arc,
    F: Fst<A> + ExpandedFst<A>,
{
    let reachable =
        LabelReachable::<A, DefaultAccumulator>::with_accumulator(fst1, false, DefaultAccumulator)?;
    Ok(std::sync::Arc::clone(reachable.data()))
}

/// As [`compose_lookahead`], for inputs already sorted the way it needs them:
/// `fst1` on its output labels and `fst2` on its input labels, and `fst2`
/// already free of input epsilons, which this does *not* check.
pub fn compose_lookahead_sorted<A, FO>(
    left: &VectorFst<A>,
    right: &VectorFst<A>,
    ofst: &mut FO,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::StateId: BiTableId,
    FO: MutableFst<A> + ExpandedFst<A>,
{
    compose_lookahead_indexed(left, &lookahead_index(left)?, right, ofst)
}

/// As [`compose_lookahead_sorted`], with the index already built.
pub fn compose_lookahead_indexed<A, FO>(
    left: &VectorFst<A>,
    index: &std::sync::Arc<LabelReachableData>,
    right: &VectorFst<A>,
    ofst: &mut FO,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::StateId: BiTableId,
    FO: MutableFst<A> + ExpandedFst<A>,
{
    let matcher1 = LabelLookAheadMatcher::from_data(
        std::sync::Arc::clone(index),
        SortedMatcher::new(left, MatchType::Output)?,
        DEFAULT_LABEL_LOOKAHEAD_FLAGS | OUTPUT_LOOKAHEAD_MATCHER,
        DefaultAccumulator,
    );
    let matcher2 = TrivialLookAheadMatcher::new(SortedMatcher::new(right, MatchType::Input)?);
    let inner = SequenceComposeFilter::new(left, matcher1, matcher2);
    let mut filter = LookAheadComposeFilter::new(inner, right)?;
    compose_with(left, right, ofst, &mut filter)?;
    crate::algorithms::connect::connect(ofst);
    Ok(())
}

/// Matches one arc of the side being walked against the other side.
fn match_arc<'f, A, Filter>(
    filter: &mut Filter,
    states: &mut GenericComposeStateTable<A, Filter::FilterState>,
    arcs: &mut Vec<A>,
    arc: &A,
    match_input: bool,
) where
    A: Arc,
    A::StateId: BiTableId,
    Filter: ComposeFilter<Arc = A>,
    Filter::Matcher1: Matcher<'f, A>,
    Filter::Matcher2: Matcher<'f, A>,
{
    // The label to look for is the one that meets the other side: the output
    // of the first FST against the input of the second.
    let label = if match_input {
        arc.olabel()
    } else {
        arc.ilabel()
    };

    // Everything the matcher finds for that label, paired with the arc.
    let found: Vec<A> = if match_input {
        let matcher = filter.matcher2_mut();
        if !matcher.find(label) {
            return;
        }
        let mut out = Vec::new();
        while !matcher.done() {
            out.push(matcher.value());
            matcher.next();
        }
        out
    } else {
        let matcher = filter.matcher1_mut();
        if !matcher.find(label) {
            return;
        }
        let mut out = Vec::new();
        while !matcher.done() {
            out.push(matcher.value());
            matcher.next();
        }
        out
    };

    for other in found {
        // `arc1` is always the first FST's, `arc2` the second's, whichever
        // side was walked.
        let (mut arc1, mut arc2) = if match_input {
            (arc.clone(), other)
        } else {
            (other, arc.clone())
        };
        let Some(fs) = filter.filter_arc(&mut arc1, &mut arc2) else {
            continue;
        };
        let next = states.find_state(&DefaultComposeStateTuple::new(
            arc1.nextstate(),
            arc2.nextstate(),
            fs,
        ));
        arcs.push(A::new(
            arc1.ilabel(),
            arc2.olabel(),
            arc1.weight().times(arc2.weight()),
            next,
        ));
    }
}

#[cfg(test)]
mod tests {

    /// An acyclic acceptor over a small alphabet, one arc in four carrying no
    /// label: the shape a look-ahead filter is meant for, and the one that has
    /// epsilons for it to look past.
    fn epsilon_acceptor(
        rng: &mut crate::algorithms::test_support::Rng,
        states: usize,
    ) -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..states {
            fst.add_state();
        }
        fst.set_start(0);
        for s in 0..states {
            for _ in 0..3 {
                let draw = rng.below(4);
                let label = if draw == 0 { 0 } else { draw as i32 };
                let room = states - s - 1;
                if room == 0 {
                    continue;
                }
                let next = s + 1 + rng.below(room);
                fst.add_arc(
                    s as i32,
                    StdArc::new(label, label, TropicalWeight::one(), next as i32),
                );
            }
            if s % 3 == 0 {
                fst.set_final(s as i32, TropicalWeight::one());
            }
        }
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    /// Looking ahead gives the same answer as not looking ahead.
    ///
    /// That is the contract: the index only lets composition skip pairs it
    /// would have built and then thrown away. The second FST has its epsilons
    /// removed first, as the entry point requires; see the next test for why.
    #[test]
    fn looking_ahead_composes_to_the_same_fst() {
        use crate::algorithms::rmepsilon::rm_epsilon;
        use crate::algorithms::test_support::{Rng, string_weights, visible_paths};

        let mut rng = Rng::new(0x_C0FFEE);
        for round in 0..40 {
            let first = epsilon_acceptor(&mut rng, 60);
            let mut second = epsilon_acceptor(&mut rng, 60);
            rm_epsilon(&mut second, true).expect("epsilons removed");

            let mut plain = StdVectorFst::new();
            compose(&first, &second, &mut plain).expect("a composition");

            let mut ahead = StdVectorFst::new();
            compose_lookahead(&first, &second, &mut ahead).expect("a composition");

            assert_eq!(
                string_weights(visible_paths(&plain, 12)),
                string_weights(visible_paths(&ahead, 12)),
                "round {round}"
            );
        }
    }

    /// A second argument with input epsilons is refused rather than quietly
    /// composed short.
    ///
    /// The index says which label the first FST can read next; an epsilon on
    /// the other side is not a label, so a state whose only way on is its own
    /// epsilon reads as a dead end. Measured on 90000 state pairs of two
    /// 300-state acceptors, 2686 pairs were refused that should not have been.
    #[test]
    fn a_second_argument_with_input_epsilons_is_refused() {
        let mut first = StdVectorFst::new();
        for _ in 0..2 {
            first.add_state();
        }
        first.set_start(0);
        first.set_final(1, TropicalWeight::one());
        first.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        first.properties(K_FST_PROPERTIES, true);

        let mut second = StdVectorFst::new();
        for _ in 0..3 {
            second.add_state();
        }
        second.set_start(0);
        second.set_final(2, TropicalWeight::one());
        second.add_arc(0, StdArc::new(0, 0, TropicalWeight::one(), 1));
        second.add_arc(1, StdArc::new(1, 1, TropicalWeight::one(), 2));
        second.properties(K_FST_PROPERTIES, true);

        let mut out = StdVectorFst::new();
        let Err(err) = compose_lookahead(&first, &second, &mut out) else {
            panic!("an input epsilon on the second side is not something to look past")
        };
        assert!(format!("{err}").contains("input epsilons"), "{err}");

        // Plain composition has no such requirement.
        let mut out = StdVectorFst::new();
        compose(&first, &second, &mut out).expect("a composition");
        assert!(out.num_states() > 0);
    }

    /// Composition builds a state for every pair it can reach, and most of them
    /// lead nowhere; they are not part of the answer.
    ///
    /// Upstream's `Compose` connects by default (`ComposeOptions::connect`),
    /// and this was left out here at first. It was caught by a benchmark: on
    /// two 2000-state acceptors the result had 3009 states against upstream's
    /// 432, accepting the same language through seven times the graph.
    #[test]
    fn the_pairs_that_lead_nowhere_are_not_in_the_answer() {
        use crate::algorithms::connect::connect;

        // 0 -1-> 1 -2-> 2(final), against 0 -1-> 1 -3-> 2(final): the pair
        // reached by label 1 exists but can never finish.
        let mut fst1 = StdVectorFst::new();
        for _ in 0..3 {
            fst1.add_state();
        }
        fst1.set_start(0);
        fst1.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst1.add_arc(1, StdArc::new(2, 2, TropicalWeight::one(), 2));
        fst1.set_final(2, TropicalWeight::one());
        fst1.properties(K_FST_PROPERTIES, true);

        let mut fst2 = StdVectorFst::new();
        for _ in 0..3 {
            fst2.add_state();
        }
        fst2.set_start(0);
        fst2.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst2.add_arc(1, StdArc::new(3, 3, TropicalWeight::one(), 2));
        fst2.set_final(2, TropicalWeight::one());
        fst2.properties(K_FST_PROPERTIES, true);

        let mut connected = StdVectorFst::new();
        compose(&fst1, &fst2, &mut connected).unwrap();
        assert_eq!(
            connected.num_states(),
            0,
            "nothing composes, so there is nothing to keep"
        );

        let mut whole = StdVectorFst::new();
        compose_options(&fst1, &fst2, &mut whole, &ComposeOptions { connect: false }).unwrap();
        assert!(
            whole.num_states() > 0,
            "without connecting, the pairs it had to build to find out are still there"
        );
        connect(&mut whole);
        assert_eq!(whole.num_states(), 0, "and connecting is what removes them");
    }
    use super::*;
    use crate::algorithms::test_support::{Rng, random_acyclic_fst, string_weights, visible_paths};
    use crate::arc::StdArc;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::K_FST_PROPERTIES;
    use crate::weights::float_weight::TropicalWeight;

    /// A transducer over the given (input, output, weight) arcs in a chain.
    fn chain(arcs: &[(i32, i32, f32)]) -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        let mut state = fst.add_state();
        fst.set_start(state);
        for (ilabel, olabel, weight) in arcs {
            let next = fst.add_state();
            fst.add_arc(
                state,
                StdArc::new(*ilabel, *olabel, TropicalWeight(*weight), next),
            );
            state = next;
        }
        fst.set_final(state, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    fn composed(fst1: &StdVectorFst, fst2: &StdVectorFst) -> StdVectorFst {
        let mut out = StdVectorFst::new();
        compose(fst1, fst2, &mut out).unwrap();
        out
    }

    /// What a transducer transduces, epsilons dropped.
    fn transduction(fst: &StdVectorFst) -> Vec<(Vec<i32>, Vec<i32>, String)> {
        string_weights(visible_paths(fst, 16))
    }

    /// Composing what one writes into what the other reads.
    #[test]
    fn composition_runs_one_output_into_the_others_input() {
        // a:b then b:c gives a:c.
        let first = chain(&[(1, 2, 1.0)]);
        let second = chain(&[(2, 3, 2.0)]);
        assert_eq!(
            transduction(&composed(&first, &second)),
            vec![(vec![1], vec![3], "3.0000".to_string())]
        );
    }

    /// Where the two do not meet, nothing comes out.
    #[test]
    fn nothing_comes_out_when_the_labels_do_not_meet() {
        let first = chain(&[(1, 2, 0.0)]);
        let second = chain(&[(9, 3, 0.0)]);
        assert!(transduction(&composed(&first, &second)).is_empty());
    }

    /// Composing with the identity leaves a transducer saying what it said.
    #[test]
    fn composing_with_the_identity_changes_nothing() {
        let first = chain(&[(1, 5, 1.0), (2, 6, 2.0)]);

        // The identity over the labels the first writes.
        let mut identity = StdVectorFst::new();
        let state = identity.add_state();
        identity.set_start(state);
        identity.set_final(state, TropicalWeight::one());
        for label in [5, 6] {
            identity.add_arc(
                state,
                StdArc::new(label, label, TropicalWeight::one(), state),
            );
        }
        identity.properties(K_FST_PROPERTIES, true);

        assert_eq!(
            transduction(&composed(&first, &identity)),
            transduction(&first)
        );
    }

    /// Composition is associative, which is the property everything built on it
    /// relies on.
    #[test]
    fn composition_is_associative() {
        let a = chain(&[(1, 2, 1.0), (3, 4, 2.0)]);
        let b = chain(&[(2, 5, 4.0), (4, 6, 8.0)]);
        let c = chain(&[(5, 7, 16.0), (6, 8, 32.0)]);

        let left = composed(&composed(&a, &b), &c);
        let right = composed(&a, &composed(&b, &c));
        assert_eq!(transduction(&left), transduction(&right));
        assert_eq!(
            transduction(&left),
            vec![(vec![1, 3], vec![7, 8], "63.0000".to_string())]
        );
    }

    /// An epsilon on the meeting side is followed on its own, and the filter
    /// makes sure the path is found once rather than once per order.
    #[test]
    fn an_epsilon_on_the_meeting_side_is_followed_alone() {
        // a:eps then b:c, against c:d. The epsilon has nothing to meet, so it
        // is taken by itself.
        let first = chain(&[(1, 0, 1.0), (2, 3, 2.0)]);
        let second = chain(&[(3, 4, 4.0)]);
        assert_eq!(
            transduction(&composed(&first, &second)),
            vec![(vec![1, 2], vec![4], "7.0000".to_string())]
        );
    }

    /// Epsilons on both sides are taken in one order only, so the path is
    /// counted once.
    #[test]
    fn epsilons_on_both_sides_do_not_double_a_path() {
        // first writes eps then b; second reads eps then b.
        let first = chain(&[(1, 0, 0.0), (2, 3, 0.0)]);
        let second = chain(&[(0, 7, 0.0), (3, 8, 0.0)]);
        let out = composed(&first, &second);
        let paths = transduction(&out);
        assert_eq!(paths.len(), 1, "{paths:?}");
        assert_eq!(paths[0].0, vec![1, 2]);
    }

    /// An FST with no start state composes to nothing.
    #[test]
    fn composing_with_nothing_gives_nothing() {
        let first = chain(&[(1, 2, 0.0)]);
        let empty = StdVectorFst::new();
        assert_eq!(composed(&first, &empty).num_states(), 0);
        assert_eq!(composed(&empty, &first).num_states(), 0);
    }

    /// The result transduces exactly what running one into the other does,
    /// checked against pairing up the two FSTs' paths by hand.
    #[test]
    fn the_result_is_what_running_one_into_the_other_gives() {
        let mut rng = Rng::new(0x0C0F_0FE5_u64);
        let mut checked = 0;
        for round in 0..200 {
            // Two acyclic transducers over a small alphabet, so their paths can
            // be paired up directly.
            let make = |rng: &mut Rng, shift: i32| {
                let mut fst = random_acyclic_fst(rng, 5);
                let states: Vec<i32> = fst.states().collect();
                for state in states {
                    fst.mutate_arcs(state, |arc| {
                        *arc = StdArc::new(
                            arc.ilabel(),
                            arc.ilabel() + shift,
                            *arc.weight(),
                            arc.nextstate(),
                        );
                    });
                }
                fst.properties(K_FST_PROPERTIES, true);
                fst
            };
            let first = make(&mut rng, 0);
            // The second reads what the first writes.
            let second = make(&mut rng, 10);

            // By hand: every pair of paths whose meeting strings agree.
            let mut want: std::collections::BTreeMap<(Vec<i32>, Vec<i32>), f32> =
                std::collections::BTreeMap::new();
            for (i1, o1, w1) in visible_paths(&first, 12) {
                for (i2, o2, w2) in visible_paths(&second, 12) {
                    if o1 != i2 {
                        continue;
                    }
                    let weight = w1.value() + w2.value();
                    want.entry((i1.clone(), o2))
                        .and_modify(|at| *at = at.min(weight))
                        .or_insert(weight);
                }
            }
            let want: Vec<(Vec<i32>, Vec<i32>, String)> = want
                .into_iter()
                .map(|((i, o), w)| (i, o, format!("{w:.4}")))
                .collect();
            if !want.is_empty() {
                checked += 1;
            }

            assert_eq!(
                transduction(&composed(&first, &second)),
                want,
                "round {round}"
            );
        }
        assert!(checked > 20, "only {checked} compositions said anything");
    }
}
