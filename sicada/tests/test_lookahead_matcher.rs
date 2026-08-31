//! The look-ahead matcher may only be wrong in one direction.
//!
//! `LabelLookAheadMatcher` answers whether a state of the second FST can still
//! be reached from a state of the first. Composition uses it as a filter, so a
//! "yes" that turns out to be wrong only costs time: the pair is expanded and
//! then goes nowhere. A "no" that is wrong deletes a path from the result and
//! is silent about it, which is the case these tests look for.
//!
//! The second FST has to be free of epsilons for the question to be the one
//! answered here. `look_ahead` reads that side's arcs as they stand, so a pair
//! whose only way forward is an epsilon move on the right is refused; in
//! composition the filter takes that move, and the matcher is never asked.
//! Epsilons on the matcher's own side are looked over and are left in below.

use sicada::algorithms::accumulator::DefaultAccumulator;
use sicada::algorithms::arcsort::{ILabelCompare, OLabelCompare, arc_sort};
use sicada::algorithms::lookahead_matcher::{
    DEFAULT_LABEL_LOOKAHEAD_FLAGS, LabelLookAheadMatcher, LookAheadMatcher,
    OUTPUT_LOOKAHEAD_MATCHER,
};
use sicada::algorithms::rmepsilon::rm_epsilon;
use sicada::arc::Arc as _;
use sicada::fst::{ExpandedFst as _, Fst as _, MatchType, MutableFst as _};
use sicada::matcher::{Matcher as _, SortedMatcher};
use sicada::prelude::{StdArc, StdVectorFst};
use sicada::weight::Weight;
use sicada::weights::float_weight::TropicalWeight;

const SEED: u64 = 0x2545_F491_4F6C_DD1D;
const STATES: usize = 300;

#[test]
fn the_matcher_never_refuses_a_pair_that_can_reach_a_final_pair() {
    let refused = refusals(acceptor(STATES, SEED ^ 1, false));
    assert_eq!(refused, 0, "{refused} pairs were refused wrongly");
}

/// The same where the second side reaches its epsilon-free form through
/// `rm_epsilon` rather than being built that way, which leaves a different
/// shape to look ahead over.
#[test]
fn the_same_holds_of_a_second_side_the_epsilons_were_removed_from() {
    let mut rhs = acceptor(STATES, SEED ^ 1, true);
    rm_epsilon(&mut rhs, true).expect("epsilons removed");
    let refused = refusals(rhs);
    assert_eq!(refused, 0, "{refused} pairs were refused wrongly");
}

/// The number of state pairs the matcher refuses although a final pair is
/// reachable from them.
fn refusals(mut rhs: StdVectorFst) -> usize {
    let mut lhs = acceptor(STATES, SEED, true);
    arc_sort(&mut lhs, &OLabelCompare);
    arc_sort(&mut rhs, &ILabelCompare);
    lhs.properties(sicada::properties::K_FST_PROPERTIES, true);
    rhs.properties(sicada::properties::K_FST_PROPERTIES, true);

    let lhs_states = lhs.num_states();
    let rhs_states = rhs.num_states();
    let truth = reachable_pairs(&lhs, &rhs, lhs_states, rhs_states);

    let mut matcher = LabelLookAheadMatcher::new(
        &lhs,
        SortedMatcher::new(&lhs, MatchType::Output).expect("a matcher"),
        MatchType::Output,
        DEFAULT_LABEL_LOOKAHEAD_FLAGS | OUTPUT_LOOKAHEAD_MATCHER,
        DefaultAccumulator,
    )
    .expect("an index");

    let mut refused = 0usize;
    for p in 0..lhs_states {
        matcher.set_state(p as i32);
        for q in 0..rhs_states {
            let said = matcher.look_ahead(&rhs, q as i32).reachable;
            if truth[p * rhs_states + q] && !said {
                refused += 1;
            }
        }
    }
    refused
}

/// Which pairs can still reach a pair final on both sides.
///
/// Both arguments are acyclic and numbered forwards, so one sweep from the
/// highest state down settles every pair.
fn reachable_pairs(
    lhs: &StdVectorFst,
    rhs: &StdVectorFst,
    lhs_states: usize,
    rhs_states: usize,
) -> Vec<bool> {
    let zero = TropicalWeight::zero();
    let mut ok = vec![false; lhs_states * rhs_states];
    for p in (0..lhs_states).rev() {
        for q in (0..rhs_states).rev() {
            let mut reachable =
                lhs.final_weight(p as i32) != zero && rhs.final_weight(q as i32) != zero;
            // An epsilon on either side advances that side alone.
            if !reachable {
                reachable = lhs
                    .arcs(p as i32)
                    .any(|arc| arc.olabel() == 0 && ok[arc.nextstate() as usize * rhs_states + q]);
            }
            if !reachable {
                reachable = rhs
                    .arcs(q as i32)
                    .any(|arc| arc.ilabel() == 0 && ok[p * rhs_states + arc.nextstate() as usize]);
            }
            if !reachable {
                reachable = lhs.arcs(p as i32).filter(|a| a.olabel() != 0).any(|a| {
                    rhs.arcs(q as i32).any(|b| {
                        b.ilabel() == a.olabel()
                            && ok[a.nextstate() as usize * rhs_states + b.nextstate() as usize]
                    })
                });
            }
            ok[p * rhs_states + q] = reachable;
        }
    }
    ok
}

/// An acyclic acceptor, as the benchmarks use, with one arc in eight left
/// unlabelled when `epsilons` is set.
fn acceptor(states: usize, seed: u64, epsilons: bool) -> StdVectorFst {
    let mut fst = StdVectorFst::new();
    for _ in 0..states {
        fst.add_state();
    }
    fst.set_start(0);
    let mut state = seed;
    let mut next_u64 = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for s in 0..states {
        for _ in 0..4 {
            let draw = next_u64() % 8;
            let label = if epsilons && draw == 0 {
                0
            } else {
                (1 + draw % 7) as i32
            };
            let weight = TropicalWeight((next_u64() % 400) as f32 / 4.0);
            let room = (states - s - 1) as u64;
            if room == 0 {
                continue;
            }
            let next = s as u64 + 1 + next_u64() % room;
            fst.add_arc(s as i32, StdArc::new(label, label, weight, next as i32));
        }
        if s % 8 == 0 {
            let weight = TropicalWeight((next_u64() % 400) as f32 / 4.0);
            fst.set_final(s as i32, weight);
        }
    }
    fst.properties(sicada::properties::K_FST_PROPERTIES, true);
    fst
}
