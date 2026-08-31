//! Deciding whether two FSTs say the same thing.
//!
//! Port of OpenFst's `equivalent.h`, which is the classic near-linear algorithm
//! for the equivalence of two deterministic acceptors:
//!
//! > Aho, A. V., Hopcroft, J. E. and Ullman, J. D. 1974. *The Design and
//! > Analysis of Computer Algorithms*, Addison-Wesley, section 4.7.
//!
//! Reading the two in step, a state of one and a state of the other reached by
//! the same string are put into the same class; the two are equivalent exactly
//! when no class ends up holding a final state and a non-final one. A missing
//! transition is treated as one into a dead state, so the two do not have to be
//! complete.

use hashbrown::HashMap;
use std::collections::VecDeque;

use crate::algorithms::arc_map::{QuantizeMapper, arc_map};
use crate::algorithms::encode::{ENCODE_FLAGS, EncodeMapper, encode};
use crate::algorithms::push::push_weights_to_initial;
use crate::arc::{Arc, ArcStateId};
use crate::data_structures::union_find::UnionFind;
use crate::error::OpenFstError;
use crate::fst::{ExpandedFst, Fst, MutableFst};
use crate::fsts::vector_fst::VectorFst;
use crate::properties::{
    K_ACCEPTOR, K_ERROR, K_FST_PROPERTIES, K_I_DETERMINISTIC, K_NO_EPSILONS, K_UNWEIGHTED,
};
use crate::symbol_table::compat_symbols_rc;
use crate::weight::{Divide, LeftSemiring, Weight};

/// The quantization delta upstream compares at.
pub const DELTA: f32 = 1.0 / 1024.0;

/// The state that stands for "there is no transition here" in both FSTs.
const DEAD_STATE: usize = 0;

/// A state of one of the two FSTs, as a single number.
///
/// The two FSTs' states have to share one union-find, so they are interleaved:
/// state `s` of the first is `2s + 1`, of the second `2s + 2`, and 0 is the
/// dead state both share.
#[inline]
fn map_state(state: Option<usize>, which: usize) -> usize {
    match state {
        None => DEAD_STATE,
        Some(state) => (state << 1) + which,
    }
}

/// The state a mapped id stands for.
#[inline]
fn unmap_state(id: usize) -> usize {
    (id - 1) >> 1
}

/// Whether the mapped state is final in the FST it belongs to.
fn is_final<A: Arc, F: Fst<A>>(fst: &F, id: usize) -> bool {
    if id == DEAD_STATE {
        return false;
    }
    fst.final_weight(A::StateId::from_usize(unmap_state(id))) != A::Weight::zero()
}

/// The class a state is in, starting one for it if it has none.
fn find_set(sets: &mut UnionFind, id: usize) -> usize {
    match sets.find_set(id) {
        Some(representative) => representative,
        None => {
            sets.make_set(id);
            id
        }
    }
}

/// Whether `fst1` and `fst2` accept the same strings at the same weights.
///
/// Both have to be epsilon-free deterministic acceptors, which is why the
/// question is answerable in near-linear time. A weighted pair is pushed,
/// quantized and encoded first, which turns the weights into part of the labels
/// and leaves two unweighted acceptors.
///
/// SICADA-DIVERGE: upstream reports "this is not an epsilon-free deterministic
/// acceptor" by returning `false` and setting an out-parameter `bool* error`
/// that defaults to null, so by default a caller cannot tell a refusal from a
/// genuine "not equivalent". It is an error here.
pub fn equivalent<A, F1, F2>(fst1: &F1, fst2: &F2, delta: f32) -> Result<bool, OpenFstError>
where
    A: Arc,
    A::Weight: Divide + LeftSemiring + std::hash::Hash + Eq,
    F1: Fst<A> + ExpandedFst<A>,
    F2: Fst<A> + ExpandedFst<A>,
    <A::Weight as Weight>::ReverseWeight: Weight<ReverseWeight = A::Weight>,
{
    if !compat_symbols_rc(fst1.input_symbols(), fst2.input_symbols())
        || !compat_symbols_rc(fst1.output_symbols(), fst2.output_symbols())
    {
        return Err(OpenFstError::SymbolTable(
            "Equivalent: the two FSTs' symbol tables do not agree".into(),
        ));
    }

    let required = K_NO_EPSILONS | K_I_DETERMINISTIC | K_ACCEPTOR;
    for (which, props) in [
        ("1st", fst1.properties(required, true)),
        ("2nd", fst2.properties(required, true)),
    ] {
        if props != required {
            return Err(OpenFstError::InvalidOperation(format!(
                "Equivalent: the {which} argument is not an epsilon-free deterministic acceptor"
            )));
        }
    }

    // Weights are compared by making them part of the labels: pushing puts each
    // path's weight in a canonical place, and encoding folds it into the label.
    if fst1.properties(K_UNWEIGHTED, true) & K_UNWEIGHTED == 0
        || fst2.properties(K_UNWEIGHTED, true) & K_UNWEIGHTED == 0
    {
        let mut copy1: VectorFst<A> = copy_of(fst1);
        let mut copy2: VectorFst<A> = copy_of(fst2);
        push_weights_to_initial(&mut copy1, delta, false)?;
        push_weights_to_initial(&mut copy2, delta, false)?;
        arc_map(&mut copy1, &mut QuantizeMapper::new(delta))?;
        arc_map(&mut copy2, &mut QuantizeMapper::new(delta))?;
        // The same encoder for both, so that the same weight becomes the same
        // label in each.
        let mut encoder = EncodeMapper::<A>::new(ENCODE_FLAGS);
        encode(&mut copy1, &mut encoder)?;
        encode(&mut copy2, &mut encoder)?;
        return equivalent(&copy1, &copy2, delta);
    }

    let mut sets = UnionFind::new(2 * (fst1.num_states() + fst2.num_states()) + 2);
    let start1 = map_state(fst1.start().map(|s| s.as_usize()), 1);
    let start2 = map_state(fst2.start().map(|s| s.as_usize()), 2);
    sets.make_set(start1);
    sets.make_set(start2);

    if is_final(fst1, start1) != is_final(fst2, start2) {
        return Ok(false);
    }

    let zero = A::Weight::zero();
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    queue.push_back((start1, start2));
    // Where the same label leads in each FST, filled in fresh for each pair.
    let mut by_label: HashMap<A::Label, (usize, usize)> = HashMap::new();

    while let Some((s1, s2)) = queue.pop_front() {
        let rep1 = find_set(&mut sets, s1);
        let rep2 = find_set(&mut sets, s2);
        if rep1 == rep2 {
            // Already known to be the same state; nothing new to check.
            continue;
        }
        sets.union(rep1, rep2);

        by_label.clear();
        // A label with no arc in one of the two leads to the dead state there,
        // which is why the entries start at DEAD_STATE.
        if s1 != DEAD_STATE {
            for arc in fst1.arcs(A::StateId::from_usize(unmap_state(s1))) {
                // A zero-weight arc leads nowhere, so it is as if it were not
                // there.
                if *arc.weight() != zero {
                    by_label
                        .entry(arc.ilabel())
                        .or_insert((DEAD_STATE, DEAD_STATE))
                        .0 = map_state(Some(arc.nextstate().as_usize()), 1);
                }
            }
        }
        if s2 != DEAD_STATE {
            for arc in fst2.arcs(A::StateId::from_usize(unmap_state(s2))) {
                if *arc.weight() != zero {
                    by_label
                        .entry(arc.ilabel())
                        .or_insert((DEAD_STATE, DEAD_STATE))
                        .1 = map_state(Some(arc.nextstate().as_usize()), 2);
                }
            }
        }

        for (_, (to1, to2)) in by_label.iter() {
            if is_final(fst1, *to1) != is_final(fst2, *to2) {
                // One accepts here and the other does not: a string tells them
                // apart.
                return Ok(false);
            }
            queue.push_back((*to1, *to2));
        }
    }

    if fst1.properties(K_ERROR, false) & K_ERROR != 0
        || fst2.properties(K_ERROR, false) & K_ERROR != 0
    {
        return Err(OpenFstError::InvalidOperation(
            "Equivalent: one of the FSTs is marked as being in error".into(),
        ));
    }
    Ok(true)
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
    use crate::algorithms::determinize::{DeterminizeOptions, determinize};
    use crate::algorithms::rmepsilon::rm_epsilon;
    use crate::algorithms::test_support::{Rng, random_acyclic_fst};
    use crate::arc::StdArc;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::weights::float_weight::TropicalWeight;

    /// A deterministic epsilon-free acceptor over `strings`.
    fn acceptor(strings: &[(&[i32], f32)]) -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        let start = fst.add_state();
        fst.set_start(start);
        for (labels, weight) in strings {
            let mut state = start;
            for label in *labels {
                // Deterministic: reuse the arc if one with this label is there.
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
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    fn same(fst1: &StdVectorFst, fst2: &StdVectorFst) -> bool {
        equivalent(fst1, fst2, DELTA).unwrap()
    }

    #[test]
    fn an_fst_is_equivalent_to_itself() {
        let fst = acceptor(&[(&[1, 2], 0.0), (&[1, 3], 0.0)]);
        assert!(same(&fst, &fst));
    }

    /// The states may be numbered differently and the answer is the same.
    #[test]
    fn the_numbering_of_the_states_does_not_matter() {
        let fst = acceptor(&[(&[1, 2], 0.0), (&[3], 0.0)]);
        let other = acceptor(&[(&[3], 0.0), (&[1, 2], 0.0)]);
        assert_ne!(
            fst.arcs(0).map(|a| a.ilabel()).collect::<Vec<_>>(),
            other.arcs(0).map(|a| a.ilabel()).collect::<Vec<_>>(),
            "the two really are built differently"
        );
        assert!(same(&fst, &other));
    }

    #[test]
    fn one_extra_string_makes_them_different() {
        let fst = acceptor(&[(&[1], 0.0)]);
        let other = acceptor(&[(&[1], 0.0), (&[2], 0.0)]);
        assert!(!same(&fst, &other));
    }

    /// A string missing from one but present in the other is found however deep
    /// it is.
    #[test]
    fn a_difference_deep_in_the_fst_is_found() {
        let fst = acceptor(&[(&[1, 2, 3, 4], 0.0)]);
        let other = acceptor(&[(&[1, 2, 3, 5], 0.0)]);
        assert!(!same(&fst, &other));
    }

    /// The weights count too.
    #[test]
    fn the_same_strings_at_different_weights_are_not_equivalent() {
        let fst = acceptor(&[(&[1, 2], 1.0)]);
        let other = acceptor(&[(&[1, 2], 2.0)]);
        assert!(!same(&fst, &other));
        assert!(same(&fst, &acceptor(&[(&[1, 2], 1.0)])));
    }

    /// The same weighted language written with the weight in a different place
    /// is still the same language, since pushing puts it back.
    #[test]
    fn where_the_weight_sits_along_a_path_does_not_matter() {
        let mut early = StdVectorFst::new();
        for _ in 0..3 {
            early.add_state();
        }
        early.set_start(0);
        early.add_arc(0, StdArc::new(1, 1, TropicalWeight(3.0), 1));
        early.add_arc(1, StdArc::new(2, 2, TropicalWeight::one(), 2));
        early.set_final(2, TropicalWeight::one());
        early.properties(K_FST_PROPERTIES, true);

        let mut late = StdVectorFst::new();
        for _ in 0..3 {
            late.add_state();
        }
        late.set_start(0);
        late.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        late.add_arc(1, StdArc::new(2, 2, TropicalWeight(3.0), 2));
        late.set_final(2, TropicalWeight::one());
        late.properties(K_FST_PROPERTIES, true);

        assert!(same(&early, &late));
    }

    /// An FST that is not an epsilon-free deterministic acceptor is refused,
    /// rather than answered "not equivalent".
    #[test]
    fn an_fst_the_algorithm_cannot_read_is_refused() {
        let good = acceptor(&[(&[1], 0.0)]);

        let mut nondeterministic = StdVectorFst::new();
        for _ in 0..3 {
            nondeterministic.add_state();
        }
        nondeterministic.set_start(0);
        nondeterministic.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        nondeterministic.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 2));
        nondeterministic.set_final(1, TropicalWeight::one());
        nondeterministic.properties(K_FST_PROPERTIES, true);

        let err = equivalent(&good, &nondeterministic, DELTA).unwrap_err();
        assert!(format!("{err}").contains("deterministic acceptor"), "{err}");

        let mut with_epsilon = StdVectorFst::new();
        for _ in 0..2 {
            with_epsilon.add_state();
        }
        with_epsilon.set_start(0);
        with_epsilon.add_arc(0, StdArc::new(0, 0, TropicalWeight::one(), 1));
        with_epsilon.set_final(1, TropicalWeight::one());
        with_epsilon.properties(K_FST_PROPERTIES, true);
        assert!(equivalent(&good, &with_epsilon, DELTA).is_err());
    }

    /// Determinizing does not change what an FST says, which is the property
    /// equivalence checks.
    #[test]
    fn determinizing_produces_an_equivalent_fst() {
        let mut rng = Rng::new(0x0E00_1A1E_u64);
        let mut checked = 0;
        for round in 0..200 {
            let mut fst = random_acyclic_fst(&mut rng, 6);
            rm_epsilon(&mut fst, true).unwrap();
            fst.properties(K_FST_PROPERTIES, true);
            if fst.start().is_none() {
                continue;
            }

            let mut det = StdVectorFst::new();
            determinize(
                &fst,
                &mut det,
                &DeterminizeOptions {
                    max_states: Some(4096),
                    ..Default::default()
                },
            )
            .unwrap();

            // Only the pairs the algorithm can read are checked: it needs both
            // sides deterministic and epsilon-free.
            let required = K_NO_EPSILONS | K_I_DETERMINISTIC | K_ACCEPTOR;
            if fst.properties(required, true) != required {
                continue;
            }
            checked += 1;
            assert!(same(&fst, &det), "round {round}");
        }
        assert!(checked > 20, "only {checked} pairs could be compared");
    }

    /// Two FSTs built from the same set of strings in a different order are
    /// equivalent; adding one string to one of them makes them not.
    #[test]
    fn equivalence_is_decided_by_the_strings_and_nothing_else() {
        let mut rng = Rng::new(0x0E00_0002_u64);
        for round in 0..200 {
            let count = 1 + rng.below(5);
            let strings: Vec<(Vec<i32>, f32)> = (0..count)
                .map(|_| {
                    let len = 1 + rng.below(4);
                    let labels: Vec<i32> = (0..len).map(|_| 1 + rng.below(3) as i32).collect();
                    (labels, rng.below(3) as f32)
                })
                .collect();

            let forwards: Vec<(&[i32], f32)> =
                strings.iter().map(|(l, w)| (l.as_slice(), *w)).collect();
            let mut backwards = forwards.clone();
            backwards.reverse();

            let a = acceptor(&forwards);
            let b = acceptor(&backwards);
            // Building in a different order can settle a repeated string on a
            // different final weight, so only the cases that agree are compared
            // as equivalent.
            let mut sorted: Vec<(Vec<i32>, f32)> = strings.clone();
            sorted.sort_by(|x, y| x.0.cmp(&y.0));
            let repeated = sorted
                .windows(2)
                .any(|w| w[0].0 == w[1].0 && w[0].1 != w[1].1);
            if repeated {
                continue;
            }
            assert!(same(&a, &b), "round {round}");

            // And one more string that is not there makes them differ.
            let mut extra = forwards.clone();
            let novel = vec![9, 9, 9];
            extra.push((novel.as_slice(), 0.0));
            assert!(!same(&a, &acceptor(&extra)), "round {round}");
        }
    }
}
