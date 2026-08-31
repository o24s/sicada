//! Leaving one path per input string.
//!
//! Port of OpenFst's `disambiguate.h`.
//!
//! > Mohri, M. and Riley, M. 2015. On the disambiguation of weighted automata.
//! > In *CIAA*, pages 263-278.
//!
//! Determinization makes each input string readable one way by merging the
//! states it could be in; disambiguation instead *chooses* one of the ways and
//! throws the others away. The result still holds the same weights, but no two
//! distinct paths spell the same input.
//!
//! It runs in three parts: a determinization that keeps track of which state of
//! the input each result state came from (its *head*) and only merges states
//! that could still agree; a search for the pairs of transitions that leave two
//! paths spelling the same thing; and the removal of one transition from each
//! such pair.

use std::collections::VecDeque;

use hashbrown::{HashMap, HashSet};

use crate::algorithms::arcsort::{ArcCompare, arc_sort};
use crate::algorithms::cc_visitors::SccVisitor;
use crate::algorithms::connect::connect;
use crate::algorithms::determinize::DELTA;
use crate::algorithms::dfs_visit::dfs_visit_any;
use crate::algorithms::project::{ProjectType, project};
use crate::arc::{Arc, ArcStateId};
use crate::data_structures::bi_table::CompactHashBiTable;
use crate::data_structures::union_find::UnionFind;
use crate::error::OpenFstError;
use crate::fst::{ExpandedFst, Fst, MutableFst};
use crate::fsts::vector_fst::VectorFst;
use crate::properties::{K_ACCEPTOR, K_FST_PROPERTIES};
use crate::weight::{Divide, DivideType, Weight};

/// How to disambiguate.
#[derive(Debug, Clone)]
pub struct DisambiguateOptions {
    /// How closely two subset weights must agree to count as the same.
    pub delta: f32,
    /// A cap on how many states to build, as
    /// [`DeterminizeOptions::max_states`](super::determinize::DeterminizeOptions::max_states):
    /// not every FST can be disambiguated, and the ones that cannot need
    /// unboundedly many states.
    pub max_states: Option<usize>,
}

impl Default for DisambiguateOptions {
    fn default() -> Self {
        Self {
            delta: DELTA,
            max_states: None,
        }
    }
}

/// Orders arcs by input label and then by where they lead, which is the order
/// the pre-disambiguation walks them in.
#[derive(Debug, Clone, Copy, Default)]
struct ByLabelAndTarget;

impl<A: Arc> ArcCompare<A> for ByLabelAndTarget {
    fn compare(&self, lhs: &A, rhs: &A) -> std::cmp::Ordering {
        (lhs.ilabel(), lhs.nextstate()).cmp(&(rhs.ilabel(), rhs.nextstate()))
    }

    fn properties(&self, props: u64) -> u64 {
        use crate::properties::{K_ARC_SORT_PROPERTIES, K_I_LABEL_SORTED, K_O_LABEL_SORTED};
        (props & K_ARC_SORT_PROPERTIES)
            | K_I_LABEL_SORTED
            | if props & K_ACCEPTOR != 0 {
                K_O_LABEL_SORTED
            } else {
                0
            }
    }
}

/// Which pairs of states could still spell the same thing to the end.
///
/// Two states are related when some path from one to a final state carries the
/// same labels as some path from the other, which is exactly co-accessibility
/// in the FST composed with itself. Composition of an acceptor with itself is
/// the product over matching labels, so it is built here directly.
fn common_future<A, F>(fst: &F) -> HashSet<(usize, usize)>
where
    A: Arc,
    F: Fst<A> + ExpandedFst<A>,
{
    let Some(start) = fst.start() else {
        return HashSet::new();
    };
    // The product's states are pairs; only the reachable ones are built.
    let mut pairs: CompactHashBiTable<usize, (usize, usize)> = CompactHashBiTable::new(1024);
    let mut product: VectorFst<A> = VectorFst::new();
    let mut pending: Vec<usize> = Vec::new();

    let start_pair = pairs
        .find_id(&(start.as_usize(), start.as_usize()), true)
        .expect("find_id inserts");
    product.add_state();
    product.set_start(A::StateId::from_usize(start_pair));
    pending.push(start_pair);

    let zero = A::Weight::zero();
    let mut by_label: HashMap<A::Label, Vec<A::StateId>> = HashMap::new();
    while let Some(id) = pending.pop() {
        let (s1, s2) = *pairs.find_entry(id).expect("just added");
        let state1 = A::StateId::from_usize(s1);
        let state2 = A::StateId::from_usize(s2);
        if fst.final_weight(state1) != zero && fst.final_weight(state2) != zero {
            product.set_final(A::StateId::from_usize(id), A::Weight::one());
        }
        by_label.clear();
        for arc in fst.arcs(state2) {
            by_label
                .entry(arc.ilabel())
                .or_default()
                .push(arc.nextstate());
        }
        let mut arcs: Vec<(A::Label, usize)> = Vec::new();
        for arc in fst.arcs(state1) {
            let Some(targets) = by_label.get(&arc.ilabel()) else {
                continue;
            };
            for target in targets {
                let before = pairs.size();
                let next = pairs
                    .find_id(&(arc.nextstate().as_usize(), target.as_usize()), true)
                    .expect("find_id inserts");
                if next == before {
                    pending.push(next);
                }
                arcs.push((arc.ilabel(), next));
            }
        }
        while product.num_states() < pairs.size() {
            product.add_state();
        }
        for (label, next) in arcs {
            product.add_arc(
                A::StateId::from_usize(id),
                A::new(label, label, A::Weight::one(), A::StateId::from_usize(next)),
            );
        }
    }

    // A pair is related when it can still reach a pair that is final on both
    // sides.
    let mut coaccess = crate::data_structures::bit_set::GrowableBitSet::new();
    let mut props = 0;
    {
        let mut visitor = SccVisitor::new(&product, None, None, Some(&mut coaccess), &mut props);
        dfs_visit_any(&product, &mut visitor);
    }
    let mut related = HashSet::new();
    for id in 0..pairs.size() {
        if coaccess.contains(id) {
            related.insert(*pairs.find_entry(id).expect("in range"));
        }
    }
    related
}

/// A state of the pre-disambiguation: a weighted subset, and the state of the
/// input it came from.
type Subset<S, W> = Vec<(S, W)>;

/// Determinizes, but only merging the states that could still agree, and
/// remembering which state of the input each result state came from.
///
/// The head is what distinguishes this from plain determinization: the arcs of
/// the result are the arcs of the *head*, and a state of the input joins a
/// destination subset only if it is related to that destination's head.
fn pre_disambiguate<A, F>(
    ifst: &F,
    ofst: &mut VectorFst<A>,
    related: &HashSet<(usize, usize)>,
    delta: f32,
    max_states: Option<usize>,
) -> Result<Vec<usize>, OpenFstError>
where
    A: Arc,
    A::Weight: Divide + std::hash::Hash + Eq,
    F: Fst<A> + ExpandedFst<A>,
{
    ofst.delete_all_states();
    ofst.set_input_symbols(ifst.input_symbols());
    ofst.set_output_symbols(ifst.output_symbols());
    let mut heads: Vec<usize> = Vec::new();

    let Some(istart) = ifst.start() else {
        return Ok(heads);
    };

    // A state is a subset together with its head.
    let mut states: CompactHashBiTable<usize, (usize, Subset<A::StateId, A::Weight>)> =
        CompactHashBiTable::new(1024);
    let mut pending: Vec<usize> = Vec::new();

    let start = states
        .find_id(&(istart.as_usize(), vec![(istart, A::Weight::one())]), true)
        .expect("find_id inserts");
    pending.push(start);
    ofst.add_state();
    ofst.set_start(A::StateId::from_usize(start));

    let zero = A::Weight::zero();
    while let Some(id) = pending.pop() {
        let (head, subset) = states.find_entry(id).expect("just added").clone();
        let state = A::StateId::from_usize(id);
        let head_state = A::StateId::from_usize(head);
        while heads.len() <= id {
            heads.push(usize::MAX);
        }
        heads[id] = head;

        // The result is final only where the head is; without that, two states
        // merged into one subset could each contribute a way of finishing.
        if ifst.final_weight(head_state) != zero {
            let mut weight = zero.clone();
            for (member, at) in &subset {
                weight = weight.plus(&at.times(&ifst.final_weight(*member)));
            }
            if weight != zero {
                ofst.set_final(state, weight);
            }
        }

        // The arcs of the result are the head's arcs, one per distinct
        // (label, destination); a multi-arc adds nothing new.
        let mut protos: Vec<(A::Label, usize, Vec<(A::StateId, A::Weight)>)> = Vec::new();
        let mut previous: Option<(A::Label, A::StateId)> = None;
        for arc in ifst.arcs(head_state) {
            if previous == Some((arc.ilabel(), arc.nextstate())) {
                continue;
            }
            previous = Some((arc.ilabel(), arc.nextstate()));
            protos.push((arc.ilabel(), arc.nextstate().as_usize(), Vec::new()));
        }

        // Each member's arcs join the destinations whose head they could still
        // agree with.
        for (member, at) in &subset {
            for arc in ifst.arcs(*member) {
                let weight = at.times(arc.weight());
                for (label, dest_head, elements) in protos.iter_mut() {
                    if *label != arc.ilabel() {
                        continue;
                    }
                    if related.contains(&(arc.nextstate().as_usize(), *dest_head)) {
                        elements.push((arc.nextstate(), weight.clone()));
                    }
                }
            }
        }

        let mut arcs: Vec<A> = Vec::with_capacity(protos.len());
        for (label, dest_head, mut elements) in protos {
            elements.sort_by_key(|(member, _)| *member);
            let mut arc_weight = zero.clone();
            for (_, weight) in &elements {
                arc_weight = arc_weight.plus(weight);
            }
            let mut merged: Subset<A::StateId, A::Weight> = Vec::with_capacity(elements.len());
            for (member, weight) in elements {
                match merged.last_mut() {
                    Some((last, at)) if *last == member => *at = at.plus(&weight),
                    _ => merged.push((member, weight)),
                }
            }
            for (_, weight) in &mut merged {
                *weight = weight.divide(&arc_weight, DivideType::Left).quantize(delta);
            }

            let before = states.size();
            let next = states
                .find_id(&(dest_head, merged), true)
                .expect("find_id inserts");
            if next == before {
                pending.push(next);
            }
            if max_states.is_some_and(|limit| states.size() > limit) {
                return Err(OpenFstError::InvalidOperation(format!(
                    "Disambiguate: more than {} states; the FST may not be disambiguable",
                    max_states.expect("just checked")
                )));
            }
            arcs.push(A::new(
                label,
                label,
                arc_weight,
                A::StateId::from_usize(next),
            ));
        }

        while ofst.num_states() < states.size() {
            ofst.add_state();
        }
        for arc in arcs {
            ofst.add_arc(state, arc);
        }
    }
    while heads.len() < states.size() {
        heads.push(usize::MAX);
    }
    Ok(heads)
}

/// A transition, as the state it leaves and its position among that state's
/// arcs; `None` for the way of finishing there.
type ArcId = (usize, Option<usize>);

/// What the ambiguity search found.
struct Ambiguities {
    /// Pairs of transitions that leave two paths spelling the same input, the
    /// first of each being the one that would go.
    candidates: Vec<(ArcId, ArcId)>,
    /// States that only quantization told apart, to be merged back.
    merge: Option<UnionFind>,
}

/// Finds the pairs of transitions that give two paths the same input.
fn find_ambiguities<A, F>(fst: &F, heads: &[usize]) -> Ambiguities
where
    A: Arc,
    F: Fst<A> + ExpandedFst<A>,
{
    let mut found = Ambiguities {
        candidates: Vec::new(),
        merge: None,
    };
    let Some(start) = fst.start() else {
        return found;
    };
    let nstates = fst.num_states();
    let head_of = |state: usize| heads.get(state).copied().unwrap_or(usize::MAX);

    let mut coreachable: HashSet<(usize, usize)> = HashSet::new();
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    coreachable.insert((start.as_usize(), start.as_usize()));
    queue.push_back((start.as_usize(), start.as_usize()));

    let zero = A::Weight::zero();
    let mut by_label: HashMap<A::Label, Vec<(usize, usize)>> = HashMap::new();

    while let Some((mut s1, mut s2)) = queue.pop_front() {
        // SICADA-DIVERGE: upstream swaps the two so that the smaller side is
        // walked and then *falls through* to do the original order as well,
        // so every pair is examined twice. The two orders find the same pairs,
        // since the candidate is normalized by head and the co-reachable pair by
        // state and both are symmetric, so only the cheaper order is run here.
        if fst.num_arcs(A::StateId::from_usize(s2)) > fst.num_arcs(A::StateId::from_usize(s1)) {
            std::mem::swap(&mut s1, &mut s2);
        }
        let state1 = A::StateId::from_usize(s1);
        let state2 = A::StateId::from_usize(s2);

        by_label.clear();
        for (position, arc) in fst.arcs(state2).enumerate() {
            by_label
                .entry(arc.ilabel())
                .or_default()
                .push((position, arc.nextstate().as_usize()));
        }

        for (position1, arc1) in fst.arcs(state1).enumerate() {
            let Some(matches) = by_label.get(&arc1.ilabel()) else {
                continue;
            };
            for (position2, next2) in matches {
                let next1 = arc1.nextstate().as_usize();
                // Two ways of reading the same label into the same state: one
                // of them has to go.
                if s1 != s2 && next1 == *next2 {
                    let a1 = (s1, Some(position1));
                    let a2 = (s2, Some(*position2));
                    found.candidates.push(if head_of(s1) > head_of(s2) {
                        (a1, a2)
                    } else {
                        (a2, a1)
                    });
                }
                let pair = if next1 <= *next2 {
                    (next1, *next2)
                } else {
                    (*next2, next1)
                };
                if !coreachable.insert(pair) {
                    continue;
                }
                if pair.0 != pair.1 && head_of(pair.0) == head_of(pair.1) {
                    // Only quantization put these apart; they are one state.
                    let merge = found.merge.get_or_insert_with(|| {
                        let mut sets = UnionFind::new(nstates);
                        sets.make_all_set(nstates);
                        sets
                    });
                    merge.union(pair.0, pair.1);
                } else {
                    queue.push_back(pair);
                }
            }
        }

        // Two ways of finishing here is the same ambiguity, at the end.
        if s1 != s2 && fst.final_weight(state1) != zero && fst.final_weight(state2) != zero {
            let a1 = (s1, None);
            let a2 = (s2, None);
            found.candidates.push(if head_of(s1) > head_of(s2) {
                (a1, a2)
            } else {
                (a2, a1)
            });
        }
    }
    found
}

/// Decides which transition of each ambiguous pair goes.
fn mark_ambiguities(candidates: &[(ArcId, ArcId)], heads: &[usize]) -> HashSet<ArcId> {
    let head_of = |state: usize| heads.get(state).copied().unwrap_or(usize::MAX);
    // Upstream keeps the candidates in a map ordered by the source's head, then
    // by the source, then by position, and walks them in that order, so which
    // of a pair survives does not depend on the order they were found in.
    let mut ordered: Vec<(ArcId, ArcId)> = candidates.to_vec();
    ordered.sort_by_key(|((state, position), _)| (head_of(*state), *state, *position));

    let mut ambiguous: HashSet<ArcId> = HashSet::new();
    for (a, b) in ordered {
        // If `b` is staying, then `a` is the one that goes.
        if !ambiguous.contains(&b) {
            ambiguous.insert(a);
        }
    }
    ambiguous
}

/// Points the merged-away states at the ones that stand for them.
fn remove_splits<A>(fst: &mut VectorFst<A>, merge: &mut UnionFind)
where
    A: Arc,
{
    let states: Vec<A::StateId> = fst.states().collect();
    for state in states {
        fst.mutate_arcs(state, |arc| {
            if let Some(to) = merge.find_set(arc.nextstate().as_usize())
                && to != arc.nextstate().as_usize()
            {
                *arc = A::new(
                    arc.ilabel(),
                    arc.olabel(),
                    arc.weight().clone(),
                    A::StateId::from_usize(to),
                );
            }
        });
    }
}

/// Takes out the transitions that were chosen to go.
fn remove_ambiguities<A>(fst: &mut VectorFst<A>, ambiguous: &HashSet<ArcId>)
where
    A: Arc,
{
    if ambiguous.is_empty() {
        return;
    }
    // A state nothing leads out of, to send the removed transitions to; connect
    // then takes them and it away.
    let dead = fst.add_state();
    let mut by_state: HashMap<usize, HashSet<usize>> = HashMap::new();
    for (state, position) in ambiguous {
        match position {
            Some(position) => {
                by_state.entry(*state).or_default().insert(*position);
            }
            None => fst.set_final(A::StateId::from_usize(*state), A::Weight::zero()),
        }
    }
    for (state, positions) in by_state {
        let mut index = 0usize;
        fst.mutate_arcs(A::StateId::from_usize(state), |arc| {
            if positions.contains(&index) {
                *arc = A::new(arc.ilabel(), arc.olabel(), arc.weight().clone(), dead);
            }
            index += 1;
        });
    }
    connect(fst);
}

/// Leaves one path per input string, keeping what the FST said.
///
/// SICADA-DIVERGE: a `max_states` cap, for the same reason
/// [`determinize`](super::determinize::determinize) has one. Not every FST can
/// be disambiguated, and upstream's delayed determinization only diverges when
/// the caller expands it all.
pub fn disambiguate<A, F1, F2>(
    ifst: &F1,
    ofst: &mut F2,
    opts: &DisambiguateOptions,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: Divide + std::hash::Hash + Eq,
    F1: Fst<A> + ExpandedFst<A>,
    F2: MutableFst<A> + ExpandedFst<A>,
{
    // The relation is over the input side alone, so a transducer is read as the
    // acceptor of what it reads.
    let mut source: VectorFst<A> = copy_of(ifst);
    connect(&mut source);
    arc_sort(&mut source, &ByLabelAndTarget);

    let related = if source.properties(K_ACCEPTOR, true) & K_ACCEPTOR != 0 {
        common_future(&source)
    } else {
        let mut inputs: VectorFst<A> = copy_of(&source);
        project(&mut inputs, ProjectType::Input)?;
        common_future(&inputs)
    };

    let mut result: VectorFst<A> = VectorFst::new();
    let mut heads = pre_disambiguate(&source, &mut result, &related, opts.delta, opts.max_states)?;
    arc_sort(&mut result, &ByLabelAndTarget);

    let mut found = find_ambiguities(&result, &heads);
    if let Some(mut merge) = found.merge.take() {
        remove_splits(&mut result, &mut merge);
        // The merge changed where arcs lead, so the search runs again on what
        // is now there.
        heads.resize(result.num_states(), usize::MAX);
        found = find_ambiguities(&result, &heads);
        if found.merge.is_some() {
            return Err(OpenFstError::InvalidOperation(
                "Disambiguate: could not remove the states quantization split apart".into(),
            ));
        }
    }
    let ambiguous = mark_ambiguities(&found.candidates, &heads);
    remove_ambiguities(&mut result, &ambiguous);

    ofst.delete_all_states();
    ofst.set_input_symbols(ifst.input_symbols());
    ofst.set_output_symbols(ifst.output_symbols());
    ofst.add_states(result.num_states());
    if let Some(start) = result.start() {
        ofst.set_start(start);
    }
    for state in result.states() {
        ofst.set_final(state, result.final_weight(state));
        for arc in result.arcs(state) {
            ofst.add_arc(state, arc);
        }
    }
    ofst.set_properties(result.properties(K_FST_PROPERTIES, false), K_FST_PROPERTIES);
    Ok(())
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
    use crate::algorithms::shortest_distance::shortest_distance;
    use crate::algorithms::test_support::{Rng, random_acyclic_fst, string_weights, visible_paths};
    use crate::arc::StdArc;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::K_FST_PROPERTIES;
    use crate::weights::float_weight::TropicalWeight;

    fn disambiguated(fst: &StdVectorFst) -> StdVectorFst {
        let mut out = StdVectorFst::new();
        disambiguate(
            fst,
            &mut out,
            &DisambiguateOptions {
                max_states: Some(4096),
                ..Default::default()
            },
        )
        .unwrap();
        out
    }

    /// The weight the FST gives each input string.
    fn language(fst: &StdVectorFst) -> Vec<(Vec<i32>, Vec<i32>, String)> {
        string_weights(visible_paths(fst, 12))
    }

    /// Whether two distinct paths spell the same input.
    fn is_unambiguous(fst: &StdVectorFst) -> bool {
        let mut inputs: Vec<Vec<i32>> = visible_paths(fst, 12)
            .into_iter()
            .map(|(ilabels, _, _)| ilabels)
            .collect();
        inputs.sort();
        let before = inputs.len();
        inputs.dedup();
        inputs.len() == before
    }

    /// Two ways of reading the same string become one, keeping the better.
    #[test]
    fn two_paths_spelling_the_same_thing_become_one() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(2.0), 1));
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(5.0), 2));
        fst.set_final(1, TropicalWeight::one());
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        assert!(!is_unambiguous(&fst));
        let out = disambiguated(&fst);
        assert!(is_unambiguous(&out));
        assert_eq!(language(&out), language(&fst));
    }

    /// An FST already unambiguous keeps saying what it said.
    #[test]
    fn an_unambiguous_fst_is_left_saying_the_same_thing() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(1.0), 1));
        fst.add_arc(0, StdArc::new(2, 2, TropicalWeight(2.0), 2));
        fst.set_final(1, TropicalWeight::one());
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let out = disambiguated(&fst);
        assert!(is_unambiguous(&out));
        assert_eq!(language(&out), language(&fst));
    }

    /// Two ways of finishing at the same place are an ambiguity too.
    #[test]
    fn two_ways_of_finishing_are_an_ambiguity() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 2));
        fst.set_final(1, TropicalWeight(3.0));
        fst.set_final(2, TropicalWeight(4.0));
        fst.properties(K_FST_PROPERTIES, true);

        let out = disambiguated(&fst);
        assert!(is_unambiguous(&out));
        assert_eq!(language(&out), language(&fst));
    }

    /// An empty FST disambiguates to an empty one.
    #[test]
    fn an_empty_fst_disambiguates_to_nothing() {
        assert_eq!(disambiguated(&StdVectorFst::new()).num_states(), 0);
    }

    /// Whatever the FST, the result is unambiguous and gives each string the
    /// same weight.
    #[test]
    fn disambiguating_keeps_the_weights_and_leaves_no_ambiguity() {
        let mut rng = Rng::new(0x00D1_5AB0_u64);
        let mut checked = 0;
        for round in 0..200 {
            let fst = random_acyclic_fst(&mut rng, 5);
            let before = language(&fst);
            if before.is_empty() {
                continue;
            }
            checked += 1;
            let out = disambiguated(&fst);
            assert!(is_unambiguous(&out), "round {round}");
            assert_eq!(language(&out), before, "round {round}");
        }
        assert!(checked > 50, "only {checked} FSTs said anything");
    }

    /// The total weight is unchanged, which is the other half of "says the
    /// same thing".
    #[test]
    fn the_total_weight_is_unchanged() {
        let mut rng = Rng::new(0x00D1_5AB1_u64);
        for round in 0..100 {
            let fst = random_acyclic_fst(&mut rng, 5);
            let before = shortest_distance(&fst, DELTA).unwrap();
            let out = disambiguated(&fst);
            let after = shortest_distance(&out, DELTA).unwrap();
            assert!(
                before.approx_equal(&after, 1e-4),
                "round {round}: {before:?} against {after:?}"
            );
        }
    }
}
