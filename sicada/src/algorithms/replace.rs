//! Expanding a recursive transition network into a single FST.
//!
//! Port of OpenFst's `replace.h`. An RTN is a set of FSTs each named by a
//! label; an arc whose output label names one of them stands for a call into
//! it. Replacement walks the network, following a call into the named FST and
//! coming back where the call left off, so that the whole thing becomes one
//! FST. It terminates exactly when the calls do not recurse in a way that
//! needs an unbounded stack.

use hashbrown::HashMap;

use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::data_structures::bi_table::CompactHashBiTable;
use crate::error::OpenFstError;
use crate::fst::{Fst, MutableFst};
use crate::properties::{
    K_COPY_PROPERTIES, K_FST_PROPERTIES, K_I_LABEL_SORTED, K_O_LABEL_SORTED, replace_properties,
};
use crate::weight::Weight;

/// Which side of a call or return arc carries a label.
///
/// Putting a label on one side only makes a transducer out of what may have
/// been acceptors, which is why the choice is offered rather than fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReplaceLabelType {
    /// Epsilon on both sides.
    Neither,
    /// A label on the input side, epsilon on the output.
    #[default]
    Input,
    /// Epsilon on the input side, a label on the output.
    Output,
    /// A label on both sides.
    Both,
}

impl ReplaceLabelType {
    /// Whether this leaves the input side epsilon.
    #[inline]
    pub fn epsilon_on_input(self) -> bool {
        matches!(self, Self::Neither | Self::Output)
    }

    /// Whether this leaves the output side epsilon.
    #[inline]
    pub fn epsilon_on_output(self) -> bool {
        matches!(self, Self::Neither | Self::Input)
    }
}

/// How to label the call and return arcs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplaceOptions<L> {
    /// The non-terminal to start from.
    pub root: L,
    /// How to label the arc into a called FST.
    pub call_label_type: ReplaceLabelType,
    /// How to label the arc back out of one.
    pub return_label_type: ReplaceLabelType,
    /// The output label to put on a call arc, or `None` to keep the one the
    /// arc already had.
    pub call_output_label: Option<L>,
    /// The label to put on a return arc.
    pub return_label: L,
}

impl<L: ArcLabel> ReplaceOptions<L> {
    /// Labels the call arc on its input side and the return arc on neither, as
    /// upstream does by default.
    pub fn new(root: L) -> Self {
        Self {
            root,
            call_label_type: ReplaceLabelType::Input,
            return_label_type: ReplaceLabelType::Neither,
            call_output_label: None,
            return_label: L::epsilon(),
        }
    }

    /// Puts epsilons on both sides of both arcs, so that the result accepts
    /// exactly what the network describes with nothing marking the calls.
    pub fn epsilon_calls(root: L) -> Self {
        Self {
            root,
            call_label_type: ReplaceLabelType::Neither,
            return_label_type: ReplaceLabelType::Neither,
            call_output_label: Some(L::epsilon()),
            return_label: L::epsilon(),
        }
    }

    /// Whether the call or return arcs can have different labels on their two
    /// sides, which can make the result a transducer even if the inputs were
    /// acceptors.
    pub fn makes_a_transducer(&self) -> bool {
        matches!(
            self.call_label_type,
            ReplaceLabelType::Input | ReplaceLabelType::Output
        ) || (self.call_label_type == ReplaceLabelType::Both && self.call_output_label.is_some())
            || matches!(
                self.return_label_type,
                ReplaceLabelType::Input | ReplaceLabelType::Output
            )
    }
}

/// A state of the expanded FST: where in which FST, under which call stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct StateTuple<S> {
    /// The call stack, as an index into the prefix table.
    prefix_id: usize,
    /// Which of the FSTs is being walked.
    fst_id: usize,
    /// Where in it.
    fst_state: S,
}

/// The call stacks, each identified by an index.
///
/// SICADA-OPT: upstream's `StackPrefix` is a vector of frames, and `PushPrefix`
/// takes it **by value**, so pushing a frame copies the whole stack and
/// interning it hashes the whole stack. A stack is only ever pushed onto or
/// popped from at the top, so it is stored here as a linked list: an entry is
/// its parent's index plus one frame. Two stacks get the same index exactly
/// when their parents do and their tops match, which by induction is exactly
/// when their contents match. That is the same identity upstream computes, at
/// O(1) per push instead of O(depth).
struct PrefixTable<S> {
    /// `(parent, fst_id, return_state)` per index; index 0 is the empty stack.
    entries: CompactHashBiTable<usize, (usize, usize, S)>,
}

impl<S: ArcStateId + std::hash::Hash> PrefixTable<S> {
    /// A table holding only the empty stack, at index 0.
    fn new() -> Self {
        let mut entries = CompactHashBiTable::new(64);
        // The empty stack is its own parent and returns nowhere.
        entries.find_id(&(0, 0, S::no_state()), true);
        Self { entries }
    }

    /// The stack `prefix` with one more frame on top.
    fn push(&mut self, prefix: usize, fst_id: usize, return_state: S) -> usize {
        self.entries
            .find_id(&(prefix, fst_id, return_state), true)
            .expect("find_id inserts when asked to")
    }

    /// The frame on top of `prefix` and what is left underneath, or `None` for
    /// the empty stack.
    fn top(&self, prefix: usize) -> Option<(usize, usize, S)> {
        if prefix == 0 {
            return None;
        }
        self.entries.find_entry(prefix).copied()
    }
}

/// Expands the recursive transition network `fsts` into `ofst`.
///
/// `fsts` pairs each non-terminal label with the FST it names. An arc whose
/// output label is one of those non-terminals is replaced by a call into that
/// FST, and a final state of a called FST by a return to where the call came
/// from.
///
/// SICADA-DIVERGE: upstream reports a root label that names none of the FSTs by
/// writing it into the non-terminal map with `operator[]`, which **inserts** a
/// zero entry, so the label then looks like a non-terminal naming the null FST
/// slot and every arc carrying it is deleted without a word. Here it is an
/// error.
///
/// SICADA-DIVERGE: a network whose calls recurse is refused. Every time a call
/// is followed the stack grows by a frame, and a state of the result is a
/// position in an FST *together with the stack that got there*, so a cycle in
/// the call graph means an unbounded number of states. Upstream's eager
/// `Replace` builds exactly this expansion and so runs until it is killed;
/// `ReplaceUtil::CyclicDependencies` exists to be asked beforehand, but nothing
/// makes the caller ask. Recursion is a property of the network, not of the
/// expansion, so the check is one pass over the arcs, the same pass the
/// properties need, and it is done here.
///
/// This applies to right recursion too, even though a right-recursive grammar
/// describes a regular language: the stack still deepens at every call. That is
/// what upstream's `kReplaceSCCRightLinear` records: such a component "can be
/// represented as finite-state despite any cyclic dependencies, but not by the
/// usual replacement operation".
pub fn replace<A, F1, F2>(
    fsts: &[(A::Label, &F1)],
    ofst: &mut F2,
    opts: &ReplaceOptions<A::Label>,
) -> Result<(), OpenFstError>
where
    A: Arc,
    F1: Fst<A>,
    F2: MutableFst<A>,
{
    ofst.delete_all_states();
    ofst.set_properties(0, K_FST_PROPERTIES);

    let mut nonterminal: HashMap<A::Label, usize> = HashMap::with_capacity(fsts.len());
    for (index, (label, _)) in fsts.iter().enumerate() {
        nonterminal.insert(*label, index);
    }
    let Some(&root) = nonterminal.get(&opts.root) else {
        return Err(OpenFstError::InvalidOperation(format!(
            "Replace: no FST for the root label {}",
            opts.root
        )));
    };

    // A fast reject before the hash lookup, exactly as upstream does it.
    let min_nonterminal = fsts.iter().map(|(label, _)| *label).min();
    let max_nonterminal = fsts.iter().map(|(label, _)| *label).max();

    if let Some(label) = recursive_nonterminal(fsts, &nonterminal, root) {
        return Err(OpenFstError::InvalidOperation(format!(
            "Replace: the non-terminal {label} calls itself, directly or through \
             others, so the expansion has no end"
        )));
    }

    ofst.set_input_symbols(fsts[root].1.input_symbols());
    ofst.set_output_symbols(fsts[root].1.output_symbols());

    let props = properties_of(fsts, root, opts);

    let Some(root_start) = fsts[root].1.start() else {
        ofst.set_properties(props, K_FST_PROPERTIES);
        return Ok(());
    };

    let mut prefixes = PrefixTable::<A::StateId>::new();
    let mut states: CompactHashBiTable<usize, StateTuple<A::StateId>> =
        CompactHashBiTable::new(1024);
    let mut pending: Vec<usize> = Vec::new();

    let find_state = |states: &mut CompactHashBiTable<usize, StateTuple<A::StateId>>,
                      pending: &mut Vec<usize>,
                      tuple: StateTuple<A::StateId>| {
        let before = states.size();
        let id = states
            .find_id(&tuple, true)
            .expect("find_id inserts when asked to");
        if id == before {
            pending.push(id);
        }
        id
    };

    let start = find_state(
        &mut states,
        &mut pending,
        StateTuple {
            prefix_id: 0,
            fst_id: root,
            fst_state: root_start,
        },
    );
    ofst.add_state();
    ofst.set_start(A::StateId::from_usize(start));

    let epsilon = A::Label::epsilon();
    let zero = A::Weight::zero();
    let mut arcs: Vec<A> = Vec::new();

    while let Some(id) = pending.pop() {
        let tuple = *states.find_entry(id).expect("the state was just added");
        let fst = fsts[tuple.fst_id].1;
        let out_state = A::StateId::from_usize(id);

        // A state of the root FST keeps its final weight; one reached through a
        // call returns instead, so its final weight goes on the return arc.
        let final_weight = fst.final_weight(tuple.fst_state);
        if tuple.prefix_id == 0 {
            ofst.set_final(out_state, final_weight.clone());
        }

        arcs.clear();
        if final_weight != zero
            && let Some((parent, caller, resume)) = prefixes.top(tuple.prefix_id)
        {
            let nextstate = find_state(
                &mut states,
                &mut pending,
                StateTuple {
                    prefix_id: parent,
                    fst_id: caller,
                    fst_state: resume,
                },
            );
            arcs.push(A::new(
                if opts.return_label_type.epsilon_on_input() {
                    epsilon
                } else {
                    opts.return_label
                },
                if opts.return_label_type.epsilon_on_output() {
                    epsilon
                } else {
                    opts.return_label
                },
                final_weight,
                A::StateId::from_usize(nextstate),
            ));
        }

        for arc in fst.arcs(tuple.fst_state) {
            let called = (arc.olabel() != epsilon
                && min_nonterminal.is_some_and(|min| arc.olabel() >= min)
                && max_nonterminal.is_some_and(|max| arc.olabel() <= max))
            .then(|| nonterminal.get(&arc.olabel()).copied())
            .flatten();

            let Some(called) = called else {
                let nextstate = find_state(
                    &mut states,
                    &mut pending,
                    StateTuple {
                        fst_state: arc.nextstate(),
                        ..tuple
                    },
                );
                arcs.push(A::new(
                    arc.ilabel(),
                    arc.olabel(),
                    arc.weight().clone(),
                    A::StateId::from_usize(nextstate),
                ));
                continue;
            };

            // A call into an FST with no start state accepts nothing, so the
            // arc is dropped rather than leading nowhere.
            let Some(called_start) = fsts[called].1.start() else {
                continue;
            };
            let prefix_id = prefixes.push(tuple.prefix_id, tuple.fst_id, arc.nextstate());
            let nextstate = find_state(
                &mut states,
                &mut pending,
                StateTuple {
                    prefix_id,
                    fst_id: called,
                    fst_state: called_start,
                },
            );
            arcs.push(A::new(
                if opts.call_label_type.epsilon_on_input() {
                    epsilon
                } else {
                    arc.ilabel()
                },
                if opts.call_label_type.epsilon_on_output() {
                    epsilon
                } else {
                    opts.call_output_label.unwrap_or_else(|| arc.olabel())
                },
                arc.weight().clone(),
                A::StateId::from_usize(nextstate),
            ));
        }

        while ofst.num_states() < states.size() {
            ofst.add_state();
        }
        for arc in arcs.drain(..) {
            ofst.add_arc(out_state, arc);
        }
    }

    ofst.set_properties(props, K_FST_PROPERTIES);
    Ok(())
}

/// The FST an arc's output label calls, if it names one.
///
/// Epsilon is never a call, whatever labels the FSTs were given: replacement
/// itself puts epsilons on the call and return arcs, so an expanded FST is full
/// of them, and a network whose root is labelled 0 would otherwise look like it
/// called itself from every one of them. Upstream tests `arc.olabel == 0`
/// before the lookup for the same reason.
pub(crate) fn called_fst<L: ArcLabel>(nonterminal: &HashMap<L, usize>, olabel: L) -> Option<usize> {
    if olabel == L::epsilon() {
        return None;
    }
    nonterminal.get(&olabel).copied()
}

/// The non-terminal on a call cycle reachable from the root, if there is one.
///
/// Only what the root can reach matters: a cycle among FSTs the root never
/// calls is never expanded. Upstream's `CyclicDependencies` reports the whole
/// dependency graph's `kCyclic`, so it is the coarser answer.
fn recursive_nonterminal<A, F>(
    fsts: &[(A::Label, &F)],
    nonterminal: &HashMap<A::Label, usize>,
    root: usize,
) -> Option<A::Label>
where
    A: Arc,
    F: Fst<A>,
{
    /// Where a node stands in the depth-first walk.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Mark {
        /// Not reached.
        White,
        /// On the stack: an edge back here closes a cycle.
        Grey,
        /// Finished.
        Black,
    }

    // The FSTs each FST calls, in the order their arcs give.
    let calls = |from: usize| -> Vec<usize> {
        let fst = fsts[from].1;
        let mut out = Vec::new();
        for state in fst.states() {
            for arc in fst.arcs(state) {
                if let Some(to) = called_fst(nonterminal, arc.olabel()) {
                    out.push(to);
                }
            }
        }
        out
    };

    let mut mark = vec![Mark::White; fsts.len()];
    // Each frame is a node and the calls out of it that are still to be walked.
    let mut stack: Vec<(usize, std::vec::IntoIter<usize>)> = Vec::new();
    mark[root] = Mark::Grey;
    stack.push((root, calls(root).into_iter()));

    while let Some((node, iter)) = stack.last_mut() {
        let node = *node;
        match iter.next() {
            Some(to) => match mark[to] {
                Mark::Grey => return Some(fsts[to].0),
                Mark::Black => {}
                Mark::White => {
                    mark[to] = Mark::Grey;
                    stack.push((to, calls(to).into_iter()));
                }
            },
            None => {
                mark[node] = Mark::Black;
                stack.pop();
            }
        }
    }
    None
}

/// The properties the result has, from those of the FSTs going in.
fn properties_of<A, F>(fsts: &[(A::Label, &F)], root: usize, opts: &ReplaceOptions<A::Label>) -> u64
where
    A: Arc,
    F: Fst<A>,
{
    let mut inprops = Vec::with_capacity(fsts.len());
    let mut all_ilabel_sorted = true;
    let mut all_olabel_sorted = true;
    let mut all_non_empty = true;
    // Either every non-terminal is negative, or they form a dense range
    // starting at 1, which lets the state table be a vector.
    let mut all_negative = true;
    let mut dense_range = true;
    let count = A::Label::from_i64(fsts.len() as i64);

    for (label, fst) in fsts {
        if *label >= A::Label::epsilon() {
            all_negative = false;
        }
        if count.is_none_or(|count| *label > count) || *label <= A::Label::epsilon() {
            dense_range = false;
        }
        if fst.start().is_none() {
            all_non_empty = false;
        }
        if fst.properties(K_I_LABEL_SORTED, false) & K_I_LABEL_SORTED == 0 {
            all_ilabel_sorted = false;
        }
        if fst.properties(K_O_LABEL_SORTED, false) & K_O_LABEL_SORTED == 0 {
            all_olabel_sorted = false;
        }
        inprops.push(fst.properties(K_COPY_PROPERTIES, false));
    }

    replace_properties(
        &inprops,
        root,
        opts.call_label_type.epsilon_on_input(),
        opts.return_label_type.epsilon_on_input(),
        opts.call_label_type.epsilon_on_output(),
        opts.return_label_type.epsilon_on_output(),
        opts.makes_a_transducer(),
        all_non_empty,
        all_ilabel_sorted,
        all_olabel_sorted,
        all_negative || dense_range,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{Path, paths, sorted};
    use crate::arc::StdArc;
    use crate::fst::ExpandedFst as _;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::{K_ACCEPTOR, K_ERROR};
    use crate::weights::float_weight::TropicalWeight;

    /// A linear acceptor over `labels`, with `weight` on the final state.
    fn chain(labels: &[i32], weight: TropicalWeight) -> StdVectorFst {
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
        fst.set_final(state, weight);
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    fn expand(
        fsts: &[(i32, &StdVectorFst)],
        opts: &ReplaceOptions<i32>,
    ) -> Result<StdVectorFst, OpenFstError> {
        let mut out = StdVectorFst::new();
        replace(fsts, &mut out, opts)?;
        Ok(out)
    }

    fn strings(fst: &StdVectorFst, max_len: usize) -> Vec<(Vec<i32>, Vec<i32>, String)> {
        sorted(paths(fst, max_len))
    }

    /// The root's non-terminal arc is replaced by the FST it names, with the
    /// call and return arcs turned into epsilons.
    #[test]
    fn a_non_terminal_becomes_the_fst_it_names() {
        // root: 1 NT 3, where NT = -1 spells 7 8.
        let mut root = StdVectorFst::new();
        for _ in 0..4 {
            root.add_state();
        }
        root.set_start(0);
        root.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        root.add_arc(1, StdArc::new(-1, -1, TropicalWeight::one(), 2));
        root.add_arc(2, StdArc::new(3, 3, TropicalWeight::one(), 3));
        root.set_final(3, TropicalWeight::one());
        root.properties(K_FST_PROPERTIES, true);
        let sub = chain(&[7, 8], TropicalWeight::one());

        let out = expand(&[(0, &root), (-1, &sub)], &ReplaceOptions::epsilon_calls(0)).unwrap();
        assert_eq!(
            strings(&out, 12),
            vec![(
                vec![1, 0, 7, 8, 0, 3],
                vec![1, 0, 7, 8, 0, 3],
                "0.0000".to_string()
            )],
            "the call and the return are the two epsilons"
        );
    }

    /// The call and return arcs are where the labels the options ask for go.
    #[test]
    fn the_label_types_decide_what_the_call_and_return_arcs_carry() {
        let mut root = StdVectorFst::new();
        for _ in 0..2 {
            root.add_state();
        }
        root.set_start(0);
        root.add_arc(0, StdArc::new(5, -1, TropicalWeight::one(), 1));
        root.set_final(1, TropicalWeight::one());
        root.properties(K_FST_PROPERTIES, true);
        let sub = chain(&[7], TropicalWeight::one());
        let fsts = [(0i32, &root), (-1, &sub)];

        // Input on the call arc, nothing on the return arc: the input side
        // keeps the call arc's own input label.
        let out = expand(&fsts, &ReplaceOptions::new(0)).unwrap();
        assert_eq!(
            strings(&out, 12),
            vec![(vec![5, 7, 0], vec![0, 7, 0], "0.0000".to_string())]
        );

        // Output on the call arc: the output side keeps the non-terminal.
        let mut opts = ReplaceOptions::new(0);
        opts.call_label_type = ReplaceLabelType::Output;
        let out = expand(&fsts, &opts).unwrap();
        assert_eq!(
            strings(&out, 12),
            vec![(vec![0, 7, 0], vec![-1, 7, 0], "0.0000".to_string())]
        );

        // Both sides, with a chosen output label.
        let mut opts = ReplaceOptions::new(0);
        opts.call_label_type = ReplaceLabelType::Both;
        opts.call_output_label = Some(99);
        let out = expand(&fsts, &opts).unwrap();
        assert_eq!(
            strings(&out, 12),
            vec![(vec![5, 7, 0], vec![99, 7, 0], "0.0000".to_string())]
        );

        // A label on the return arc marks where the call came back.
        let mut opts = ReplaceOptions::epsilon_calls(0);
        opts.return_label_type = ReplaceLabelType::Both;
        opts.return_label = 42;
        let out = expand(&fsts, &opts).unwrap();
        assert_eq!(
            strings(&out, 12),
            vec![(vec![0, 7, 42], vec![0, 7, 42], "0.0000".to_string())]
        );
    }

    /// The weight of a call arc and of the called FST's final state both land
    /// on the path, exactly once each.
    #[test]
    fn the_weights_of_the_call_and_the_return_are_both_kept() {
        let mut root = StdVectorFst::new();
        for _ in 0..2 {
            root.add_state();
        }
        root.set_start(0);
        root.add_arc(0, StdArc::new(-1, -1, TropicalWeight(1.0), 1));
        root.set_final(1, TropicalWeight(2.0));
        root.properties(K_FST_PROPERTIES, true);
        let sub = chain(&[7], TropicalWeight(4.0));

        let out = expand(&[(0, &root), (-1, &sub)], &ReplaceOptions::epsilon_calls(0)).unwrap();
        assert_eq!(
            strings(&out, 12),
            vec![(vec![0, 7, 0], vec![0, 7, 0], "7.0000".to_string())],
            "1 on the call, 4 on the return, 2 at the end"
        );
    }

    /// The same FST called from two places comes back to the right place each
    /// time, which is the whole point of the call stack.
    #[test]
    fn a_call_returns_to_where_it_came_from() {
        let mut root = StdVectorFst::new();
        for _ in 0..5 {
            root.add_state();
        }
        root.set_start(0);
        // Two branches, each calling the same non-terminal and continuing
        // somewhere different.
        root.add_arc(0, StdArc::new(-1, -1, TropicalWeight::one(), 1));
        root.add_arc(1, StdArc::new(1, 1, TropicalWeight::one(), 2));
        root.add_arc(0, StdArc::new(-1, -1, TropicalWeight::one(), 3));
        root.add_arc(3, StdArc::new(2, 2, TropicalWeight::one(), 4));
        root.set_final(2, TropicalWeight::one());
        root.set_final(4, TropicalWeight::one());
        root.properties(K_FST_PROPERTIES, true);
        let sub = chain(&[9], TropicalWeight::one());

        let out = expand(&[(0, &root), (-1, &sub)], &ReplaceOptions::epsilon_calls(0)).unwrap();
        let got: Vec<Vec<i32>> = strings(&out, 12)
            .into_iter()
            .map(|(ilabels, _, _)| ilabels.into_iter().filter(|l| *l != 0).collect())
            .collect();
        assert_eq!(got, vec![vec![9, 1], vec![9, 2]]);
    }

    /// A non-terminal inside a called FST nests, and the stack unwinds in
    /// order.
    #[test]
    fn calls_nest() {
        let mut root = StdVectorFst::new();
        for _ in 0..3 {
            root.add_state();
        }
        root.set_start(0);
        root.add_arc(0, StdArc::new(-1, -1, TropicalWeight::one(), 1));
        root.add_arc(1, StdArc::new(1, 1, TropicalWeight::one(), 2));
        root.set_final(2, TropicalWeight::one());
        root.properties(K_FST_PROPERTIES, true);

        let mut middle = StdVectorFst::new();
        for _ in 0..3 {
            middle.add_state();
        }
        middle.set_start(0);
        middle.add_arc(0, StdArc::new(-2, -2, TropicalWeight::one(), 1));
        middle.add_arc(1, StdArc::new(2, 2, TropicalWeight::one(), 2));
        middle.set_final(2, TropicalWeight::one());
        middle.properties(K_FST_PROPERTIES, true);

        let inner = chain(&[3], TropicalWeight::one());

        let out = expand(
            &[(0, &root), (-1, &middle), (-2, &inner)],
            &ReplaceOptions::epsilon_calls(0),
        )
        .unwrap();
        let got: Vec<i32> = strings(&out, 20)[0]
            .0
            .iter()
            .copied()
            .filter(|l| *l != 0)
            .collect();
        assert_eq!(got, vec![3, 2, 1]);
    }

    /// Right recursion terminates: the stack never grows, because the call is
    /// A rule that calls itself has no finite expansion: a state of the result
    /// is a position *plus the call stack that reached it*, and every call
    /// deepens the stack. This holds for right recursion too, even though the
    /// language it describes is regular.
    #[test]
    fn a_recursive_rule_is_refused() {
        // NT -> 1 | 1 NT, with the recursive call last: right recursion.
        let mut right = StdVectorFst::new();
        for _ in 0..2 {
            right.add_state();
        }
        right.set_start(0);
        right.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        right.add_arc(1, StdArc::new(-1, -1, TropicalWeight::one(), 1));
        right.set_final(1, TropicalWeight::one());
        right.properties(K_FST_PROPERTIES, true);

        let Err(err) = expand(&[(-1, &right)], &ReplaceOptions::epsilon_calls(-1)) else {
            panic!("a recursive rule has no finite expansion")
        };
        assert!(format!("{err}").contains("calls itself"), "{err}");

        // And through another non-terminal rather than directly.
        let mut a = StdVectorFst::new();
        for _ in 0..2 {
            a.add_state();
        }
        a.set_start(0);
        a.add_arc(0, StdArc::new(-2, -2, TropicalWeight::one(), 1));
        a.set_final(1, TropicalWeight::one());

        let mut b = StdVectorFst::new();
        for _ in 0..2 {
            b.add_state();
        }
        b.set_start(0);
        b.add_arc(0, StdArc::new(-1, -1, TropicalWeight::one(), 1));
        b.set_final(1, TropicalWeight::one());

        assert!(expand(&[(-1, &a), (-2, &b)], &ReplaceOptions::epsilon_calls(-1)).is_err());
    }

    /// A cycle among rules the root never calls is never expanded, so it is not
    /// a reason to refuse. Upstream's `CyclicDependencies` reports the whole
    /// graph and would call this cyclic.
    #[test]
    fn a_cycle_the_root_never_calls_does_not_matter() {
        let mut root = StdVectorFst::new();
        for _ in 0..2 {
            root.add_state();
        }
        root.set_start(0);
        root.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        root.set_final(1, TropicalWeight::one());

        // -2 calls itself, but nothing calls -2.
        let mut loner = StdVectorFst::new();
        for _ in 0..2 {
            loner.add_state();
        }
        loner.set_start(0);
        loner.add_arc(0, StdArc::new(-2, -2, TropicalWeight::one(), 1));
        loner.set_final(1, TropicalWeight::one());

        let out = expand(
            &[(-1, &root), (-2, &loner)],
            &ReplaceOptions::epsilon_calls(-1),
        )
        .unwrap();
        assert_eq!(strings(&out, 4).len(), 1);
    }

    /// An FST with no start state accepts nothing, so a call into it leads
    /// nowhere and the arc is dropped.
    #[test]
    fn a_call_into_an_empty_fst_is_dropped() {
        let mut root = StdVectorFst::new();
        for _ in 0..3 {
            root.add_state();
        }
        root.set_start(0);
        root.add_arc(0, StdArc::new(-1, -1, TropicalWeight::one(), 1));
        root.add_arc(0, StdArc::new(5, 5, TropicalWeight::one(), 2));
        root.set_final(1, TropicalWeight::one());
        root.set_final(2, TropicalWeight::one());
        root.properties(K_FST_PROPERTIES, true);
        let empty = StdVectorFst::new();

        let out = expand(
            &[(0, &root), (-1, &empty)],
            &ReplaceOptions::epsilon_calls(0),
        )
        .unwrap();
        assert_eq!(
            strings(&out, 12),
            vec![(vec![5], vec![5], "0.0000".to_string())],
            "only the path that does not call the empty FST survives"
        );
    }

    /// A root that names none of the FSTs is refused. Upstream writes the
    /// missing label into its non-terminal map with `operator[]`, which inserts
    /// a zero, so the label then names the null FST slot and every arc carrying
    /// it disappears without a word.
    #[test]
    fn a_root_that_names_no_fst_is_refused() {
        let sub = chain(&[1], TropicalWeight::one());
        let Err(err) = expand(&[(-1, &sub)], &ReplaceOptions::epsilon_calls(7)) else {
            panic!("a root naming no FST must be refused")
        };
        assert!(format!("{err}").contains("root label"), "{err}");
    }

    /// An empty root gives an empty result rather than an error.
    #[test]
    fn an_empty_root_gives_an_empty_result() {
        let empty = StdVectorFst::new();
        let out = expand(&[(0, &empty)], &ReplaceOptions::epsilon_calls(0)).unwrap();
        assert_eq!(out.start(), None);
        assert_eq!(out.num_states(), 0);
    }

    /// One FST and no non-terminal arcs is a copy.
    #[test]
    fn a_network_of_one_fst_is_a_copy_of_it() {
        let root = chain(&[1, 2, 3], TropicalWeight(4.0));
        let out = expand(&[(0, &root)], &ReplaceOptions::epsilon_calls(0)).unwrap();
        assert_eq!(strings(&out, 12), strings(&root, 12));
        assert_eq!(out.num_states(), root.num_states());
    }

    /// The properties claimed have to be ones the result actually has.
    #[test]
    fn the_claimed_properties_are_the_ones_the_result_has() {
        let mut root = StdVectorFst::new();
        for _ in 0..3 {
            root.add_state();
        }
        root.set_start(0);
        root.add_arc(0, StdArc::new(-1, -1, TropicalWeight::one(), 1));
        root.add_arc(1, StdArc::new(4, 4, TropicalWeight::one(), 2));
        root.set_final(2, TropicalWeight::one());
        root.properties(K_FST_PROPERTIES, true);
        let sub = chain(&[7, 8], TropicalWeight::one());
        let fsts = [(0i32, &root), (-1, &sub)];

        for opts in [
            ReplaceOptions::epsilon_calls(0),
            ReplaceOptions::new(0),
            ReplaceOptions {
                call_label_type: ReplaceLabelType::Both,
                return_label_type: ReplaceLabelType::Both,
                return_label: 3,
                ..ReplaceOptions::new(0)
            },
        ] {
            let out = expand(&fsts, &opts).unwrap();
            let claimed = out.properties(K_FST_PROPERTIES, false);
            let actual = out.properties(K_FST_PROPERTIES, true);
            assert_eq!(claimed & K_ERROR, 0);
            assert_eq!(
                claimed & !actual & K_FST_PROPERTIES,
                0,
                "claimed {:#x} that the result does not have",
                claimed & !actual
            );
        }

        // Epsilon calls between acceptors keep the result an acceptor; a label
        // on one side only does not.
        let out = expand(&fsts, &ReplaceOptions::epsilon_calls(0)).unwrap();
        assert_ne!(out.properties(K_ACCEPTOR, true) & K_ACCEPTOR, 0);
        let out = expand(&fsts, &ReplaceOptions::new(0)).unwrap();
        assert_eq!(out.properties(K_ACCEPTOR, true) & K_ACCEPTOR, 0);
    }

    /// The states of the result are the reachable (stack, FST, state) triples,
    /// with nothing left over.
    #[test]
    fn every_state_of_the_result_is_reachable() {
        let mut root = StdVectorFst::new();
        for _ in 0..3 {
            root.add_state();
        }
        root.set_start(0);
        root.add_arc(0, StdArc::new(-1, -1, TropicalWeight::one(), 1));
        root.add_arc(1, StdArc::new(-1, -1, TropicalWeight::one(), 2));
        root.set_final(2, TropicalWeight::one());
        root.properties(K_FST_PROPERTIES, true);
        let sub = chain(&[7], TropicalWeight::one());

        let out = expand(&[(0, &root), (-1, &sub)], &ReplaceOptions::epsilon_calls(0)).unwrap();

        let mut seen = vec![false; out.num_states()];
        let mut stack = vec![out.start().unwrap()];
        seen[out.start().unwrap() as usize] = true;
        while let Some(s) = stack.pop() {
            for arc in out.arcs(s) {
                if !seen[arc.nextstate() as usize] {
                    seen[arc.nextstate() as usize] = true;
                    stack.push(arc.nextstate());
                }
            }
        }
        assert!(seen.iter().all(|s| *s), "{seen:?}");
    }

    /// Every path of the result is a path of the network, and vice versa: an
    /// RTN whose calls all bottom out is the same as substituting the FSTs in
    /// by hand.
    #[test]
    fn expanding_agrees_with_substituting_by_hand() {
        // root = A NT B, NT = C | D. By hand: A C B and A D B.
        let mut root = StdVectorFst::new();
        for _ in 0..4 {
            root.add_state();
        }
        root.set_start(0);
        root.add_arc(0, StdArc::new(1, 1, TropicalWeight(1.0), 1));
        root.add_arc(1, StdArc::new(-1, -1, TropicalWeight(2.0), 2));
        root.add_arc(2, StdArc::new(4, 4, TropicalWeight(3.0), 3));
        root.set_final(3, TropicalWeight(5.0));
        root.properties(K_FST_PROPERTIES, true);

        let mut sub = StdVectorFst::new();
        for _ in 0..2 {
            sub.add_state();
        }
        sub.set_start(0);
        sub.add_arc(0, StdArc::new(2, 2, TropicalWeight(7.0), 1));
        sub.add_arc(0, StdArc::new(3, 3, TropicalWeight(11.0), 1));
        sub.set_final(1, TropicalWeight(13.0));
        sub.properties(K_FST_PROPERTIES, true);

        let out = expand(&[(0, &root), (-1, &sub)], &ReplaceOptions::epsilon_calls(0)).unwrap();

        // Substituting by hand: two chains, each weighing the sum of the parts.
        let mut want = Vec::new();
        for (middle, weight) in [(2, 7.0), (3, 11.0)] {
            want.push((
                vec![1, middle, 4],
                vec![1, middle, 4],
                format!("{:.4}", 1.0 + 2.0 + weight + 13.0 + 3.0 + 5.0),
            ));
        }
        want.sort();

        let got: Vec<(Vec<i32>, Vec<i32>, String)> = strings(&out, 12)
            .into_iter()
            .map(|(ilabels, olabels, weight)| {
                (
                    ilabels.into_iter().filter(|l| *l != 0).collect(),
                    olabels.into_iter().filter(|l| *l != 0).collect(),
                    weight,
                )
            })
            .collect();
        assert_eq!(got, want);
    }

    /// A stack that reaches the same contents twice gets the same index, so an
    /// FST that calls the same rule from the same place converges rather than
    /// growing without bound.
    #[test]
    fn stacks_with_the_same_contents_are_the_same_state() {
        let mut prefixes = PrefixTable::<i32>::new();
        assert_eq!(prefixes.top(0), None, "index 0 is the empty stack");

        let a = prefixes.push(0, 1, 5);
        let b = prefixes.push(0, 1, 5);
        assert_eq!(a, b);
        assert_ne!(a, prefixes.push(0, 1, 6));
        assert_ne!(a, prefixes.push(0, 2, 5));

        let deep = prefixes.push(a, 3, 9);
        assert_eq!(prefixes.top(deep), Some((a, 3, 9)));
        assert_eq!(prefixes.top(a), Some((0, 1, 5)));
        assert_eq!(deep, prefixes.push(a, 3, 9));
    }

    #[allow(unused)]
    fn paths_type_check(fst: &StdVectorFst) -> Vec<Path> {
        paths(fst, 4)
    }
}
