//! Rewriting an FST a state at a time.
//!
//! Port of OpenFst's `state-map.h`. Where [`arc_map`](super::arc_map) rewrites
//! each arc on its own and cannot change how many there are, a state mapper is
//! handed a whole state and returns whatever arcs it likes, so it can combine
//! duplicates or drop them.

use crate::algorithms::arc_map::MapSymbolsAction;
use crate::arc::{Arc, ArcStateId};
use crate::fst::{Fst, MutableFst};
use crate::properties::{
    K_ARC_SORT_PROPERTIES, K_COPY_PROPERTIES, K_DELETE_ARCS_PROPERTIES, K_FST_PROPERTIES,
    K_WEIGHT_INVARIANT_PROPERTIES,
};
use crate::weight::Weight;

/// Produces the arcs and final weight a state should have.
pub trait StateMapper<From: Arc, To: Arc> {
    /// The initial state of the result.
    fn start(&self) -> Option<To::StateId>;

    /// The final weight the result's `state` should have.
    fn final_weight(&self, state: From::StateId) -> To::Weight;

    /// The arcs the result's `state` should have.
    fn arcs(&mut self, state: From::StateId) -> Vec<To>;

    /// What to do about the input symbol table.
    fn input_symbols_action(&self) -> MapSymbolsAction {
        MapSymbolsAction::Copy
    }

    /// What to do about the output symbol table.
    fn output_symbols_action(&self) -> MapSymbolsAction {
        MapSymbolsAction::Copy
    }

    /// The properties the result has, given those of the input.
    fn properties(&self, props: u64) -> u64;
}

/// Rewrites `fst` in place, one state at a time.
///
/// SICADA-DIVERGE: upstream's mapper is an iterator (`SetState`, then `Done`,
/// `Value`, `Next`), which forces every one of them to keep a cursor and a
/// buffer as members. Returning the arcs says the same thing, and the two
/// mappers here were building exactly such a buffer anyway.
pub fn state_map<A, F, M>(fst: &mut F, mapper: &mut M)
where
    A: Arc,
    F: MutableFst<A>,
    M: StateMapper<A, A>,
{
    if mapper.input_symbols_action() == MapSymbolsAction::Clear {
        fst.set_input_symbols(None);
    }
    if mapper.output_symbols_action() == MapSymbolsAction::Clear {
        fst.set_output_symbols(None);
    }
    if fst.start().is_none() {
        return;
    }

    let props = fst.properties(K_FST_PROPERTIES, false);
    let states: Vec<A::StateId> = fst.states().collect();
    // Every state's replacement is worked out before anything is changed, since
    // a mapper reads the FST it is rewriting.
    let replacements: Vec<(Vec<A>, A::Weight)> = states
        .iter()
        .map(|&state| (mapper.arcs(state), mapper.final_weight(state)))
        .collect();

    if let Some(start) = mapper.start() {
        fst.set_start(start);
    }
    for (&state, (arcs, weight)) in states.iter().zip(replacements) {
        fst.delete_arcs(state);
        for arc in arcs {
            fst.add_arc(state, arc);
        }
        fst.set_final(state, weight);
    }
    fst.set_properties(mapper.properties(props), K_FST_PROPERTIES);
}

/// Rewrites `ifst` into `ofst`, one state at a time.
pub fn state_map_to<From, To, F1, F2, M>(ifst: &F1, ofst: &mut F2, mapper: &mut M)
where
    From: Arc,
    To: Arc<StateId = From::StateId>,
    F1: Fst<From>,
    F2: MutableFst<To>,
    M: StateMapper<From, To>,
{
    ofst.delete_all_states();
    match mapper.input_symbols_action() {
        MapSymbolsAction::Copy => ofst.set_input_symbols(ifst.input_symbols()),
        MapSymbolsAction::Clear => ofst.set_input_symbols(None),
        MapSymbolsAction::Noop => {}
    }
    match mapper.output_symbols_action() {
        MapSymbolsAction::Copy => ofst.set_output_symbols(ifst.output_symbols()),
        MapSymbolsAction::Clear => ofst.set_output_symbols(None),
        MapSymbolsAction::Noop => {}
    }

    let iprops = ifst.properties(K_COPY_PROPERTIES, false);
    if ifst.start().is_none() {
        return;
    }
    if let Some(num_states) = ifst.num_states_if_known() {
        ofst.reserve_states(num_states);
    }
    for _ in ifst.states() {
        ofst.add_state();
    }
    if let Some(start) = mapper.start() {
        ofst.set_start(start);
    }
    for state in ifst.states() {
        for arc in mapper.arcs(state) {
            ofst.add_arc(state, arc);
        }
        ofst.set_final(state, mapper.final_weight(state));
    }
    let oprops = ofst.properties(K_FST_PROPERTIES, false);
    ofst.set_properties(mapper.properties(iprops) | oprops, K_FST_PROPERTIES);
}

/// The order two arcs are compared in when looking for duplicates: by labels,
/// then by where they lead. Weights are deliberately not part of it, since two
/// arcs that differ only in weight are the duplicates being looked for.
fn arc_key<A: Arc>(arc: &A) -> (A::Label, A::Label, usize) {
    (arc.ilabel(), arc.olabel(), arc.nextstate().as_usize())
}

/// Replaces arcs that agree on labels and destination with a single arc whose
/// weight is their sum.
///
/// Two such arcs are two ways of doing the same thing, so in a semiring they
/// combine with ⊕.
pub struct ArcSumMapper<'a, A: Arc, F: Fst<A>> {
    fst: &'a F,
    _marker: std::marker::PhantomData<A>,
}

impl<'a, A: Arc, F: Fst<A>> ArcSumMapper<'a, A, F> {
    /// Sums the duplicate arcs of `fst`.
    pub fn new(fst: &'a F) -> Self {
        Self {
            fst,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<A: Arc, F: Fst<A>> StateMapper<A, A> for ArcSumMapper<'_, A, F> {
    fn start(&self) -> Option<A::StateId> {
        self.fst.start()
    }

    fn final_weight(&self, state: A::StateId) -> A::Weight {
        self.fst.final_weight(state)
    }

    fn arcs(&mut self, state: A::StateId) -> Vec<A> {
        let mut arcs: Vec<A> = self.fst.arcs(state).collect();
        arcs.sort_by_key(arc_key::<A>);
        let mut out: Vec<A> = Vec::with_capacity(arcs.len());
        for arc in arcs {
            match out.last_mut() {
                Some(last) if arc_key::<A>(last) == arc_key::<A>(&arc) => {
                    *last = A::new(
                        last.ilabel(),
                        last.olabel(),
                        last.weight().plus(arc.weight()),
                        last.nextstate(),
                    );
                }
                _ => out.push(arc),
            }
        }
        out
    }

    fn properties(&self, props: u64) -> u64 {
        props & K_ARC_SORT_PROPERTIES & K_DELETE_ARCS_PROPERTIES & K_WEIGHT_INVARIANT_PROPERTIES
    }
}

/// Keeps one of each set of arcs that agree on labels, destination *and*
/// weight, dropping the rest.
///
/// Unlike [`ArcSumMapper`], two arcs differing only in weight are both kept:
/// they are different arcs.
pub struct ArcUniqueMapper<'a, A: Arc, F: Fst<A>> {
    fst: &'a F,
    _marker: std::marker::PhantomData<A>,
}

impl<'a, A: Arc, F: Fst<A>> ArcUniqueMapper<'a, A, F> {
    /// Removes the repeated arcs of `fst`.
    pub fn new(fst: &'a F) -> Self {
        Self {
            fst,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<A: Arc, F: Fst<A>> StateMapper<A, A> for ArcUniqueMapper<'_, A, F> {
    fn start(&self) -> Option<A::StateId> {
        self.fst.start()
    }

    fn final_weight(&self, state: A::StateId) -> A::Weight {
        self.fst.final_weight(state)
    }

    fn arcs(&mut self, state: A::StateId) -> Vec<A> {
        let mut arcs: Vec<A> = self.fst.arcs(state).collect();
        arcs.sort_by_key(arc_key::<A>);

        // SICADA-DIVERGE: upstream sorts by labels and destination only, then
        // calls `std::unique` with an equality that also compares the weight.
        // Arcs differing only in weight sort as equal, so whether two identical
        // ones end up adjacent, and hence whether one is removed, depends on
        // how `std::sort` happened to order equal elements, which is
        // unspecified. Comparing against every arc already kept from the same
        // run makes the result the same every time: the first of each set
        // survives. A run is the arcs of one state sharing labels and a
        // destination, which is a handful at most.
        let mut out: Vec<A> = Vec::with_capacity(arcs.len());
        let mut run_start = 0;
        for arc in arcs {
            let key = arc_key::<A>(&arc);
            if out
                .get(run_start)
                .is_none_or(|first| arc_key::<A>(first) != key)
            {
                run_start = out.len();
                out.push(arc);
                continue;
            }
            if !out[run_start..]
                .iter()
                .any(|kept| kept.weight() == arc.weight())
            {
                out.push(arc);
            }
        }
        out
    }

    fn properties(&self, props: u64) -> u64 {
        props & K_ARC_SORT_PROPERTIES & K_DELETE_ARCS_PROPERTIES
    }
}

/// Leaves every state as it is.
pub struct IdentityStateMapper<'a, A: Arc, F: Fst<A>> {
    fst: &'a F,
    _marker: std::marker::PhantomData<A>,
}

impl<'a, A: Arc, F: Fst<A>> IdentityStateMapper<'a, A, F> {
    /// Reproduces `fst`.
    pub fn new(fst: &'a F) -> Self {
        Self {
            fst,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<A: Arc, F: Fst<A>> StateMapper<A, A> for IdentityStateMapper<'_, A, F> {
    fn start(&self) -> Option<A::StateId> {
        self.fst.start()
    }

    fn final_weight(&self, state: A::StateId) -> A::Weight {
        self.fst.final_weight(state)
    }

    fn arcs(&mut self, state: A::StateId) -> Vec<A> {
        self.fst.arcs(state).collect()
    }

    fn properties(&self, props: u64) -> u64 {
        props
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{string_weights, visible_paths};
    use crate::arc::StdArc;
    use crate::fst::ExpandedFst as _;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::weights::float_weight::TropicalWeight;

    /// Two arcs from 0 to 1 with the same labels and different weights, one
    /// exact duplicate of the first, and one arc that differs.
    fn with_duplicates() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.set_final(1, TropicalWeight::one());
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(5.0), 1));
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(3.0), 1));
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(5.0), 1));
        fst.add_arc(0, StdArc::new(2, 2, TropicalWeight(1.0), 1));
        fst
    }

    fn arcs(fst: &StdVectorFst) -> Vec<(i32, i32, f32, i32)> {
        (0..fst.num_states() as i32)
            .flat_map(|s| {
                fst.arcs(s)
                    .map(|a| (a.ilabel(), a.olabel(), a.weight().value(), a.nextstate()))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Arcs that do the same thing become one arc whose weight is their sum,
    /// which in the tropical semiring is the minimum.
    #[test]
    fn summing_combines_arcs_that_do_the_same_thing() {
        let source = with_duplicates();
        let mut fst = source.clone();
        {
            let mut mapper = ArcSumMapper::new(&source);
            state_map(&mut fst, &mut mapper);
        }
        assert_eq!(arcs(&fst), vec![(1, 1, 3.0, 1), (2, 2, 1.0, 1)]);
    }

    /// Summing cannot change what the FST accepts: the arcs it merges were
    /// alternatives for the same string.
    #[test]
    fn summing_preserves_what_the_fst_accepts() {
        let source = with_duplicates();
        let mut fst = source.clone();
        let before = string_weights(visible_paths(&source, 6));
        {
            let mut mapper = ArcSumMapper::new(&source);
            state_map(&mut fst, &mut mapper);
        }
        assert_eq!(string_weights(visible_paths(&fst, 6)), before);
    }

    /// Uniquing keeps arcs that differ in weight, because they are different
    /// arcs; only exact repeats go.
    #[test]
    fn uniquing_keeps_arcs_that_differ_in_weight() {
        let source = with_duplicates();
        let mut fst = source.clone();
        {
            let mut mapper = ArcUniqueMapper::new(&source);
            state_map(&mut fst, &mut mapper);
        }
        assert_eq!(
            arcs(&fst),
            vec![(1, 1, 5.0, 1), (1, 1, 3.0, 1), (2, 2, 1.0, 1)],
            "the exact repeat of the 5.0 arc is gone; the 3.0 one stays"
        );
    }

    /// The result does not depend on how equal-keyed arcs happened to be
    /// ordered, which upstream's does.
    #[test]
    fn uniquing_gives_the_same_answer_however_the_arcs_arrive() {
        let orderings = [
            [(5.0, 0), (3.0, 1), (5.0, 2)],
            [(3.0, 0), (5.0, 1), (5.0, 2)],
            [(5.0, 0), (5.0, 1), (3.0, 2)],
        ];
        let mut results = Vec::new();
        for ordering in orderings {
            let mut source = StdVectorFst::new();
            for _ in 0..2 {
                source.add_state();
            }
            source.set_start(0);
            source.set_final(1, TropicalWeight::one());
            for (weight, _) in ordering {
                source.add_arc(0, StdArc::new(1, 1, TropicalWeight(weight), 1));
            }

            let mut fst = source.clone();
            let mut mapper = ArcUniqueMapper::new(&source);
            state_map(&mut fst, &mut mapper);
            let mut weights: Vec<f32> = fst.arcs(0).map(|a| a.weight().value()).collect();
            weights.sort_by(f32::total_cmp);
            results.push(weights);
        }
        assert_eq!(results[0], vec![3.0, 5.0]);
        assert!(
            results.iter().all(|r| *r == results[0]),
            "the answer moved with the input order: {results:?}"
        );
    }

    #[test]
    fn the_identity_mapper_reproduces_the_fst() {
        let source = with_duplicates();
        let mut ofst = StdVectorFst::new();
        {
            let mut mapper = IdentityStateMapper::new(&source);
            state_map_to(&source, &mut ofst, &mut mapper);
        }
        assert_eq!(arcs(&ofst), arcs(&source));
        assert_eq!(ofst.start(), source.start());
        assert_eq!(ofst.final_weight(1), source.final_weight(1));
    }

    #[test]
    fn mapping_into_another_fst_leaves_the_input_alone() {
        let source = with_duplicates();
        let mut ofst = StdVectorFst::new();
        {
            let mut mapper = ArcSumMapper::new(&source);
            state_map_to(&source, &mut ofst, &mut mapper);
        }
        assert_eq!(arcs(&ofst), vec![(1, 1, 3.0, 1), (2, 2, 1.0, 1)]);
        assert_eq!(arcs(&source).len(), 4, "the input still has its duplicates");
    }

    #[test]
    fn an_fst_with_no_start_state_is_left_alone() {
        let source = StdVectorFst::new();
        let mut ofst = StdVectorFst::new();
        ofst.add_state();
        {
            let mut mapper = ArcSumMapper::new(&source);
            state_map_to(&source, &mut ofst, &mut mapper);
        }
        assert_eq!(ofst.num_states(), 0);
    }
}
