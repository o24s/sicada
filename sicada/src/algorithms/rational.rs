//! Building a union, concatenation or closure without doing the work yet.
//!
//! Port of OpenFst's `rational.h`. The three rational operations combine whole
//! FSTs, and doing each one eagerly copies everything it touches, so a
//! sequence of them copies the operands over and over. What this holds instead
//! is a *recipe*: a small FST whose arcs name the operands, exactly the shape
//! [`replace`] expands. Operations extend the recipe;
//! the operands are never copied more than once.
//!
//! SICADA-DIVERGE: upstream's `RationalFst` is a delayed FST: it answers
//! `Start`, `Final` and `NumArcs` by expanding through a cached `ReplaceFst` on
//! demand, so a caller sees an FST that is being built as it is walked. sicada
//! does not have the delayed FST wrappers yet, so this is a builder, where the
//! recipe is accumulated and [`expand`](RationalFst::expand) runs it. The saving
//! that motivates the type, not copying the operands once per operation, is the
//! same either way.

use crate::AtomicRc;
use crate::algorithms::closure::{ClosureType, closure};
use crate::algorithms::concat::{concat, concat_onto};
use crate::algorithms::replace::{ReplaceOptions, replace};
use crate::algorithms::union::union;
use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::error::OpenFstError;
use crate::fst::{Fst, MutableFst};
use crate::fsts::vector_fst::VectorFst;
use crate::properties::{
    K_COPY_PROPERTIES, K_FST_PROPERTIES, closure_properties, concat_properties, union_properties,
};
use crate::symbol_table::SymbolTable;
use crate::weight::Weight;

/// A union, concatenation or closure of FSTs, held as a recipe.
pub struct RationalFst<A: Arc> {
    /// The recipe: an FST whose arcs are calls to the operands.
    recipe: VectorFst<A>,
    /// The operands, each under the non-terminal naming it.
    operands: Vec<(A::Label, VectorFst<A>)>,
    /// How many non-terminals have been handed out.
    nonterminals: i64,
    /// The properties the expansion will have.
    properties: u64,
    input_symbols: Option<AtomicRc<SymbolTable>>,
    output_symbols: Option<AtomicRc<SymbolTable>>,
}

impl<A: Arc> Default for RationalFst<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Arc> RationalFst<A> {
    /// An empty recipe, standing for the FST that accepts nothing.
    pub fn new() -> Self {
        Self {
            recipe: VectorFst::new(),
            operands: Vec::new(),
            nonterminals: 0,
            properties: 0,
            input_symbols: None,
            output_symbols: None,
        }
    }

    /// The next non-terminal, counting down from -1 so that it cannot collide
    /// with a real label, which upstream also relies on.
    fn next_nonterminal(&mut self) -> Option<A::Label> {
        self.nonterminals += 1;
        A::Label::from_i64(-self.nonterminals)
    }

    /// A two-state FST whose one arc calls `label`.
    fn call(label: A::Label) -> VectorFst<A> {
        let mut fst = VectorFst::new();
        let start = fst.add_state();
        let end = fst.add_state();
        fst.set_start(start);
        fst.set_final(end, A::Weight::one());
        fst.add_arc(
            start,
            A::new(A::Label::epsilon(), label, A::Weight::one(), end),
        );
        fst
    }

    /// Records `fst` under a fresh non-terminal and returns a recipe fragment
    /// that calls it.
    fn adopt<F: Fst<A>>(&mut self, fst: &F) -> Result<VectorFst<A>, OpenFstError> {
        let Some(label) = self.next_nonterminal() else {
            return Err(OpenFstError::InvalidOperation(
                "RationalFst: the label type has run out of non-terminals".into(),
            ));
        };
        let mut owned = VectorFst::new();
        copy_into(fst, &mut owned);
        self.operands.push((label, owned));
        Ok(Self::call(label))
    }

    /// Adds everything `fst` accepts.
    pub fn union_with<F: Fst<A>>(&mut self, fst: &F) -> Result<(), OpenFstError> {
        let props = self.combine(fst, |a, b| union_properties(a, b, true));
        let fragment = self.adopt(fst)?;
        union(&mut self.recipe, &fragment)?;
        self.properties = props;
        Ok(())
    }

    /// Appends everything `fst` accepts.
    pub fn concat_with<F: Fst<A>>(&mut self, fst: &F) -> Result<(), OpenFstError> {
        let props = self.combine(fst, |a, b| concat_properties(a, b, true));
        let fragment = self.adopt(fst)?;
        if self.recipe.start().is_none() {
            // Nothing to append to yet, so the fragment is the whole recipe.
            self.recipe = fragment;
        } else {
            concat(&mut self.recipe, &fragment)?;
        }
        self.properties = props;
        Ok(())
    }

    /// Prepends everything `fst` accepts.
    pub fn concat_onto<F: Fst<A>>(&mut self, fst: &F) -> Result<(), OpenFstError> {
        let props = self.combine(fst, |a, b| concat_properties(b, a, true));
        let fragment = self.adopt(fst)?;
        if self.recipe.start().is_none() {
            self.recipe = fragment;
        } else {
            concat_onto(&fragment, &mut self.recipe)?;
        }
        self.properties = props;
        Ok(())
    }

    /// Lets what has been built so far repeat.
    pub fn closure(&mut self, closure_type: ClosureType) {
        self.properties =
            closure_properties(self.properties, closure_type == ClosureType::Star, true);
        closure(&mut self.recipe, closure_type);
    }

    /// The properties of combining what is here with `fst`, and the symbol
    /// tables carried over.
    fn combine<F: Fst<A>>(&mut self, fst: &F, how: impl Fn(u64, u64) -> u64) -> u64 {
        if self.operands.is_empty() {
            self.input_symbols = fst.input_symbols();
            self.output_symbols = fst.output_symbols();
        }
        how(self.properties, fst.properties(K_FST_PROPERTIES, false))
    }

    /// The properties the expansion will have.
    pub fn properties(&self) -> u64 {
        self.properties
    }

    /// How many operands the recipe names.
    pub fn len(&self) -> usize {
        self.operands.len()
    }

    /// Whether nothing has been combined yet.
    pub fn is_empty(&self) -> bool {
        self.operands.is_empty()
    }

    /// Runs the recipe, writing the result to `ofst`.
    pub fn expand<F: MutableFst<A>>(&self, ofst: &mut F) -> Result<(), OpenFstError> {
        ofst.delete_all_states();
        if self.recipe.start().is_none() {
            return Ok(());
        }
        // The recipe itself is the root of the network, under a non-terminal of
        // its own that nothing calls.
        let Some(root) = A::Label::from_i64(-(self.nonterminals + 1)) else {
            return Err(OpenFstError::InvalidOperation(
                "RationalFst: the label type has run out of non-terminals".into(),
            ));
        };
        let mut network: Vec<(A::Label, &VectorFst<A>)> = Vec::with_capacity(self.len() + 1);
        network.push((root, &self.recipe));
        for (label, fst) in &self.operands {
            network.push((*label, fst));
        }
        // The calls are epsilons on both sides: what the recipe adds is only
        // where the operands go, not anything to read.
        replace(&network, ofst, &ReplaceOptions::epsilon_calls(root))?;
        ofst.set_input_symbols(self.input_symbols.clone());
        ofst.set_output_symbols(self.output_symbols.clone());
        ofst.set_properties(self.properties, K_COPY_PROPERTIES);
        Ok(())
    }
}

/// Copies an FST state for state.
fn copy_into<A, F1, F2>(ifst: &F1, ofst: &mut F2)
where
    A: Arc,
    F1: Fst<A>,
    F2: MutableFst<A>,
{
    ofst.delete_all_states();
    ofst.set_input_symbols(ifst.input_symbols());
    ofst.set_output_symbols(ifst.output_symbols());
    let mut nstates = 0usize;
    for state in ifst.states() {
        while nstates <= state.as_usize() {
            ofst.add_state();
            nstates += 1;
        }
    }
    if let Some(start) = ifst.start() {
        ofst.set_start(start);
    }
    for state in ifst.states() {
        ofst.set_final(state, ifst.final_weight(state));
        for arc in ifst.arcs(state) {
            ofst.add_arc(state, arc);
        }
    }
    ofst.set_properties(ifst.properties(K_FST_PROPERTIES, false), K_FST_PROPERTIES);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{Rng, random_acyclic_fst, visible_paths};
    use crate::arc::StdArc;
    use crate::fst::ExpandedFst as _;
    use crate::fsts::vector_fst::StdVectorFst;
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

    /// The lightest way to each string, up to a length in arcs.
    fn language(fst: &StdVectorFst, max_len: usize) -> Vec<(Vec<i32>, String)> {
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

    fn expanded(rational: &RationalFst<StdArc>) -> StdVectorFst {
        let mut out = StdVectorFst::new();
        rational.expand(&mut out).unwrap();
        out
    }

    /// A recipe of one union expands to the union.
    #[test]
    fn a_union_recipe_expands_to_the_union() {
        let mut rational = RationalFst::<StdArc>::new();
        rational.union_with(&chain(&[1, 2], 1.0)).unwrap();
        rational.union_with(&chain(&[3], 2.0)).unwrap();

        assert_eq!(
            language(&expanded(&rational), 16),
            vec![
                (vec![1, 2], "1.0000".to_string()),
                (vec![3], "2.0000".to_string()),
            ]
        );
    }

    /// And one of concatenations to the concatenation.
    #[test]
    fn a_concat_recipe_expands_to_the_concatenation() {
        let mut rational = RationalFst::<StdArc>::new();
        rational.concat_with(&chain(&[1], 1.0)).unwrap();
        rational.concat_with(&chain(&[2], 2.0)).unwrap();
        rational.concat_with(&chain(&[3], 4.0)).unwrap();

        assert_eq!(
            language(&expanded(&rational), 16),
            vec![(vec![1, 2, 3], "7.0000".to_string())]
        );
    }

    /// Prepending puts the operand at the front.
    #[test]
    fn prepending_puts_the_operand_first() {
        let mut rational = RationalFst::<StdArc>::new();
        rational.concat_with(&chain(&[2], 0.0)).unwrap();
        rational.concat_onto(&chain(&[1], 0.0)).unwrap();

        assert_eq!(
            language(&expanded(&rational), 16),
            vec![(vec![1, 2], "0.0000".to_string())]
        );
    }

    /// Closure of a recipe repeats what the recipe stands for.
    #[test]
    fn a_closure_recipe_repeats_what_was_built() {
        let mut rational = RationalFst::<StdArc>::new();
        rational.concat_with(&chain(&[1], 2.0)).unwrap();
        rational.closure(ClosureType::Star);

        let language = language(&expanded(&rational), 24);
        for repetitions in 0..3 {
            assert!(
                language.contains(&(
                    vec![1; repetitions],
                    format!("{:.4}", 2.0 * repetitions as f32)
                )),
                "{repetitions} repetitions missing from {language:?}"
            );
        }
    }

    /// An operand is stored once however many operations follow it, which is
    /// the reason the type exists.
    #[test]
    fn an_operand_is_stored_once() {
        let big = chain(&[1, 2, 3, 4, 5, 6, 7, 8], 0.0);
        let mut rational = RationalFst::<StdArc>::new();
        rational.concat_with(&big).unwrap();
        rational.closure(ClosureType::Plus);
        rational.union_with(&chain(&[9], 0.0)).unwrap();
        rational.closure(ClosureType::Star);

        assert_eq!(rational.len(), 2, "one entry per operand");
        // The recipe is small whatever the operands weigh: two states per call,
        // plus what the rational operations add around them.
        assert!(
            rational.recipe.num_states() < big.num_states(),
            "the recipe has {} states against the operand's {}",
            rational.recipe.num_states(),
            big.num_states()
        );
    }

    /// A recipe built up out of operations expands to what doing those
    /// operations eagerly gives.
    ///
    /// Only strings of at most a few labels are compared: the expansion reaches
    /// them through the epsilon arcs the calls and returns add, so the same
    /// string costs more arcs there than it does in the eager result, and the
    /// two enumerations would otherwise stop at different depths. The operands
    /// are acyclic and no closure is taken here, so both enumerations are
    /// finite; the repetition case is checked directly in
    /// `a_closure_recipe_repeats_what_was_built`.
    #[test]
    fn the_recipe_expands_to_what_the_eager_operations_give() {
        use crate::algorithms::concat::concat as eager_concat;
        use crate::algorithms::union::union as eager_union;

        /// The strings of at most `labels` labels, with their weights.
        fn short(fst: &StdVectorFst, budget: usize, labels: usize) -> Vec<(Vec<i32>, String)> {
            language(fst, budget)
                .into_iter()
                .filter(|(string, _)| string.len() <= labels)
                .collect()
        }

        let mut rng = Rng::new(0x0000_5A71_u64);
        let mut compared = 0;
        for round in 0..100 {
            let first = random_acyclic_fst(&mut rng, 4);
            let second = random_acyclic_fst(&mut rng, 4);
            let third = random_acyclic_fst(&mut rng, 4);

            let mut rational = RationalFst::<StdArc>::new();
            rational.concat_with(&first).unwrap();
            rational.union_with(&second).unwrap();
            rational.concat_with(&third).unwrap();

            let mut eager = first.clone();
            eager_union(&mut eager, &second).unwrap();
            eager_concat(&mut eager, &third).unwrap();

            let want = short(&eager, 24, 4);
            if !want.is_empty() {
                compared += 1;
            }
            assert_eq!(short(&expanded(&rational), 48, 4), want, "round {round}");
        }
        assert!(compared > 20, "only {compared} rounds accepted anything");
    }

    /// An empty recipe stands for the FST that accepts nothing.
    #[test]
    fn an_empty_recipe_expands_to_nothing() {
        let rational = RationalFst::<StdArc>::new();
        assert!(rational.is_empty());
        assert_eq!(expanded(&rational).num_states(), 0);
    }
}
