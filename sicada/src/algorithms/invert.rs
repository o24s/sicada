//! Swapping the two sides of a transducer.
//!
//! Port of OpenFst's `invert.h`.

use crate::algorithms::arc_map::{
    ArcMapper, MapFinalAction, MapSymbolsAction, arc_map, arc_map_to,
};
use crate::arc::Arc;
use crate::error::OpenFstError;
use crate::fst::{Fst, MutableFst};
use crate::properties::invert_properties;

/// Swaps an arc's input and output labels.
#[derive(Debug, Clone, Copy, Default)]
pub struct InvertMapper;

impl<A: Arc> ArcMapper<A, A> for InvertMapper {
    #[inline]
    fn map(&mut self, arc: &A) -> A {
        A::new(
            arc.olabel(),
            arc.ilabel(),
            arc.weight().clone(),
            arc.nextstate(),
        )
    }

    fn final_action(&self) -> MapFinalAction {
        MapFinalAction::NoSuperfinal
    }

    /// Both tables are cleared by the mapping and put back the other way round
    /// by [`invert`], which is the only thing that knows they should swap.
    fn input_symbols_action(&self) -> MapSymbolsAction {
        MapSymbolsAction::Clear
    }

    fn output_symbols_action(&self) -> MapSymbolsAction {
        MapSymbolsAction::Clear
    }

    fn properties(&self, props: u64) -> u64 {
        invert_properties(props)
    }
}

/// Inverts `fst` in place: what it read it now writes, and the other way round.
pub fn invert<A: Arc, F: MutableFst<A>>(fst: &mut F) -> Result<(), OpenFstError> {
    let input = fst.input_symbols();
    let output = fst.output_symbols();
    arc_map(fst, &mut InvertMapper)?;
    fst.set_input_symbols(output);
    fst.set_output_symbols(input);
    Ok(())
}

/// Writes the inverse of `ifst` into `ofst`.
pub fn invert_to<A: Arc, F1: Fst<A>, F2: MutableFst<A>>(
    ifst: &F1,
    ofst: &mut F2,
) -> Result<(), OpenFstError> {
    let input = ifst.input_symbols();
    let output = ifst.output_symbols();
    arc_map_to(ifst, ofst, &mut InvertMapper)?;
    ofst.set_input_symbols(output);
    ofst.set_output_symbols(input);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AtomicRc;
    use crate::arc::StdArc;
    use crate::fst::ExpandedFst as _;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::symbol_table::SymbolTable;
    use crate::weights::float_weight::TropicalWeight;

    fn transducer() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 10, TropicalWeight(1.0), 1));
        fst.add_arc(1, StdArc::new(2, 20, TropicalWeight(2.0), 2));
        fst.set_final(2, TropicalWeight(3.0));
        fst
    }

    fn labels(fst: &StdVectorFst) -> Vec<(i32, i32)> {
        (0..fst.num_states() as i32)
            .flat_map(|s| {
                fst.arcs(s)
                    .map(|a| (a.ilabel(), a.olabel()))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn inverting_swaps_the_two_sides() {
        let mut fst = transducer();
        invert(&mut fst).unwrap();
        assert_eq!(labels(&fst), vec![(10, 1), (20, 2)]);
        assert_eq!(fst.num_states(), 3);
        assert_eq!(fst.start(), Some(0));
        assert_eq!(fst.final_weight(2), TropicalWeight(3.0));
    }

    /// Inverting twice is the identity, which is the property that defines it.
    #[test]
    fn inverting_twice_gives_back_the_original() {
        let original = transducer();
        let mut fst = original.clone();
        invert(&mut fst).unwrap();
        invert(&mut fst).unwrap();
        assert_eq!(labels(&fst), labels(&original));
    }

    /// The symbol tables swap with the sides they describe.
    #[test]
    fn the_symbol_tables_swap_too() {
        let mut input = SymbolTable::new("input".to_string());
        input.add_symbol("<eps>", 0);
        input.add_symbol("a", 1);
        let mut output = SymbolTable::new("output".to_string());
        output.add_symbol("<eps>", 0);
        output.add_symbol("x", 10);

        let mut fst = transducer();
        fst.set_input_symbols(Some(AtomicRc::new(input)));
        fst.set_output_symbols(Some(AtomicRc::new(output)));

        invert(&mut fst).unwrap();
        assert_eq!(fst.input_symbols().unwrap().find_symbol(10), Some("x"));
        assert_eq!(fst.output_symbols().unwrap().find_symbol(1), Some("a"));
    }

    #[test]
    fn inverting_into_another_fst_leaves_the_input_alone() {
        let ifst = transducer();
        let mut ofst = StdVectorFst::new();
        invert_to(&ifst, &mut ofst).unwrap();
        assert_eq!(labels(&ofst), vec![(10, 1), (20, 2)]);
        assert_eq!(labels(&ifst), vec![(1, 10), (2, 20)]);
    }
}
