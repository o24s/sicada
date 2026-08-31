//! Turning a transducer into an acceptor by keeping one side.
//!
//! Port of OpenFst's `project.h`.

use crate::algorithms::arc_map::{
    ArcMapper, MapFinalAction, MapSymbolsAction, arc_map, arc_map_to,
};
use crate::arc::Arc;
use crate::error::OpenFstError;
use crate::fst::{Fst, MutableFst};
use crate::properties::project_properties;

/// Which side to keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectType {
    /// Keep the input labels, copying them onto the output side.
    Input,
    /// Keep the output labels, copying them onto the input side.
    Output,
}

/// Copies one side of an arc onto both.
#[derive(Debug, Clone, Copy)]
pub struct ProjectMapper {
    project_type: ProjectType,
}

impl ProjectMapper {
    /// Keeps the side named by `project_type`.
    pub fn new(project_type: ProjectType) -> Self {
        Self { project_type }
    }
}

impl<A: Arc> ArcMapper<A, A> for ProjectMapper {
    #[inline]
    fn map(&mut self, arc: &A) -> A {
        let label = match self.project_type {
            ProjectType::Input => arc.ilabel(),
            ProjectType::Output => arc.olabel(),
        };
        A::new(label, label, arc.weight().clone(), arc.nextstate())
    }

    fn final_action(&self) -> MapFinalAction {
        MapFinalAction::NoSuperfinal
    }

    fn input_symbols_action(&self) -> MapSymbolsAction {
        match self.project_type {
            ProjectType::Input => MapSymbolsAction::Copy,
            ProjectType::Output => MapSymbolsAction::Clear,
        }
    }

    fn output_symbols_action(&self) -> MapSymbolsAction {
        match self.project_type {
            ProjectType::Output => MapSymbolsAction::Copy,
            ProjectType::Input => MapSymbolsAction::Clear,
        }
    }

    fn properties(&self, props: u64) -> u64 {
        project_properties(props, self.project_type == ProjectType::Input)
    }
}

/// Projects `fst` onto one side in place, leaving an acceptor.
pub fn project<A: Arc, F: MutableFst<A>>(
    fst: &mut F,
    project_type: ProjectType,
) -> Result<(), OpenFstError> {
    arc_map(fst, &mut ProjectMapper::new(project_type))?;
    // The side that was kept now describes both, so its table describes both.
    match project_type {
        ProjectType::Input => fst.set_output_symbols(fst.input_symbols()),
        ProjectType::Output => fst.set_input_symbols(fst.output_symbols()),
    }
    Ok(())
}

/// Writes the projection of `ifst` into `ofst`.
pub fn project_to<A: Arc, F1: Fst<A>, F2: MutableFst<A>>(
    ifst: &F1,
    ofst: &mut F2,
    project_type: ProjectType,
) -> Result<(), OpenFstError> {
    arc_map_to(ifst, ofst, &mut ProjectMapper::new(project_type))?;
    match project_type {
        ProjectType::Input => ofst.set_output_symbols(ifst.input_symbols()),
        ProjectType::Output => ofst.set_input_symbols(ifst.output_symbols()),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AtomicRc;
    use crate::arc::StdArc;
    use crate::fst::ExpandedFst as _;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::{K_ACCEPTOR, K_FST_PROPERTIES};
    use crate::symbol_table::SymbolTable;
    use crate::weight::Weight;
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
    fn projecting_keeps_one_side_on_both() {
        let mut fst = transducer();
        project(&mut fst, ProjectType::Input).unwrap();
        assert_eq!(labels(&fst), vec![(1, 1), (2, 2)]);

        let mut fst = transducer();
        project(&mut fst, ProjectType::Output).unwrap();
        assert_eq!(labels(&fst), vec![(10, 10), (20, 20)]);
    }

    /// What a projection is for: the result is an acceptor, and says so.
    #[test]
    fn the_result_is_an_acceptor() {
        let mut fst = transducer();
        project(&mut fst, ProjectType::Input).unwrap();
        assert_ne!(fst.properties(K_ACCEPTOR, true) & K_ACCEPTOR, 0);
        assert_ne!(
            fst.properties(K_FST_PROPERTIES, false) & K_ACCEPTOR,
            0,
            "and claims it without needing to be asked to check"
        );
    }

    /// The kept side's table now describes both sides.
    #[test]
    fn the_kept_sides_table_describes_both() {
        let mut input = SymbolTable::new("input".to_string());
        input.add_symbol("<eps>", 0);
        input.add_symbol("a", 1);
        let mut output = SymbolTable::new("output".to_string());
        output.add_symbol("<eps>", 0);
        output.add_symbol("x", 10);

        let mut fst = transducer();
        fst.set_input_symbols(Some(AtomicRc::new(input)));
        fst.set_output_symbols(Some(AtomicRc::new(output.clone())));
        project(&mut fst, ProjectType::Input).unwrap();
        assert_eq!(fst.input_symbols().unwrap().find_symbol(1), Some("a"));
        assert_eq!(fst.output_symbols().unwrap().find_symbol(1), Some("a"));

        let mut fst = transducer();
        fst.set_output_symbols(Some(AtomicRc::new(output)));
        project(&mut fst, ProjectType::Output).unwrap();
        assert_eq!(fst.input_symbols().unwrap().find_symbol(10), Some("x"));
    }

    /// Projecting an acceptor changes nothing.
    #[test]
    fn projecting_an_acceptor_is_a_no_op() {
        let mut fst = StdVectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(5, 5, TropicalWeight::one(), 1));
        fst.set_final(1, TropicalWeight::one());
        let before = labels(&fst);

        project(&mut fst, ProjectType::Input).unwrap();
        assert_eq!(labels(&fst), before);
        project(&mut fst, ProjectType::Output).unwrap();
        assert_eq!(labels(&fst), before);
    }

    #[test]
    fn projecting_into_another_fst_leaves_the_input_alone() {
        let ifst = transducer();
        let mut ofst = StdVectorFst::new();
        project_to(&ifst, &mut ofst, ProjectType::Output).unwrap();
        assert_eq!(labels(&ofst), vec![(10, 10), (20, 20)]);
        assert_eq!(labels(&ifst), vec![(1, 10), (2, 20)]);
    }
}
