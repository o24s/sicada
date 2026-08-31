//! Checking that an FST holds together.
//!
//! Port of OpenFst's `verify.h`. The checks are the ones a reader of a file, or
//! a caller handed an FST built elsewhere, cannot make for itself: that state
//! IDs point at states that exist, that labels are in their symbol tables, that
//! weights are members of their semiring, and that the property bits the FST
//! carries do not contradict what it actually contains.

use crate::algorithms::test_properties::compute_properties;
use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::error::OpenFstError;
use crate::fst::Fst;
use crate::properties::{K_ERROR, K_FST_PROPERTIES, internal::compat_properties};
use crate::symbol_table::SymbolTable;
use crate::weight::Weight;

/// Checks that `fst` holds together, describing the first fault found.
///
/// With `allow_negative_labels`, negative labels pass. They are not valid in a
/// file, but the library uses them itself: ρ in
/// [`ComplementFst`](crate::fsts::complement_fst::ComplementFst) is -2, and the
/// matchers reserve others.
///
/// SICADA-DIVERGE: upstream has two of these: `VerifyWithStatus`, returning a
/// status, and a deprecated `Verify`, returning a bool after logging. One is
/// enough when the return type carries the reason.
pub fn verify<A: Arc, F: Fst<A>>(fst: &F, allow_negative_labels: bool) -> Result<(), OpenFstError> {
    let fault = |message: String| Err(OpenFstError::VerificationFailed(message));

    let nstates = fst.count_states();
    match fst.start() {
        None => {
            if nstates > 0 {
                return fault("FST start state ID not set".to_string());
            }
        }
        // Compared against zero before being used as an index: verify is
        // handed FSTs it has no reason to trust, so a negative state ID has to
        // come back as a fault rather than as a panic.
        Some(start) if start < A::StateId::from_usize(0) || start.as_usize() >= nstates => {
            return fault(format!(
                "FST start state ID out of valid range: [0, {nstates})"
            ));
        }
        Some(_) => {}
    }

    let isymbols = fst.input_symbols();
    let osymbols = fst.output_symbols();

    for state in fst.states() {
        for (position, arc) in fst.arcs(state).enumerate() {
            let at = || format!("of arc at position {position} of state {state:?}");
            if !allow_negative_labels && arc.ilabel() < A::Label::epsilon() {
                return fault(format!("FST input label ID {} is negative", at()));
            }
            if let Some(table) = &isymbols
                && !contains(table, arc.ilabel())
            {
                return fault(format!(
                    "FST input label ID {} {} is missing from input symbol table \"{}\"",
                    arc.ilabel(),
                    at(),
                    table.name()
                ));
            }
            if !allow_negative_labels && arc.olabel() < A::Label::epsilon() {
                return fault(format!("FST output label ID {} is negative", at()));
            }
            if let Some(table) = &osymbols
                && !contains(table, arc.olabel())
            {
                return fault(format!(
                    "FST output label ID {} {} is missing from output symbol table \"{}\"",
                    arc.olabel(),
                    at(),
                    table.name()
                ));
            }
            if !arc.weight().is_member() {
                return fault(format!("FST weight {} is invalid", at()));
            }
            if arc.nextstate() < A::StateId::from_usize(0) {
                return fault(format!("FST destination state ID {} is negative", at()));
            }
            if arc.nextstate().as_usize() >= nstates {
                return fault(format!(
                    "FST destination state ID {} exceeds number of states",
                    at()
                ));
            }
        }
        if !fst.final_weight(state).is_member() {
            return fault(format!("FST final weight of state {state:?} is invalid"));
        }
    }

    let stored = fst.properties(K_FST_PROPERTIES, false);
    if stored & K_ERROR != 0 {
        return fault("FST error property is set".to_string());
    }
    if !compat_properties(stored, compute_properties(fst, K_FST_PROPERTIES).props) {
        return fault("stored FST properties contradict its contents".to_string());
    }
    Ok(())
}

/// Whether `table` has an entry for `label`.
///
/// SICADA-DIVERGE: upstream's negative-label check happens first and returns,
/// so `Member` never sees one unless negative labels were allowed, and then it
/// looks one up in a table that cannot hold it. A label that does not fit an
/// `i64` cannot be in the table either.
fn contains<L: ArcLabel>(table: &SymbolTable, label: L) -> bool {
    match label.to_i64() {
        Some(label) => table.member_key(label),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AtomicRc;
    use crate::arc::StdArc;
    use crate::fst::MutableFst;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::{K_ACCEPTOR, K_NOT_ACCEPTOR};
    use crate::weights::float_weight::TropicalWeight;

    fn chain() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        fst.add_state();
        fst.add_state();
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.set_final(1, TropicalWeight::one());
        fst
    }

    fn message(fst: &StdVectorFst, allow_negative: bool) -> String {
        match verify(fst, allow_negative) {
            Ok(()) => panic!("expected a fault"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn a_well_formed_fst_passes() {
        verify(&chain(), false).unwrap();
    }

    /// An FST with no states needs no start state; one with states does.
    #[test]
    fn a_missing_start_state_is_a_fault_only_once_there_are_states() {
        let empty = StdVectorFst::new();
        assert_eq!(empty.start(), None);
        verify(&empty, false).unwrap();

        let mut fst = StdVectorFst::new();
        fst.add_state();
        assert_eq!(fst.start(), None);
        assert!(message(&fst, false).contains("start state ID not set"));
    }

    #[test]
    fn a_start_state_outside_the_fst_is_a_fault() {
        let mut fst = chain();
        fst.set_start(7);
        assert!(message(&fst, false).contains("out of valid range"));

        // Negative, which would panic if it reached an index.
        let mut fst = chain();
        fst.set_start(-3);
        assert!(message(&fst, false).contains("out of valid range"));
    }

    #[test]
    fn an_arc_leading_to_a_negative_state_is_a_fault() {
        let mut fst = chain();
        fst.add_arc(1, StdArc::new(1, 1, TropicalWeight::one(), -5));
        assert!(message(&fst, false).contains("is negative"));
    }

    #[test]
    fn an_arc_leading_outside_the_fst_is_a_fault() {
        let mut fst = chain();
        fst.add_arc(1, StdArc::new(1, 1, TropicalWeight::one(), 9));
        assert!(message(&fst, false).contains("exceeds number of states"));
    }

    /// Negative labels are refused by default and allowed on request, because
    /// the library uses them itself: rho is -2.
    #[test]
    fn negative_labels_are_refused_unless_allowed() {
        let mut fst = chain();
        fst.add_arc(1, StdArc::new(-2, 1, TropicalWeight::one(), 1));
        assert!(message(&fst, false).contains("input label ID"));
        verify(&fst, true).unwrap();

        let mut fst = chain();
        fst.add_arc(1, StdArc::new(1, -2, TropicalWeight::one(), 1));
        assert!(message(&fst, false).contains("output label ID"));
        verify(&fst, true).unwrap();
    }

    /// A weight outside its semiring, such as NaN for a tropical weight, is a
    /// fault wherever it sits.
    #[test]
    fn a_weight_outside_the_semiring_is_a_fault() {
        let mut fst = chain();
        fst.add_arc(1, StdArc::new(1, 1, TropicalWeight(f32::NAN), 1));
        assert!(message(&fst, false).contains("weight of arc"));

        let mut fst = chain();
        fst.set_final(0, TropicalWeight(f32::NAN));
        assert!(message(&fst, false).contains("final weight of state"));
    }

    /// A symbol table on the FST is a claim that every label appears in it.
    #[test]
    fn a_label_missing_from_its_symbol_table_is_a_fault() {
        let mut table = SymbolTable::new("input".to_string());
        table.add_symbol("<eps>", 0);
        table.add_symbol("a", 1);

        let mut fst = chain();
        fst.set_input_symbols(Some(AtomicRc::new(table.clone())));
        verify(&fst, false).unwrap();

        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight::one(), 1));
        assert!(message(&fst, false).contains("missing from input symbol table"));

        let mut fst = chain();
        fst.set_output_symbols(Some(AtomicRc::new(table)));
        fst.add_arc(1, StdArc::new(1, 2, TropicalWeight::one(), 1));
        assert!(message(&fst, false).contains("missing from output symbol table"));
    }

    /// The last check, and the one that needs a full scan: an FST that says it
    /// is one thing while being another.
    #[test]
    fn property_bits_that_contradict_the_contents_are_a_fault() {
        let mut fst = chain();
        // The chain is an acceptor; say it is not.
        fst.set_properties(K_NOT_ACCEPTOR, K_ACCEPTOR | K_NOT_ACCEPTOR);
        assert!(message(&fst, false).contains("stored FST properties contradict"));

        // Saying nothing is not a contradiction.
        let mut fst = chain();
        fst.set_properties(0, K_ACCEPTOR | K_NOT_ACCEPTOR);
        verify(&fst, false).unwrap();
    }

    #[test]
    fn an_fst_carrying_the_error_bit_is_a_fault() {
        let mut fst = chain();
        fst.set_properties(K_ERROR, K_ERROR);
        assert!(message(&fst, false).contains("error property is set"));
    }
}
