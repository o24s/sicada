use crate::arc::Arc;
use crate::fst::Fst;
use crate::properties::{K_COPY_PROPERTIES, internal::compat_properties};
use crate::symbol_table::compat_symbols_rc;
use crate::weight::Weight;

pub const EQUAL_FSTS: u8 = 0x01;
pub const EQUAL_FST_TYPES: u8 = 0x02;
pub const EQUAL_COMPAT_PROPERTIES: u8 = 0x04;
pub const EQUAL_COMPAT_SYMBOLS: u8 = 0x08;
pub const EQUAL_ALL: u8 =
    EQUAL_FSTS | EQUAL_FST_TYPES | EQUAL_COMPAT_PROPERTIES | EQUAL_COMPAT_SYMBOLS;

/// Whether two FSTs are equal, comparing weights with `weight_equal`.
pub fn equal_with<A, F1, F2, WEq>(fst1: &F1, fst2: &F2, mut weight_equal: WEq, etype: u8) -> bool
where
    A: Arc,
    A::StateId: PartialEq,
    F1: Fst<A>,
    F2: Fst<A>,
    WEq: FnMut(&A::Weight, &A::Weight) -> bool,
{
    if (etype & EQUAL_FST_TYPES) != 0 && fst1.fst_type() != fst2.fst_type() {
        return false;
    }

    if (etype & EQUAL_COMPAT_PROPERTIES) != 0
        && !compat_properties(
            fst1.properties(K_COPY_PROPERTIES, false),
            fst2.properties(K_COPY_PROPERTIES, false),
        )
    {
        return false;
    }

    // Whether the symbol tables are compatible.
    if (etype & EQUAL_COMPAT_SYMBOLS) != 0 {
        let isyms1 = fst1.input_symbols();
        let isyms2 = fst2.input_symbols();
        if !compat_symbols_rc(isyms1, isyms2) {
            return false;
        }

        let osyms1 = fst1.output_symbols();
        let osyms2 = fst2.output_symbols();
        if !compat_symbols_rc(osyms1, osyms2) {
            return false;
        }
    }

    // Nothing else to check when the topology is not being compared.
    if (etype & EQUAL_FSTS) == 0 {
        return true;
    }

    // The start states.
    if fst1.start() != fst2.start() {
        return false;
    }

    let mut siter1 = fst1.states();
    let mut siter2 = fst2.states();

    // The states have to come out in the same order.
    loop {
        match (siter1.next(), siter2.next()) {
            (Some(s1), Some(s2)) => {
                if s1 != s2 {
                    return false; // Mismatched states
                }

                let final1 = fst1.final_weight(s1);
                let final2 = fst2.final_weight(s2);
                if !weight_equal(&final1, &final2) {
                    return false; // Mismatched final weights
                }

                let mut aiter1 = fst1.arcs(s1);
                let mut aiter2 = fst2.arcs(s2);

                // So do the arcs leaving each of them.
                loop {
                    match (aiter1.next(), aiter2.next()) {
                        (Some(a1), Some(a2)) => {
                            if a1.ilabel() != a2.ilabel() {
                                return false;
                            }
                            if a1.olabel() != a2.olabel() {
                                return false;
                            }
                            if a1.nextstate() != a2.nextstate() {
                                return false;
                            }
                            if !weight_equal(a1.weight(), a2.weight()) {
                                return false;
                            }
                        }
                        (None, None) => break,
                        _ => return false, // Mismatched number of arcs
                    }
                }

                #[cfg(debug_assertions)]
                {
                    // Neither of these should be reachable; they catch an `ArcIter` that
                    // disagrees with the arc count it reported.
                    if fst1.num_arcs(s1) != fst2.num_arcs(s2) {
                        return false;
                    }
                    if fst1.num_input_epsilons(s1) != fst2.num_input_epsilons(s2) {
                        return false;
                    }
                    if fst1.num_output_epsilons(s1) != fst2.num_output_epsilons(s2) {
                        return false;
                    }
                }
            }
            (None, None) => break,
            _ => return false, // Mismatched number of states
        }
    }

    true
}

/// Whether two FSTs are equal, comparing weights with `approx_equal` to
/// within `delta`.
#[inline]
pub fn equal<A, F1, F2>(fst1: &F1, fst2: &F2, delta: f32, etype: u8) -> bool
where
    A: Arc,
    A::StateId: PartialEq,
    A::Weight: Weight,
    F1: Fst<A>,
    F2: Fst<A>,
{
    equal_with(
        fst1,
        fst2,
        |w1, w2| A::Weight::approx_equal(w1, w2, delta),
        etype,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::fst::MutableFst;
    use crate::fsts::vector_fst::VectorFst;
    use crate::weight::Weight;
    use crate::weights::float_weight::TropicalWeight;

    /// Two states, one arc between them, both final-weighted.
    fn simple(weight: f32) -> VectorFst<StdArc> {
        let mut fst = VectorFst::new();
        let start = fst.add_state();
        let end = fst.add_state();
        fst.set_start(start);
        fst.set_final(end, TropicalWeight::one());
        fst.add_arc(start, StdArc::new(1, 2, TropicalWeight(weight), end));
        fst
    }

    #[test]
    fn an_fst_equals_itself() {
        let fst = simple(1.0);
        assert!(equal(&fst, &fst, 1e-6, EQUAL_FSTS));
        assert!(equal(&fst, &simple(1.0), 1e-6, EQUAL_ALL));
    }

    #[test]
    fn a_different_weight_is_not_equal() {
        assert!(!equal(&simple(1.0), &simple(2.0), 1e-6, EQUAL_FSTS));
    }

    /// Equality on weights is approximate, so a difference below the tolerance
    /// is not a difference. That is why the algorithms compare with a delta at
    /// all: a float semiring is not exactly associative.
    #[test]
    fn a_weight_within_the_tolerance_is_equal() {
        assert!(equal(&simple(1.0), &simple(1.000_001), 1e-3, EQUAL_FSTS));
        assert!(!equal(&simple(1.0), &simple(1.000_001), 1e-9, EQUAL_FSTS));
    }

    #[test]
    fn a_missing_start_state_is_not_equal_to_a_present_one() {
        let with_start = simple(1.0);
        let mut without_start = VectorFst::<StdArc>::new();
        without_start.add_state();
        assert!(!equal(&with_start, &without_start, 1e-6, EQUAL_FSTS));
    }

    #[test]
    fn a_different_state_count_is_not_equal() {
        let mut extra = simple(1.0);
        extra.add_state();
        assert!(!equal(&simple(1.0), &extra, 1e-6, EQUAL_FSTS));
    }

    #[test]
    fn a_different_final_weight_is_not_equal() {
        let base = simple(1.0);
        let mut changed = simple(1.0);
        changed.set_final(1, TropicalWeight(5.0));
        assert!(!equal(&base, &changed, 1e-6, EQUAL_FSTS));
    }

    /// Arc order is part of the comparison: `Equal` checks the arcs are in the
    /// same order, not merely that the sets match. `Isomorphic` is the test that
    /// ignores order.
    #[test]
    fn arc_order_matters() {
        let mut forwards = VectorFst::<StdArc>::new();
        let start = forwards.add_state();
        forwards.set_start(start);
        forwards.set_final(start, TropicalWeight::one());
        forwards.add_arc(start, StdArc::new(1, 1, TropicalWeight::one(), start));
        forwards.add_arc(start, StdArc::new(2, 2, TropicalWeight::one(), start));

        let mut backwards = VectorFst::<StdArc>::new();
        let start = backwards.add_state();
        backwards.set_start(start);
        backwards.set_final(start, TropicalWeight::one());
        backwards.add_arc(start, StdArc::new(2, 2, TropicalWeight::one(), start));
        backwards.add_arc(start, StdArc::new(1, 1, TropicalWeight::one(), start));

        assert!(!equal(&forwards, &backwards, 1e-6, EQUAL_FSTS));
    }

    #[test]
    fn two_empty_fsts_are_equal() {
        let left = VectorFst::<StdArc>::new();
        let right = VectorFst::<StdArc>::new();
        assert!(equal(&left, &right, 1e-6, EQUAL_ALL));
    }

    #[test]
    fn the_equality_flags_match_openfst() {
        assert_eq!(EQUAL_FSTS, 0x01);
        assert_eq!(EQUAL_FST_TYPES, 0x02);
        assert_eq!(EQUAL_COMPAT_PROPERTIES, 0x04);
        assert_eq!(EQUAL_COMPAT_SYMBOLS, 0x08);
        assert_eq!(EQUAL_ALL, 0x0F);
    }
}
