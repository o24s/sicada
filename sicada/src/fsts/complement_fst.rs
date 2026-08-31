//! The complement of a deterministic acceptor.
//!
//! Port of OpenFst's `complement.h`.
//!
//! The complement of an FST accepting a language *L* accepts everything but
//! *L*. It is built lazily by adding one state, numbered 0 so that every state
//! of the input shifts up by one, which is where every symbol the input has no
//! transition for leads, and by swapping final for non-final everywhere. The
//! new state is final and loops to itself, so once a string leaves the input's
//! language it stays out.

use crate::AtomicRc;
use crate::algorithms::test_properties::cached_properties;
use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::error::OpenFstError;
use crate::fst::{Fst, PropertyCache};
use crate::properties::{
    K_ACCEPTOR, K_ERROR, K_I_DETERMINISTIC, K_I_LABEL_SORTED, K_NO_EPSILONS, K_UNWEIGHTED,
    complement_properties,
};
use crate::symbol_table::SymbolTable;
use crate::weight::Weight;

/// A label type with room for the ρ ("everything else") transition.
///
/// ρ has to sort below every real label, so that a complement of a label-sorted
/// FST stays label-sorted with the ρ arc emitted first. Upstream picks -2, a
/// value it keeps private to the library for exactly this reason.
///
/// SICADA-DIVERGE: upstream leaves the label type free and would produce an
/// out-of-order arc for an unsigned one, silently breaking the sortedness the
/// property bits then claim. Here an unsigned label type simply has no ρ, so
/// complementing over one does not compile.
pub trait RhoLabel: ArcLabel {
    /// The label standing for every symbol with no transition of its own.
    fn rho_label() -> Self;
}

impl RhoLabel for i32 {
    #[inline(always)]
    fn rho_label() -> Self {
        -2
    }
}

impl RhoLabel for i64 {
    #[inline(always)]
    fn rho_label() -> Self {
        -2
    }
}

/// The complement of `F`, produced a state at a time.
pub struct ComplementFst<A: Arc, F: Fst<A>> {
    fst: F,
    properties: PropertyCache,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Arc, F: Fst<A>> ComplementFst<A, F> {
    pub fn new(fst: F) -> Result<Self, OpenFstError> {
        let required_props = K_UNWEIGHTED | K_NO_EPSILONS | K_I_DETERMINISTIC | K_ACCEPTOR;
        let actual_props = fst.properties(required_props, true);

        if (actual_props & required_props) != required_props {
            return Err(OpenFstError::InvalidOperation(
                "ComplementFst: Argument not an unweighted epsilon-free deterministic acceptor"
                    .into(),
            ));
        }

        // Only sortedness carries over, as upstream does it. `complement_properties`
        // reads more bits than that, but handing it a wider input makes it
        // contradict itself: from an accessible, label-sorted FST it would
        // return both `K_I_LABEL_SORTED` and `K_NOT_I_LABEL_SORTED`.
        let child_props = fst.properties(K_I_LABEL_SORTED, false);
        let properties = complement_properties(child_props);

        Ok(Self {
            fst,
            properties: PropertyCache::new(properties),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Picks up an error the input has fallen into since construction.
    #[inline]
    fn is_error(&self) -> bool {
        if self.fst.properties(K_ERROR, false) & K_ERROR != 0 {
            self.properties.mark_error();
        }
        self.properties.get() & K_ERROR != 0
    }
}

pub struct ComplementStateIter<'a, A: Arc, F: Fst<A> + 'a> {
    siter: F::StateIter<'a>,
    s: usize,
}

impl<'a, A: Arc, F: Fst<A> + 'a> Iterator for ComplementStateIter<'a, A, F> {
    type Item = A::StateId;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.s == 0 {
            self.s += 1;
            Some(A::StateId::from_usize(0))
        } else {
            match self.siter.next() {
                Some(_) => {
                    let ret = self.s;
                    self.s += 1;
                    Some(A::StateId::from_usize(ret))
                }
                None => None,
            }
        }
    }
}

pub struct ComplementArcIter<'a, A: Arc, F: Fst<A> + 'a> {
    aiter: Option<F::ArcIter<'a>>,
    s: usize,
    pos: usize,
}

impl<'a, A: Arc, F: Fst<A> + 'a> Clone for ComplementArcIter<'a, A, F> {
    fn clone(&self) -> Self {
        Self {
            aiter: self.aiter.clone(),
            s: self.s,
            pos: self.pos,
        }
    }
}

impl<'a, A: Arc, F: Fst<A> + 'a> Iterator for ComplementArcIter<'a, A, F>
where
    A::Label: RhoLabel,
{
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos == 0 {
            self.pos += 1;
            Some(A::new(
                <A::Label as RhoLabel>::rho_label(),
                <A::Label as RhoLabel>::rho_label(),
                A::Weight::one(),
                A::StateId::from_usize(0),
            ))
        } else {
            if let Some(ref mut iter) = self.aiter {
                match iter.next() {
                    Some(arc) => {
                        self.pos += 1;
                        Some(A::new(
                            arc.ilabel(),
                            arc.olabel(),
                            arc.weight().clone(),
                            A::StateId::from_usize(arc.nextstate().as_usize() + 1),
                        ))
                    }
                    None => None,
                }
            } else {
                None
            }
        }
    }
}

impl<A: Arc, F: Fst<A>> Fst<A> for ComplementFst<A, F>
where
    A::Label: RhoLabel,
{
    type StateIter<'a>
        = ComplementStateIter<'a, A, F>
    where
        Self: 'a;
    type ArcIter<'a>
        = ComplementArcIter<'a, A, F>
    where
        Self: 'a;

    #[inline]
    fn start(&self) -> Option<A::StateId> {
        if self.is_error() {
            return None;
        }
        // An input with no start state accepts nothing, so its complement
        // accepts everything: the search starts at the new state, which is
        // final and loops to itself on ρ.
        Some(match self.fst.start() {
            Some(s) => A::StateId::from_usize(s.as_usize() + 1),
            None => A::StateId::from_usize(0),
        })
    }

    #[inline]
    fn final_weight(&self, state: A::StateId) -> A::Weight {
        let s = state.as_usize();
        if s == 0 || self.fst.final_weight(A::StateId::from_usize(s - 1)) == A::Weight::zero() {
            A::Weight::one()
        } else {
            A::Weight::zero()
        }
    }

    #[inline]
    fn num_arcs(&self, state: A::StateId) -> usize {
        let s = state.as_usize();
        if s == 0 {
            1
        } else {
            self.fst.num_arcs(A::StateId::from_usize(s - 1)) + 1
        }
    }

    #[inline]
    fn num_input_epsilons(&self, state: A::StateId) -> usize {
        let s = state.as_usize();
        if s == 0 {
            0
        } else {
            self.fst.num_input_epsilons(A::StateId::from_usize(s - 1))
        }
    }

    #[inline]
    fn num_output_epsilons(&self, state: A::StateId) -> usize {
        let s = state.as_usize();
        if s == 0 {
            0
        } else {
            self.fst.num_output_epsilons(A::StateId::from_usize(s - 1))
        }
    }

    #[inline]
    fn num_states_if_known(&self) -> Option<usize> {
        self.fst.num_states_if_known().map(|n| n + 1)
    }

    #[inline]
    fn properties(&self, mask: u64, test: bool) -> u64 {
        if mask & K_ERROR != 0 {
            self.is_error();
        }
        cached_properties(self, &self.properties, mask, test)
    }

    #[inline]
    fn fst_type(&self) -> &str {
        "complement"
    }

    #[inline]
    fn input_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
        self.fst.input_symbols()
    }

    #[inline]
    fn output_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
        self.fst.output_symbols()
    }

    #[inline]
    fn states<'a>(&'a self) -> Self::StateIter<'a> {
        ComplementStateIter {
            siter: self.fst.states(),
            s: 0,
        }
    }

    #[inline]
    fn arcs<'a>(&'a self, state: A::StateId) -> Self::ArcIter<'a> {
        let s = state.as_usize();
        let aiter = if s == 0 {
            None
        } else {
            Some(self.fst.arcs(A::StateId::from_usize(s - 1)))
        };

        ComplementArcIter { aiter, s, pos: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::fst::MutableFst;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::{K_ACCESSIBLE, K_CYCLIC, K_FST_PROPERTIES, K_NOT_I_LABEL_SORTED};
    use crate::weights::float_weight::TropicalWeight;

    /// Builds a deterministic, epsilon-free, unweighted acceptor over `{1, 2}`
    /// from a transition table, and tells it its own properties honestly.
    fn acceptor(
        nstates: usize,
        start: Option<i32>,
        finals: &[i32],
        arcs: &[(i32, i32, i32)],
    ) -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..nstates {
            fst.add_state();
        }
        if let Some(start) = start {
            fst.set_start(start);
        }
        for &(from, label, to) in arcs {
            fst.add_arc(from, StdArc::new(label, label, TropicalWeight::one(), to));
        }
        for &state in finals {
            fst.set_final(state, TropicalWeight::one());
        }
        fst
    }

    /// Whether `fst` accepts `input`, following the ρ arc for any symbol with no
    /// transition of its own, which is how the complement works.
    fn accepts<A, F>(fst: &F, input: &[i32]) -> bool
    where
        A: Arc<Label = i32, StateId = i32, Weight = TropicalWeight>,
        F: Fst<A>,
    {
        let Some(mut state) = fst.start() else {
            return false;
        };
        for &symbol in input {
            let mut next = None;
            let mut rho = None;
            for arc in fst.arcs(state) {
                if arc.ilabel() == symbol {
                    next = Some(arc.nextstate());
                    break;
                }
                if arc.ilabel() == <i32 as RhoLabel>::rho_label() {
                    rho = Some(arc.nextstate());
                }
            }
            match next.or(rho) {
                Some(s) => state = s,
                None => return false,
            }
        }
        fst.final_weight(state) != TropicalWeight::zero()
    }

    /// Every string over `{1, 2}` up to `max_len` symbols.
    fn strings(max_len: usize) -> Vec<Vec<i32>> {
        let mut out = vec![Vec::new()];
        let mut frontier = vec![Vec::new()];
        for _ in 0..max_len {
            let mut next = Vec::new();
            for prefix in &frontier {
                for symbol in [1, 2] {
                    let mut s = prefix.clone();
                    s.push(symbol);
                    next.push(s);
                }
            }
            out.extend(next.iter().cloned());
            frontier = next;
        }
        out
    }

    /// The property a complement has to have, checked string by string.
    fn assert_complements(fst: StdVectorFst) {
        let complement = ComplementFst::new(fst.clone()).expect("valid input");
        for input in strings(5) {
            assert_eq!(
                accepts(&complement, &input),
                !accepts(&fst, &input),
                "{input:?}"
            );
        }
    }

    #[test]
    fn the_complement_accepts_exactly_what_the_input_rejects() {
        // Accepts exactly "1".
        assert_complements(acceptor(2, Some(0), &[1], &[(0, 1, 1)]));
        // Accepts strings of 1s of even length, over an alphabet that also has 2.
        assert_complements(acceptor(2, Some(0), &[0], &[(0, 1, 1), (1, 1, 0)]));
        // Accepts everything.
        assert_complements(acceptor(1, Some(0), &[0], &[(0, 1, 0), (0, 2, 0)]));
        // Accepts only the empty string.
        assert_complements(acceptor(1, Some(0), &[0], &[]));
        // Accepts nothing, though it has states.
        assert_complements(acceptor(2, Some(0), &[], &[(0, 1, 1)]));
    }

    /// An input accepting nothing complements to one accepting everything,
    /// including when it has no start state at all, which is where the search
    /// would otherwise have nowhere to begin.
    #[test]
    fn an_input_with_no_start_state_complements_to_everything() {
        let fst = acceptor(2, None, &[1], &[(0, 1, 1)]);
        assert_eq!(fst.start(), None);

        let complement = ComplementFst::new(fst).expect("valid input");
        assert_eq!(complement.start(), Some(0));
        for input in strings(4) {
            assert!(accepts(&complement, &input), "{input:?} was not accepted");
        }
    }

    #[test]
    fn the_new_state_is_final_and_loops_to_itself() {
        let complement =
            ComplementFst::new(acceptor(2, Some(0), &[1], &[(0, 1, 1)])).expect("valid input");

        assert_eq!(complement.start(), Some(1));
        assert_eq!(complement.count_states(), 3);

        assert_eq!(complement.final_weight(0), TropicalWeight::one());
        assert_eq!(complement.num_arcs(0), 1);
        let rho = complement.arcs(0).next().unwrap();
        assert_eq!(rho.ilabel(), -2);
        assert_eq!(rho.olabel(), -2);
        assert_eq!(rho.nextstate(), 0);
        assert_eq!(rho.weight(), &TropicalWeight::one());

        // The input's start state was not final, so its image is.
        assert_eq!(complement.final_weight(1), TropicalWeight::one());
        assert_eq!(complement.num_arcs(1), 2);
        let arcs: Vec<_> = complement.arcs(1).collect();
        assert_eq!((arcs[0].ilabel(), arcs[0].nextstate()), (-2, 0));
        assert_eq!((arcs[1].ilabel(), arcs[1].nextstate()), (1, 2));

        // The input's final state was final, so its image is not.
        assert_eq!(complement.final_weight(2), TropicalWeight::zero());
        assert_eq!(complement.num_arcs(2), 1);
    }

    #[test]
    fn an_input_that_is_not_a_deterministic_unweighted_acceptor_is_refused() {
        // A transducer rather than an acceptor.
        let mut fst = StdVectorFst::new();
        fst.add_state();
        fst.set_start(0);
        fst.set_final(0, TropicalWeight::one());
        fst.add_arc(0, StdArc::new(1, 2, TropicalWeight::one(), 0));
        assert!(ComplementFst::new(fst).is_err());

        // Non-deterministic on the input side.
        let mut fst = StdVectorFst::new();
        fst.add_state();
        fst.add_state();
        fst.set_start(0);
        fst.set_final(1, TropicalWeight::one());
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 0));
        assert!(ComplementFst::new(fst).is_err());

        // Weighted.
        let mut fst = StdVectorFst::new();
        fst.add_state();
        fst.set_start(0);
        fst.set_final(0, TropicalWeight::one());
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(2.0), 0));
        assert!(ComplementFst::new(fst).is_err());
    }

    /// The property bits a complement claims must not contradict themselves.
    /// Feeding `complement_properties` more than sortedness, which the port used
    /// to do, makes it return a bit and its opposite at once.
    #[test]
    fn the_claimed_properties_are_consistent() {
        let fst = acceptor(2, Some(0), &[1], &[(0, 1, 1)]);
        // The input is accessible and label-sorted, the combination that used
        // to produce the contradiction.
        assert_ne!(fst.properties(K_I_LABEL_SORTED, true) & K_I_LABEL_SORTED, 0);
        assert_ne!(fst.properties(K_ACCESSIBLE, true) & K_ACCESSIBLE, 0);

        let complement = ComplementFst::new(fst).expect("valid input");
        let props = complement.properties(K_FST_PROPERTIES, false);
        assert_ne!(props & K_I_LABEL_SORTED, 0);
        assert_eq!(props & K_NOT_I_LABEL_SORTED, 0);
        assert_eq!(props & K_CYCLIC, 0, "cyclicity is not claimed here");

        // What a complement always is.
        assert_ne!(props & K_ACCEPTOR, 0);
        assert_ne!(props & K_UNWEIGHTED, 0);
        assert_ne!(props & K_NO_EPSILONS, 0);
        assert_ne!(props & K_I_DETERMINISTIC, 0);
    }
}
