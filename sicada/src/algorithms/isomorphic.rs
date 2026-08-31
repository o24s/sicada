use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::hash::Hash;

use hashbrown::HashMap;

use crate::arc::Arc;
use crate::fst::Fst;
use crate::weight::Weight;

pub const EQUAL_FSTS: u8 = 0x01;
pub const EQUAL_FST_TYPES: u8 = 0x02;
pub const EQUAL_COMPAT_PROPERTIES: u8 = 0x04;
pub const EQUAL_COMPAT_SYMBOLS: u8 = 0x08;
pub const EQUAL_ALL: u8 =
    EQUAL_FSTS | EQUAL_FST_TYPES | EQUAL_COMPAT_PROPERTIES | EQUAL_COMPAT_SYMBOLS;

/// Tests if two FSTs are isomorphic, i.e., they are equal up to a state
/// and arc re-ordering.
///
/// FSTs should be deterministic when viewed as unweighted automata.
/// False negatives (but not false positives) are possible when the inputs
/// are nondeterministic.
///
/// # Arguments
/// - `fst1`, `fst2` - Input FSTs to compare.
/// - `delta` - Weight equality delta.
/// - `weight_cmp` - Ordering function for weights when they are not approx equal.
///
/// # Returns
/// `Ok(true)` if they are isomorphic, `Ok(false)` otherwise.
/// Returns `Err(String)` if nondeterminism prevents determining isomorphism.
/// Tests whether two FSTs are isomorphic: equal up to state renumbering and arc
/// reordering.
///
/// Weights are ordered by their natural comparison when they are not
/// approximately equal, as upstream's default does.
///
/// SICADA-DIVERGE: upstream logs and returns `false` when nondeterminism makes
/// the answer undecidable, so a caller cannot tell "not isomorphic" from "could
/// not tell". Here that is an `Err`.
pub fn isomorphic<A, F1, F2>(fst1: &F1, fst2: &F2, delta: f32) -> Result<bool, String>
where
    A: Arc,
    A::StateId: Copy + Eq + Hash + Ord + Debug,
    A::Label: Ord + Debug,
    A::Weight: Weight + Debug + PartialOrd,
    F1: Fst<A>,
    F2: Fst<A>,
{
    isomorphic_with(fst1, fst2, delta, |lhs, rhs| {
        lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal)
    })
}

pub fn isomorphic_with<A, F1, F2, WCmp>(
    fst1: &F1,
    fst2: &F2,
    delta: f32,
    weight_cmp: WCmp,
) -> Result<bool, String>
where
    A: Arc,
    A::StateId: Copy + Eq + Hash + Ord + Debug,
    A::Label: Ord + Debug,
    A::Weight: Weight + Debug,
    F1: Fst<A>,
    F2: Fst<A>,
    WCmp: FnMut(&A::Weight, &A::Weight) -> Ordering,
{
    let start1 = fst1.start();
    let start2 = fst2.start();

    match (start1, start2) {
        (None, None) => Ok(true),
        (None, Some(_)) | (Some(_), None) => {
            log::debug!("Isomorphic: Only one of the FSTs is empty.");
            Ok(false)
        }
        (Some(s1), Some(s2)) => {
            let mut iso = Isomorphism {
                fst1,
                fst2,
                delta,
                weight_cmp,
                state_pairs: HashMap::new(),
                queue: VecDeque::new(),
                nondet: false,
            };

            iso.pair_state(s1, s2)?;

            while let Some((state1, state2)) = iso.queue.pop_front() {
                if !iso.is_isomorphic_state(state1, state2)? {
                    if iso.nondet {
                        let msg = format!(
                            "Isomorphic: Non-determinism as an unweighted automaton. state1: {:?} state2: {:?}",
                            state1, state2
                        );
                        log::error!("{}", msg);
                        return Err(msg);
                    }
                    return Ok(false);
                }
            }
            Ok(true)
        }
    }
}

struct Isomorphism<'a, A: Arc, F1: Fst<A>, F2: Fst<A>, WCmp> {
    fst1: &'a F1,
    fst2: &'a F2,
    delta: f32,
    weight_cmp: WCmp,
    state_pairs: HashMap<A::StateId, A::StateId>,
    queue: VecDeque<(A::StateId, A::StateId)>,
    nondet: bool,
}

impl<'a, A, F1, F2, WCmp> Isomorphism<'a, A, F1, F2, WCmp>
where
    A: Arc,
    A::StateId: Copy + Eq + Hash + Ord + Debug,
    A::Label: Ord + Debug,
    A::Weight: Weight + Debug,
    F1: Fst<A>,
    F2: Fst<A>,
    WCmp: FnMut(&A::Weight, &A::Weight) -> Ordering,
{
    fn pair_state(&mut self, s1: A::StateId, s2: A::StateId) -> Result<bool, String> {
        if let Some(&existing_s2) = self.state_pairs.get(&s1) {
            if existing_s2 == s2 {
                return Ok(true); // Already seen this pair.
            } else {
                return Ok(false); // s1 already paired with another s2.
            }
        }

        log::trace!("Pairing states: ({:?}, {:?})", s1, s2);
        self.state_pairs.insert(s1, s2);
        self.queue.push_back((s1, s2));
        Ok(true)
    }

    fn is_isomorphic_state(&mut self, s1: A::StateId, s2: A::StateId) -> Result<bool, String> {
        let final1 = self.fst1.final_weight(s1);
        let final2 = self.fst2.final_weight(s2);

        if !A::Weight::approx_equal(&final1, &final2, self.delta) {
            log::debug!(
                "Isomorphic: Final weights not equal to within delta={}: fst1.final({:?}) = {:?}, fst2.final({:?}) = {:?}",
                self.delta,
                s1,
                final1,
                s2,
                final2
            );
            return Ok(false);
        }

        let narcs1 = self.fst1.num_arcs(s1);
        let narcs2 = self.fst2.num_arcs(s2);

        if narcs1 != narcs2 {
            log::debug!(
                "Isomorphic: NumArcs not equal. fst1.num_arcs({:?}) = {}, fst2.num_arcs({:?}) = {}",
                s1,
                narcs1,
                s2,
                narcs2
            );
            return Ok(false);
        }

        let mut arcs1: Vec<A> = self.fst1.arcs(s1).collect();
        let mut arcs2: Vec<A> = self.fst2.arcs(s2).collect();

        // Orders arcs for equality checking.
        let mut cmp_func = |a1: &A, a2: &A| -> Ordering {
            match a1.ilabel().cmp(&a2.ilabel()) {
                Ordering::Equal => match a1.olabel().cmp(&a2.olabel()) {
                    Ordering::Equal => {
                        if A::Weight::approx_equal(a1.weight(), a2.weight(), self.delta) {
                            a1.nextstate().cmp(&a2.nextstate())
                        } else {
                            (self.weight_cmp)(a1.weight(), a2.weight())
                        }
                    }
                    other => other,
                },
                other => other,
            }
        };

        arcs1.sort_by(&mut cmp_func);
        arcs2.sort_by(&mut cmp_func);

        for i in 0..arcs1.len() {
            let arc1 = &arcs1[i];
            let arc2 = &arcs2[i];

            if arc1.ilabel() != arc2.ilabel() {
                log::debug!(
                    "Isomorphic: ilabels not equal. state1: {:?} arc1: *{:?}* state2: {:?} arc2: *{:?}*",
                    s1,
                    arc1.ilabel(),
                    s2,
                    arc2.ilabel()
                );
                return Ok(false);
            }
            if arc1.olabel() != arc2.olabel() {
                log::debug!(
                    "Isomorphic: olabels not equal. state1: {:?} arc1: *{:?}* state2: {:?} arc2: *{:?}*",
                    s1,
                    arc1.olabel(),
                    s2,
                    arc2.olabel()
                );
                return Ok(false);
            }
            if !A::Weight::approx_equal(arc1.weight(), arc2.weight(), self.delta) {
                log::debug!(
                    "Isomorphic: weights not ApproxEqual. state1: {:?} arc1: *{:?}* state2: {:?} arc2: *{:?}*",
                    s1,
                    arc1.weight(),
                    s2,
                    arc2.weight()
                );
                return Ok(false);
            }
            if !self.pair_state(arc1.nextstate(), arc2.nextstate())? {
                log::debug!(
                    "Isomorphic: nextstates could not be paired. state1: {:?} arc1: *{:?}* state2: {:?} arc2: *{:?}*",
                    s1,
                    arc1.nextstate(),
                    s2,
                    arc2.nextstate()
                );
                return Ok(false);
            }

            // Checks for non-determinism.
            if i > 0 {
                let arc0 = &arcs1[i - 1];
                if arc1.ilabel() == arc0.ilabel()
                    && arc1.olabel() == arc0.olabel()
                    && A::Weight::approx_equal(arc1.weight(), arc0.weight(), self.delta)
                {
                    log::debug!(
                        "Isomorphic: Detected non-determinism as an unweighted automaton; deferring error. state: {:?} arc: {:?}",
                        s1,
                        arc1.ilabel()
                    );
                    self.nondet = true;
                }
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::fst::MutableFst;
    use crate::fsts::vector_fst::VectorFst;
    use crate::weight::Weight;
    use crate::weights::float_weight::TropicalWeight;

    fn isomorphic_ok(fst1: &VectorFst<StdArc>, fst2: &VectorFst<StdArc>, delta: f32) -> bool {
        isomorphic(fst1, fst2, delta).expect("these inputs are decidable")
    }

    fn two_loops(first: i32, second: i32) -> VectorFst<StdArc> {
        let mut fst = VectorFst::new();
        let start = fst.add_state();
        fst.set_start(start);
        fst.set_final(start, TropicalWeight::one());
        fst.add_arc(
            start,
            StdArc::new(first, first, TropicalWeight::one(), start),
        );
        fst.add_arc(
            start,
            StdArc::new(second, second, TropicalWeight::one(), start),
        );
        fst
    }

    /// Isomorphism is the comparison that ignores how the states happen to be
    /// numbered and how the arcs happen to be ordered, which is exactly what
    /// `Equal` does not ignore.
    #[test]
    fn arc_order_does_not_matter() {
        assert!(isomorphic_ok(&two_loops(1, 2), &two_loops(2, 1), 1e-6));
    }

    #[test]
    fn state_numbering_does_not_matter() {
        // A path 0 -> 1, and the same path with the states swapped.
        let mut forwards = VectorFst::<StdArc>::new();
        let a = forwards.add_state();
        let b = forwards.add_state();
        forwards.set_start(a);
        forwards.set_final(b, TropicalWeight::one());
        forwards.add_arc(a, StdArc::new(1, 1, TropicalWeight::one(), b));

        let mut renumbered = VectorFst::<StdArc>::new();
        let b = renumbered.add_state();
        let a = renumbered.add_state();
        renumbered.set_start(a);
        renumbered.set_final(b, TropicalWeight::one());
        renumbered.add_arc(a, StdArc::new(1, 1, TropicalWeight::one(), b));

        assert!(isomorphic_ok(&forwards, &renumbered, 1e-6));
    }

    #[test]
    fn different_labels_are_not_isomorphic() {
        assert!(!isomorphic_ok(&two_loops(1, 2), &two_loops(1, 3), 1e-6));
    }

    #[test]
    fn a_different_shape_is_not_isomorphic() {
        let mut one_loop = VectorFst::<StdArc>::new();
        let start = one_loop.add_state();
        one_loop.set_start(start);
        one_loop.set_final(start, TropicalWeight::one());
        one_loop.add_arc(start, StdArc::new(1, 1, TropicalWeight::one(), start));

        assert!(!isomorphic_ok(&two_loops(1, 2), &one_loop, 1e-6));
    }

    #[test]
    fn weights_still_have_to_match() {
        let mut heavy = two_loops(1, 2);
        heavy.set_final(0, TropicalWeight(9.0));
        assert!(!isomorphic_ok(&two_loops(1, 2), &heavy, 1e-6));
    }

    #[test]
    fn two_empty_fsts_are_isomorphic() {
        let left = VectorFst::<StdArc>::new();
        let right = VectorFst::<StdArc>::new();
        assert!(isomorphic_ok(&left, &right, 1e-6));
    }

    #[test]
    fn an_empty_fst_is_not_isomorphic_to_a_non_empty_one() {
        let empty = VectorFst::<StdArc>::new();
        assert!(!isomorphic_ok(&empty, &two_loops(1, 2), 1e-6));
        assert!(!isomorphic_ok(&two_loops(1, 2), &empty, 1e-6));
    }

    /// Isomorphism is only decidable here when the inputs are deterministic as
    /// unweighted automata, since two arcs leaving a state on the same label
    /// leave no way to know which should match which. sicada says so; upstream logs and
    /// answers "not isomorphic", which a caller cannot tell from a real answer.
    #[test]
    fn nondeterministic_input_is_reported_as_undecidable() {
        let build = |first_final: f32, second_final: f32| {
            let mut fst = VectorFst::<StdArc>::new();
            let start = fst.add_state();
            let first = fst.add_state();
            let second = fst.add_state();
            fst.set_start(start);
            fst.set_final(first, TropicalWeight(first_final));
            fst.set_final(second, TropicalWeight(second_final));
            // Both arcs carry label 1, so the automaton is nondeterministic.
            fst.add_arc(start, StdArc::new(1, 1, TropicalWeight::one(), first));
            fst.add_arc(start, StdArc::new(1, 1, TropicalWeight::one(), second));
            fst
        };
        let error = isomorphic(&build(1.0, 2.0), &build(2.0, 1.0), 1e-6)
            .expect_err("nondeterministic input cannot be decided");
        assert!(error.contains("Non-determinism"), "{error}");
    }
}
