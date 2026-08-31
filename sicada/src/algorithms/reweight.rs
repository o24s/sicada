//! Pushing weight along an FST's paths.
//!
//! Port of OpenFst's `reweight.h`. Reweighting redistributes the weight of
//! every path without changing what that path's weight comes to: an arc from a
//! state of potential `p` to one of potential `q` gives up `p` and takes on
//! `q`, and the potentials cancel along the path. Which side they cancel on is
//! what the direction chooses, and it is why one direction needs a left
//! distributive semiring and the other a right one.

use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::fst::MutableFst;
use crate::properties::{K_FST_PROPERTIES, K_INITIAL_ACYCLIC, reweight_properties};
use crate::weight::{Divide, DivideType, LeftSemiring, RightSemiring, Weight};

/// Reweights `fst` towards its initial state.
///
/// An arc of weight `w` from a state of potential `p` to one of potential `q`
/// becomes `p⁻¹ ⊗ (w ⊗ q)`. States past the end of `potential`, and states whose
/// potential is `Zero`, are left alone.
///
/// SICADA-DIVERGE: upstream checks at run time that the weight is left
/// distributive and sets `K_ERROR` if it is not. The bound says the same thing
/// where it cannot be got wrong: an FST over a weight that does not qualify
/// will not compile against this.
///
/// SICADA-DIVERGE: a potential that is not a member of its semiring is treated
/// as `Zero`, so the states it belongs to are skipped rather than dividing by
/// an invalid weight.
pub fn reweight_to_initial<A, F>(fst: &mut F, potential: &[A::Weight])
where
    A: Arc,
    F: MutableFst<A>,
    A::Weight: LeftSemiring + Divide,
{
    if fst.num_states() == 0 {
        return;
    }

    let input_props = fst.properties(K_FST_PROPERTIES, false);
    let states: Vec<_> = fst.states().collect();

    for &s in &states {
        let s_idx = s.as_usize();
        if s_idx >= potential.len() {
            // Note: states past the potential length are untouched for INITIAL.
            continue;
        }

        let weight = potential[s_idx].clone();
        if weight.is_member() && weight != A::Weight::zero() {
            fst.mutate_arcs(s, |arc| {
                let Some(nextweight) = potential.get(arc.nextstate().as_usize()) else {
                    return;
                };
                if !nextweight.is_member() || *nextweight == A::Weight::zero() {
                    return;
                }
                let reweighted = arc
                    .weight()
                    .times(nextweight)
                    .divide(&weight, DivideType::Left);
                *arc = A::new(arc.ilabel(), arc.olabel(), reweighted, arc.nextstate());
            });

            let fin = fst.final_weight(s);
            fst.set_final(s, fin.divide(&weight, DivideType::Left));
        }
    }

    let startweight = if let Some(start) = fst.start() {
        let idx = start.as_usize();
        if idx < potential.len() {
            potential[idx].clone()
        } else {
            A::Weight::zero()
        }
    } else {
        A::Weight::zero()
    };

    let mut added_start_epsilon = false;
    if startweight != A::Weight::one() && startweight != A::Weight::zero() {
        if (fst.properties(K_INITIAL_ACYCLIC, true) & K_INITIAL_ACYCLIC) != 0 {
            let s = fst
                .start()
                .expect("a start weight means there is a start state");
            fst.mutate_arcs(s, |arc| {
                let pushed = startweight.times(arc.weight());
                *arc = A::new(arc.ilabel(), arc.olabel(), pushed, arc.nextstate());
            });

            let fin = fst.final_weight(s);
            fst.set_final(s, startweight.times(&fin));
        } else {
            let s = fst.add_state();
            let arc = A::new(
                A::Label::epsilon(),
                A::Label::epsilon(),
                startweight,
                fst.start()
                    .expect("a start weight means there is a start state"),
            );
            fst.add_arc(s, arc);
            fst.set_start(s);
            added_start_epsilon = true;
        }
    }

    let oprops = fst.properties(K_FST_PROPERTIES, false);
    fst.set_properties(
        reweight_properties(input_props, added_start_epsilon) | oprops,
        K_FST_PROPERTIES,
    );
}

/// Reweights `fst` towards its final states.
///
/// An arc of weight `w` from a state of potential `p` to one of potential `q`
/// becomes `(p ⊗ w) ⊗ q⁻¹`. A state whose potential is `Zero`, or which has none
/// because it lies past the end of `potential`, stops being final: it cannot
/// reach a final state, so no path through it can succeed.
///
/// See [`reweight_to_initial`] for the two divergences from upstream, which
/// apply here with the sides swapped.
pub fn reweight_to_final<A, F>(fst: &mut F, potential: &[A::Weight])
where
    A: Arc,
    F: MutableFst<A>,
    A::Weight: RightSemiring + Divide,
{
    if fst.num_states() == 0 {
        return;
    }

    let input_props = fst.properties(K_FST_PROPERTIES, false);
    let states: Vec<_> = fst.states().collect();

    for &s in &states {
        let s_idx = s.as_usize();

        if s_idx >= potential.len() {
            // This handles elements past the end of the potentials array for REWEIGHT_TO_FINAL.
            let fin = fst.final_weight(s);
            fst.set_final(s, A::Weight::zero().times(&fin));
            continue;
        }

        let weight = potential[s_idx].clone();
        if weight.is_member() && weight != A::Weight::zero() {
            fst.mutate_arcs(s, |arc| {
                let Some(nextweight) = potential.get(arc.nextstate().as_usize()) else {
                    return;
                };
                if !nextweight.is_member() || *nextweight == A::Weight::zero() {
                    return;
                }
                let reweighted = weight
                    .times(arc.weight())
                    .divide(nextweight, DivideType::Right);
                *arc = A::new(arc.ilabel(), arc.olabel(), reweighted, arc.nextstate());
            });
        }

        // Outside the test above: a state whose potential is `Zero` reaches no
        // final state, so `Zero ⊗ final` makes it non-final.
        let fin = fst.final_weight(s);
        fst.set_final(s, weight.times(&fin));
    }

    let startweight = if let Some(start) = fst.start() {
        let idx = start.as_usize();
        if idx < potential.len() {
            potential[idx].clone()
        } else {
            A::Weight::zero()
        }
    } else {
        A::Weight::zero()
    };

    let mut added_start_epsilon = false;
    if startweight != A::Weight::one() && startweight != A::Weight::zero() {
        let inv_startweight = A::Weight::one().divide(&startweight, DivideType::Right);

        if (fst.properties(K_INITIAL_ACYCLIC, true) & K_INITIAL_ACYCLIC) != 0 {
            let s = fst
                .start()
                .expect("a start weight means there is a start state");
            fst.mutate_arcs(s, |arc| {
                let pushed = inv_startweight.times(arc.weight());
                *arc = A::new(arc.ilabel(), arc.olabel(), pushed, arc.nextstate());
            });

            let fin = fst.final_weight(s);
            fst.set_final(s, inv_startweight.times(&fin));
        } else {
            let s = fst.add_state();
            let arc = A::new(
                A::Label::epsilon(),
                A::Label::epsilon(),
                inv_startweight,
                fst.start()
                    .expect("a start weight means there is a start state"),
            );
            fst.add_arc(s, arc);
            fst.set_start(s);
            added_start_epsilon = true;
        }
    }

    let oprops = fst.properties(K_FST_PROPERTIES, false);
    fst.set_properties(
        reweight_properties(input_props, added_start_epsilon) | oprops,
        K_FST_PROPERTIES,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{Rng, paths, random_acyclic_fst, sorted};
    use crate::arc::StdArc;
    use crate::fst::{ExpandedFst as _, Fst, MutableFst};
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::weights::float_weight::TropicalWeight;

    #[test]
    fn test_reweight_to_initial() {
        let mut fst = StdVectorFst::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight(5.0), s1));

        // Potential mapping:
        // p[0] = 2.0
        // p[1] = 1.0
        let potentials = vec![TropicalWeight(2.0), TropicalWeight(1.0)];
        reweight_to_initial(&mut fst, &potentials);

        // - Reweight arcs: w' = p[s0]^-1 * w * p[s1]
        // Tropical division is subtraction: (5.0 + 1.0) - 2.0 = 4.0
        // - Start state correction (added to arcs leaving the start state):
        // p[s0] * w' = 2.0 + 4.0 = 6.0
        let mut arcs = fst.arcs(s0);
        let arc = arcs.next().unwrap();
        assert_eq!(*arc.weight(), TropicalWeight(6.0));
    }

    #[test]
    fn test_reweight_to_final() {
        let mut fst = StdVectorFst::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight(0.0));

        fst.add_arc(s0, StdArc::new(1, 1, TropicalWeight(5.0), s1));

        // Potential mapping:
        // p[0] = 2.0
        // p[1] = 1.0
        let potentials = vec![TropicalWeight(2.0), TropicalWeight(1.0)];
        reweight_to_final(&mut fst, &potentials);

        // - Reweight arcs: w' = p[s0] * w * p[s1]^-1
        // Tropical division is subtraction: (2.0 + 5.0) - 1.0 = 6.0
        // - Start state correction (added to arcs leaving the start state):
        // (1 / p[s0]) * w' = (0.0 - 2.0) + 6.0 = 4.0
        let mut arcs = fst.arcs(s0);
        let arc = arcs.next().unwrap();
        assert_eq!(*arc.weight(), TropicalWeight(4.0));
    }

    /// What reweighting is for: the weight of every accepting path comes out
    /// the same. Reweighting moves weight from one arc to another along a path;
    /// the potentials cancel, so the product over the path does not move.
    ///
    /// The potentials here are arbitrary rather than the shortest distances a
    /// caller would normally use, because the property does not depend on where
    /// they came from.
    fn assert_preserves_path_weights(
        fst: &StdVectorFst,
        potential: &[TropicalWeight],
        to_final: bool,
    ) {
        let before = sorted(paths(fst, 8));

        let mut after_fst = fst.clone();
        if to_final {
            reweight_to_final(&mut after_fst, potential);
        } else {
            reweight_to_initial(&mut after_fst, potential);
        }
        // Reweighting may prepend an epsilon arc carrying the start weight, so
        // one more step is allowed and epsilon labels drop out.
        let after = sorted(
            paths(&after_fst, 9)
                .into_iter()
                .map(|(i, o, w)| {
                    (
                        i.into_iter().filter(|&l| l != 0).collect(),
                        o.into_iter().filter(|&l| l != 0).collect(),
                        w,
                    )
                })
                .collect(),
        );
        assert_eq!(after, before, "to_final={to_final}");
    }

    #[test]
    fn reweighting_leaves_every_path_weighing_what_it_did() {
        let mut rng = Rng::new(0x5EED_1234);
        for _ in 0..200 {
            let fst = random_acyclic_fst(&mut rng, 5);
            let nstates = fst.num_states();
            // Every potential a member and non-zero, so no state drops out;
            // the zero cases have their own test below.
            let potential: Vec<TropicalWeight> = (0..nstates)
                .map(|_| TropicalWeight(rng.below(5) as f32))
                .collect();

            assert_preserves_path_weights(&fst, &potential, false);
            assert_preserves_path_weights(&fst, &potential, true);
        }
    }

    /// Reweighting towards the final states drops a state whose potential says
    /// it reaches none, whether the potential is `Zero` or missing entirely.
    #[test]
    fn a_state_with_no_potential_stops_being_final_when_reweighting_to_final() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight::one(), 2));
        fst.set_final(1, TropicalWeight::one());
        fst.set_final(2, TropicalWeight::one());

        // State 1's potential is Zero; state 2 has none at all.
        let potential = [TropicalWeight::one(), TropicalWeight::zero()];
        reweight_to_final(&mut fst, &potential);

        assert_eq!(fst.final_weight(1), TropicalWeight::zero());
        assert_eq!(fst.final_weight(2), TropicalWeight::zero());
    }

    /// Reweighting towards the initial state leaves such states alone, as
    /// upstream does: the final-weight update sits inside the potential test
    /// there and outside it for the other direction.
    #[test]
    fn a_state_with_no_potential_is_untouched_when_reweighting_to_initial() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.set_final(1, TropicalWeight(2.0));
        fst.set_final(2, TropicalWeight(3.0));

        let potential = [TropicalWeight::one(), TropicalWeight::zero()];
        reweight_to_initial(&mut fst, &potential);

        assert_eq!(fst.final_weight(1), TropicalWeight(2.0));
        assert_eq!(fst.final_weight(2), TropicalWeight(3.0));
    }

    #[test]
    fn an_empty_fst_is_left_alone() {
        let mut fst = StdVectorFst::new();
        reweight_to_initial(&mut fst, &[]);
        reweight_to_final(&mut fst, &[]);
        assert_eq!(fst.num_states(), 0);
    }
}
