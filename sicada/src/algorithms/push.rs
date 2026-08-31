//! Moving weight, and labels, towards one end of an FST.
//!
//! Port of OpenFst's `push.h`. Pushing towards the initial state makes the sum
//! of the arcs leaving each non-initial state, together with its final weight,
//! come to [`Weight::one`], so a weight is committed as early on a path as it
//! can be, which is why a search over the result prunes well. Pushing
//! towards the final states does the same on the reversed machine.
//!
//! What the FST accepts does not change; only where along a path the weight
//! sits.

use crate::algorithms::arc_map::{FromGallicMapper, RmWeightMapper, ToGallicMapper, arc_map_to};
use crate::algorithms::factor_weight::{
    FactorIterator, FactorWeightOptions, GallicFactor, factor_weight,
};
use crate::algorithms::reweight::{reweight_to_final, reweight_to_initial};
use crate::algorithms::shortest_distance::{shortest_distance_forward, shortest_distance_reverse};
use crate::arc::{Arc, ArcStateId, GallicArc};
use crate::error::OpenFstError;
use crate::fst::{Fst, MutableFst};
use crate::fsts::vector_fst::VectorFst;
use crate::weight::{Divide, DivideType, LeftSemiring, RightSemiring, Weight};
use crate::weights::string_weight::{
    GallicLeft, GallicRight, GallicTypeMarker, GallicWeight, StringWeight,
};

/// Which end of the FST to move weight towards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReweightType {
    /// Towards the initial state.
    ToInitial,
    /// Towards the final states.
    ToFinal,
}

/// Move the weights.
pub const PUSH_WEIGHTS: u8 = 0x01;
/// Move the labels.
pub const PUSH_LABELS: u8 = 0x02;
/// Divide the total weight out, so what is left sums to [`Weight::one`].
pub const PUSH_REMOVE_TOTAL_WEIGHT: u8 = 0x04;
/// Divide the common prefix or suffix out of the labels.
pub const PUSH_REMOVE_COMMON_AFFIX: u8 = 0x08;

/// The ⊕-sum of every accepting path, read off a distance vector already
/// computed.
///
/// Which end the distance was computed from decides how to read it: from the
/// final states it is the start state's entry, from the initial state it is the
/// sum over states of distance ⊗ final weight.
pub fn total_weight<A, F>(fst: &F, distance: &[A::Weight], reverse: bool) -> A::Weight
where
    A: Arc,
    F: Fst<A>,
{
    if reverse {
        return fst
            .start()
            .and_then(|start| distance.get(start.as_usize()).cloned())
            .unwrap_or_else(A::Weight::zero);
    }
    let mut sum = A::Weight::zero();
    for (index, weight) in distance.iter().enumerate() {
        sum = sum.plus(&weight.times(&fst.final_weight(A::StateId::from_usize(index))));
    }
    sum
}

/// Divides `weight` out of every accepting path, at the final states when
/// `at_final` and at the initial state otherwise.
pub fn remove_weight<A, F>(fst: &mut F, weight: &A::Weight, at_final: bool)
where
    A: Arc,
    A::Weight: Divide,
    F: MutableFst<A>,
{
    if *weight == A::Weight::one() || *weight == A::Weight::zero() {
        return;
    }
    if at_final {
        let states: Vec<A::StateId> = fst.states().collect();
        for state in states {
            let divided = fst.final_weight(state).divide(weight, DivideType::Right);
            fst.set_final(state, divided);
        }
        return;
    }
    let Some(start) = fst.start() else { return };
    fst.mutate_arcs(start, |arc| {
        let divided = arc.weight().divide(weight, DivideType::Left);
        *arc = A::new(arc.ilabel(), arc.olabel(), divided, arc.nextstate());
    });
    let divided = fst.final_weight(start).divide(weight, DivideType::Left);
    fst.set_final(start, divided);
}

/// Moves the weights of `fst` towards one end, in place.
///
/// The direction is chosen at run time, so the weight has to distribute on
/// both sides; a semiring with only one should use [`push_to_initial`] or
/// [`push_to_final`], which each ask for the side they actually divide on.
pub fn push_weights<A, F>(
    fst: &mut F,
    to: ReweightType,
    delta: f32,
    remove_total_weight: bool,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: Divide + LeftSemiring + RightSemiring,
    F: MutableFst<A>,
    <A::Weight as Weight>::ReverseWeight: Weight<ReverseWeight = A::Weight>,
{
    // Pushing towards the initial state needs to know what each state can still
    // reach, which is the distance computed backwards.
    let reverse = to == ReweightType::ToInitial;
    let distance = if reverse {
        shortest_distance_reverse(fst, delta)?
    } else {
        shortest_distance_forward(fst, delta)?
    };

    let total = remove_total_weight.then(|| total_weight(fst, &distance, reverse));
    match to {
        ReweightType::ToInitial => reweight_to_initial(fst, &distance),
        ReweightType::ToFinal => reweight_to_final(fst, &distance),
    }
    if let Some(total) = total {
        remove_weight(fst, &total, !reverse);
    }
    Ok(())
}

/// Moves the weights towards the initial state, in place.
///
/// Only left distributivity is asked for, since that is the only side this
/// direction divides on, which lets a weight that has one side but not the
/// other, such as the left gallic weight, be pushed.
pub fn push_weights_to_initial<A, F>(
    fst: &mut F,
    delta: f32,
    remove_total_weight: bool,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: Divide + LeftSemiring,
    F: MutableFst<A>,
    <A::Weight as Weight>::ReverseWeight: Weight<ReverseWeight = A::Weight>,
{
    let distance = shortest_distance_reverse(fst, delta)?;
    let total = remove_total_weight.then(|| total_weight(fst, &distance, true));
    reweight_to_initial(fst, &distance);
    if let Some(total) = total {
        remove_weight(fst, &total, false);
    }
    Ok(())
}

/// As [`push_weights_to_initial`], the other way.
pub fn push_weights_to_final<A, F>(
    fst: &mut F,
    delta: f32,
    remove_total_weight: bool,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: Divide + RightSemiring,
    F: MutableFst<A>,
    <A::Weight as Weight>::ReverseWeight: Weight<ReverseWeight = A::Weight>,
{
    let distance = shortest_distance_forward(fst, delta)?;
    let total = remove_total_weight.then(|| total_weight(fst, &distance, false));
    reweight_to_final(fst, &distance);
    if let Some(total) = total {
        remove_weight(fst, &total, true);
    }
    Ok(())
}

/// Moves the weights and labels of `ifst` towards the initial state.
///
/// `flags` is some combination of [`PUSH_WEIGHTS`], [`PUSH_LABELS`],
/// [`PUSH_REMOVE_TOTAL_WEIGHT`] and [`PUSH_REMOVE_COMMON_AFFIX`].
///
/// Labels travel through the **left** gallic semiring, whose string half keeps
/// the longest common prefix and so distributes on the left, which is the side
/// this direction divides on. That pairing is not a choice: upstream picks the
/// gallic type from the direction for the same reason, and here the bound makes
/// the wrong pairing fail to compile rather than to converge.
pub fn push_to_initial<A, F1, F2>(
    ifst: &F1,
    ofst: &mut F2,
    flags: u8,
    delta: f32,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: Divide + LeftSemiring + std::hash::Hash + Eq,
    F1: Fst<A>,
    F2: MutableFst<A>,
    <A::Weight as Weight>::ReverseWeight: Weight<ReverseWeight = A::Weight>,
    GallicWeight<A::Label, A::Weight, GallicLeft>: Divide + LeftSemiring,
    <GallicWeight<A::Label, A::Weight, GallicLeft> as Weight>::ReverseWeight:
        Weight<ReverseWeight = GallicWeight<A::Label, A::Weight, GallicLeft>>,
{
    push_impl::<A, GallicLeft, F1, F2>(
        ifst,
        ofst,
        ReweightType::ToInitial,
        flags,
        delta,
        reweight_to_initial,
        reweight_to_initial,
    )
}

/// Moves the weights and labels of `ifst` towards the final states.
///
/// Labels travel through the **right** gallic semiring, for the mirror of the
/// reason [`push_to_initial`] uses the left one.
pub fn push_to_final<A, F1, F2>(
    ifst: &F1,
    ofst: &mut F2,
    flags: u8,
    delta: f32,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: Divide + RightSemiring + std::hash::Hash + Eq,
    F1: Fst<A>,
    F2: MutableFst<A>,
    <A::Weight as Weight>::ReverseWeight: Weight<ReverseWeight = A::Weight>,
    GallicWeight<A::Label, A::Weight, GallicRight>: Divide + RightSemiring,
    <GallicWeight<A::Label, A::Weight, GallicRight> as Weight>::ReverseWeight:
        Weight<ReverseWeight = GallicWeight<A::Label, A::Weight, GallicRight>>,
{
    push_impl::<A, GallicRight, F1, F2>(
        ifst,
        ofst,
        ReweightType::ToFinal,
        flags,
        delta,
        reweight_to_final,
        reweight_to_final,
    )
}

/// What both directions do, with the two reweightings handed in.
///
/// SICADA-DIVERGE: upstream logs a warning and copies the input when asked to
/// push neither weights nor labels. Doing nothing is what was asked for, so it
/// is not a warning; the copy is made either way.
///
/// The reweightings are passed as function pointers rather than chosen here
/// because which one is used decides which side the semiring has to distribute
/// on, and that is a bound the caller carries. Choosing at run time would mean
/// demanding both sides of every weight.
#[allow(clippy::too_many_arguments)]
fn push_impl<A, G, F1, F2>(
    ifst: &F1,
    ofst: &mut F2,
    to: ReweightType,
    flags: u8,
    delta: f32,
    reweight_plain: fn(&mut F2, &[A::Weight]),
    reweight_gallic: fn(&mut VectorFst<GallicArc<A, G>>, &[GallicWeight<A::Label, A::Weight, G>]),
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: Divide + std::hash::Hash + Eq,
    G: GallicTypeMarker,
    F1: Fst<A>,
    F2: MutableFst<A>,
    <A::Weight as Weight>::ReverseWeight: Weight<ReverseWeight = A::Weight>,
    GallicWeight<A::Label, A::Weight, G>: Divide,
    <GallicWeight<A::Label, A::Weight, G> as Weight>::ReverseWeight:
        Weight<ReverseWeight = GallicWeight<A::Label, A::Weight, G>>,
{
    let reverse = to == ReweightType::ToInitial;

    if flags & PUSH_LABELS == 0 {
        copy_fst(ifst, ofst);
        if flags & PUSH_WEIGHTS != 0 {
            let distance = distance_of(ofst, reverse, delta)?;
            let total = (flags & PUSH_REMOVE_TOTAL_WEIGHT != 0)
                .then(|| total_weight(ofst, &distance, reverse));
            reweight_plain(ofst, &distance);
            if let Some(total) = total {
                remove_weight(ofst, &total, !reverse);
            }
        }
        return Ok(());
    }

    // Labels are pushed by moving them into the weight, where the machinery for
    // weights already applies, and taking them back out again.
    let mut gfst: VectorFst<GallicArc<A, G>> = VectorFst::new();
    arc_map_to(ifst, &mut gfst, &mut ToGallicMapper::<G>::new())?;

    let gdistance = if flags & PUSH_WEIGHTS != 0 {
        distance_of(&gfst, reverse, delta)?
    } else {
        // Only the labels are being moved, so the distances have to be worked
        // out over the labels alone.
        let mut unweighted: VectorFst<A> = VectorFst::new();
        arc_map_to(ifst, &mut unweighted, &mut RmWeightMapper)?;
        let mut gunweighted: VectorFst<GallicArc<A, G>> = VectorFst::new();
        arc_map_to(
            &unweighted,
            &mut gunweighted,
            &mut ToGallicMapper::<G>::new(),
        )?;
        distance_of(&gunweighted, reverse, delta)?
    };

    let total = (flags & (PUSH_REMOVE_TOTAL_WEIGHT | PUSH_REMOVE_COMMON_AFFIX) != 0).then(|| {
        let total = total_weight(&gfst, &gdistance, reverse);
        // Only the halves that were asked for are divided out.
        GallicWeight::<A::Label, A::Weight, G>::from_parts(
            if flags & PUSH_REMOVE_COMMON_AFFIX != 0 {
                total.labels().clone()
            } else {
                StringWeight::one()
            },
            if flags & PUSH_REMOVE_TOTAL_WEIGHT != 0 {
                total.weight().clone()
            } else {
                A::Weight::one()
            },
        )
    });

    reweight_gallic(&mut gfst, &gdistance);
    if let Some(total) = total {
        remove_weight(&mut gfst, &total, to == ReweightType::ToFinal);
    }

    // A gallic weight may now hold several labels, which one arc cannot carry,
    // so it is spread over a chain of arcs before being taken apart.
    let mut factored: VectorFst<GallicArc<A, G>> = VectorFst::new();
    factor_weight(
        &gfst,
        &mut factored,
        GallicFactor::new,
        &FactorWeightOptions::<A::Label>::default(),
    );
    let mut mapper = FromGallicMapper::<A::Label, G>::new();
    arc_map_to(&factored, ofst, &mut mapper)?;
    if mapper.error() {
        return Err(OpenFstError::InvalidOperation(
            "Push: a weight came out that no single arc can carry".into(),
        ));
    }
    ofst.set_output_symbols(ifst.output_symbols());
    Ok(())
}

/// The distance from or to the final states, whichever pushing needs.
fn distance_of<A, F>(fst: &F, reverse: bool, delta: f32) -> Result<Vec<A::Weight>, OpenFstError>
where
    A: Arc,
    F: Fst<A>,
    <A::Weight as Weight>::ReverseWeight: Weight<ReverseWeight = A::Weight>,
{
    if reverse {
        shortest_distance_reverse(fst, delta)
    } else {
        shortest_distance_forward(fst, delta)
    }
}

/// Copies an FST state for state.
fn copy_fst<A, F1, F2>(ifst: &F1, ofst: &mut F2)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::shortest_distance::SHORTEST_DELTA;
    use crate::algorithms::test_support::{Rng, paths, random_acyclic_fst, sorted};
    use crate::arc::StdArc;
    use crate::fst::ExpandedFst as _;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::K_FST_PROPERTIES;
    use crate::weights::float_weight::TropicalWeight;

    /// The arcs and final weight leaving each non-initial state should come to
    /// One once the weight has been pushed to the front.
    fn leaving(fst: &StdVectorFst, state: i32) -> TropicalWeight {
        let mut sum = fst.final_weight(state);
        for arc in fst.arcs(state) {
            sum = sum.plus(arc.weight());
        }
        sum
    }

    /// Two branches costing 1 and 3, joining and then costing 2 more.
    fn branches() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..4 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(1.0), 1));
        fst.add_arc(0, StdArc::new(2, 2, TropicalWeight(3.0), 2));
        fst.add_arc(1, StdArc::new(3, 3, TropicalWeight(2.0), 3));
        fst.add_arc(2, StdArc::new(4, 4, TropicalWeight(2.0), 3));
        fst.set_final(3, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    fn pushed(to: ReweightType, remove_total: bool) -> StdVectorFst {
        let mut fst = branches();
        push_weights(&mut fst, to, SHORTEST_DELTA, remove_total).unwrap();
        fst
    }

    /// Pushing to the front commits weight as early as it can, so every
    /// non-initial state's outgoing weight sums to One.
    #[test]
    fn pushing_to_the_initial_state_leaves_one_behind_every_state() {
        let fst = pushed(ReweightType::ToInitial, true);
        for state in 1..fst.num_states() as i32 {
            assert_eq!(leaving(&fst, state), TropicalWeight::one(), "state {state}");
        }
        // The lighter branch is now free, and the difference sits on the other.
        let from_start: Vec<f32> = fst.arcs(0).map(|a| a.weight().value()).collect();
        assert_eq!(from_start, vec![0.0, 2.0]);
    }

    /// Pushing does not change what the FST accepts, only where the weight
    /// sits along each path.
    #[test]
    fn pushing_does_not_change_the_paths() {
        let before = sorted(paths(&branches(), 12));
        assert_eq!(
            sorted(paths(&pushed(ReweightType::ToInitial, false), 12)),
            before
        );
        assert_eq!(
            sorted(paths(&pushed(ReweightType::ToFinal, false), 12)),
            before
        );
    }

    /// Dividing the total out makes the shortest path weigh One; the others
    /// weigh what they cost above it.
    #[test]
    fn removing_the_total_weight_makes_the_best_path_free() {
        for to in [ReweightType::ToInitial, ReweightType::ToFinal] {
            let fst = pushed(to, true);
            let mut weights: Vec<f32> = paths(&fst, 12)
                .into_iter()
                .map(|(_, _, weight)| weight.value())
                .collect();
            weights.sort_by(f32::total_cmp);
            assert_eq!(weights, vec![0.0, 2.0], "{to:?}");
        }
    }

    /// Whatever the FST, pushing changes only where the weight sits.
    #[test]
    fn pushing_never_changes_what_a_path_costs_in_total() {
        let mut rng = Rng::new(0x00F0_54ED_u64);
        for round in 0..200 {
            let fst = random_acyclic_fst(&mut rng, 6);
            let before = sorted(paths(&fst, 12));
            if before.is_empty() {
                continue;
            }
            for to in [ReweightType::ToInitial, ReweightType::ToFinal] {
                let mut copy = fst.clone();
                push_weights(&mut copy, to, SHORTEST_DELTA, false).unwrap();
                assert_eq!(sorted(paths(&copy, 12)), before, "round {round}, {to:?}");
            }
        }
    }

    /// Pushing labels moves the output side towards the front, leaving what the
    /// FST transduces unchanged.
    #[test]
    fn pushing_labels_moves_the_output_side_forward() {
        // 0 -a:eps-> 1 -b:xy-> 2, final.
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 0, TropicalWeight::one(), 1));
        fst.add_arc(1, StdArc::new(2, 7, TropicalWeight::one(), 2));
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let before = sorted(paths(&fst, 12));
        let mut out = StdVectorFst::new();
        push_to_initial(&fst, &mut out, PUSH_LABELS, SHORTEST_DELTA).unwrap();

        // The output label 7 has moved onto the first arc.
        let first: Vec<i32> = out.arcs(out.start().unwrap()).map(|a| a.olabel()).collect();
        assert_eq!(first, vec![7]);

        // And the transduction is the same, once epsilons are set aside.
        let visible =
            |paths: Vec<(Vec<i32>, Vec<i32>, String)>| -> Vec<(Vec<i32>, Vec<i32>, String)> {
                paths
                    .into_iter()
                    .map(|(i, o, w)| {
                        (
                            i.into_iter().filter(|l| *l != 0).collect(),
                            o.into_iter().filter(|l| *l != 0).collect(),
                            w,
                        )
                    })
                    .collect()
            };
        assert_eq!(visible(sorted(paths(&out, 12))), visible(before));
    }

    /// Pushing labels and weights together leaves the transduction alone.
    #[test]
    fn pushing_labels_and_weights_keeps_the_transduction() {
        let mut rng = Rng::new(0x001A_B315_u64);
        let visible =
            |paths: Vec<(Vec<i32>, Vec<i32>, String)>| -> Vec<(Vec<i32>, Vec<i32>, String)> {
                paths
                    .into_iter()
                    .map(|(i, o, w)| {
                        (
                            i.into_iter().filter(|l| *l != 0).collect(),
                            o.into_iter().filter(|l| *l != 0).collect(),
                            w,
                        )
                    })
                    .collect()
            };
        for round in 0..100 {
            let fst = random_acyclic_fst(&mut rng, 5);
            let before = visible(sorted(paths(&fst, 12)));
            if before.is_empty() {
                continue;
            }
            let mut out = StdVectorFst::new();
            push_to_initial(&fst, &mut out, PUSH_LABELS | PUSH_WEIGHTS, SHORTEST_DELTA).unwrap();
            assert_eq!(visible(sorted(paths(&out, 12))), before, "round {round}");
        }
    }

    /// Asking for neither weights nor labels copies the FST.
    #[test]
    fn pushing_nothing_copies_the_fst() {
        let fst = branches();
        let mut out = StdVectorFst::new();
        push_to_initial(&fst, &mut out, 0, SHORTEST_DELTA).unwrap();
        assert_eq!(sorted(paths(&out, 12)), sorted(paths(&fst, 12)));
        assert_eq!(out.num_states(), fst.num_states());
    }

    /// The total weight read off a distance vector is the sum over accepting
    /// paths, whichever end it was computed from.
    #[test]
    fn the_total_weight_is_the_same_from_either_end() {
        let fst = branches();
        let forward = shortest_distance_forward(&fst, SHORTEST_DELTA).unwrap();
        let backward = shortest_distance_reverse(&fst, SHORTEST_DELTA).unwrap();
        assert_eq!(
            total_weight(&fst, &forward, false),
            total_weight(&fst, &backward, true)
        );
        assert_eq!(total_weight(&fst, &forward, false), TropicalWeight(3.0));
    }
}
