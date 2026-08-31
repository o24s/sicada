//! Splitting weights across arcs.
//!
//! Port of OpenFst's `factor-weight.h`.
//!
//! Some weights are made of pieces: a string weight is a sequence of labels,
//! and a gallic weight is a string paired with a weight. An FST carrying them
//! can be rewritten so that each piece sits on its own arc. That is how the
//! output of a determinization over the gallic semiring is turned back into an
//! ordinary transducer.
//!
//! What can be split, and how, is a [`FactorIterator`]: given a weight, it
//! enumerates the ways of writing it as a first piece and a remainder.

use rustc_hash::FxHashMap;

use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::fst::{Fst, MutableFst};
use crate::properties::{K_FST_PROPERTIES, factor_weight_properties};
use crate::weight::{DELTA, Weight};
use crate::weights::string_weight::{
    GallicTypeMarker, GallicWeight, StringTypeMarker, StringWeight, StringWeightValue,
};

bitflags::bitflags! {
    /// Which weights to split.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct FactorMode: u8 {
        /// Split the final weights, putting the pieces on arcs to new states.
        const FINAL_WEIGHTS = 0x01;
        /// Split the arc weights, putting the remainder on the state reached.
        const ARC_WEIGHTS = 0x02;
    }
}

/// How to split weights.
#[derive(Debug, Clone)]
pub struct FactorWeightOptions<L> {
    /// How closely two remainders must agree to be treated as the same state.
    pub delta: f32,
    /// Which weights to split.
    pub mode: FactorMode,
    /// The input label of an arc made from a final weight.
    pub final_ilabel: L,
    /// The output label of such an arc.
    pub final_olabel: L,
    /// Whether to increment the input label when a final weight splits into
    /// more than one arc, so the arcs stay distinguishable.
    pub increment_final_ilabel: bool,
    /// As above, for the output label.
    pub increment_final_olabel: bool,
}

impl<L: ArcLabel> Default for FactorWeightOptions<L> {
    fn default() -> Self {
        Self {
            delta: DELTA,
            mode: FactorMode::FINAL_WEIGHTS | FactorMode::ARC_WEIGHTS,
            final_ilabel: L::epsilon(),
            final_olabel: L::epsilon(),
            increment_final_ilabel: false,
            increment_final_olabel: false,
        }
    }
}

/// Enumerates the ways of writing a weight as a first piece and a remainder.
///
/// An iterator that yields nothing says the weight cannot be split, and it is
/// left where it is.
pub trait FactorIterator<W: Weight>: Sized {
    /// Begins splitting `weight`.
    fn new(weight: W) -> Self;

    /// The next `(first, remainder)` pair, if any.
    fn next(&mut self) -> Option<(W, W)>;
}

/// Splits nothing. The result is the input, restructured only by whatever
/// else the options ask for.
pub struct IdentityFactor<W>(std::marker::PhantomData<W>);

impl<W: Weight> FactorIterator<W> for IdentityFactor<W> {
    fn new(_weight: W) -> Self {
        Self(std::marker::PhantomData)
    }

    fn next(&mut self) -> Option<(W, W)> {
        None
    }
}

/// Moves every weight onto the state it leads to.
///
/// Only meaningful with [`FactorMode::ARC_WEIGHTS`]: it unfolds the FST until
/// any two paths reaching the same state carry the same weight.
pub struct OneFactor<W> {
    weight: Option<W>,
}

impl<W: Weight> FactorIterator<W> for OneFactor<W> {
    fn new(weight: W) -> Self {
        Self {
            weight: (weight != W::one()).then_some(weight),
        }
    }

    fn next(&mut self) -> Option<(W, W)> {
        self.weight.take().map(|weight| (W::one(), weight))
    }
}

/// Splits a string weight into its first label and the rest.
pub struct StringFactor<L: ArcLabel, S: StringTypeMarker> {
    weight: StringWeight<L, S>,
    done: bool,
}

impl<L: ArcLabel, S: StringTypeMarker> FactorIterator<StringWeight<L, S>> for StringFactor<L, S> {
    fn new(weight: StringWeight<L, S>) -> Self {
        let done = weight.size() <= 1;
        Self { weight, done }
    }

    fn next(&mut self) -> Option<(StringWeight<L, S>, StringWeight<L, S>)> {
        if self.done {
            return None;
        }
        self.done = true;
        let StringWeightValue::Labels(labels) = &self.weight.value else {
            return None;
        };
        Some((
            StringWeight::new(vec![labels[0]]),
            StringWeight::new(labels[1..].to_vec()),
        ))
    }
}

/// Splits a gallic weight by splitting its string part, leaving the weight
/// part on the first piece.
pub struct GallicFactor<L: ArcLabel, W: Weight, G: GallicTypeMarker> {
    weight: GallicWeight<L, W, G>,
    done: bool,
}

impl<L, W, G> FactorIterator<GallicWeight<L, W, G>> for GallicFactor<L, W, G>
where
    L: ArcLabel,
    W: Weight,
    G: GallicTypeMarker,
{
    fn new(weight: GallicWeight<L, W, G>) -> Self {
        let done = weight.labels().size() <= 1;
        Self { weight, done }
    }

    fn next(&mut self) -> Option<(GallicWeight<L, W, G>, GallicWeight<L, W, G>)> {
        if self.done {
            return None;
        }
        self.done = true;
        let StringWeightValue::Labels(labels) = &self.weight.labels().value else {
            return None;
        };
        // The weight travels with the first label; the rest of the string
        // carries nothing, so that the two multiply back to the original.
        Some((
            GallicWeight::from_parts(
                StringWeight::new(vec![labels[0]]),
                self.weight.weight().clone(),
            ),
            GallicWeight::from_parts(StringWeight::new(labels[1..].to_vec()), W::one()),
        ))
    }
}

/// A state of the result: a state of the input plus the remainder owed to it.
///
/// The input state is `None` for the states that exist only to pay out a final
/// weight that was split.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Element<S, W> {
    state: Option<S>,
    weight: W,
}

/// Rewrites `ifst` into `ofst` with its weights split according to `opts`.
///
/// `factor` says how to split one weight, and is given as the constructor
/// itself: `GallicFactor::new`, `StringFactor::new` or `IdentityFactor::new`.
///
/// SICADA-DIVERGE: upstream names the factor iterator as a template argument
/// (`FactorWeightFst<Arc, GallicFactor<…>>`) and so did this port, which meant
/// spelling every other parameter alongside it:
/// `factor_weight::<_, GallicFactor<A::Label, A::Weight, G>, _, _>(…)`.
/// Passing the constructor makes it an ordinary argument, so nothing has to be named.
/// It costs nothing: a function item is a zero-sized type of its own, so the
/// call below is as direct as `FI::new` was.
///
/// SICADA-DIVERGE: upstream provides only the delayed `FactorWeightFst`, and a
/// caller wanting a concrete result assigns one to a `MutableFst`. Building the
/// result directly is the same work without a cache in the middle; the delayed
/// wrapper is still outstanding.
pub fn factor_weight<A, FI, MakeFactor, F1, F2>(
    ifst: &F1,
    ofst: &mut F2,
    factor: MakeFactor,
    opts: &FactorWeightOptions<A::Label>,
) where
    A: Arc,
    A::Weight: std::hash::Hash + Eq,
    FI: FactorIterator<A::Weight>,
    MakeFactor: Fn(A::Weight) -> FI,
    F1: Fst<A>,
    F2: MutableFst<A>,
{
    ofst.delete_all_states();
    ofst.set_input_symbols(ifst.input_symbols());
    ofst.set_output_symbols(ifst.output_symbols());

    let iprops = ifst.properties(K_FST_PROPERTIES, false);
    let Some(istart) = ifst.start() else {
        ofst.set_properties(factor_weight_properties(iprops), K_FST_PROPERTIES);
        return;
    };

    let mut elements: Vec<Element<A::StateId, A::Weight>> = Vec::new();
    let mut ids: FxHashMap<Element<A::StateId, A::Weight>, A::StateId> = FxHashMap::default();

    let mut find_state = |element: Element<A::StateId, A::Weight>,
                          elements: &mut Vec<Element<A::StateId, A::Weight>>,
                          ofst: &mut F2|
     -> A::StateId {
        if let Some(&id) = ids.get(&element) {
            return id;
        }
        let id = ofst.add_state();
        elements.push(element.clone());
        ids.insert(element, id);
        id
    };

    let start = find_state(
        Element {
            state: Some(istart),
            weight: A::Weight::one(),
        },
        &mut elements,
        ofst,
    );
    ofst.set_start(start);

    let zero = A::Weight::zero();
    let mut next = 0;
    while next < elements.len() {
        let element = elements[next].clone();
        let state = A::StateId::from_usize(next);
        next += 1;

        if let Some(input_state) = element.state {
            for arc in ifst.arcs(input_state) {
                let weight = element.weight.times(arc.weight());
                let mut factors = factor(weight.clone());
                let first = opts
                    .mode
                    .contains(FactorMode::ARC_WEIGHTS)
                    .then(|| factors.next())
                    .flatten();
                match first {
                    None => {
                        // Nothing to split, so the whole weight rides the arc.
                        let dest = find_state(
                            Element {
                                state: Some(arc.nextstate()),
                                weight: A::Weight::one(),
                            },
                            &mut elements,
                            ofst,
                        );
                        ofst.add_arc(state, A::new(arc.ilabel(), arc.olabel(), weight, dest));
                    }
                    Some(pair) => {
                        let mut pair = Some(pair);
                        while let Some((head, rest)) = pair.take() {
                            let dest = find_state(
                                Element {
                                    state: Some(arc.nextstate()),
                                    weight: rest.quantize(opts.delta),
                                },
                                &mut elements,
                                ofst,
                            );
                            ofst.add_arc(state, A::new(arc.ilabel(), arc.olabel(), head, dest));
                            pair = factors.next();
                        }
                    }
                }
            }
        }

        // The final weight owed here: what was carried in, times whatever the
        // input state itself is final with.
        let is_final_source =
            element.state.is_none() || ifst.final_weight(element.state.expect("checked")) != zero;
        if opts.mode.contains(FactorMode::FINAL_WEIGHTS) && is_final_source {
            let weight = match element.state {
                Some(input_state) => element.weight.times(&ifst.final_weight(input_state)),
                None => element.weight.clone(),
            };
            let mut factors = factor(weight);
            let mut ilabel = opts.final_ilabel;
            let mut olabel = opts.final_olabel;
            while let Some((head, rest)) = factors.next() {
                let dest = find_state(
                    Element {
                        state: None,
                        weight: rest.quantize(opts.delta),
                    },
                    &mut elements,
                    ofst,
                );
                ofst.add_arc(state, A::new(ilabel, olabel, head, dest));
                if opts.increment_final_ilabel {
                    ilabel = A::Label::from_i64(ilabel.to_i64().unwrap_or(0) + 1).unwrap_or(ilabel);
                }
                if opts.increment_final_olabel {
                    olabel = A::Label::from_i64(olabel.to_i64().unwrap_or(0) + 1).unwrap_or(olabel);
                }
            }
        }

        // A state is final when what it owes cannot be split any further.
        let weight = match element.state {
            Some(input_state) => element.weight.times(&ifst.final_weight(input_state)),
            None => element.weight.clone(),
        };
        let splittable = opts.mode.contains(FactorMode::FINAL_WEIGHTS)
            && factor(weight.clone()).next().is_some();
        if !splittable {
            ofst.set_final(state, weight);
        }
    }

    ofst.set_properties(factor_weight_properties(iprops), K_FST_PROPERTIES);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{string_weights, visible_paths};
    use crate::arc::{ArcTpl, StdArc};
    use crate::fst::ExpandedFst as _;
    use crate::fsts::vector_fst::{StdVectorFst, VectorFst};
    use crate::weights::float_weight::TropicalWeight;
    use crate::weights::string_weight::{GallicLeft, StringLeft};

    type StringArc = ArcTpl<StringWeight<i32, StringLeft>>;
    type StringFst = VectorFst<StringArc>;

    /// Splitting nothing leaves the FST as it was.
    #[test]
    fn the_identity_factor_leaves_the_paths_alone() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(1.0), 1));
        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight(2.0), 2));
        fst.set_final(2, TropicalWeight(3.0));

        let mut ofst = StdVectorFst::new();
        factor_weight(
            &fst,
            &mut ofst,
            IdentityFactor::new,
            &FactorWeightOptions::default(),
        );
        assert_eq!(
            string_weights(visible_paths(&ofst, 8)),
            string_weights(visible_paths(&fst, 8))
        );
    }

    /// A string weight of several labels becomes one label per arc.
    #[test]
    fn a_string_weight_is_split_one_label_to_an_arc() {
        let mut fst = StringFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StringArc::new(1, 1, StringWeight::new(vec![7, 8, 9]), 1));
        fst.set_final(1, StringWeight::one());

        let mut ofst = StringFst::new();
        factor_weight(
            &fst,
            &mut ofst,
            StringFactor::new,
            &FactorWeightOptions::default(),
        );

        // Each arc carries one label, and the last piece stays where a piece
        // can stay: as the final weight. So three labels become two arcs and a
        // final weight, not three arcs.
        let weights: Vec<Vec<i32>> = (0..ofst.num_states() as i32)
            .flat_map(|s| {
                ofst.arcs(s)
                    .map(|a| match &a.weight().value {
                        StringWeightValue::Labels(v) => v.clone(),
                        _ => Vec::new(),
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(weights, vec![vec![7], vec![8]]);

        // What matters is that the pieces multiply back to what they came from.
        let mut along = StringWeight::<i32, StringLeft>::one();
        let mut state = ofst.start().unwrap();
        while let Some(arc) = ofst.arcs(state).next() {
            along = along.times(arc.weight());
            state = arc.nextstate();
        }
        along = along.times(&ofst.final_weight(state));
        assert_eq!(along, StringWeight::new(vec![7, 8, 9]));
    }

    /// A string weight of one label has nothing to split.
    #[test]
    fn a_single_label_weight_is_left_where_it_is() {
        let mut fst = StringFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StringArc::new(1, 1, StringWeight::new(vec![7]), 1));
        fst.set_final(1, StringWeight::one());

        let mut ofst = StringFst::new();
        factor_weight(
            &fst,
            &mut ofst,
            StringFactor::new,
            &FactorWeightOptions::default(),
        );
        assert_eq!(ofst.num_states(), 2);
        let arc = ofst.arcs(0).next().unwrap();
        assert_eq!(arc.weight(), &StringWeight::new(vec![7]));
    }

    /// A final weight that can be split becomes arcs to states that pay it out
    /// a piece at a time.
    #[test]
    fn a_splittable_final_weight_becomes_arcs() {
        let mut fst = StringFst::new();
        fst.add_state();
        fst.set_start(0);
        fst.set_final(0, StringWeight::new(vec![4, 5]));

        let mut ofst = StringFst::new();
        factor_weight(
            &fst,
            &mut ofst,
            StringFactor::new,
            &FactorWeightOptions::default(),
        );

        // The start state is no longer final: what it owed became an arc.
        assert_eq!(ofst.final_weight(0), StringWeight::zero());
        let arc = ofst.arcs(0).next().unwrap();
        assert_eq!(arc.weight(), &StringWeight::new(vec![4]));
        assert_eq!(
            ofst.final_weight(arc.nextstate()),
            StringWeight::new(vec![5])
        );
    }

    /// Splitting only the arc weights leaves the final weights whole.
    #[test]
    fn the_mode_selects_which_weights_are_split() {
        let mut fst = StringFst::new();
        fst.add_state();
        fst.set_start(0);
        fst.set_final(0, StringWeight::new(vec![4, 5]));

        let mut ofst = StringFst::new();
        factor_weight(
            &fst,
            &mut ofst,
            StringFactor::new,
            &FactorWeightOptions {
                mode: FactorMode::ARC_WEIGHTS,
                ..Default::default()
            },
        );
        assert_eq!(ofst.num_states(), 1);
        assert_eq!(ofst.final_weight(0), StringWeight::new(vec![4, 5]));
    }

    /// A gallic weight splits by its string part, and the weight rides the
    /// first piece so the pieces multiply back to what they came from.
    #[test]
    fn a_gallic_weight_keeps_its_weight_on_the_first_piece() {
        type GW = GallicWeight<i32, TropicalWeight, GallicLeft>;
        let original = GW::from_parts(StringWeight::new(vec![1, 2, 3]), TropicalWeight(5.0));

        let mut factors = GallicFactor::<i32, TropicalWeight, GallicLeft>::new(original.clone());
        let (head, rest) = factors.next().expect("three labels can be split");
        assert!(factors.next().is_none(), "one split, then done");

        assert_eq!(head.labels(), &StringWeight::new(vec![1]));
        assert_eq!(head.weight(), &TropicalWeight(5.0));
        assert_eq!(rest.labels(), &StringWeight::new(vec![2, 3]));
        assert_eq!(rest.weight(), &TropicalWeight::one());
        assert_eq!(head.times(&rest), original, "the pieces multiply back");
    }

    #[test]
    fn an_fst_with_no_start_state_factors_to_nothing() {
        let ifst = StdVectorFst::new();
        let mut ofst = StdVectorFst::new();
        ofst.add_state();
        factor_weight(
            &ifst,
            &mut ofst,
            IdentityFactor::new,
            &FactorWeightOptions::default(),
        );
        assert_eq!(ofst.num_states(), 0);
    }
}
