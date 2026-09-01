//! Making an FST read each input string one way only.
//!
//! Port of OpenFst's `determinize.h`. A state of the result is a *weighted
//! subset* of the input's states: where the input could be in any of several
//! places after reading a prefix, the result is in the one state standing for
//! all of them, and each carries the weight by which it is still in the
//! running.
//!
//! > Mohri, M. 1997. Finite-state transducers in language and speech
//! > processing. *Computational Linguistics* 23(2): 269-311.
//!
//! Epsilon is treated as an ordinary label here; removing it first is
//! [`rm_epsilon`](super::rmepsilon)'s job.
//!
//! Not every weighted FST can be determinized. The ones that cannot need
//! unboundedly many subsets, and the algorithm simply keeps producing them.
//! Every unweighted or acyclic FST can be. See
//! [`DeterminizeOptions::max_states`].

use crate::algorithms::arc_map::{FromGallicMapper, ToGallicMapper, arc_map_to};
use crate::algorithms::factor_weight::{
    FactorIterator, FactorMode, FactorWeightOptions, GallicFactor, factor_weight,
};
use crate::arc::{Arc, ArcLabel, ArcStateId, GallicArc};
use crate::data_structures::bi_table::CompactHashBiTable;
use crate::error::OpenFstError;
use crate::fst::{Fst, MutableFst};
use crate::fsts::vector_fst::VectorFst;
use crate::properties::{K_ACCEPTOR, K_FST_PROPERTIES, determinize_properties};
use crate::weight::{Divide, DivideType, Weight};
use crate::weights::string_weight::{
    GallicRestrict, GallicTypeMarker, GallicWeight, StringTypeMarker, StringWeight,
    StringWeightValue,
};

/// The quantization delta upstream uses by default.
pub const DELTA: f32 = 1.0 / 1024.0;

/// What weight to put on the arc leaving a subset, given the weights the subset
/// members contribute.
///
/// Whatever is put there is divided back out of the members, so any choice
/// gives an equivalent FST; the choice decides how much weight moves forward
/// and so how many distinct subsets appear. Taking ⊕ is the usual one.
pub trait CommonDivisor<W: Weight> {
    /// Combines a running divisor with one more member's weight.
    fn divisor(&self, w1: &W, w2: &W) -> W;
}

/// The semiring's own ⊕.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultCommonDivisor;

impl<W: Weight> CommonDivisor<W> for DefaultCommonDivisor {
    #[inline]
    fn divisor(&self, w1: &W, w2: &W) -> W {
        w1.plus(w2)
    }
}

/// The first letter the two label sequences agree on, or none.
///
/// Used on the string half of a gallic weight, so that at most one output label
/// is committed per arc, which keeps the result a transducer with one label per
/// arc rather than one with strings on its arcs.
#[derive(Debug, Clone, Copy, Default)]
pub struct LabelCommonDivisor;

impl<L: ArcLabel, S: StringTypeMarker> CommonDivisor<StringWeight<L, S>> for LabelCommonDivisor {
    fn divisor(&self, w1: &StringWeight<L, S>, w2: &StringWeight<L, S>) -> StringWeight<L, S> {
        let first = |w: &StringWeight<L, S>| match &w.value {
            StringWeightValue::Labels(labels) => labels.first().copied(),
            _ => None,
        };
        // SICADA-DIVERGE: upstream tests `Size() == 0` before it tests for
        // zero, which works there only because its zero is a sequence holding
        // one sentinel label and so has size 1. sicada's `size()` reports 0 for
        // zero, as it does for every non-sequence value, so the two tests have
        // to be the other way round; otherwise zero would be read as "no
        // letter to agree on" and every divisor over it would come out empty.
        let zero = StringWeight::<L, S>::zero();
        match (w1 == &zero, w2 == &zero) {
            (true, true) => return StringWeight::one(),
            // Zero contributes nothing, so the other one's letter stands.
            (true, false) => {
                return first(w2).map_or_else(StringWeight::one, |l| StringWeight::new(vec![l]));
            }
            (false, true) => {
                return first(w1).map_or_else(StringWeight::one, |l| StringWeight::new(vec![l]));
            }
            (false, false) => {}
        }
        // An empty sequence has no letter to agree on.
        if w1.size() == 0 || w2.size() == 0 {
            return StringWeight::one();
        }
        match (first(w1), first(w2)) {
            (Some(a), Some(b)) if a == b => StringWeight::new(vec![a]),
            _ => StringWeight::one(),
        }
    }
}

/// The label divisor on the string half and `D` on the weight half.
#[derive(Debug, Clone, Copy, Default)]
pub struct GallicCommonDivisor<D> {
    /// What to use on the weight half.
    pub weights: D,
}

impl<L, W, G, D> CommonDivisor<GallicWeight<L, W, G>> for GallicCommonDivisor<D>
where
    L: ArcLabel,
    W: Weight,
    G: GallicTypeMarker,
    D: CommonDivisor<W>,
    GallicWeight<L, W, G>: Weight,
{
    fn divisor(
        &self,
        w1: &GallicWeight<L, W, G>,
        w2: &GallicWeight<L, W, G>,
    ) -> GallicWeight<L, W, G> {
        GallicWeight::from_parts(
            LabelCommonDivisor.divisor(w1.labels(), w2.labels()),
            self.weights.divisor(w1.weight(), w2.weight()),
        )
    }
}

/// Which reading of a transducer to determinize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeterminizeType {
    /// The transducer maps each input string to at most one output string.
    #[default]
    Functional,
    /// It may map one input to several outputs, which are kept.
    ///
    /// SICADA-DIVERGE: not implemented. It needs the union gallic weight and
    /// upstream's relation filter.
    NonFunctional,
    /// Keep only one of several outputs.
    ///
    /// SICADA-DIVERGE: not implemented. It is `disambiguate.h`'s filter, which
    /// is a later tier.
    Disambiguate,
}

/// How to determinize.
#[derive(Debug, Clone)]
pub struct DeterminizeOptions<L> {
    /// How closely two subset weights must agree to count as the same subset.
    pub delta: f32,
    /// The label on the arc a leftover final output becomes.
    pub subsequential_label: L,
    /// Whether to make those labels distinct when a final weight becomes
    /// several arcs.
    pub increment_subsequential_label: bool,
    /// Which reading of the transducer to take.
    pub det_type: DeterminizeType,
    /// A cap on how many states to build.
    ///
    /// SICADA-DIVERGE: upstream's determinization is a delayed FST, so an input
    /// that cannot be determinized only diverges when the caller expands it
    /// all; the eager form here would run until it is killed. `None` is
    /// upstream's behaviour; a limit turns it into an error. Every unweighted
    /// or acyclic FST can be determinized, so the limit only matters for a
    /// weighted cyclic input.
    pub max_states: Option<usize>,
}

impl<L: ArcLabel> Default for DeterminizeOptions<L> {
    fn default() -> Self {
        Self {
            delta: DELTA,
            subsequential_label: L::epsilon(),
            increment_subsequential_label: false,
            det_type: DeterminizeType::Functional,
            max_states: None,
        }
    }
}

/// A weighted subset of the input's states, kept sorted by state so that two
/// subsets holding the same thing compare equal.
type Subset<S, W> = Vec<(S, W)>;

/// Where the distances go, when the caller asked for them.
struct Distances<'a, W> {
    /// How far each state of the *input* is from the final states.
    to_final: &'a [W],
    /// The same for each state built here, indexed by its state id.
    out: &'a mut Vec<W>,
}

/// What a subset is worth: how far it is from the final states, given how far
/// each of its members is.
///
/// Port of upstream's `DeterminizeFsaImpl::ComputeDistance`. A member past the
/// end of `to_final` counts as unable to reach one, as it does there.
fn subset_distance<S: ArcStateId, W: Weight>(subset: &Subset<S, W>, to_final: &[W]) -> W {
    let mut out = W::zero();
    for (state, weight) in subset {
        let ind = to_final
            .get(state.as_usize())
            .cloned()
            .unwrap_or_else(W::zero);
        out = out.plus(&weight.times(&ind));
    }
    out
}

/// The state standing for `subset`, added if it is new.
fn find_state<S, W>(
    subsets: &mut CompactHashBiTable<usize, Subset<S, W>>,
    pending: &mut Vec<usize>,
    distances: &mut Option<Distances<'_, W>>,
    subset: &Subset<S, W>,
) -> usize
where
    S: ArcStateId,
    W: Weight + std::hash::Hash + Eq,
{
    let before = subsets.size();
    let id = subsets
        .find_id(subset, true)
        .expect("find_id inserts when asked to");
    if id == before {
        pending.push(id);
        // Filled the moment a subset first becomes a state, as upstream does,
        // which is what leaves `out` indexed by state id.
        if let Some(distances) = distances.as_mut() {
            let distance = subset_distance(subset, distances.to_final);
            distances.out.push(distance);
        }
    }
    id
}

/// Determinizes an acceptor.
///
/// The weight has to be left divisible, since the arc weight is divided back out
/// of the subset, which holds for the tropical and log semirings.
pub fn determinize_fsa<A, D, F1, F2>(
    ifst: &F1,
    ofst: &mut F2,
    divisor: &D,
    delta: f32,
    max_states: Option<usize>,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: Divide + std::hash::Hash + Eq,
    D: CommonDivisor<A::Weight>,
    F1: Fst<A>,
    F2: MutableFst<A>,
{
    determinize_fsa_impl(ifst, ofst, divisor, delta, max_states, None)
}

/// Determinizes an acceptor, reporting how far each state of the result is from
/// the final states.
///
/// `in_dist[q]` is that distance for state `q` of the input; `out_dist` is
/// filled with it for each state of the result, which is `⊕ w ⊗ in_dist[q]`
/// over the subset `(q, w)` that state stands for.
///
/// It is done here because the subsets are gone afterwards: this costs one ⊗
/// and one ⊕ per subset member, where asking later costs a shortest-distance
/// pass over the larger result.
/// [`shortest_path_unique`](super::shortest_path::shortest_path_unique) is the
/// caller.
pub fn determinize_fsa_with_distance<A, D, F1, F2>(
    ifst: &F1,
    ofst: &mut F2,
    divisor: &D,
    delta: f32,
    max_states: Option<usize>,
    in_dist: &[A::Weight],
    out_dist: &mut Vec<A::Weight>,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: Divide + std::hash::Hash + Eq,
    D: CommonDivisor<A::Weight>,
    F1: Fst<A>,
    F2: MutableFst<A>,
{
    out_dist.clear();
    determinize_fsa_impl(
        ifst,
        ofst,
        divisor,
        delta,
        max_states,
        Some(Distances {
            to_final: in_dist,
            out: out_dist,
        }),
    )
}

fn determinize_fsa_impl<A, D, F1, F2>(
    ifst: &F1,
    ofst: &mut F2,
    divisor: &D,
    delta: f32,
    max_states: Option<usize>,
    mut distances: Option<Distances<'_, A::Weight>>,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: Divide + std::hash::Hash + Eq,
    D: CommonDivisor<A::Weight>,
    F1: Fst<A>,
    F2: MutableFst<A>,
{
    ofst.delete_all_states();
    ofst.set_input_symbols(ifst.input_symbols());
    ofst.set_output_symbols(ifst.output_symbols());
    let iprops = ifst.properties(K_FST_PROPERTIES, false);

    let Some(istart) = ifst.start() else {
        ofst.set_properties(
            determinize_properties(iprops, true, false),
            K_FST_PROPERTIES,
        );
        return Ok(());
    };

    let mut subsets: CompactHashBiTable<usize, Subset<A::StateId, A::Weight>> =
        CompactHashBiTable::new(1024);
    let mut pending: Vec<usize> = Vec::new();
    // Reused across states. Expanding one subset used to build a fresh
    // `HashMap<Label, Vec<_>>` and a fresh `Vec` of its keys, so a
    // determinization of n states paid n hash tables and one allocation per
    // label per state. One sort by (label, destination) groups the labels and
    // orders each group at once, and the labels come out in order for free.
    let mut current: Subset<A::StateId, A::Weight> = Vec::new();
    let mut transitions: Vec<(A::Label, A::StateId, A::Weight)> = Vec::new();

    let initial: Subset<A::StateId, A::Weight> = vec![(istart, A::Weight::one())];
    let start = find_state(&mut subsets, &mut pending, &mut distances, &initial);
    ofst.add_state();
    ofst.set_start(A::StateId::from_usize(start));

    let zero = A::Weight::zero();
    while let Some(id) = pending.pop() {
        current.clear();
        current.extend_from_slice(subsets.find_entry(id).expect("just added"));
        let subset = &current;
        let state = A::StateId::from_usize(id);

        // A subset is final when any state in it is, weighted by how it got
        // there.
        let mut final_weight = zero.clone();
        for (member, weight) in subset {
            final_weight = final_weight.plus(&weight.times(&ifst.final_weight(*member)));
        }
        if final_weight != zero {
            if !final_weight.is_member() {
                return Err(OpenFstError::InvalidOperation(
                    "Determinize: a subset's final weight left the semiring".into(),
                ));
            }
            ofst.set_final(state, final_weight);
        }

        // Every arc out of every member. Sorting by (label, destination) both
        // groups the labels, in an order the result must not depend on, and
        // orders each group by destination, which lets duplicates be merged by
        // looking only at the entry before.
        transitions.clear();
        for (member, weight) in subset {
            for arc in ifst.arcs(*member) {
                transitions.push((arc.ilabel(), arc.nextstate(), weight.times(arc.weight())));
            }
        }
        // Stable, so that two contributions alike in label and destination keep
        // the order they were read in: the divisor below is folded over them in
        // that order and floating-point ⊕ is not associative.
        transitions.sort_by_key(|(label, member, _)| (*label, *member));

        let mut begin = 0;
        while begin < transitions.len() {
            let label = transitions[begin].0;
            let mut end = begin + 1;
            while end < transitions.len() && transitions[end].0 == label {
                end += 1;
            }
            let destinations = &transitions[begin..end];
            begin = end;

            // What goes on the arc: the divisor over every contribution, taken
            // before duplicates are merged, as upstream does.
            let mut arc_weight = zero.clone();
            for (_, _, weight) in destinations {
                arc_weight = divisor.divisor(&arc_weight, weight);
            }

            // A state reached more than once is in the subset once, at the ⊕ of
            // the ways of reaching it.
            let mut merged: Subset<A::StateId, A::Weight> = Vec::with_capacity(destinations.len());
            for (_, member, weight) in destinations.iter().cloned() {
                match merged.last_mut() {
                    Some((last, at)) if *last == member => {
                        *at = at.plus(&weight);
                        if !at.is_member() {
                            return Err(OpenFstError::InvalidOperation(
                                "Determinize: a subset weight left the semiring".into(),
                            ));
                        }
                    }
                    _ => merged.push((member, weight)),
                }
            }
            // What the arc took is divided back out, so the subset holds only
            // what is still to be decided. Quantizing makes two subsets that
            // agree to within `delta` the same state.
            for (_, weight) in &mut merged {
                *weight = weight.divide(&arc_weight, DivideType::Left).quantize(delta);
            }

            let next = find_state(&mut subsets, &mut pending, &mut distances, &merged);
            if max_states.is_some_and(|limit| subsets.size() > limit) {
                return Err(OpenFstError::InvalidOperation(format!(
                    "Determinize: more than {} states; the FST may not be determinizable",
                    max_states.expect("just checked")
                )));
            }
            while ofst.num_states() < subsets.size() {
                ofst.add_state();
            }
            ofst.add_arc(
                state,
                A::new(label, label, arc_weight, A::StateId::from_usize(next)),
            );
        }
    }

    ofst.set_properties(
        determinize_properties(iprops, true, false),
        K_FST_PROPERTIES,
    );
    Ok(())
}

/// Determinizes a weighted transducer.
///
/// An acceptor is determinized directly. A transducer is turned into an
/// acceptor over the gallic semiring, where the output labels ride along in the
/// weight, determinized there, and turned back; the leftover output that could
/// not be committed becomes an arc labelled
/// [`subsequential_label`](DeterminizeOptions::subsequential_label).
///
/// The transducer has to be functional: one input string, at most one output.
pub fn determinize<A, F1, F2>(
    ifst: &F1,
    ofst: &mut F2,
    opts: &DeterminizeOptions<A::Label>,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: Divide + std::hash::Hash + Eq,
    F1: Fst<A>,
    F2: MutableFst<A>,
    GallicWeight<A::Label, A::Weight, GallicRestrict>: Weight + Divide,
{
    match opts.det_type {
        DeterminizeType::Functional => {}
        other => {
            return Err(OpenFstError::InvalidOperation(format!(
                "Determinize: the {other:?} reading is not implemented"
            )));
        }
    }

    if ifst.properties(K_ACCEPTOR, true) & K_ACCEPTOR != 0 {
        return determinize_fsa(
            ifst,
            ofst,
            &DefaultCommonDivisor,
            opts.delta,
            opts.max_states,
        );
    }

    // The output labels travel in the weight, so that determinizing the input
    // side carries them along.
    type G = GallicRestrict;
    let mut gfst: VectorFst<GallicArc<A, G>> = VectorFst::new();
    arc_map_to(ifst, &mut gfst, &mut ToGallicMapper::<G>::new())?;

    let mut determinized: VectorFst<GallicArc<A, G>> = VectorFst::new();
    determinize_fsa(
        &gfst,
        &mut determinized,
        &GallicCommonDivisor {
            weights: DefaultCommonDivisor,
        },
        opts.delta,
        opts.max_states,
    )?;

    // A weight may now carry several output labels, which one arc cannot hold.
    let mut factored: VectorFst<GallicArc<A, G>> = VectorFst::new();
    factor_weight(
        &determinized,
        &mut factored,
        GallicFactor::new,
        &FactorWeightOptions {
            delta: opts.delta,
            mode: FactorMode::FINAL_WEIGHTS,
            final_ilabel: opts.subsequential_label,
            final_olabel: opts.subsequential_label,
            increment_final_ilabel: opts.increment_subsequential_label,
            increment_final_olabel: opts.increment_subsequential_label,
        },
    );

    let mut mapper =
        FromGallicMapper::<A::Label, G>::with_superfinal_label(opts.subsequential_label);
    arc_map_to(&factored, ofst, &mut mapper)?;
    if mapper.error() {
        return Err(OpenFstError::InvalidOperation(
            "Determinize: a weight came out that no single arc can carry".into(),
        ));
    }
    ofst.set_input_symbols(ifst.input_symbols());
    ofst.set_output_symbols(ifst.output_symbols());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{
        Rng, paths, random_acyclic_fst, sorted, string_weights, visible_paths,
    };
    use crate::arc::StdArc;
    use crate::fst::ExpandedFst as _;
    use crate::fst::MutableFst;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::weights::float_weight::TropicalWeight;

    /// Whether no state has two arcs with the same input label.
    fn is_deterministic(fst: &StdVectorFst) -> bool {
        fst.states().all(|state| {
            let mut seen: Vec<i32> = fst.arcs(state).map(|arc| arc.ilabel()).collect();
            seen.sort_unstable();
            let before = seen.len();
            seen.dedup();
            seen.len() == before
        })
    }

    fn determinized(fst: &StdVectorFst) -> StdVectorFst {
        let mut out = StdVectorFst::new();
        determinize(
            fst,
            &mut out,
            &DeterminizeOptions {
                max_states: Some(4096),
                ..Default::default()
            },
        )
        .unwrap();
        out
    }

    /// Two ways of reading `a` at different weights become one, keeping the
    /// lighter.
    #[test]
    fn two_arcs_with_the_same_label_become_one() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(2.0), 1));
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(5.0), 2));
        fst.set_final(1, TropicalWeight(1.0));
        fst.set_final(2, TropicalWeight(1.0));
        fst.properties(K_FST_PROPERTIES, true);

        let out = determinized(&fst);
        assert!(is_deterministic(&out));
        assert_eq!(out.num_arcs(out.start().unwrap()), 1);
        assert_eq!(
            sorted(paths(&out, 8)),
            vec![(vec![1], vec![1], "3.0000".to_string())],
            "the lighter of 2 + 1 and 5 + 1"
        );
    }

    /// The residual weight travels: two branches that agree on a prefix commit
    /// only what they agree on.
    #[test]
    fn the_weight_a_subset_cannot_commit_travels_with_it() {
        let mut fst = StdVectorFst::new();
        for _ in 0..5 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(1.0), 1));
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(3.0), 2));
        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight::one(), 3));
        fst.add_arc(2, StdArc::new(3, 3, TropicalWeight::one(), 4));
        fst.set_final(3, TropicalWeight::one());
        fst.set_final(4, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let out = determinized(&fst);
        assert!(is_deterministic(&out));
        // The first arc commits 1, the shared prefix's cost; the branch to 3
        // then carries the extra 2.
        let first: Vec<f32> = out
            .arcs(out.start().unwrap())
            .map(|arc| arc.weight().value())
            .collect();
        assert_eq!(first, vec![1.0]);
        assert_eq!(
            sorted(paths(&out, 8)),
            vec![
                (vec![1, 2], vec![1, 2], "1.0000".to_string()),
                (vec![1, 3], vec![1, 3], "3.0000".to_string()),
            ]
        );
    }

    /// An FST that is already deterministic comes back saying the same thing.
    #[test]
    fn an_already_deterministic_fst_is_unchanged_in_what_it_accepts() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(1.0), 1));
        fst.add_arc(0, StdArc::new(2, 2, TropicalWeight(2.0), 2));
        fst.set_final(1, TropicalWeight::one());
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        assert_eq!(
            sorted(paths(&determinized(&fst), 8)),
            sorted(paths(&fst, 8))
        );
    }

    /// An FST with no start state determinizes to one with no states.
    #[test]
    fn an_empty_fst_determinizes_to_an_empty_one() {
        let out = determinized(&StdVectorFst::new());
        assert_eq!(out.num_states(), 0);
    }

    /// The result is deterministic and says exactly what the input said, over
    /// random acyclic FSTs, which are all determinizable.
    #[test]
    fn determinizing_keeps_the_language_and_makes_it_deterministic() {
        let mut rng = Rng::new(0x0DE7_E401_u64);
        let mut checked = 0;
        for round in 0..200 {
            let fst = random_acyclic_fst(&mut rng, 6);
            // Determinization merges two paths spelling the same string into
            // one at their sum, which is the weight the FST gave that string
            // all along, so the weight per string is what has to agree, not the
            // list of paths.
            let before = string_weights(paths(&fst, 12));
            if before.is_empty() {
                continue;
            }
            checked += 1;
            let out = determinized(&fst);
            assert!(is_deterministic(&out), "round {round}");
            assert_eq!(string_weights(paths(&out, 12)), before, "round {round}");
        }
        assert!(checked > 50, "only {checked} FSTs accepted anything");
    }

    /// A transducer goes through the gallic semiring: the input side becomes
    /// deterministic and the output side is preserved.
    #[test]
    fn a_transducer_is_determinized_on_its_input_side() {
        // Reading 1 can produce 7 or 8; the two are told apart by what follows.
        let mut fst = StdVectorFst::new();
        for _ in 0..5 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 7, TropicalWeight::one(), 1));
        fst.add_arc(0, StdArc::new(1, 8, TropicalWeight::one(), 2));
        fst.add_arc(1, StdArc::new(2, 9, TropicalWeight::one(), 3));
        fst.add_arc(2, StdArc::new(3, 9, TropicalWeight::one(), 4));
        fst.set_final(3, TropicalWeight::one());
        fst.set_final(4, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let out = determinized(&fst);
        assert!(is_deterministic(&out), "the input side is deterministic");

        // The transduction is unchanged, once the epsilons the gallic round
        // trip introduces are set aside.
        let visible = |fst: &StdVectorFst| -> Vec<(Vec<i32>, Vec<i32>, String)> {
            sorted(paths(fst, 12))
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
        assert_eq!(visible(&out), visible(&fst));
    }

    /// A functional transducer keeps what it transduces, over random inputs.
    #[test]
    fn determinizing_a_transducer_keeps_the_transduction() {
        let mut rng = Rng::new(0x0000_7A11_u64);
        let visible = |fst: &StdVectorFst| string_weights(visible_paths(fst, 14));

        let mut checked = 0;
        for round in 0..100 {
            let mut fst = random_acyclic_fst(&mut rng, 5);
            // Give the arcs distinct output labels, keeping it functional by
            // deriving the output from the input.
            let states: Vec<i32> = fst.states().collect();
            for state in states {
                fst.mutate_arcs(state, |arc| {
                    *arc = StdArc::new(
                        arc.ilabel(),
                        arc.ilabel() + 10,
                        *arc.weight(),
                        arc.nextstate(),
                    );
                });
            }
            fst.properties(K_FST_PROPERTIES, true);

            let before = visible(&fst);
            if before.is_empty() {
                continue;
            }
            // Functional means one output per input; a random FST need not be,
            // so only the ones that are get checked.
            let mut inputs: Vec<Vec<i32>> = before.iter().map(|(i, _, _)| i.clone()).collect();
            inputs.sort();
            let unique = inputs.len();
            inputs.dedup();
            if inputs.len() != unique {
                continue;
            }
            checked += 1;
            assert_eq!(visible(&determinized(&fst)), before, "round {round}");
        }
        assert!(checked > 20, "only {checked} FSTs were functional");
    }

    /// An FST that cannot be determinized is refused rather than run forever.
    #[test]
    fn an_undeterminizable_fst_is_refused_when_a_limit_is_given() {
        // The classic non-determinizable example: two cycles on the same label
        // whose weights never let a common prefix be committed.
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(1.0), 1));
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(2.0), 2));
        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight(1.0), 1));
        fst.add_arc(2, StdArc::new(2, 2, TropicalWeight(2.0), 2));
        fst.set_final(1, TropicalWeight::one());
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let mut out = StdVectorFst::new();
        let err = determinize(
            &fst,
            &mut out,
            &DeterminizeOptions {
                max_states: Some(64),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(format!("{err}").contains("determinizable"), "{err}");
    }

    /// The readings that are not implemented say so rather than doing the
    /// functional thing quietly.
    #[test]
    fn an_unimplemented_reading_is_refused() {
        let fst = StdVectorFst::new();
        for det_type in [
            DeterminizeType::NonFunctional,
            DeterminizeType::Disambiguate,
        ] {
            let mut out = StdVectorFst::new();
            let err = determinize(
                &fst,
                &mut out,
                &DeterminizeOptions {
                    det_type,
                    ..Default::default()
                },
            )
            .unwrap_err();
            assert!(format!("{err}").contains("not implemented"), "{err}");
        }
    }

    /// The distances it reports are the ones the result itself has.
    ///
    /// Recomputing them on the result is an oracle that shares no code with the
    /// line inside the determinization that produces them.
    #[test]
    fn the_distances_it_reports_are_the_result_s_own() {
        use crate::algorithms::shortest_distance::shortest_distance_reverse;

        let mut rng = Rng::new(0x0000_D157_u64);
        let mut checked = 0;
        for round in 0..200 {
            // `random_acyclic_fst` builds an acceptor, which is what
            // `determinize_fsa` takes.
            let fst = random_acyclic_fst(&mut rng, 6);
            let in_dist = shortest_distance_reverse::<StdArc, _>(&fst, DELTA).unwrap();

            let mut out = StdVectorFst::new();
            let mut out_dist = Vec::new();
            determinize_fsa_with_distance(
                &fst,
                &mut out,
                &DefaultCommonDivisor,
                DELTA,
                Some(4096),
                &in_dist,
                &mut out_dist,
            )
            .unwrap();

            assert_eq!(
                out_dist.len(),
                out.num_states(),
                "round {round}: one distance per state"
            );
            if out.num_states() == 0 {
                continue;
            }
            checked += 1;
            // A state the reverse pass never reached cannot reach a final
            // state, which is `zero`; comparing only the entries it returned
            // would leave those states unchecked.
            let want = shortest_distance_reverse::<StdArc, _>(&out, DELTA).unwrap();
            for (state, got) in out_dist.iter().enumerate() {
                let want = want
                    .get(state)
                    .cloned()
                    .unwrap_or_else(TropicalWeight::zero);
                assert!(
                    got.approx_equal(&want, 1e-3),
                    "round {round}, state {state}: {got} against {want}"
                );
            }
        }
        assert!(checked > 50, "only {checked} FSTs had any states");
    }

    /// Asking for distances over an input with no start state asks for none.
    #[test]
    fn an_empty_input_reports_no_distances() {
        let mut out = StdVectorFst::new();
        let mut out_dist = vec![TropicalWeight(7.0)];
        determinize_fsa_with_distance(
            &StdVectorFst::new(),
            &mut out,
            &DefaultCommonDivisor,
            DELTA,
            None,
            &[],
            &mut out_dist,
        )
        .unwrap();
        assert!(
            out_dist.is_empty(),
            "the caller's vector is not appended to"
        );
    }

    /// The single-letter divisor commits a letter only where the two agree.
    #[test]
    fn the_label_divisor_commits_only_what_both_agree_on() {
        type S = StringWeight<i32, crate::weights::string_weight::StringLeft>;
        let divisor = LabelCommonDivisor;

        assert_eq!(
            divisor.divisor(&S::new(vec![1, 2]), &S::new(vec![1, 3])),
            S::new(vec![1]),
            "they agree on the first letter"
        );
        assert_eq!(
            divisor.divisor(&S::new(vec![1, 2]), &S::new(vec![4, 5])),
            S::one(),
            "they agree on nothing"
        );
        assert_eq!(
            divisor.divisor(&S::zero(), &S::new(vec![7])),
            S::new(vec![7]),
            "zero contributes nothing, so the other stands"
        );
        assert_eq!(
            divisor.divisor(&S::one(), &S::new(vec![7])),
            S::one(),
            "an empty sequence has no letter to agree on"
        );
    }
}
