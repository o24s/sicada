//! Merging the states that cannot be told apart.
//!
//! Port of OpenFst's `minimize.h`. Two states are equivalent when no string
//! read from one is read differently from the other; minimization finds the
//! coarsest such partition and keeps one state per class.
//!
//! Two algorithms, as upstream has:
//!
//! - Hopcroft's, `O(E log V)`, which works on any input the semiring allows;
//! - Revuz's, `O(E)`, for an acyclic deterministic input.
//!
//! > Hopcroft, J. 1971. An n log n algorithm for minimizing states in a finite
//! > automaton. Ms, Stanford University.
//! >
//! > Revuz, D. 1992. Minimization of acyclic deterministic automata in linear
//! > time. *Theoretical Computer Science* 92(1): 181-189.
//!
//! Both are stated for an unweighted acceptor. A weighted one is made
//! unweighted by pushing its weights to the front and then encoding them into
//! the labels; a transducer, by moving its output side into a gallic weight and
//! doing the same.

use std::hash::{Hash, Hasher};

use hashbrown::HashMap;

use crate::algorithms::arc_map::{
    FromGallicMapper, QuantizeMapper, ToGallicMapper, arc_map, arc_map_to,
};
use crate::algorithms::arcsort::{ILabelCompare, arc_sort};
use crate::algorithms::connect::connect;
use crate::algorithms::dfs_visit::DfsVisitor;
use crate::algorithms::dfs_visit::dfs_visit_any;
use crate::algorithms::encode::{ENCODE_FLAGS, EncodeMapper, decode, encode};
use crate::algorithms::factor_weight::{
    FactorIterator, FactorMode, FactorWeightOptions, GallicFactor, factor_weight,
};
use crate::algorithms::push::push_weights_to_initial;
use crate::algorithms::reverse::reverse;
use crate::algorithms::state_map::{ArcUniqueMapper, state_map_to};
use crate::arc::{Arc, ArcStateId, GallicArc};
use crate::data_structures::partition::Partition;
use crate::error::OpenFstError;
use crate::fst::{ExpandedFst, Fst, MutableFst};
use crate::fsts::vector_fst::VectorFst;
use crate::properties::{K_ACCEPTOR, K_ACYCLIC, K_I_DETERMINISTIC, K_WEIGHTED};
use crate::queue::{LifoQueue, Queue};
use crate::weight::{Divide, LeftSemiring, Weight};
use crate::weights::string_weight::{GallicLeft, GallicWeight};

/// The quantization delta upstream minimizes at.
pub const DELTA: f32 = 1e-6;

/// The hash of a weight, for grouping states by their final weight.
fn weight_hash<W: Weight + Hash>(weight: &W) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    weight.hash(&mut hasher);
    hasher.finish()
}

/// Hopcroft's minimization, for an FST that may have cycles.
///
/// Works by splitting: a class is split whenever two of its states are told
/// apart by where a label leads. The reverse FST makes "the states that reach
/// this class on this label" cheap to walk.
fn hopcroft_partition<A, F>(fst: &F) -> Result<Partition, OpenFstError>
where
    A: Arc,
    A::Weight: Weight + Hash,
    F: Fst<A> + ExpandedFst<A>,
{
    let mut reversed: VectorFst<A::Reverse> = VectorFst::new();
    reverse(fst, &mut reversed, true);
    arc_sort(&mut reversed, &ILabelCompare);

    let nstates = fst.num_states();
    let mut partition = Partition::new(nstates);
    let mut active: LifoQueue<usize> = LifoQueue::new();

    // The starting partition. Final and non-final states can never be
    // equivalent, and states whose outgoing labels differ as a set cannot be
    // either, so both are separated up front. Hopcroft's bound does not
    // depend on this; it only saves splitting later.
    let zero = A::Weight::zero();
    let mut classes: HashMap<(bool, u64, Vec<A::Label>), usize> = HashMap::new();
    let mut initial = vec![0usize; nstates];
    let mut next_class = 0usize;
    for state in fst.states() {
        let mut labels: Vec<A::Label> = fst.arcs(state).map(|arc| arc.ilabel()).collect();
        labels.sort_unstable();
        labels.dedup();
        let final_weight = fst.final_weight(state);
        let key = (final_weight != zero, weight_hash(&final_weight), labels);
        let class = *classes.entry(key).or_insert_with(|| {
            let class = next_class;
            next_class += 1;
            class
        });
        initial[state.as_usize()] = class;
    }
    partition.allocate_classes(next_class);
    for (state, class) in initial.into_iter().enumerate() {
        partition.add(state, class);
    }
    for class in 0..next_class {
        active.enqueue(class);
    }

    // Splitting on a class: for every label, the states that reach the class on
    // that label are told apart from those that do not.
    while let Some(class) = active.dequeue() {
        // The arcs entering this class, by the label they carry. State `s` of
        // the input is state `s + 1` of the reverse, which added a superinitial
        // state.
        let mut entering: Vec<(A::Label, A::StateId)> = Vec::new();
        for member in partition.iter_class(class) {
            let in_reverse = A::StateId::from_usize(member + 1);
            for arc in reversed.arcs(in_reverse) {
                // The reverse's arcs point back at where they came from, again
                // offset by the superinitial state.
                entering.push((
                    arc.ilabel(),
                    A::StateId::from_usize(arc.nextstate().as_usize() - 1),
                ));
            }
        }
        // Splitting one label at a time makes the split well defined: all the
        // states reaching the class on a label move together.
        entering.sort_unstable_by_key(|(label, _)| *label);

        let mut previous: Option<A::Label> = None;
        for (label, from) in entering {
            if previous != Some(label) {
                partition.finalize_split(|class| active.enqueue(class));
                previous = Some(label);
            }
            let from_class = partition.class_id(from.as_usize());
            if partition.class_size(from_class) > 1 {
                partition.split_on(from.as_usize());
            }
        }
        partition.finalize_split(|class| active.enqueue(class));
    }
    Ok(partition)
}

/// How far each state is from a final state, which Revuz's algorithm partitions
/// by first.
struct HeightVisitor<A: Arc> {
    /// The height of each state, or `None` until it is finished.
    height: Vec<i64>,
    max_height: i64,
    num_states: usize,
    _marker: std::marker::PhantomData<A>,
}

impl<A: Arc> HeightVisitor<A> {
    fn new() -> Self {
        Self {
            height: Vec::new(),
            max_height: 0,
            num_states: 0,
            _marker: std::marker::PhantomData,
        }
    }

    fn ensure(&mut self, index: usize) {
        while self.height.len() <= index {
            self.height.push(-1);
        }
    }
}

impl<A: Arc> DfsVisitor<A> for HeightVisitor<A> {
    fn init_visit<F: Fst<A>>(&mut self, _fst: &F) {}

    fn init_state(&mut self, state: A::StateId, _root: A::StateId) -> bool {
        let index = state.as_usize();
        self.ensure(index);
        self.num_states = self.num_states.max(index + 1);
        true
    }

    fn tree_arc(&mut self, _state: A::StateId, _arc: &A) -> bool {
        true
    }

    fn back_arc(&mut self, _state: A::StateId, _arc: &A) -> bool {
        true
    }

    fn forward_or_cross_arc(&mut self, state: A::StateId, arc: &A) -> bool {
        let next = arc.nextstate().as_usize();
        self.ensure(next.max(state.as_usize()));
        if self.height[next] + 1 > self.height[state.as_usize()] {
            self.height[state.as_usize()] = self.height[next] + 1;
        }
        true
    }

    fn finish_state(&mut self, state: A::StateId, parent: Option<A::StateId>, _arc: Option<&A>) {
        let index = state.as_usize();
        self.ensure(index);
        if self.height[index] == -1 {
            self.height[index] = 0;
        }
        let h = self.height[index] + 1;
        if let Some(parent) = parent {
            self.ensure(parent.as_usize());
            if h > self.height[parent.as_usize()] {
                self.height[parent.as_usize()] = h;
            }
            if h > self.max_height {
                self.max_height = h;
            }
        }
    }

    fn finish_visit(&mut self) {}
}

/// Revuz's minimization, for an acyclic deterministic FST.
///
/// States are grouped by their distance to a final state, and each group is
/// then split by what its states' arcs look like, which is decidable in one
/// pass because the arcs of a state at height `h` all lead to states at height
/// below `h`, whose classes are already settled.
fn revuz_partition<A, F>(fst: &F) -> Partition
where
    A: Arc,
    A::Weight: Weight + Hash,
    F: Fst<A> + ExpandedFst<A>,
{
    let mut visitor = HeightVisitor::<A>::new();
    dfs_visit_any(fst, &mut visitor);

    let mut partition = Partition::new(visitor.num_states);
    partition.allocate_classes((visitor.max_height + 1) as usize);
    for (state, height) in visitor.height.iter().enumerate() {
        partition.add(state, (*height).max(0) as usize);
    }

    // Working up from the states nearest the end: by the time a height is
    // reached, everything its arcs lead to has been classified.
    let heights = partition.num_classes();
    for height in 0..heights {
        let members: Vec<usize> = partition.iter_class(height).collect();
        if members.len() < 2 {
            continue;
        }
        // Two states are the same when their final weights agree and their
        // arcs agree label for label on where they lead.
        let mut groups: HashMap<(u64, Vec<(A::Label, usize)>), usize> = HashMap::new();
        let mut moves: Vec<(usize, usize)> = Vec::new();
        for member in members {
            let state = A::StateId::from_usize(member);
            let signature: Vec<(A::Label, usize)> = fst
                .arcs(state)
                .map(|arc| (arc.ilabel(), partition.class_id(arc.nextstate().as_usize())))
                .collect();
            let key = (weight_hash(&fst.final_weight(state)), signature);
            match groups.get(&key) {
                Some(&class) => moves.push((member, class)),
                None => {
                    // The first state of a signature keeps the height class;
                    // the rest of the signatures get classes of their own.
                    let class = if groups.is_empty() {
                        height
                    } else {
                        partition.add_class()
                    };
                    groups.insert(key, class);
                    if class != height {
                        moves.push((member, class));
                    }
                }
            }
        }
        // SICADA-DIVERGE: upstream moves elements while iterating the class,
        // stepping the iterator forward first because a move invalidates it --
        // a hazard its own comment points out. The moves are collected first
        // here, so there is nothing to invalidate.
        for (member, class) in moves {
            if partition.class_id(member) != class {
                partition.move_element(member, class);
            }
        }
    }
    partition
}

/// Keeps one state per class and points everything at it.
fn merge_states<A, F>(partition: &Partition, fst: &mut F)
where
    A: Arc,
    F: MutableFst<A> + ExpandedFst<A>,
{
    let nclasses = partition.num_classes();
    // The first state of each class stands for it.
    let representative: Vec<usize> = (0..nclasses)
        .map(|class| partition.iter_class(class).next().unwrap_or(0))
        .collect();

    // Everything is read before anything is written, so that an arc is never
    // remapped through a state whose own arcs have already been replaced.
    let mut merged: Vec<Vec<A>> = Vec::with_capacity(nclasses);
    for class in 0..nclasses {
        let mut arcs = Vec::new();
        for member in partition.iter_class(class) {
            for arc in fst.arcs(A::StateId::from_usize(member)) {
                let to = representative[partition.class_id(arc.nextstate().as_usize())];
                arcs.push(A::new(
                    arc.ilabel(),
                    arc.olabel(),
                    arc.weight().clone(),
                    A::StateId::from_usize(to),
                ));
            }
        }
        merged.push(arcs);
    }

    for (class, arcs) in merged.into_iter().enumerate() {
        let state = A::StateId::from_usize(representative[class]);
        fst.delete_arcs(state);
        for arc in arcs {
            fst.add_arc(state, arc);
        }
    }
    if let Some(start) = fst.start() {
        let class = partition.class_id(start.as_usize());
        fst.set_start(A::StateId::from_usize(representative[class]));
    }
    // The states that were merged away are now unreachable.
    connect(fst);
}

/// A `VectorFst` holding what `fst` holds.
fn copy_of<A, F>(fst: &F) -> VectorFst<A>
where
    A: Arc,
    F: Fst<A> + ExpandedFst<A>,
{
    let mut out = VectorFst::new();
    out.add_states(fst.num_states());
    out.set_input_symbols(fst.input_symbols());
    out.set_output_symbols(fst.output_symbols());
    if let Some(start) = fst.start() {
        out.set_start(start);
    }
    for state in fst.states() {
        out.set_final(state, fst.final_weight(state));
        for arc in fst.arcs(state) {
            out.add_arc(state, arc);
        }
    }
    out.set_properties(
        fst.properties(crate::properties::K_FST_PROPERTIES, false),
        crate::properties::K_FST_PROPERTIES,
    );
    out
}

/// Minimizes an unweighted acceptor.
fn acceptor_minimize<A, F>(fst: &mut F) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: Weight + Hash,
    F: MutableFst<A> + ExpandedFst<A>,
{
    // States nothing reaches would otherwise be merged into classes of their
    // own and kept.
    connect(fst);
    if fst.start().is_none() {
        return Ok(());
    }
    // Revuz's algorithm reads a state's arcs as a sequence, which only says
    // what the state does when the input is deterministic and acyclic.
    let revuz = K_ACYCLIC | K_I_DETERMINISTIC;
    let partition = if fst.properties(revuz, true) & revuz == revuz {
        arc_sort(fst, &ILabelCompare);
        revuz_partition(&*fst)
    } else {
        hopcroft_partition::<A, F>(&*fst)?
    };
    merge_states(&partition, fst);
    // Merging can leave a state with two arcs alike but for their weights,
    // which the semiring says are one arc.
    //
    // SICADA-DIVERGE: upstream's `ArcUniqueMapper` holds a reference to the
    // very FST `StateMap` is writing through, reading each state's arcs just
    // before they are replaced, which is the same alias `ArcSort` relies on.
    // The source is copied here instead, in place of that alias.
    let source: VectorFst<A> = copy_of(&*fst);
    let mut mapper = ArcUniqueMapper::new(&source);
    state_map_to(&source, fst, &mut mapper);
    Ok(())
}

/// Merges the states of `fst` that cannot be told apart, in place.
///
/// A non-deterministic input needs an idempotent semiring and `allow_nondet`:
/// two of its arcs may lead to states that are about to be merged, and the
/// algorithm has no way to combine their weights.
pub fn minimize<A, F>(fst: &mut F, delta: f32, allow_nondet: bool) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: Weight + Hash + Eq + Divide + LeftSemiring,
    F: MutableFst<A> + ExpandedFst<A>,
    <A::Weight as Weight>::ReverseWeight: Weight<ReverseWeight = A::Weight>,
    GallicWeight<A::Label, A::Weight, GallicLeft>: Weight + Hash + Eq + Divide + LeftSemiring,
    <GallicWeight<A::Label, A::Weight, GallicLeft> as Weight>::ReverseWeight:
        Weight<ReverseWeight = GallicWeight<A::Label, A::Weight, GallicLeft>>,
{
    let mask = K_ACCEPTOR | K_I_DETERMINISTIC | K_WEIGHTED;
    let props = fst.properties(mask, true);

    if props & K_I_DETERMINISTIC == 0 {
        if A::Weight::properties() & crate::weight::IDEMPOTENT == 0 {
            return Err(OpenFstError::InvalidOperation(format!(
                "Minimize: a non-deterministic FST over the non-idempotent semiring {} cannot \
                 be minimized",
                A::Weight::type_name()
            )));
        }
        if !allow_nondet {
            return Err(OpenFstError::InvalidOperation(
                "Minimize: refusing a non-deterministic FST without allow_nondet".into(),
            ));
        }
    }

    if props & K_ACCEPTOR == 0 {
        // A transducer: the output side moves into a gallic weight, and what is
        // left is an acceptor that the weighted case below can handle.
        let mut gfst: VectorFst<GallicArc<A, GallicLeft>> = VectorFst::new();
        arc_map_to(&*fst, &mut gfst, &mut ToGallicMapper::<GallicLeft>::new())?;
        gfst.set_properties(K_ACCEPTOR, K_ACCEPTOR);

        push_weights_to_initial(&mut gfst, delta, false)?;
        arc_map(&mut gfst, &mut QuantizeMapper::new(delta))?;
        let mut encoder = EncodeMapper::<GallicArc<A, GallicLeft>>::new(ENCODE_FLAGS);
        encode(&mut gfst, &mut encoder)?;
        acceptor_minimize(&mut gfst)?;
        decode(&mut gfst, &encoder)?;

        // A gallic weight may hold several labels, which one arc cannot carry.
        //
        // SICADA-DIVERGE: upstream factors only the final weights here, which
        // assumes no arc weight is left holding more than one label. It can be.
        // Pushing over the gallic semiring commits every output label as early
        // as it goes, and `Reweight` finishes by multiplying the start state's
        // own arcs by whatever is left over, so on two paths spelling the
        // same output, the start's arcs come out carrying the whole output.
        // `FromGallicMapper` then has a weight it cannot represent; upstream
        // marks the result `kError` and returns it anyway.
        //
        // Arc weights are factored too here. That costs nothing when upstream's
        // assumption holds, since a weight of one label is already factored,
        // and turns the case where it does not into a correct FST rather than a
        // broken one.
        let mut factored: VectorFst<GallicArc<A, GallicLeft>> = VectorFst::new();
        factor_weight(
            &gfst,
            &mut factored,
            GallicFactor::new,
            &FactorWeightOptions {
                delta,
                mode: FactorMode::FINAL_WEIGHTS | FactorMode::ARC_WEIGHTS,
                ..Default::default()
            },
        );
        let osymbols = fst.output_symbols();
        let mut mapper = FromGallicMapper::<A::Label, GallicLeft>::new();
        arc_map_to(&factored, fst, &mut mapper)?;
        if mapper.error() {
            return Err(OpenFstError::InvalidOperation(
                "Minimize: a weight came out that no single arc can carry".into(),
            ));
        }
        fst.set_output_symbols(osymbols);
        return Ok(());
    }

    if props & K_WEIGHTED != 0 {
        // Pushing puts each path's weight as near the front as it goes, so that
        // two states that differ only in where their weight sits become the
        // same; encoding then makes the weights part of the labels, leaving an
        // unweighted acceptor.
        push_weights_to_initial::<A, F>(fst, delta, false)?;
        arc_map(fst, &mut QuantizeMapper::new(delta))?;
        let mut encoder = EncodeMapper::<A>::new(ENCODE_FLAGS);
        encode(fst, &mut encoder)?;
        acceptor_minimize::<A, F>(fst)?;
        decode(fst, &encoder)?;
        return Ok(());
    }

    acceptor_minimize::<A, F>(fst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{Rng, random_acyclic_fst, string_weights, visible_paths};
    use crate::arc::StdArc;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::K_FST_PROPERTIES;
    use crate::weights::float_weight::TropicalWeight;

    fn language(fst: &StdVectorFst, max_len: usize) -> Vec<(Vec<i32>, Vec<i32>, String)> {
        string_weights(visible_paths(fst, max_len))
    }

    fn minimized(fst: &StdVectorFst, allow_nondet: bool) -> StdVectorFst {
        let mut copy = fst.clone();
        minimize(&mut copy, DELTA, allow_nondet).unwrap();
        copy
    }

    /// Two states that lead to the same place on the same labels are one state.
    #[test]
    fn states_that_cannot_be_told_apart_become_one() {
        // 0 --1--> 1 --3--> 3, and 0 --2--> 2 --3--> 3: states 1 and 2 do the
        // same thing.
        let mut fst = StdVectorFst::new();
        for _ in 0..4 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(0, StdArc::new(2, 2, TropicalWeight::one(), 2));
        fst.add_arc(1, StdArc::new(3, 3, TropicalWeight::one(), 3));
        fst.add_arc(2, StdArc::new(3, 3, TropicalWeight::one(), 3));
        fst.set_final(3, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let before = language(&fst, 8);
        let out = minimized(&fst, false);
        assert_eq!(out.num_states(), 3, "states 1 and 2 merged");
        assert_eq!(language(&out, 8), before);
    }

    /// Two states that differ in whether they are final stay apart.
    #[test]
    fn a_final_state_is_never_merged_with_a_non_final_one() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(0, StdArc::new(2, 2, TropicalWeight::one(), 2));
        fst.set_final(1, TropicalWeight::one());
        // State 2 is not final and leads nowhere, so connect removes it.
        fst.properties(K_FST_PROPERTIES, true);

        let out = minimized(&fst, false);
        assert_eq!(language(&out, 8), language(&fst, 8));
        assert!(out.num_states() >= 2);
    }

    /// An FST that is already minimal comes back the same size.
    #[test]
    fn an_already_minimal_fst_is_not_made_smaller() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight::one(), 2));
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let out = minimized(&fst, false);
        assert_eq!(out.num_states(), 3);
        assert_eq!(language(&out, 8), language(&fst, 8));
    }

    /// A cycle is minimized by Hopcroft's algorithm rather than Revuz's.
    #[test]
    fn a_cyclic_fst_is_minimized() {
        // Two copies of the same one-state loop, which are the same state.
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight::one(), 2));
        fst.add_arc(2, StdArc::new(2, 2, TropicalWeight::one(), 2));
        fst.set_final(1, TropicalWeight::one());
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let before = language(&fst, 8);
        let out = minimized(&fst, false);
        assert_eq!(language(&out, 8), before);
        assert!(out.num_states() <= 3);
    }

    /// A weighted acceptor is pushed and encoded first, so states that differ
    /// only in where their weight sits become one.
    #[test]
    fn a_weighted_acceptor_is_minimized_after_pushing() {
        let mut fst = StdVectorFst::new();
        for _ in 0..4 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(1.0), 1));
        fst.add_arc(0, StdArc::new(2, 2, TropicalWeight(1.0), 2));
        fst.add_arc(1, StdArc::new(3, 3, TropicalWeight(2.0), 3));
        fst.add_arc(2, StdArc::new(3, 3, TropicalWeight(2.0), 3));
        fst.set_final(3, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let before = language(&fst, 8);
        let out = minimized(&fst, false);
        assert_eq!(language(&out, 8), before);
        assert!(out.num_states() < 4, "{} states", out.num_states());
    }

    /// A non-deterministic FST needs saying so.
    #[test]
    fn a_non_deterministic_fst_needs_permission() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(2.0), 2));
        fst.set_final(1, TropicalWeight::one());
        fst.set_final(2, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let mut refused = fst.clone();
        let err = minimize(&mut refused, DELTA, false).unwrap_err();
        assert!(format!("{err}").contains("allow_nondet"), "{err}");

        let before = language(&fst, 8);
        let out = minimized(&fst, true);
        assert_eq!(language(&out, 8), before);
    }

    /// Minimizing never changes what the FST says, and never makes it bigger.
    #[test]
    fn minimizing_keeps_the_language_and_does_not_grow_the_fst() {
        let mut rng = Rng::new(0x0011_1111_u64);
        let mut checked = 0;
        for round in 0..200 {
            let fst = random_acyclic_fst(&mut rng, 6);
            let before = language(&fst, 12);
            if before.is_empty() {
                continue;
            }
            checked += 1;
            let out = minimized(&fst, true);
            assert_eq!(language(&out, 12), before, "round {round}");
            assert!(
                out.num_states() <= fst.num_states(),
                "round {round}: {} states from {}",
                out.num_states(),
                fst.num_states()
            );
        }
        assert!(checked > 50, "only {checked} FSTs accepted anything");
    }

    /// Minimizing twice changes nothing the second time.
    #[test]
    fn minimizing_is_idempotent() {
        let mut rng = Rng::new(0x0000_1DEE_u64);
        for round in 0..100 {
            let fst = random_acyclic_fst(&mut rng, 6);
            if language(&fst, 12).is_empty() {
                continue;
            }
            let once = minimized(&fst, true);
            let twice = minimized(&once, true);
            assert_eq!(twice.num_states(), once.num_states(), "round {round}");
            assert_eq!(language(&twice, 12), language(&once, 12), "round {round}");
        }
    }

    /// A transducer goes through the gallic semiring and comes back saying the
    /// same thing.
    #[test]
    fn a_transducer_is_minimized_through_the_gallic_semiring() {
        let mut fst = StdVectorFst::new();
        for _ in 0..4 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 7, TropicalWeight::one(), 1));
        fst.add_arc(0, StdArc::new(2, 7, TropicalWeight::one(), 2));
        fst.add_arc(1, StdArc::new(3, 8, TropicalWeight::one(), 3));
        fst.add_arc(2, StdArc::new(3, 8, TropicalWeight::one(), 3));
        fst.set_final(3, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let before = language(&fst, 12);
        let out = minimized(&fst, false);
        assert_eq!(language(&out, 12), before);
    }
}
