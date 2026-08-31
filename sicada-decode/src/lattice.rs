//! Decoding to a lattice rather than to a single answer.
//!
//! [`viterbi_decode`](crate::viterbi::viterbi_decode) returns the best path and
//! throws the rest away. That is enough to read out a transcript and nothing
//! else: no confidence, no *n*-best, and above all no second pass, because
//! rescoring needs the alternatives the first pass considered.
//!
//! A lattice keeps them. It is the part of `graph ∘ dense` that survived the
//! beam, with states as `(frame, graph state)` pairs and arcs as the graph arcs
//! between two surviving states. Each arc's graph cost and acoustic cost are
//! kept apart in a [`LatticeWeight`], so a rescoring pass can rebuild one half
//! without disturbing the other.
//!
//! The construction follows Kaldi's `LatticeFasterDecoder`. The one thing worth
//! stating explicitly, because it is what separates a lattice from a backtrace:
//! **an arc is kept because both its endpoints survived, not because it was the
//! best way to reach its endpoint.** Keeping only the improving arcs would give
//! a tree, namely the Viterbi backtrace, and no alternatives at all.

use rustc_hash::FxHashMap;

use sicada::algorithms::connect::connect;
use sicada::algorithms::prune::{PruneOptions, prune as prune_fst};
use sicada::arc::{Arc, ArcLabel, ArcStateId};
use sicada::error::OpenFstError;
use sicada::fst::{Fst, MutableFst};
use sicada::fsts::vector_fst::VectorFst;

use crate::dense::{DenseFst, FromScore};
use crate::frontier::{DecodeOptions, NO_AUX, Token, prune, relax_cost};
use crate::lattice_weight::{LatticeArc, LatticeWeight};

/// How wide to search, and how much of what was found to keep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatticeDecodeOptions {
    /// The beam the search itself runs under.
    pub search: DecodeOptions,
    /// Paths worse than the best by more than this are dropped from the
    /// lattice.
    ///
    /// Separate from `search.beam`, and usually smaller: the search beam has to
    /// be generous because a path that looks bad now may win later, while the
    /// lattice beam judges finished paths and can afford to be strict.
    /// `f32::INFINITY` keeps everything the search saw.
    pub lattice_beam: f32,
}

impl Default for LatticeDecodeOptions {
    fn default() -> Self {
        Self {
            search: DecodeOptions::default(),
            // Kaldi's default.
            lattice_beam: 8.0,
        }
    }
}

impl LatticeDecodeOptions {
    /// No beam anywhere: the lattice is the whole of `graph ∘ dense`.
    ///
    /// What the tests compare against, since it is the only setting under which
    /// the lattice and the composition are the same object.
    pub fn exhaustive() -> Self {
        Self {
            search: DecodeOptions::exhaustive(),
            lattice_beam: f32::INFINITY,
        }
    }
}

/// The lattice type for a decoding graph over `A`.
///
/// Same labels and state ids as the graph; the weight is the one that keeps the
/// two costs apart.
pub type Lattice<A> = VectorFst<LatticeArc<A>>;

/// The lattice state an `aux` names.
#[inline(always)]
fn state_of<A: Arc>(aux: u32) -> A::StateId {
    A::StateId::from_usize(aux as usize)
}

/// A graph arc that reached a state, kept until it is known whether that state
/// survived the beam.
struct Pending<A: Arc> {
    from: u32,
    to: A::StateId,
    ilabel: A::Label,
    olabel: A::Label,
    graph: f32,
    acoustic: f32,
}

/// Decodes to a lattice, or to `None` if the beam killed every path.
///
/// The graph's input labels are matched against the acoustic model's columns
/// and survive on the lattice's input side, so an alignment can still be read
/// off it; the output labels are the words.
///
/// # Errors
///
/// As [`viterbi_decode`](crate::viterbi::viterbi_decode): an input label naming
/// no column is a graph/model mismatch, and epsilon arcs that do not settle
/// mean a cycle of them costs less than nothing.
pub fn lattice_decode<A, G>(
    graph: &G,
    dense: &DenseFst<'_, A>,
    opts: &LatticeDecodeOptions,
) -> Result<Option<Lattice<A>>, OpenFstError>
where
    A: Arc,
    A::Weight: FromScore,
    A::StateId: ArcStateId,
    G: Fst<A>,
{
    let Some(start) = graph.start() else {
        return Ok(None);
    };

    let mut lattice: Lattice<A> = VectorFst::new();
    let mut current: FxHashMap<A::StateId, Token> = FxHashMap::default();
    let mut next: FxHashMap<A::StateId, Token> = FxHashMap::default();
    let mut queue: Vec<A::StateId> = Vec::new();
    let mut costs: Vec<f32> = Vec::new();
    let mut pending: Vec<Pending<A>> = Vec::new();

    current.insert(
        start,
        Token {
            cost: 0.0,
            aux: NO_AUX,
        },
    );
    settle_epsilons(graph, &mut current, &mut queue, f32::INFINITY)?;
    allocate(&mut lattice, &mut current);
    lattice.set_start(state_of::<A>(current[&start].aux));
    emit_epsilons(graph, &current, &mut lattice);

    for t in 0..dense.num_frames() {
        let frame = dense.frame(t);
        next.clear();
        pending.clear();

        for (&state, &token) in &current {
            for arc in graph.arcs(state) {
                if arc.ilabel() == A::Label::epsilon() {
                    continue;
                }
                let Some(column) = dense.column_of(arc.ilabel()) else {
                    return Err(OpenFstError::InvalidOperation(format!(
                        "lattice_decode: the graph has input label {} at state {state:?}, which \
                         names no column of a {}-symbol acoustic matrix",
                        arc.ilabel(),
                        dense.num_symbols()
                    )));
                };
                let graph_cost = arc.weight().to_cost();
                let acoustic = frame[column];
                relax_cost(
                    &mut next,
                    arc.nextstate(),
                    token.cost + graph_cost + acoustic,
                    NO_AUX,
                );
                // Kept whatever the relaxation decided: the lattice wants every
                // arc between two surviving states, not only the winning one.
                pending.push(Pending {
                    from: token.aux,
                    to: arc.nextstate(),
                    ilabel: arc.ilabel(),
                    olabel: arc.olabel(),
                    graph: graph_cost,
                    acoustic,
                });
            }
        }

        if next.is_empty() {
            return Ok(None);
        }
        settle_and_prune(graph, &mut next, &mut queue, &mut costs, &opts.search)?;
        allocate(&mut lattice, &mut next);

        for step in &pending {
            let Some(to) = next.get(&step.to) else {
                continue;
            };
            lattice.add_arc(
                state_of::<A>(step.from),
                LatticeArc::<A>::new(
                    step.ilabel,
                    step.olabel,
                    LatticeWeight::new(step.graph, step.acoustic),
                    state_of::<A>(to.aux),
                ),
            );
        }
        emit_epsilons(graph, &next, &mut lattice);

        std::mem::swap(&mut current, &mut next);
    }

    let mut reached_the_end = false;
    for (&state, &token) in &current {
        let final_cost = graph.final_weight(state).to_cost();
        if !final_cost.is_finite() {
            continue;
        }
        reached_the_end = true;
        lattice.set_final(
            state_of::<A>(token.aux),
            LatticeWeight::new(final_cost, 0.0),
        );
    }
    if !reached_the_end {
        return Ok(None);
    }

    // Most of what was allocated leads nowhere: a token survives the beam by
    // being cheap to *reach*, which says nothing about whether the rest of the
    // audio can be decoded from it.
    connect(&mut lattice);
    if lattice.start().is_none() {
        return Ok(None);
    }

    if opts.lattice_beam.is_finite() {
        prune_fst(
            &mut lattice,
            &PruneOptions::threshold(LatticeWeight::new(opts.lattice_beam, 0.0)),
        )?;
        if lattice.start().is_none() {
            return Ok(None);
        }
    }

    Ok(Some(lattice))
}

/// Gives every state in `frontier` a lattice state.
///
/// Written over the *lattice's* arc type rather than the graph's: `Lattice<A>`
/// mentions `A` only behind its associated types, and an associated type does
/// not determine the type it came from, so the graph's arc could not be
/// inferred here.
fn allocate<LA: Arc>(lattice: &mut VectorFst<LA>, frontier: &mut FxHashMap<LA::StateId, Token>) {
    for token in frontier.values_mut() {
        token.aux = lattice.add_state().as_usize() as u32;
    }
}

/// Relaxes the graph's epsilon arcs over `frontier` until nothing improves.
///
/// The same fixed point [`viterbi`](crate::viterbi) computes, without the
/// backpointers: here the arcs are read off the survivors afterwards.
fn settle_epsilons<A, G>(
    graph: &G,
    frontier: &mut FxHashMap<A::StateId, Token>,
    queue: &mut Vec<A::StateId>,
    cutoff: f32,
) -> Result<(), OpenFstError>
where
    A: Arc,
    A::Weight: FromScore,
    G: Fst<A>,
{
    queue.clear();
    queue.extend(frontier.keys().copied());

    let budget = frontier.len().saturating_mul(64).saturating_add(1024);
    let mut steps = 0usize;

    while let Some(state) = queue.pop() {
        steps += 1;
        if steps > budget {
            return Err(OpenFstError::InvalidOperation(
                "lattice_decode: the graph's epsilon arcs do not settle, which means a cycle of \
                 them costs less than nothing"
                    .into(),
            ));
        }
        let token = frontier[&state];
        for arc in graph.arcs(state) {
            if arc.ilabel() != A::Label::epsilon() {
                continue;
            }
            let cost = token.cost + arc.weight().to_cost();
            if cost > cutoff {
                continue;
            }
            if relax_cost(frontier, arc.nextstate(), cost, NO_AUX) {
                queue.push(arc.nextstate());
            }
        }
    }
    Ok(())
}

/// Settles the epsilons, prunes, and settles again under the tighter cutoff.
fn settle_and_prune<A, G>(
    graph: &G,
    frontier: &mut FxHashMap<A::StateId, Token>,
    queue: &mut Vec<A::StateId>,
    costs: &mut Vec<f32>,
    opts: &DecodeOptions,
) -> Result<f32, OpenFstError>
where
    A: Arc,
    A::Weight: FromScore,
    G: Fst<A>,
{
    let cutoff = prune(frontier, opts, costs);
    settle_epsilons(graph, frontier, queue, cutoff)?;
    // Epsilon arcs only added states at or under the cutoff, so the beam still
    // holds; the cap may not.
    if frontier.len() > opts.max_active {
        return Ok(prune(frontier, opts, costs));
    }
    Ok(cutoff)
}

/// Adds a lattice arc for every epsilon arc between two states of `frontier`.
fn emit_epsilons<A, G>(graph: &G, frontier: &FxHashMap<A::StateId, Token>, lattice: &mut Lattice<A>)
where
    A: Arc,
    A::Weight: FromScore,
    G: Fst<A>,
{
    for (&state, &token) in frontier {
        for arc in graph.arcs(state) {
            if arc.ilabel() != A::Label::epsilon() {
                continue;
            }
            let Some(to) = frontier.get(&arc.nextstate()) else {
                continue;
            };
            lattice.add_arc(
                state_of::<A>(token.aux),
                LatticeArc::<A>::new(
                    arc.ilabel(),
                    arc.olabel(),
                    // Consuming no frame, an epsilon arc costs the acoustic
                    // model nothing.
                    LatticeWeight::new(arc.weight().to_cost(), 0.0),
                    state_of::<A>(to.aux),
                ),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sicada::algorithms::arcsort::{ILabelCompare, arc_sort};
    use sicada::algorithms::compose::compose;
    use sicada::algorithms::shortest_path::{ShortestPathOptions, shortest_path};
    use sicada::arc::StdArc;
    use sicada::fst::ExpandedFst;
    use sicada::fsts::vector_fst::StdVectorFst;
    use sicada::properties::K_FST_PROPERTIES;
    use sicada::string::string_fst_to_output_labels;
    use sicada::weight::Weight;
    use sicada::weights::float_weight::TropicalWeight;

    use crate::viterbi::viterbi_decode;

    /// A small xorshift, so the random cases are the same every run.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn cost(&mut self) -> f32 {
            self.below(4096) as f32 / 64.0
        }
    }

    fn random_graph(rng: &mut Rng, symbols: usize) -> StdVectorFst {
        let states = 1 + rng.below(6);
        let mut graph: StdVectorFst = VectorFst::new();
        for _ in 0..states {
            graph.add_state();
        }
        graph.set_start(0);
        for from in 0..states as i32 {
            for _ in 0..1 + rng.below(4) {
                let ilabel = if rng.below(4) == 0 {
                    0
                } else {
                    1 + rng.below(symbols) as i32
                };
                let olabel = if rng.below(3) == 0 {
                    0
                } else {
                    10 * (1 + rng.below(symbols) as i32)
                };
                let to = rng.below(states) as i32;
                graph.add_arc(
                    from,
                    StdArc::new(ilabel, olabel, TropicalWeight(rng.cost()), to),
                );
            }
            if rng.below(3) == 0 {
                graph.set_final(from, TropicalWeight(rng.cost()));
            }
        }
        graph.properties(K_FST_PROPERTIES, true);
        graph
    }

    /// `dense ∘ graph`, connected: what an exhaustive lattice should be.
    fn composition(graph: &StdVectorFst, dense: &DenseFst<'_, StdArc>) -> StdVectorFst {
        let mut sorted = graph.clone();
        arc_sort(&mut sorted, &ILabelCompare);
        let mut composed: StdVectorFst = VectorFst::new();
        compose(dense, &sorted, &mut composed).expect("a composition");
        composed
    }

    /// The lattice's own best path, read out the same way the oracle's is.
    fn best_of(lattice: &Lattice<StdArc>) -> (Vec<i32>, f32) {
        let mut best: Lattice<StdArc> = VectorFst::new();
        shortest_path(lattice, &mut best, &ShortestPathOptions::default()).expect("a best path");
        let (labels, weight) = string_fst_to_output_labels(&best).expect("a single path");
        (
            labels.into_iter().filter(|&l| l != 0).collect(),
            weight.total(),
        )
    }

    fn free_graph() -> StdVectorFst {
        let mut fst = VectorFst::new();
        fst.add_state();
        fst.set_start(0);
        fst.set_final(0, TropicalWeight::one());
        for label in 1..=3 {
            fst.add_arc(0, StdArc::new(label, label * 10, TropicalWeight::one(), 0));
        }
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    #[test]
    fn its_best_path_is_the_one_the_viterbi_decoder_finds() {
        let scores = [
            5.0, 1.0, 9.0, //
            0.5, 4.0, 4.0, //
            3.0, 0.25, 3.0,
        ];
        let dense = DenseFst::<StdArc>::new(&scores, 3, 3).unwrap();
        let graph = free_graph();

        let lattice = lattice_decode(&graph, &dense, &LatticeDecodeOptions::exhaustive())
            .unwrap()
            .expect("a lattice");
        let expected = viterbi_decode(&graph, &dense, &DecodeOptions::exhaustive())
            .unwrap()
            .expect("a path");

        let (labels, total) = best_of(&lattice);
        assert_eq!(labels, expected.labels);
        assert!((total - expected.weight.0).abs() < 1e-5);
    }

    /// The two halves are the point of the whole weight, so they are checked
    /// rather than only their sum: the acoustic half of the best path has to be
    /// the frame scores it actually used.
    #[test]
    fn the_two_costs_stay_apart() {
        let scores = [
            5.0, 1.0, 9.0, //
            0.5, 4.0, 4.0,
        ];
        let dense = DenseFst::<StdArc>::new(&scores, 2, 3).unwrap();
        // Every arc costs the graph 0.25, so the graph half is 2 x 0.25.
        let mut graph: StdVectorFst = VectorFst::new();
        graph.add_state();
        graph.set_start(0);
        graph.set_final(0, TropicalWeight::one());
        for label in 1..=3 {
            graph.add_arc(0, StdArc::new(label, label * 10, TropicalWeight(0.25), 0));
        }
        graph.properties(K_FST_PROPERTIES, true);

        let lattice = lattice_decode(&graph, &dense, &LatticeDecodeOptions::exhaustive())
            .unwrap()
            .unwrap();
        let mut best: Lattice<StdArc> = VectorFst::new();
        shortest_path(&lattice, &mut best, &ShortestPathOptions::default()).unwrap();
        let (_, weight) = string_fst_to_output_labels(&best).unwrap();

        assert!((weight.graph - 0.5).abs() < 1e-6, "{weight}");
        // Frame 0 picks symbol 2 (1.0), frame 1 picks symbol 1 (0.5).
        assert!((weight.acoustic - 1.5).abs() < 1e-6, "{weight}");
        // And rescoring the acoustic half is what keeping them apart is for.
        assert!((weight.total_scaled(0.5) - (0.5 + 0.75)).abs() < 1e-6);
    }

    /// With no beam anywhere, the lattice *is* `graph ∘ dense`: the same states
    /// and the same arcs, because every pair survives and every arc between two
    /// survivors is kept. That equality is the strongest statement available
    /// about the construction, so it is the one the random cases check.
    #[test]
    fn an_unpruned_lattice_is_the_composition() {
        let symbols = 4;
        let mut rng = Rng(0x00C0_FFEE_1234_5678);
        let mut compared = 0;

        for round in 0..200 {
            let graph = random_graph(&mut rng, symbols);
            let frames = 1 + rng.below(5);
            let scores: Vec<f32> = (0..frames * symbols).map(|_| rng.cost()).collect();
            let dense = DenseFst::<StdArc>::new(&scores, frames, symbols).unwrap();

            let expected = composition(&graph, &dense);
            let lattice =
                lattice_decode(&graph, &dense, &LatticeDecodeOptions::exhaustive()).unwrap();

            let Some(lattice) = lattice else {
                assert!(
                    expected.start().is_none(),
                    "round {round}: no lattice, but the composition has {} states",
                    expected.num_states()
                );
                continue;
            };
            compared += 1;

            assert_eq!(
                lattice.num_states(),
                expected.num_states(),
                "round {round}: states"
            );
            assert_eq!(
                lattice.count_arcs(),
                expected.count_arcs(),
                "round {round}: arcs"
            );

            let mut best: StdVectorFst = VectorFst::new();
            shortest_path(&expected, &mut best, &ShortestPathOptions::default()).unwrap();
            let (labels, weight) = string_fst_to_output_labels(&best).unwrap();
            let labels: Vec<i32> = labels.into_iter().filter(|&l| l != 0).collect();

            let (mine, total) = best_of(&lattice);
            assert!(
                (total - weight.0).abs() < 1e-4,
                "round {round}: lattice {total} vs composition {}",
                weight.0
            );
            assert_eq!(mine, labels, "round {round}");
        }

        assert!(
            compared > 100,
            "only {compared} rounds had a lattice at all"
        );
    }

    /// The lattice beam drops paths, so the lattice shrinks, but never so far
    /// that the best path gets worse.
    ///
    /// The *cost* is what is asserted, not the labels. Two paths can tie
    /// exactly, and then which one `shortest_path` returns is not determined;
    /// removing one of them legitimately changes the answer without changing
    /// how good it is. Round 131 of these seeds is such a case.
    #[test]
    fn pruning_keeps_the_best_path() {
        let symbols = 4;
        let mut rng = Rng(0xBEEF_4321_9876);
        let mut shrank = 0;

        for round in 0..200 {
            let graph = random_graph(&mut rng, symbols);
            let frames = 1 + rng.below(5);
            let scores: Vec<f32> = (0..frames * symbols).map(|_| rng.cost()).collect();
            let dense = DenseFst::<StdArc>::new(&scores, frames, symbols).unwrap();

            let whole =
                lattice_decode(&graph, &dense, &LatticeDecodeOptions::exhaustive()).unwrap();
            let pruned = lattice_decode(
                &graph,
                &dense,
                &LatticeDecodeOptions {
                    search: DecodeOptions::exhaustive(),
                    lattice_beam: 2.0,
                },
            )
            .unwrap();

            match (whole, pruned) {
                (None, None) => {}
                (Some(whole), Some(pruned)) => {
                    assert!(
                        pruned.count_arcs() <= whole.count_arcs(),
                        "round {round}: pruning grew the lattice"
                    );
                    if pruned.count_arcs() < whole.count_arcs() {
                        shrank += 1;
                    }
                    let (whole_labels, whole_cost) = best_of(&whole);
                    let (pruned_labels, pruned_cost) = best_of(&pruned);
                    assert!(
                        (pruned_cost - whole_cost).abs() < 1e-4,
                        "round {round}: {pruned_cost} vs {whole_cost}, {pruned_labels:?} vs {whole_labels:?}"
                    );
                }
                (whole, pruned) => panic!(
                    "round {round}: whole {:?}, pruned {:?}",
                    whole.is_some(),
                    pruned.is_some()
                ),
            }
        }

        assert!(shrank > 20, "the beam never removed anything in {shrank}");
    }

    #[test]
    fn a_graph_that_reaches_no_final_state_has_no_lattice() {
        let mut graph: StdVectorFst = VectorFst::new();
        graph.add_state();
        graph.set_start(0);
        graph.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 0));
        graph.properties(K_FST_PROPERTIES, true);

        let scores = [1.0, 1.0, 1.0];
        let dense = DenseFst::<StdArc>::new(&scores, 1, 3).unwrap();
        assert!(
            lattice_decode(&graph, &dense, &LatticeDecodeOptions::exhaustive())
                .unwrap()
                .is_none()
        );
    }

    /// The alignment has to survive: the input labels say which acoustic column
    /// each frame used, which a second pass needs in order to rescore it.
    #[test]
    fn the_input_labels_are_still_the_acoustic_columns() {
        let scores = [5.0, 1.0, 9.0];
        let dense = DenseFst::<StdArc>::new(&scores, 1, 3).unwrap();
        let lattice = lattice_decode(&free_graph(), &dense, &LatticeDecodeOptions::exhaustive())
            .unwrap()
            .unwrap();

        for state in lattice.states() {
            for arc in lattice.arcs(state) {
                assert_eq!(
                    arc.olabel(),
                    arc.ilabel() * 10,
                    "the graph maps label n to output 10n"
                );
                assert!(dense.column_of(arc.ilabel()).is_some());
            }
        }
    }
}
