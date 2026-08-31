//! Frame-synchronous Viterbi beam search over a decoding graph.
//!
//! What this computes is the best path of `graph ∘ dense`, without ever building
//! that composition. The composition's states are pairs `(graph state, frame)`,
//! and the frame moves in lockstep for every one of them, so the whole thing
//! can be walked one frame at a time keeping only the graph states alive at
//! that frame. That is the standard decoder shape (Kaldi's `SimpleDecoder` and
//! `FasterDecoder`, k2's `intersect_dense_pruned`), and it keeps decoding linear
//! in `T` rather than in the size of a composition nobody wants to store.
//!
//! Two kinds of arc leave a graph state:
//!
//! - an **emitting** arc, whose input label names a column of the acoustic
//!   matrix. Taking it consumes one frame and pays that frame's score.
//! - an **epsilon** arc, which consumes no frame. These are relaxed to a fixed
//!   point within the frame, before and after each emitting step.
//!
//! [`viterbi_decode`] returns the best path only. To keep the alternatives as
//! well, so that a second pass has something to rescore, use
//! [`lattice_decode`](crate::lattice::lattice_decode).

use rustc_hash::FxHashMap;

use sicada::arc::{Arc, ArcLabel};
use sicada::error::OpenFstError;
use sicada::fst::Fst;
use sicada::weight::PathWeight;

use crate::dense::{DenseFst, FromScore};
use crate::frontier::{DecodeOptions, NO_AUX, Token, prune};

/// The best path through the graph, given the acoustic scores.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded<A: Arc> {
    /// The output labels along the path, epsilons removed.
    pub labels: Vec<A::Label>,
    /// The path's weight: the graph's costs, the acoustic scores and the final
    /// weight, all multiplied together.
    pub weight: A::Weight,
}

/// One step of a path, kept so the answer can be read back.
///
/// A frame keeps at most `max_active` of these, but never releases the ones
/// behind it, so the arena grows with `T × max_active`. Reference-counted
/// tokens that free a dead prefix are what Kaldi does and are outstanding here.
#[derive(Debug, Clone, Copy)]
struct Link<L> {
    prev: u32,
    olabel: L,
}

/// Returns the best path of `graph ∘ dense`, or `None` if the beam killed every
/// path before the last frame.
///
/// The graph's *input* labels are matched against the acoustic model's columns
/// and its *output* labels are what comes back, which is the usual arrangement:
/// input side indexed by the model, output side in words.
///
/// # Errors
///
/// A non-epsilon input label that names no column of `dense` is a mismatch
/// between the graph and the acoustic model, and is reported rather than
/// skipped: dropping such an arc would return a worse path with no indication
/// that anything was wrong. Remove disambiguation symbols from the graph before
/// decoding.
pub fn viterbi_decode<A, G>(
    graph: &G,
    dense: &DenseFst<'_, A>,
    opts: &DecodeOptions,
) -> Result<Option<Decoded<A>>, OpenFstError>
where
    A: Arc,
    A::Weight: FromScore + PathWeight,
    G: Fst<A>,
{
    let Some(start) = graph.start() else {
        return Ok(None);
    };

    let mut links: Vec<Link<A::Label>> = Vec::new();
    let mut current: FxHashMap<A::StateId, Token> = FxHashMap::default();
    let mut next: FxHashMap<A::StateId, Token> = FxHashMap::default();
    let mut queue: Vec<A::StateId> = Vec::new();
    let mut costs: Vec<f32> = Vec::new();

    current.insert(
        start,
        Token {
            cost: 0.0,
            aux: NO_AUX,
        },
    );
    relax_epsilons(graph, &mut current, &mut links, &mut queue, f32::INFINITY)?;

    for t in 0..dense.num_frames() {
        let frame = dense.frame(t);
        next.clear();

        for (&state, &token) in &current {
            for arc in graph.arcs(state) {
                if arc.ilabel() == A::Label::epsilon() {
                    continue;
                }
                let Some(column) = dense.column_of(arc.ilabel()) else {
                    return Err(OpenFstError::InvalidOperation(format!(
                        "viterbi_decode: the graph has input label {} at state {state:?}, which \
                         names no column of a {}-symbol acoustic matrix",
                        arc.ilabel(),
                        dense.num_symbols()
                    )));
                };
                let cost = token.cost + arc.weight().to_cost() + frame[column];
                relax(
                    &mut next,
                    &mut links,
                    arc.nextstate(),
                    cost,
                    token.aux,
                    arc.olabel(),
                );
            }
        }

        if next.is_empty() {
            return Ok(None);
        }
        let cutoff = prune(&mut next, opts, &mut costs);
        relax_epsilons(graph, &mut next, &mut links, &mut queue, cutoff)?;
        // Epsilon arcs can only have added states at or under the cutoff, so
        // the beam still holds; the cap may not, and is re-applied.
        if next.len() > opts.max_active {
            prune(&mut next, opts, &mut costs);
        }

        std::mem::swap(&mut current, &mut next);
    }

    let mut best: Option<(f32, u32)> = None;
    for (&state, &token) in &current {
        let final_cost = graph.final_weight(state).to_cost();
        if !final_cost.is_finite() {
            continue;
        }
        let total = token.cost + final_cost;
        if best.is_none_or(|(so_far, _)| total < so_far) {
            best = Some((total, token.aux));
        }
    }

    Ok(best.map(|(total, link)| Decoded {
        labels: trace_back(&links, link),
        weight: A::Weight::from_cost(total),
    }))
}

/// Records `cost` at `state` if it beats what is already there.
///
/// Written over the state-id and label types rather than over the arc: `A`
/// would appear only behind `A::StateId` and `A::Label`, and an associated type
/// does not determine the type it came from, so every call would have to name
/// the arc.
#[inline]
fn relax<S, L>(
    frontier: &mut FxHashMap<S, Token>,
    links: &mut Vec<Link<L>>,
    state: S,
    cost: f32,
    prev_link: u32,
    olabel: L,
) -> bool
where
    S: std::hash::Hash + Eq,
    L: ArcLabel,
{
    match frontier.get_mut(&state) {
        Some(token) if token.cost <= cost => false,
        slot => {
            // An epsilon output label adds nothing to read back, so the path
            // keeps pointing at whatever came before it.
            let link = if olabel == L::epsilon() {
                prev_link
            } else {
                links.push(Link {
                    prev: prev_link,
                    olabel,
                });
                (links.len() - 1) as u32
            };
            let token = Token { cost, aux: link };
            match slot {
                Some(existing) => *existing = token,
                None => {
                    frontier.insert(state, token);
                }
            }
            true
        }
    }
}

/// Relaxes the graph's epsilon arcs over `frontier` until nothing improves.
///
/// Epsilon arcs consume no frame, so they may be taken any number of times
/// within one; the fixed point is the shortest distance over them. A decoding
/// graph's epsilon arcs cost nothing or cost something, never less than
/// nothing, so every relaxation strictly lowers a cost and the loop ends. A
/// graph that breaks that is reported rather than spun on.
fn relax_epsilons<A, G>(
    graph: &G,
    frontier: &mut FxHashMap<A::StateId, Token>,
    links: &mut Vec<Link<A::Label>>,
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

    // Generous: every state may legitimately be re-reached once per other
    // state on its epsilon path. Anything past this is a negative cycle.
    let budget = frontier.len().saturating_mul(64).saturating_add(1024);
    let mut steps = 0usize;

    while let Some(state) = queue.pop() {
        steps += 1;
        if steps > budget {
            return Err(OpenFstError::InvalidOperation(
                "viterbi_decode: the graph's epsilon arcs do not settle, which means a cycle of \
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
            if relax(
                frontier,
                links,
                arc.nextstate(),
                cost,
                token.aux,
                arc.olabel(),
            ) {
                queue.push(arc.nextstate());
            }
        }
    }
    Ok(())
}

/// Walks the links back to the start, yielding the labels in order.
fn trace_back<L: Copy>(links: &[Link<L>], mut link: u32) -> Vec<L> {
    let mut labels = Vec::new();
    while link != NO_AUX {
        let step = links[link as usize];
        labels.push(step.olabel);
        link = step.prev;
    }
    labels.reverse();
    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use sicada::algorithms::arcsort::{ILabelCompare, arc_sort};
    use sicada::algorithms::compose::compose;
    use sicada::algorithms::shortest_path::{ShortestPathOptions, shortest_path};
    use sicada::arc::StdArc;
    use sicada::fst::MutableFst;
    use sicada::fsts::vector_fst::{StdVectorFst, VectorFst};
    use sicada::properties::K_FST_PROPERTIES;
    use sicada::string::string_fst_to_output_labels;
    use sicada::weight::Weight;
    use sicada::weights::float_weight::TropicalWeight;

    /// The answer the decoder is supposed to agree with: build the whole
    /// composition and take its shortest path.
    ///
    /// This is exactly the work the decoder exists to avoid, which is why it
    /// makes a good oracle: it shares no code with the thing under test beyond
    /// the FST types themselves.
    fn by_composition(
        graph: &StdVectorFst,
        dense: &DenseFst<'_, StdArc>,
    ) -> Option<(Vec<i32>, f32)> {
        // `dense ∘ graph`, not the other way round: composition matches the
        // left FST's *output* labels against the right one's *input* labels,
        // and it is the acoustic symbols that meet, leaving the graph's words
        // on the output side.
        let mut sorted = graph.clone();
        arc_sort(&mut sorted, &ILabelCompare);
        let mut composed: StdVectorFst = VectorFst::new();
        compose(dense, &sorted, &mut composed).expect("a composition");
        composed.start()?;
        let mut best: StdVectorFst = VectorFst::new();
        shortest_path(&composed, &mut best, &ShortestPathOptions::default()).expect("a best path");
        best.start()?;
        let (labels, weight) = string_fst_to_output_labels(&best).expect("a single path");
        // The decoder reports the labels a reader wants, so epsilon outputs,
        // which the path does carry, are dropped on both sides.
        Some((labels.into_iter().filter(|&l| l != 0).collect(), weight.0))
    }

    /// A graph over 3 symbols (columns 0..2, labels 1..3) that accepts any
    /// sequence, mapping label 1 to output 10, 2 to 20, 3 to 30.
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
    fn it_picks_the_best_symbol_in_every_frame() {
        // Frame 0 likes symbol 2, frame 1 likes symbol 0, frame 2 likes 1.
        let scores = [
            5.0, 1.0, 9.0, //
            0.5, 4.0, 4.0, //
            3.0, 0.25, 3.0,
        ];
        let dense = DenseFst::<StdArc>::new(&scores, 3, 3).unwrap();
        let decoded = viterbi_decode(&free_graph(), &dense, &DecodeOptions::exhaustive())
            .unwrap()
            .expect("a path");

        assert_eq!(decoded.labels, vec![20, 10, 20]);
        assert!((decoded.weight.0 - (1.0 + 0.5 + 0.25)).abs() < 1e-6);
    }

    #[test]
    fn it_agrees_with_composing_and_taking_the_shortest_path() {
        let scores = [
            5.0, 1.0, 9.0, //
            0.5, 4.0, 4.0, //
            3.0, 0.25, 3.0, //
            2.0, 2.5, 0.75,
        ];
        let dense = DenseFst::<StdArc>::new(&scores, 4, 3).unwrap();
        let graph = free_graph();

        let (labels, weight) = by_composition(&graph, &dense).expect("an answer");
        let decoded = viterbi_decode(&graph, &dense, &DecodeOptions::exhaustive())
            .unwrap()
            .expect("a path");

        assert_eq!(decoded.labels, labels);
        assert!((decoded.weight.0 - weight).abs() < 1e-5);
    }

    /// The same agreement over graphs that constrain what may follow what, and
    /// that carry their own costs.
    #[test]
    fn it_agrees_on_a_graph_that_forbids_repeats() {
        // 3 states: after emitting symbol s you may not emit s again.
        let mut graph: StdVectorFst = VectorFst::new();
        for _ in 0..4 {
            graph.add_state();
        }
        graph.set_start(0);
        for from in 0..4 {
            for label in 1..=3i32 {
                if from == label {
                    continue;
                }
                graph.add_arc(
                    from,
                    StdArc::new(label, label * 10, TropicalWeight(label as f32 * 0.1), label),
                );
            }
        }
        for state in 1..4 {
            graph.set_final(state, TropicalWeight(0.5));
        }
        graph.properties(K_FST_PROPERTIES, true);

        let scores = [
            5.0, 1.0, 9.0, //
            0.5, 4.0, 4.0, //
            3.0, 0.25, 3.0, //
            2.0, 2.5, 0.75, //
            1.0, 1.0, 1.0,
        ];
        let dense = DenseFst::<StdArc>::new(&scores, 5, 3).unwrap();

        let (labels, weight) = by_composition(&graph, &dense).expect("an answer");
        let decoded = viterbi_decode(&graph, &dense, &DecodeOptions::exhaustive())
            .unwrap()
            .expect("a path");

        assert_eq!(decoded.labels, labels);
        assert!((decoded.weight.0 - weight).abs() < 1e-5);
    }

    /// Epsilon arcs consume no frame, so a path may take several of them
    /// between two frames. The composition oracle handles them by construction,
    /// which is why it is worth comparing against.
    #[test]
    fn it_agrees_when_the_graph_has_epsilon_arcs() {
        let mut graph: StdVectorFst = VectorFst::new();
        for _ in 0..3 {
            graph.add_state();
        }
        graph.set_start(0);
        graph.set_final(2, TropicalWeight::one());
        // 0 --1:10--> 0, and an epsilon chain 0 -> 1 -> 2 that emits 99.
        graph.add_arc(0, StdArc::new(1, 10, TropicalWeight::one(), 0));
        graph.add_arc(0, StdArc::new(2, 20, TropicalWeight(0.2), 0));
        graph.add_arc(0, StdArc::new(0, 0, TropicalWeight(0.3), 1));
        graph.add_arc(1, StdArc::new(0, 99, TropicalWeight(0.4), 2));
        graph.add_arc(2, StdArc::new(1, 10, TropicalWeight::one(), 0));
        graph.properties(K_FST_PROPERTIES, true);

        let scores = [
            0.5, 2.0, 9.0, //
            2.0, 0.5, 9.0, //
            0.5, 2.0, 9.0,
        ];
        let dense = DenseFst::<StdArc>::new(&scores, 3, 3).unwrap();

        let (labels, weight) = by_composition(&graph, &dense).expect("an answer");
        let decoded = viterbi_decode(&graph, &dense, &DecodeOptions::exhaustive())
            .unwrap()
            .expect("a path");

        assert_eq!(decoded.labels, labels);
        assert!((decoded.weight.0 - weight).abs() < 1e-5);
    }

    /// A small xorshift, so the random cases below are the same every run.
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

        /// A non-negative cost with enough distinct values that two different
        /// paths rarely land on the same total.
        fn cost(&mut self) -> f32 {
            self.below(4096) as f32 / 64.0
        }
    }

    /// The agreement, over graphs nobody chose.
    ///
    /// The deterministic cases above each aim at one thing; this one exists to
    /// find what none of them thought of, which for a beam search is nearly
    /// always the interaction between epsilon closure and pruning.
    #[test]
    fn it_agrees_with_the_composition_on_random_graphs() {
        let symbols = 4;
        let mut rng = Rng(0x5EED_1234_9ABC_DEF1);
        let mut compared = 0;

        for round in 0..200 {
            let states = 1 + rng.below(6);
            let mut graph: StdVectorFst = VectorFst::new();
            for _ in 0..states {
                graph.add_state();
            }
            graph.set_start(0);
            for from in 0..states as i32 {
                for _ in 0..1 + rng.below(4) {
                    // A quarter of the arcs consume no frame.
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

            let frames = 1 + rng.below(5);
            let scores: Vec<f32> = (0..frames * symbols).map(|_| rng.cost()).collect();
            let dense = DenseFst::<StdArc>::new(&scores, frames, symbols).unwrap();

            let expected = by_composition(&graph, &dense);
            let decoded = viterbi_decode(&graph, &dense, &DecodeOptions::exhaustive()).unwrap();

            match (expected, decoded) {
                (None, None) => {}
                (Some((labels, weight)), Some(decoded)) => {
                    compared += 1;
                    assert!(
                        (decoded.weight.0 - weight).abs() < 1e-4,
                        "round {round}: decoder {} vs composition {weight}",
                        decoded.weight.0
                    );
                    assert_eq!(decoded.labels, labels, "round {round}");
                }
                (expected, decoded) => {
                    panic!("round {round}: composition {expected:?}, decoder {decoded:?}")
                }
            }
        }

        // Guards against the graphs degenerating into ones that decode to
        // nothing, which would make the whole test vacuous.
        assert!(compared > 100, "only {compared} rounds had a path at all");
    }

    #[test]
    fn a_beam_that_keeps_the_best_path_does_not_change_the_answer() {
        let scores = [
            5.0, 1.0, 9.0, //
            0.5, 4.0, 4.0, //
            3.0, 0.25, 3.0,
        ];
        let dense = DenseFst::<StdArc>::new(&scores, 3, 3).unwrap();
        let graph = free_graph();

        let wide = viterbi_decode(&graph, &dense, &DecodeOptions::exhaustive())
            .unwrap()
            .unwrap();
        let narrow = viterbi_decode(
            &graph,
            &dense,
            &DecodeOptions {
                beam: 0.001,
                max_active: 1,
                min_active: 0,
            },
        )
        .unwrap()
        .unwrap();

        assert_eq!(narrow.labels, wide.labels);
        assert!((narrow.weight.0 - wide.weight.0).abs() < 1e-6);
    }

    #[test]
    fn a_graph_the_model_does_not_match_is_reported() {
        let mut graph: StdVectorFst = VectorFst::new();
        graph.add_state();
        graph.set_start(0);
        graph.set_final(0, TropicalWeight::one());
        // Column 7 does not exist in a 3-symbol matrix.
        graph.add_arc(0, StdArc::new(8, 1, TropicalWeight::one(), 0));
        graph.properties(K_FST_PROPERTIES, true);

        let scores = [1.0, 1.0, 1.0];
        let dense = DenseFst::<StdArc>::new(&scores, 1, 3).unwrap();
        let err = viterbi_decode(&graph, &dense, &DecodeOptions::exhaustive()).unwrap_err();
        assert!(format!("{err}").contains("names no column"), "{err}");
    }

    #[test]
    fn a_graph_that_reaches_no_final_state_decodes_to_nothing() {
        let mut graph: StdVectorFst = VectorFst::new();
        graph.add_state();
        graph.set_start(0);
        graph.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 0));
        graph.properties(K_FST_PROPERTIES, true);

        let scores = [1.0, 1.0, 1.0];
        let dense = DenseFst::<StdArc>::new(&scores, 1, 3).unwrap();
        assert_eq!(
            viterbi_decode(&graph, &dense, &DecodeOptions::exhaustive()).unwrap(),
            None
        );
    }

    #[test]
    fn an_epsilon_cycle_that_costs_less_than_nothing_is_reported() {
        let mut graph: StdVectorFst = VectorFst::new();
        graph.add_state();
        graph.add_state();
        graph.set_start(0);
        graph.set_final(1, TropicalWeight::one());
        graph.add_arc(0, StdArc::new(0, 0, TropicalWeight(-1.0), 1));
        graph.add_arc(1, StdArc::new(0, 0, TropicalWeight(-1.0), 0));
        graph.properties(K_FST_PROPERTIES, true);

        let scores = [1.0, 1.0, 1.0];
        let dense = DenseFst::<StdArc>::new(&scores, 1, 3).unwrap();
        let err = viterbi_decode(&graph, &dense, &DecodeOptions::exhaustive()).unwrap_err();
        assert!(format!("{err}").contains("less than nothing"), "{err}");
    }

    #[test]
    fn no_frames_decodes_the_graphs_own_best_path() {
        let graph = free_graph();
        let dense = DenseFst::<StdArc>::new(&[], 0, 3).unwrap();
        let decoded = viterbi_decode(&graph, &dense, &DecodeOptions::exhaustive())
            .unwrap()
            .expect("the empty path");
        assert!(decoded.labels.is_empty());
        assert_eq!(decoded.weight, TropicalWeight::one());
    }
}
