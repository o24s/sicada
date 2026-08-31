//! Drawing paths from an FST at random.
//!
//! Port of OpenFst's `randgen.h`. Walking from the start state and picking one
//! arc at each step gives one path; doing it `npath` times and keeping the
//! paths as a tree gives a sample of what the FST accepts. How the arc is
//! picked is left to an [`ArcSelector`]: uniformly, or by treating the weights
//! as negative log probabilities.

use std::collections::BTreeMap;

use crate::arc::{Arc, ArcLabel};
use crate::error::OpenFstError;
use crate::fst::{Fst, MutableFst};
use crate::weight::Weight;
use crate::weights::float_weight::Log64Weight;

/// A small deterministic random number generator.
///
/// SICADA-DIVERGE: upstream draws from `absl::BitGen`, or from `std::mt19937_64`
/// when a seed is given. Sampling here is always from an explicit generator, so
/// that a run can be reproduced without a global one being involved; the
/// sequence is not upstream's, and nothing about the algorithm depends on which
/// it is.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// A generator started from `seed`.
    pub fn new(seed: u64) -> Self {
        // Any odd increment gives a full period; this is the usual LCG.
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    /// The next value.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // The low bits of an LCG are weak; the high ones are what is used.
        self.0 ^ (self.0 >> 31)
    }

    /// A number below `bound`.
    #[inline]
    pub fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() >> 11) as usize % bound
    }

    /// A number in `[0, 1)`.
    #[inline]
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Picks one of the choices leaving a state.
///
/// The answer is in `0..=num_arcs`: `num_arcs` itself means "stop here", which
/// is only offered when the state is final.
pub trait ArcSelector<A: Arc> {
    /// Which choice to take at `state`.
    fn select<F: Fst<A>>(&self, rng: &mut Rng, fst: &F, state: A::StateId) -> usize;
}

/// Every choice equally likely, the weights ignored.
#[derive(Debug, Clone, Copy, Default)]
pub struct UniformArcSelector;

impl<A: Arc> ArcSelector<A> for UniformArcSelector {
    fn select<F: Fst<A>>(&self, rng: &mut Rng, fst: &F, state: A::StateId) -> usize {
        let choices =
            fst.num_arcs(state) + usize::from(fst.final_weight(state) != A::Weight::zero());
        rng.below(choices)
    }
}

/// The weights read as negative log probabilities, normalized over what leaves
/// the state.
///
/// A zero-weight arc is never chosen, since a zero weight means "this path does
/// not exist" in any semiring.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogProbArcSelector;

impl<A: Arc> ArcSelector<A> for LogProbArcSelector
where
    Log64Weight: From<A::Weight>,
{
    fn select<F: Fst<A>>(&self, rng: &mut Rng, fst: &F, state: A::StateId) -> usize {
        let as_log = |weight: A::Weight| Log64Weight::from(weight);
        // Everything leaving the state, including the choice of stopping.
        let mut total = Log64Weight::zero();
        for arc in fst.arcs(state) {
            total = total.plus(&as_log(arc.weight().clone()));
        }
        total = total.plus(&as_log(fst.final_weight(state)));

        // Drawing uniformly in the probability the total stands for, and then
        // walking until the running sum passes it.
        let threshold = rng.unit();
        let log_threshold = -threshold.ln() + total.value();
        let mut running = Log64Weight::zero();
        for (index, arc) in fst.arcs(state).enumerate() {
            running = running.plus(&as_log(arc.weight().clone()));
            if running.value() < log_threshold {
                return index;
            }
        }
        fst.num_arcs(state)
    }
}

/// How to draw.
#[derive(Debug, Clone)]
pub struct RandGenOptions<S> {
    /// How to pick an arc.
    pub selector: S,
    /// The longest path to draw, or `None` for no limit.
    pub max_length: Option<usize>,
    /// How many paths to draw.
    pub npath: usize,
    /// Whether the result carries how often each path was drawn, rather than
    /// being a plain tree of the paths.
    pub weighted: bool,
    /// Whether to divide the number of draws out of the weights.
    pub remove_total_weight: bool,
}

impl<S> RandGenOptions<S> {
    /// One unweighted path of any length.
    pub fn new(selector: S) -> Self {
        Self {
            selector,
            max_length: None,
            npath: 1,
            weighted: false,
            remove_total_weight: false,
        }
    }
}

/// A state of the result: where in the input, and how many of the draws are
/// still following this path.
struct RandState<S> {
    state: S,
    nsamples: usize,
    length: usize,
}

/// Draws `opts.npath` paths from `ifst` into `ofst`.
///
/// The result is the tree of what was drawn: a path drawn twice is one path,
/// and with `weighted` set its weight says how often. Without it, the tree is
/// unweighted and a path drawn twice appears once.
pub fn rand_gen<A, B, F1, F2, S>(
    ifst: &F1,
    ofst: &mut F2,
    rng: &mut Rng,
    opts: &RandGenOptions<S>,
) -> Result<(), OpenFstError>
where
    A: Arc,
    B: Arc<Label = A::Label, StateId = A::StateId>,
    B::Weight: From<Log64Weight>,
    F1: Fst<A>,
    F2: MutableFst<B>,
    S: ArcSelector<A>,
{
    ofst.delete_all_states();
    ofst.set_input_symbols(ifst.input_symbols());
    ofst.set_output_symbols(ifst.output_symbols());

    let Some(start) = ifst.start() else {
        return Ok(());
    };
    if opts.npath == 0 {
        return Ok(());
    }

    let out_start = ofst.add_state();
    ofst.set_start(out_start);
    let mut pending: Vec<(B::StateId, RandState<A::StateId>)> = vec![(
        out_start,
        RandState {
            state: start,
            nsamples: opts.npath,
            length: 0,
        },
    )];
    // Only made when a path stops and the result is unweighted, since then
    // stopping has to be an arc rather than a final weight.
    let mut superfinal: Option<B::StateId> = None;
    let epsilon = A::Label::epsilon();
    let zero = A::Weight::zero();

    while let Some((here, rstate)) = pending.pop() {
        let narcs = ifst.num_arcs(rstate.state);
        let can_stop = ifst.final_weight(rstate.state) != zero;
        // A state with nowhere to go and no way to stop ends the path, as does
        // running out of length.
        if (narcs == 0 && !can_stop) || opts.max_length == Some(rstate.length) {
            continue;
        }

        // Every draw at this state, counted by which choice it made. Counting
        // is how a path drawn twice becomes one path of the result.
        let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
        for _ in 0..rstate.nsamples {
            *counts
                .entry(opts.selector.select(rng, ifst, rstate.state))
                .or_insert(0) += 1;
        }

        let arcs: Vec<A> = ifst.arcs(rstate.state).collect();
        for (choice, count) in counts {
            let share = count as f64 / rstate.nsamples as f64;
            if let Some(arc) = arcs.get(choice) {
                let weight = if opts.weighted {
                    B::Weight::from(Log64Weight(-share.ln()))
                } else {
                    B::Weight::one()
                };
                let next = ofst.add_state();
                ofst.add_arc(here, B::new(arc.ilabel(), arc.olabel(), weight, next));
                pending.push((
                    next,
                    RandState {
                        state: arc.nextstate(),
                        nsamples: count,
                        length: rstate.length + 1,
                    },
                ));
                continue;
            }
            // The choice was to stop.
            if opts.weighted {
                let value = if opts.remove_total_weight {
                    -share.ln()
                } else {
                    -(share * opts.npath as f64).ln()
                };
                ofst.set_final(here, B::Weight::from(Log64Weight(value)));
            } else {
                // Without weights, the number of times a path was drawn shows
                // as that many epsilon arcs into one shared final state, as
                // upstream does, so that the unweighted result is still a tree
                // with one leaf per draw.
                let target = *superfinal.get_or_insert_with(|| {
                    let state = ofst.add_state();
                    ofst.set_final(state, B::Weight::one());
                    state
                });
                for _ in 0..count {
                    ofst.add_arc(here, B::new(epsilon, epsilon, B::Weight::one(), target));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::visible_paths;
    use crate::arc::{LogArc, StdArc};
    use crate::fst::ExpandedFst as _;
    use crate::fsts::vector_fst::{StdVectorFst, VectorFst};
    use crate::properties::K_FST_PROPERTIES;
    use crate::weights::float_weight::{LogWeight, TropicalWeight};

    /// A three-way branch, each branch one label, all final.
    fn branches() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        for label in 1..=3 {
            fst.add_arc(0, StdArc::new(label, label, TropicalWeight::one(), 1));
        }
        fst.set_final(1, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    fn drawn(
        fst: &StdVectorFst,
        opts: &RandGenOptions<UniformArcSelector>,
        seed: u64,
    ) -> StdVectorFst {
        let mut out = StdVectorFst::new();
        let mut rng = Rng::new(seed);
        rand_gen(fst, &mut out, &mut rng, opts).unwrap();
        out
    }

    /// The strings a drawn tree holds.
    fn strings(fst: &StdVectorFst) -> Vec<Vec<i32>> {
        let mut out: Vec<Vec<i32>> = visible_paths(fst, 24)
            .into_iter()
            .map(|(ilabels, _, _)| ilabels)
            .collect();
        out.sort();
        out
    }

    /// A drawn path is a path of the input.
    #[test]
    fn every_drawn_path_is_a_path_of_the_input() {
        let fst = branches();
        let allowed = strings(&fst);
        for seed in 0..50 {
            let out = drawn(&fst, &RandGenOptions::new(UniformArcSelector), seed);
            for path in strings(&out) {
                assert!(
                    allowed.contains(&path),
                    "{path:?} is not a path of the input"
                );
            }
        }
    }

    /// One draw gives one path.
    #[test]
    fn one_draw_gives_one_path() {
        let fst = branches();
        for seed in 0..20 {
            let out = drawn(&fst, &RandGenOptions::new(UniformArcSelector), seed);
            assert_eq!(strings(&out).len(), 1, "seed {seed}");
        }
    }

    /// The same seed gives the same answer, and different seeds eventually give
    /// different ones.
    #[test]
    fn the_seed_decides_what_is_drawn() {
        let fst = branches();
        let opts = RandGenOptions::new(UniformArcSelector);
        assert_eq!(
            strings(&drawn(&fst, &opts, 7)),
            strings(&drawn(&fst, &opts, 7))
        );

        let seen: std::collections::BTreeSet<Vec<Vec<i32>>> = (0..60)
            .map(|seed| strings(&drawn(&fst, &opts, seed)))
            .collect();
        assert!(seen.len() > 1, "every seed drew the same path");
    }

    /// Drawing many paths reaches every branch of a small FST.
    #[test]
    fn drawing_enough_paths_reaches_every_branch() {
        let fst = branches();
        let opts = RandGenOptions {
            npath: 200,
            ..RandGenOptions::new(UniformArcSelector)
        };
        let out = drawn(&fst, &opts, 1);
        let mut labels: Vec<i32> = strings(&out).into_iter().flatten().collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels, vec![1, 2, 3]);
    }

    /// A limit on the length stops the walk.
    #[test]
    fn the_length_limit_stops_the_walk() {
        // A loop that could go round forever.
        let mut fst = StdVectorFst::new();
        fst.add_state();
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 0));
        fst.set_final(0, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let opts = RandGenOptions {
            max_length: Some(4),
            npath: 20,
            ..RandGenOptions::new(UniformArcSelector)
        };
        let out = drawn(&fst, &opts, 3);
        for path in strings(&out) {
            assert!(path.len() <= 4, "{path:?} is longer than the limit");
        }
    }

    /// An FST with no start state draws nothing, and so does asking for no
    /// paths.
    #[test]
    fn there_is_nothing_to_draw_from_nothing() {
        let empty = StdVectorFst::new();
        assert_eq!(
            drawn(&empty, &RandGenOptions::new(UniformArcSelector), 1).num_states(),
            0
        );

        let opts = RandGenOptions {
            npath: 0,
            ..RandGenOptions::new(UniformArcSelector)
        };
        assert_eq!(drawn(&branches(), &opts, 1).num_states(), 0);
    }

    /// A weighted draw records how often each path was taken, so a path taken
    /// more often weighs less.
    #[test]
    fn a_weighted_draw_records_how_often_each_path_was_taken() {
        // One branch is far more likely than the other under the log selector.
        let mut fst: VectorFst<LogArc> = VectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, LogArc::new(1, 1, LogWeight(0.0), 1));
        fst.add_arc(0, LogArc::new(2, 2, LogWeight(6.0), 1));
        fst.set_final(1, LogWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        let mut out: VectorFst<LogArc> = VectorFst::new();
        let mut rng = Rng::new(11);
        rand_gen(
            &fst,
            &mut out,
            &mut rng,
            &RandGenOptions {
                npath: 400,
                weighted: true,
                ..RandGenOptions::new(LogProbArcSelector)
            },
        )
        .unwrap();

        // The likely branch was drawn far more often, so it weighs far less.
        let mut by_label: Vec<(i32, f32)> = out
            .arcs(out.start().unwrap())
            .map(|arc| (arc.ilabel(), arc.weight().value()))
            .collect();
        by_label.sort_by_key(|(label, _)| *label);
        assert!(!by_label.is_empty());
        if by_label.len() == 2 {
            assert!(
                by_label[0].1 < by_label[1].1,
                "label 1 should be the lighter: {by_label:?}"
            );
        } else {
            assert_eq!(by_label[0].0, 1, "only the likely branch was drawn");
        }
    }

    /// The log selector never picks an arc that weighs zero, because no path
    /// goes through it.
    #[test]
    fn a_zero_weight_arc_is_never_drawn() {
        let mut fst: VectorFst<LogArc> = VectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, LogArc::new(1, 1, LogWeight::one(), 1));
        fst.add_arc(0, LogArc::new(2, 2, LogWeight::zero(), 1));
        fst.set_final(1, LogWeight::one());
        fst.properties(K_FST_PROPERTIES, true);

        for seed in 0..40 {
            let mut out: VectorFst<LogArc> = VectorFst::new();
            let mut rng = Rng::new(seed);
            rand_gen(
                &fst,
                &mut out,
                &mut rng,
                &RandGenOptions {
                    npath: 20,
                    ..RandGenOptions::new(LogProbArcSelector)
                },
            )
            .unwrap();
            for state in out.states() {
                for arc in out.arcs(state) {
                    assert_ne!(
                        arc.ilabel(),
                        2,
                        "seed {seed}: the zero-weight arc was drawn"
                    );
                }
            }
        }
    }
}
