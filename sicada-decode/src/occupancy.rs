//! The alignment chain again, in the log semiring: what every alignment says,
//! not just the best one.
//!
//! [`align`](crate::align::align) walks the chain of
//! [`AlignChain`] in the tropical semiring, so its ⊕
//! is `min` and one path survives. Walking the same chain with the log
//! semiring's ⊕, which is `-log(e^-a + e^-b)` and so adds probabilities,
//! discards nothing: the forward pass ends holding the total probability of the
//! reference over *all* its alignments, and pairing it with a backward pass
//! gives the posterior of every transition, frame by frame.
//!
//! The graph is not restated here. Both passes read
//! `transitions_into` and its dual, which is
//! the single description of the chain's shape, so there is no way for the two
//! semirings to end up walking different graphs.
//!
//! What it is for:
//!
//! - **label priors.** A CTC model trained with them needs an estimate of how
//!   often each column is the right answer, and [`Occupancy::label_prior`] is
//!   that estimate taken over the alignment rather than over a hard decision.
//! - **diagnosis.** [`Occupancy::skip_posteriors`] gives, per phone, how much
//!   of the probability mass gave it up. That is the instrument the tropical
//!   answer cannot supply: a skip in [`Alignment::skipped`] is a decision, and
//!   a decision does not say whether it was close. Silent skips are the
//!   failure mode this whole module exists to catch, so the soft count is
//!   worth its cost.
//!
//! # What it costs
//!
//! The backward pass needs the forward scores, so where
//! [`align`](crate::align::align) keeps two rows and a packed traceback, this
//! keeps the whole `(T + 1) × (N + 1)` plane of them: 651 MB for a ten-minute
//! utterance against a 5 385-phone reference. That still fits, but it is 16
//! times the aligner, so [`occupancy`] is the second call to make and not the
//! first.
//!
//! [`Alignment::skipped`]: crate::align::Alignment::skipped

use sicada::arc::Arc;
use sicada::error::OpenFstError;

use crate::align::{AlignChain, column_read};
use crate::dense::{DenseFst, FromScore};
use crate::trellis::posteriors;

/// How the reference's probability is spread over the frames.
#[derive(Debug, Clone, PartialEq)]
pub struct Occupancy {
    /// `T × C`, row-major: the posterior of each column in each frame.
    posteriors: Vec<f32>,
    /// Per position, the expected number of frames sounding it.
    durations: Vec<f32>,
    /// Per position, the posterior that the alignment gave it up.
    skips: Vec<f32>,
    num_frames: usize,
    num_symbols: usize,
    cost: f32,
}

impl Occupancy {
    /// The number of frames.
    #[inline(always)]
    pub fn num_frames(&self) -> usize {
        self.num_frames
    }

    /// The number of columns the acoustic model scores.
    #[inline(always)]
    pub fn num_symbols(&self) -> usize {
        self.num_symbols
    }

    /// The posterior of each column in `frame`, which sums to one.
    ///
    /// # Panics
    ///
    /// If `frame` is past the last one.
    #[inline(always)]
    pub fn frame(&self, frame: usize) -> &[f32] {
        &self.posteriors[frame * self.num_symbols..(frame + 1) * self.num_symbols]
    }

    /// `-log P(reference | frames)`: the cost of the whole chain, over every
    /// alignment of it at once.
    ///
    /// Always at most [`Alignment::cost`](crate::align::Alignment::cost), which
    /// is the single best alignment's, and equal to it only when there is just
    /// one. The gap between them is how undecided the alignment is, and it is
    /// the honest number to compare two references with, since the best path's
    /// cost can be beaten by a reference that merely has one good alignment.
    #[inline(always)]
    pub fn cost(&self) -> f32 {
        self.cost
    }

    /// How often each column is the right answer, averaged over the frames.
    ///
    /// Sums to one. This is the estimate a CTC model trained with label priors
    /// wants: the count is taken over the whole alignment posterior rather than
    /// over a hard decision, so a frame the model is unsure about contributes
    /// to both readings in proportion.
    pub fn label_prior(&self) -> Vec<f32> {
        let mut prior = vec![0f64; self.num_symbols];
        for frame in self.posteriors.chunks_exact(self.num_symbols.max(1)) {
            for (total, &value) in prior.iter_mut().zip(frame) {
                *total += value as f64;
            }
        }
        let frames = self.num_frames.max(1) as f64;
        prior
            .into_iter()
            .map(|total| (total / frames) as f32)
            .collect()
    }

    /// The expected number of frames sounding each position.
    ///
    /// The soft counterpart of [`Alignment::spans`](crate::align::Alignment::spans),
    /// and it answers a question spans cannot: a phone whose span is one frame
    /// but whose expected duration is four is a boundary the acoustics did not
    /// decide, not a short phone.
    #[inline(always)]
    pub fn expected_durations(&self) -> &[f32] {
        &self.durations
    }

    /// Per position, how much of the probability took the skip transition into
    /// it.
    ///
    /// Zero everywhere unless the chain allows skipping. It makes a skip
    /// auditable: [`Alignment::skipped`](crate::align::Alignment::skipped)
    /// reports the decision, and this reports how close it was. A phone at 0.51
    /// was a coin toss, and one at 0.999 really is not in the audio.
    ///
    /// It is an upper bound on the posterior that a phone got no frames at all,
    /// and not quite the same thing. Skipping into `s_i` and then *holding*
    /// phone `i` is a path the chain allows, and it sounds a phone the skip gave
    /// up. It never wins in the tropical semiring, because waiting at `s_{i-1}`
    /// and committing a frame later reaches the same cell having sounded the
    /// same phone and costs exactly `skip(i)` less, so an [`Alignment`] never
    /// contains one. In the log semiring nothing is discarded, so those paths
    /// keep their share, which at a skip cost of several nats is a small one.
    ///
    /// [`Alignment`]: crate::align::Alignment
    #[inline(always)]
    pub fn skip_posteriors(&self) -> &[f32] {
        &self.skips
    }
}

/// Forward-backward over `chain` against `dense`, in the log semiring.
///
/// Returns `None` in the same cases [`align`](crate::align::align) does: a
/// reference too long for the frames, or one no path can complete.
///
/// # Errors
///
/// A phone naming a column the acoustic model does not have, or a matrix so
/// large that the forward plane does not fit in memory; see
/// [the module docs](self#what-it-costs) for how large that is.
pub fn occupancy<A>(
    chain: &AlignChain,
    dense: &DenseFst<'_, A>,
) -> Result<Option<Occupancy>, OpenFstError>
where
    A: Arc,
    A::Weight: FromScore,
{
    let num_symbols = dense.num_symbols();
    let num_frames = dense.num_frames();
    let num_phones = chain.num_phones();
    let trellis = chain.against(dense)?;

    let cells = num_frames.checked_mul(num_symbols).ok_or_else(|| {
        OpenFstError::InvalidOperation(format!(
            "occupancy: posteriors for {num_frames} frames of {num_symbols} symbols do not fit"
        ))
    })?;
    // Summed in double precision: a frame's mass arrives in up to `4N` pieces,
    // and the per-position totals in up to `4T` of them.
    let mut posterior = vec![0f64; cells];
    let mut durations = vec![0f64; num_phones];
    let mut skips = vec![0f64; num_phones];

    // The whole of what this module adds to the solver: which column each
    // transition read, and what it meant for the phone it landed on.
    let total = posteriors(&trellis, |taken, mass| {
        let column = column_read(chain, taken.code, taken.position);
        posterior[taken.frame * num_symbols + column as usize] += mass;
        if AlignChain::sounds(taken.code) {
            durations[taken.position - 1] += mass;
        } else if taken.code == AlignChain::SKIP {
            skips[taken.position - 1] += mass;
        }
    })?;
    let Some(cost) = total else {
        return Ok(None);
    };

    Ok(Some(Occupancy {
        posteriors: posterior.into_iter().map(|mass| mass as f32).collect(),
        durations: durations.into_iter().map(|value| value as f32).collect(),
        skips: skips.into_iter().map(|value| value as f32).collect(),
        num_frames,
        num_symbols,
        cost,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sicada::arc::StdArc;

    use crate::align::{Alignment, align};

    /// Blank plus three phones.
    const SYMBOLS: usize = 4;

    /// One frame of a path: the column it read, the position it sounded, and
    /// the position it gave up.
    #[derive(Clone, Copy)]
    struct Step {
        column: usize,
        sounded: Option<usize>,
        skipped: Option<usize>,
    }

    /// Every complete alignment, weighed by its probability.
    ///
    /// The definition of a forward-backward, written out: enumerate the paths,
    /// give each `e^-cost`, and normalise. Exponential, so only for the smallest
    /// cases, but it shares nothing with the recurrences under test, not even
    /// the idea of a recurrence.
    #[derive(Debug)]
    struct Enumerated {
        cost: f32,
        posteriors: Vec<f64>,
        durations: Vec<f64>,
        skips: Vec<f64>,
    }

    fn by_enumeration(
        chain: &AlignChain,
        dense: &DenseFst<'_, StdArc>,
        num_frames: usize,
    ) -> Option<Enumerated> {
        #[allow(clippy::too_many_arguments)]
        fn walk(
            chain: &AlignChain,
            dense: &DenseFst<'_, StdArc>,
            num_frames: usize,
            frame: usize,
            position: usize,
            cost: f32,
            path: &mut Vec<Step>,
            paths: &mut Vec<(f64, Vec<Step>)>,
        ) {
            if frame == num_frames {
                if position == chain.num_phones() {
                    paths.push(((-cost as f64).exp(), path.clone()));
                }
                return;
            }
            let scores = dense.frame(frame);
            let blank = chain.blank() as usize;
            let mut take = |step: Step, extra: f32, to: usize| {
                path.push(step);
                walk(
                    chain,
                    dense,
                    num_frames,
                    frame + 1,
                    to,
                    cost + extra,
                    path,
                    paths,
                );
                path.pop();
            };

            let silent = Step {
                column: blank,
                sounded: None,
                skipped: None,
            };
            take(silent, scores[blank], position);
            if position > 0 {
                let column = chain.phones()[position - 1] as usize;
                let step = Step {
                    column,
                    sounded: Some(position - 1),
                    skipped: None,
                };
                take(step, scores[column], position);
            }
            if position < chain.num_phones() {
                let column = chain.phones()[position] as usize;
                let step = Step {
                    column,
                    sounded: Some(position),
                    skipped: None,
                };
                take(step, scores[column], position + 1);

                let skip = chain.skip_costs()[position];
                if skip.is_finite() {
                    let step = Step {
                        skipped: Some(position),
                        ..silent
                    };
                    take(step, skip + scores[blank], position + 1);
                }
            }
        }

        let mut paths = Vec::new();
        walk(
            chain,
            dense,
            num_frames,
            0,
            0,
            0.0,
            &mut Vec::new(),
            &mut paths,
        );
        if paths.is_empty() {
            return None;
        }

        let total: f64 = paths.iter().map(|(weight, _)| weight).sum();
        let num_phones = chain.num_phones();
        let mut posteriors = vec![0f64; num_frames * SYMBOLS];
        let mut durations = vec![0f64; num_phones];
        let mut skips = vec![0f64; num_phones];
        for (weight, path) in &paths {
            let share = weight / total;
            for (frame, step) in path.iter().enumerate() {
                posteriors[frame * SYMBOLS + step.column] += share;
                if let Some(position) = step.sounded {
                    durations[position] += share;
                }
                if let Some(position) = step.skipped {
                    skips[position] += share;
                }
            }
        }
        Some(Enumerated {
            cost: -(total.ln() as f32),
            posteriors,
            durations,
            skips,
        })
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

        /// Costs in a range where several alignments have real mass, so the
        /// posteriors under test are not all zero and one.
        fn cost(&mut self) -> f32 {
            self.below(1 << 14) as f32 / 4096.0
        }
    }

    #[test]
    fn it_agrees_with_enumerating_every_alignment() {
        let mut rng = Rng(0x0CC0_9E37_79B9_7C15);
        let mut compared = 0;

        for round in 0..200 {
            let num_frames = 1 + rng.below(6);
            let num_phones = rng.below(num_frames.min(3) + 1);
            let phones: Vec<u32> = (0..num_phones)
                .map(|_| 1 + rng.below(SYMBOLS - 1) as u32)
                .collect();
            let chain = AlignChain::new(phones);
            let chain = if rng.below(2) == 0 {
                chain.with_uniform_skip_cost(rng.cost()).unwrap()
            } else {
                chain
            };

            let scores: Vec<f32> = (0..num_frames * SYMBOLS).map(|_| rng.cost()).collect();
            let dense = DenseFst::<StdArc>::new(&scores, num_frames, SYMBOLS).unwrap();

            let expected = by_enumeration(&chain, &dense, num_frames);
            let measured = occupancy(&chain, &dense).unwrap();

            match (expected, measured) {
                (None, None) => {}
                (Some(expected), Some(measured)) => {
                    compared += 1;
                    assert!(
                        (measured.cost() - expected.cost).abs() < 1e-3,
                        "round {round}: total {} against every path's {}",
                        measured.cost(),
                        expected.cost
                    );
                    for frame in 0..num_frames {
                        for column in 0..SYMBOLS {
                            let want = expected.posteriors[frame * SYMBOLS + column];
                            let got = measured.frame(frame)[column] as f64;
                            assert!(
                                (got - want).abs() < 1e-4,
                                "round {round}: frame {frame} column {column}, {got} against {want}"
                            );
                        }
                    }
                    for position in 0..chain.num_phones() {
                        assert!(
                            (measured.expected_durations()[position] as f64
                                - expected.durations[position])
                                .abs()
                                < 1e-4,
                            "round {round}: duration of position {position}"
                        );
                        assert!(
                            (measured.skip_posteriors()[position] as f64
                                - expected.skips[position])
                                .abs()
                                < 1e-4,
                            "round {round}: skip of position {position}"
                        );
                    }
                }
                (expected, measured) => {
                    panic!("round {round}: enumeration {expected:?}, occupancy {measured:?}")
                }
            }
        }

        assert!(compared > 150, "only {compared} rounds had an alignment");
    }

    #[test]
    fn every_frame_is_a_distribution() {
        let mut rng = Rng(0xABCD_1234_5678_9EF1);
        for _ in 0..50 {
            let num_frames = 2 + rng.below(20);
            let num_phones = rng.below(num_frames.min(8) + 1);
            let phones: Vec<u32> = (0..num_phones)
                .map(|_| 1 + rng.below(SYMBOLS - 1) as u32)
                .collect();
            let chain = AlignChain::new(phones).with_uniform_skip_cost(2.0).unwrap();
            let scores: Vec<f32> = (0..num_frames * SYMBOLS).map(|_| rng.cost()).collect();
            let dense = DenseFst::<StdArc>::new(&scores, num_frames, SYMBOLS).unwrap();

            let measured = occupancy(&chain, &dense).unwrap().expect("an occupancy");
            for frame in 0..num_frames {
                let mass: f32 = measured.frame(frame).iter().sum();
                assert!((mass - 1.0).abs() < 1e-4, "frame {frame} carries {mass}");
            }
            // Every frame either sounds a phone or does not, so the expected
            // durations and the blank's mass share out the frames between them.
            let sounded: f32 = measured.expected_durations().iter().sum();
            let silent: f32 = (0..num_frames)
                .map(|frame| measured.frame(frame)[chain.blank() as usize])
                .sum();
            assert!(
                (sounded + silent - num_frames as f32).abs() < 1e-2,
                "{sounded} sounding and {silent} silent, of {num_frames}"
            );
        }
    }

    /// The two semirings answer different questions, and the difference is the
    /// point of having both.
    #[test]
    fn the_total_is_over_every_alignment_not_the_best_one() {
        // Three frames, one phone: the phone can sound in any non-empty run, so
        // there are several alignments and the sum beats the best of them.
        let scores = [
            1.0, 0.5, 9.0, 9.0, //
            1.0, 0.5, 9.0, 9.0, //
            1.0, 0.5, 9.0, 9.0,
        ];
        let dense = DenseFst::<StdArc>::new(&scores, 3, SYMBOLS).unwrap();
        let chain = AlignChain::new(vec![1]);

        let best = align(&chain, &dense).unwrap().expect("an alignment");
        let all = occupancy(&chain, &dense).unwrap().expect("an occupancy");
        assert!(
            all.cost() < best.cost() - 0.1,
            "sum {} against best {}",
            all.cost(),
            best.cost()
        );

        // With a reference as long as the audio there is exactly one alignment,
        // and then the two agree.
        let chain = AlignChain::new(vec![1, 1, 1]);
        let best = align(&chain, &dense).unwrap().expect("an alignment");
        let all = occupancy(&chain, &dense).unwrap().expect("an occupancy");
        assert!(
            (all.cost() - best.cost()).abs() < 1e-5,
            "sum {} against best {}",
            all.cost(),
            best.cost()
        );
    }

    #[test]
    fn a_confident_model_puts_the_mass_on_the_alignment() {
        let mut scores = vec![20.0; 5 * SYMBOLS];
        for (frame, column) in [1usize, 1, 0, 2, 0].into_iter().enumerate() {
            scores[frame * SYMBOLS + column] = 0.0;
        }
        let dense = DenseFst::<StdArc>::new(&scores, 5, SYMBOLS).unwrap();
        let chain = AlignChain::new(vec![1, 2]);

        let alignment = align(&chain, &dense).unwrap().expect("an alignment");
        let measured = occupancy(&chain, &dense).unwrap().expect("an occupancy");

        for frame in 0..5 {
            let column = match alignment.sounding(frame) {
                Some(position) => chain.phones()[position] as usize,
                None => chain.blank() as usize,
            };
            assert!(
                measured.frame(frame)[column] > 0.99,
                "frame {frame}: {:?}",
                measured.frame(frame)
            );
        }
        assert!((measured.expected_durations()[0] - 2.0).abs() < 0.01);
        assert!((measured.expected_durations()[1] - 1.0).abs() < 0.01);
        assert!(measured.skip_posteriors().iter().all(|&mass| mass == 0.0));
    }

    /// The instrument the tropical answer cannot supply: how close the skip was.
    #[test]
    fn a_skip_that_is_a_coin_toss_shows_as_one() {
        // Sounding phone 2 in frame 1 costs exactly what giving it up costs.
        let scores = [
            10.0, 0.0, 10.0, 10.0, //
            0.0, 10.0, 4.0, 10.0,
        ];
        let dense = DenseFst::<StdArc>::new(&scores, 2, SYMBOLS).unwrap();
        let chain = AlignChain::new(vec![1, 2])
            .with_skip_costs(&[9.0, 4.0])
            .unwrap();

        let measured = occupancy(&chain, &dense).unwrap().expect("an occupancy");
        assert!(
            (measured.skip_posteriors()[1] - 0.5).abs() < 1e-3,
            "{:?}",
            measured.skip_posteriors()
        );
        assert!(measured.skip_posteriors()[0] < 1e-6, "no reason to skip it");

        // And the aligner, which has to decide, keeps the phone, so the
        // decision alone would not have shown the toss.
        let alignment = align(&chain, &dense).unwrap().expect("an alignment");
        assert!(alignment.skipped().is_empty());
    }

    #[test]
    fn the_label_prior_is_the_posterior_averaged_over_the_frames() {
        let scores = [
            1.0, 0.5, 9.0, 9.0, //
            1.0, 0.5, 9.0, 9.0, //
            1.0, 0.5, 9.0, 9.0,
        ];
        let dense = DenseFst::<StdArc>::new(&scores, 3, SYMBOLS).unwrap();
        let chain = AlignChain::new(vec![1]);
        let measured = occupancy(&chain, &dense).unwrap().expect("an occupancy");

        let prior = measured.label_prior();
        assert_eq!(prior.len(), SYMBOLS);
        assert!((prior.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        for (column, &averaged) in prior.iter().enumerate() {
            let by_hand: f32 = (0..3)
                .map(|frame| measured.frame(frame)[column])
                .sum::<f32>()
                / 3.0;
            assert!((averaged - by_hand).abs() < 1e-6);
        }
        // Columns the reference never names take none of it.
        assert_eq!(prior[2], 0.0);
        assert_eq!(prior[3], 0.0);
    }

    #[test]
    fn a_reference_longer_than_the_audio_has_no_occupancy() {
        let scores = vec![1.0; 2 * SYMBOLS];
        let dense = DenseFst::<StdArc>::new(&scores, 2, SYMBOLS).unwrap();
        assert_eq!(
            occupancy(&AlignChain::new(vec![1, 2, 3]), &dense).unwrap(),
            None
        );

        let err = occupancy(&AlignChain::new(vec![9]), &dense).unwrap_err();
        assert!(format!("{err}").contains("does not have"), "{err}");
    }

    #[test]
    fn an_empty_reference_is_all_blank() {
        let scores = [0.25, 9.0, 9.0, 9.0].repeat(3);
        let dense = DenseFst::<StdArc>::new(&scores, 3, SYMBOLS).unwrap();
        let measured = occupancy(&AlignChain::new(vec![]), &dense)
            .unwrap()
            .expect("an occupancy");

        assert!((measured.cost() - 0.75).abs() < 1e-5, "{}", measured.cost());
        assert!(measured.expected_durations().is_empty());
        for frame in 0..3 {
            assert!((measured.frame(frame)[0] - 1.0).abs() < 1e-6);
        }
    }

    /// The alignment and the occupancy have to be talking about the same
    /// frames, since a caller reads them side by side.
    #[test]
    fn it_lines_up_with_the_alignment_frame_for_frame() {
        let mut rng = Rng(0x5151_2727_3939_4B4B);
        for _ in 0..40 {
            let num_frames = 2 + rng.below(12);
            let num_phones = rng.below(num_frames.min(5) + 1);
            let phones: Vec<u32> = (0..num_phones)
                .map(|_| 1 + rng.below(SYMBOLS - 1) as u32)
                .collect();
            let chain = AlignChain::new(phones);
            let scores: Vec<f32> = (0..num_frames * SYMBOLS).map(|_| rng.cost()).collect();
            let dense = DenseFst::<StdArc>::new(&scores, num_frames, SYMBOLS).unwrap();

            let alignment: Alignment = align(&chain, &dense).unwrap().expect("an alignment");
            let measured = occupancy(&chain, &dense).unwrap().expect("an occupancy");
            assert_eq!(measured.num_frames(), alignment.num_frames());
            assert_eq!(measured.num_symbols(), SYMBOLS);
            assert_eq!(measured.expected_durations().len(), alignment.num_phones());
            // The best path is one of the paths, so it can never cost less than
            // all of them together.
            assert!(measured.cost() <= alignment.cost() + 1e-4);
        }
    }
}
