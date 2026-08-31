//! The exact solver forced alignment is built on, over any trellis of the same
//! shape.
//!
//! [`align`](crate::align::align) is one instance of something more general.
//! Strip the phones out of it and what is left is `T` frames against `N + 1`
//! positions, where every transition consumes exactly one frame and advances the
//! position by a bounded number of places, and the path starts at position 0 and
//! ends at position `N`. Nothing else about the chain matters to the search.
//!
//! That shape is why the search can be exact rather than pruned, and it is worth
//! not rewriting. This module holds the parts that are fiddly to get right: the
//! band of reachable cells and its off-by-ones, the traceback packed to the bits
//! a code actually needs, the two score rows and the discipline that lets them
//! be reused, and a numerically careful ⊕ in the log semiring. The part that is
//! *yours* is left to a trait.
//!
//! | fixed | free |
//! |---|---|
//! | one frame per transition | how many transitions there are |
//! | position never goes backwards | what each of them costs |
//! | an advance of at most [`Trellis::REACH`] | what it reads, and what it means |
//! | starts at 0, ends at `N` | whether the cost depends on the frame |
//!
//! # Writing one
//!
//! [`Trellis`] answers "what enters this cell, and what does it cost". The cost
//! is the *whole* cost, meaning a structural penalty and whatever the frame
//! charges for what the transition reads, already multiplied together, so a
//! penalty that varies by position, by frame, or by both needs no extra
//! machinery.
//!
//! ```
//! use sicada_decode::trellis::{Step, Trellis, best_path};
//!
//! /// A reference that must be sounded in order, one frame at a time, with no
//! /// silence and nothing skippable: the smallest trellis there is.
//! struct Rigid<'a> {
//!     scores: &'a [f32],
//!     num_symbols: usize,
//!     phones: &'a [u32],
//! }
//!
//! impl Trellis<2> for Rigid<'_> {
//!     type Frame<'a> = &'a [f32] where Self: 'a;
//!
//!     fn num_frames(&self) -> usize {
//!         self.scores.len() / self.num_symbols
//!     }
//!     fn num_positions(&self) -> usize {
//!         self.phones.len()
//!     }
//!     fn frame(&self, frame: usize) -> &[f32] {
//!         &self.scores[frame * self.num_symbols..(frame + 1) * self.num_symbols]
//!     }
//!
//!     fn steps_into(&self, frame: &[f32], position: usize) -> [Step; 2] {
//!         if position == 0 {
//!             // Nothing reaches `s_0` after the start: the reference has to
//!             // begin in the first frame.
//!             return [Step::ABSENT; 2];
//!         }
//!         let sounding = frame[self.phones[position - 1] as usize];
//!         // Hold this phone, or arrive at it. Listed best-first, which is the
//!         // tie-break: a phone is held rather than started again.
//!         [Step::new(0, sounding), Step::new(1, sounding)]
//!     }
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let scores = [
//!     9.0, 0.0, 9.0, // phone 1
//!     9.0, 0.0, 9.0, // still phone 1
//!     9.0, 9.0, 0.0, // phone 2
//! ];
//! let path = best_path(&Rigid { scores: &scores, num_symbols: 3, phones: &[1, 2] })?
//!     .expect("the reference fits");
//! assert_eq!(path.positions(), [1, 1, 2]);
//! assert_eq!(path.codes(), [1, 0, 1]); // arrive, hold, arrive
//! # Ok(())
//! # }
//! ```
//!
//! [`ReversibleTrellis`] adds the same transitions read backwards, which is all
//! [`posteriors`] needs. It has nothing to implement, because the backward
//! reading is derived from the forward one. Write it out only to make the
//! backward pass faster, and then put [`axioms::check`] in a test: a
//! forward-backward over two graphs that differ does not fail, it returns
//! numbers that look entirely reasonable.
//!
//! # What it will not do
//!
//! There is no beam here and no place to put one. The band is the set of cells
//! a complete path can stand in at all, so leaving out the rest costs nothing;
//! narrowing it further would be a search decision, and
//! [`align`](mod@crate::align) exists because that decision fails silently. See
//! that module for the measurements.

use std::ops::Range;

use sicada::error::OpenFstError;
use sicada::weight::Weight;
use sicada::weights::float_weight::LogWeight;

/// One transition of a [`Trellis`]: how far it moves, and what it costs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Step {
    /// How many positions it advances. Read into a cell it comes from
    /// `position - advance`; read out of one it goes to `position + advance`.
    /// Must not exceed [`Trellis::REACH`].
    pub advance: u8,
    /// The whole cost of taking it in this frame: a structural penalty and
    /// whatever the frame charges for what it reads, already multiplied
    /// together. Costs are negative log probabilities, so smaller is better.
    pub cost: f32,
}

impl Step {
    /// A transition that is not available at this cell.
    ///
    /// A trellis has the same number of transitions everywhere, which is the
    /// meaning of [`Trellis`]'s `DEGREE`, so an edge case is expressed by a
    /// transition costing infinity rather than by returning fewer of them. It
    /// is also the only way to say "this one would run off the start", which
    /// the solver relies on: a step whose `advance` exceeds its cell's position
    /// **must** be absent.
    pub const ABSENT: Self = Self {
        advance: 0,
        cost: f32::INFINITY,
    };

    /// A transition advancing `advance` positions at cost `cost`.
    #[inline(always)]
    pub const fn new(advance: u8, cost: f32) -> Self {
        Self { advance, cost }
    }
}

/// `T` frames against `N + 1` positions, one frame consumed per transition.
///
/// `DEGREE` is how many transitions enter every cell, four for the chain
/// [`align`](mod@crate::align) uses. **They are listed best-first**: the solver
/// keeps the first of them that is strictly better than what it has, so the
/// order is the tie-break, and a trellis states its own by the order it lists
/// them in.
///
/// See [the module docs](self) for a worked implementation.
pub trait Trellis<const DEGREE: usize> {
    /// The most positions any one transition advances. One for a chain, where a
    /// frame either holds its position or moves to the next; more for a trellis
    /// that can pass over several positions at once, such as one that gives up
    /// a whole word.
    ///
    /// It sets how wide the band has to be, so an overstated `REACH` costs
    /// work while an understated one is a contract violation.
    const REACH: u8 = 1;

    /// What one frame's transitions are read from, usually a row of acoustic
    /// scores.
    ///
    /// Taken once per frame rather than once per cell, which for a matrix means
    /// the row is sliced `T` times rather than `T × N` times.
    type Frame<'a>: Copy
    where
        Self: 'a;

    /// The number of frames to be accounted for.
    fn num_frames(&self) -> usize;

    /// The last position. There are `num_positions() + 1` cells in a frame,
    /// `s_0` through `s_N`; a path starts at `s_0` and has to finish at `s_N`.
    fn num_positions(&self) -> usize;

    /// One frame's scores.
    fn frame(&self, frame: usize) -> Self::Frame<'_>;

    /// The `DEGREE` transitions entering `position`, best-first.
    ///
    /// A transition that does not exist at this cell is [`Step::ABSENT`], and so
    /// is one that would come from before `s_0`, since the solver indexes
    /// `position - advance` without checking it.
    fn steps_into(&self, frame: Self::Frame<'_>, position: usize) -> [Step; DEGREE];
}

/// A [`Trellis`] that can also be read backwards, as a forward-backward
/// requires.
///
/// **There is nothing to implement.** The backward reading is derived from the
/// forward one, so the two cannot disagree:
///
/// ```
/// # use sicada_decode::trellis::{ReversibleTrellis, Step, Trellis};
/// # struct Chain;
/// # impl Trellis<2> for Chain {
/// #     type Frame<'a> = ();
/// #     fn num_frames(&self) -> usize { 1 }
/// #     fn num_positions(&self) -> usize { 1 }
/// #     fn frame(&self, _: usize) {}
/// #     fn steps_into(&self, _: (), p: usize) -> [Step; 2] {
/// #         if p == 0 { [Step::new(0, 1.0), Step::ABSENT] }
/// #         else { [Step::new(0, 1.0), Step::new(1, 1.0)] }
/// #     }
/// # }
/// impl ReversibleTrellis<2> for Chain {}
/// ```
///
/// Overriding it buys speed, since the derived reading asks
/// [`steps_into`](Trellis::steps_into) once per advance where a written one
/// answers in a single call. It is also the only way the two readings can come
/// apart, and that matters more than it looks: [`posteriors`] over two different
/// graphs does not fail, it returns numbers that look entirely reasonable and
/// are wrong. An override is therefore a claim, and [`axioms::check`] is how
/// that claim is checked.
pub trait ReversibleTrellis<const DEGREE: usize>: Trellis<DEGREE> {
    /// The `DEGREE` transitions leaving `position`, in the same order
    /// [`steps_into`](Trellis::steps_into) lists them.
    ///
    /// A transition running past `s_N` is [`Step::ABSENT`]; unlike the forward
    /// direction the solver does bound the target, because it has to anyway.
    ///
    /// The default is [`derive_steps_out_of`]. Override it only to make the
    /// backward pass faster, and put [`axioms::check`] in a test when you do.
    fn steps_out_of(&self, frame: Self::Frame<'_>, position: usize) -> [Step; DEGREE] {
        derive_steps_out_of(self, frame, position)
    }
}

/// The transitions leaving `position`, read off the ones entering the cells
/// they could reach.
///
/// A transition coded `c` leaves `position` for `position + a` exactly when the
/// cell `a` along says code `c` arrived from `a` back. So asking each cell
/// within [`REACH`](Trellis::REACH) recovers the backward reading from the
/// forward one, which makes [`ReversibleTrellis`]'s default correct rather than
/// merely plausible.
///
/// # Panics
///
/// In debug builds, if one code leaves `position` by two different advances. The
/// trellis is then ambiguous, because the code names two transitions out of one
/// cell, and no backward reading of it exists. [`axioms::check`] reports it in
/// any build.
pub fn derive_steps_out_of<const DEGREE: usize, T>(
    trellis: &T,
    frame: T::Frame<'_>,
    position: usize,
) -> [Step; DEGREE]
where
    T: Trellis<DEGREE> + ?Sized,
{
    let mut out = [Step::ABSENT; DEGREE];
    let last = trellis.num_positions();
    for advance in 0..=usize::from(T::REACH) {
        let to = position + advance;
        if to > last {
            break;
        }
        for (code, step) in trellis.steps_into(frame, to).iter().enumerate() {
            // `Step::ABSENT` also advances zero, so a finite cost is what tells
            // a real transition from a missing one.
            if usize::from(step.advance) == advance && step.cost.is_finite() {
                debug_assert!(
                    !out[code].cost.is_finite(),
                    "transition {code} leaves position {position} by two different advances"
                );
                out[code] = *step;
            }
        }
    }
    out
}

/// The cells a complete path can stand in after `frame` frames.
///
/// Every transition consumes one frame and advances at most `reach` positions,
/// so reaching `s_N` from `s_0` in `T` frames pins the position after `t` of
/// them to `max(0, N - reach(T - t)) ..= min(reach · t, N)`.
///
/// This is not a beam. Outside it there is no complete path at all, so the
/// cells left out could not have contributed. It is exact when a trellis uses
/// every advance from 0 to `reach`; one that uses only some of them leaves some
/// cells in the band unreachable, and they carry their infinities harmlessly.
#[inline(always)]
pub fn band(frame: usize, num_frames: usize, num_positions: usize, reach: usize) -> Range<usize> {
    let left = num_frames - frame.min(num_frames);
    let lo = num_positions.saturating_sub(reach.saturating_mul(left));
    let hi = reach.saturating_mul(frame).min(num_positions);
    lo..hi + 1
}

/// The best path through a trellis: which transition each frame took.
#[derive(Debug, Clone, PartialEq)]
pub struct Path {
    codes: Vec<u8>,
    positions: Vec<u32>,
    cost: f32,
}

impl Path {
    /// The number of frames, which is the number of transitions taken.
    #[inline(always)]
    pub fn num_frames(&self) -> usize {
        self.codes.len()
    }

    /// Which transition each frame took, as an index into what
    /// [`Trellis::steps_into`] returns.
    #[inline(always)]
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    /// The position each frame landed in. The last is `N`.
    #[inline(always)]
    pub fn positions(&self) -> &[u32] {
        &self.positions
    }

    /// The path's total cost: every transition's, multiplied together.
    #[inline(always)]
    pub fn cost(&self) -> f32 {
        self.cost
    }
}

/// One transition of a trellis, as [`posteriors`] visits it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Transition {
    /// The frame it consumed.
    pub frame: usize,
    /// The position it landed in. This is the target, so that a caller can ask
    /// what was read there without also tracking the advance.
    pub position: usize,
    /// Which transition it was, as an index into what
    /// [`Trellis::steps_into`] returns.
    pub code: u8,
}

/// The best path through `trellis`, exactly.
///
/// Returns `None` when no path completes: a reference longer than the frames
/// can carry, or one every route through which costs infinity.
///
/// # Errors
///
/// A degree of zero or above 256, which no code could name; or a trellis so
/// large that its traceback does not fit in memory, reported rather than
/// attempted.
pub fn best_path<const DEGREE: usize, T>(trellis: &T) -> Result<Option<Path>, OpenFstError>
where
    T: Trellis<DEGREE> + ?Sized,
{
    let num_frames = trellis.num_frames();
    let num_positions = trellis.num_positions();
    let reach = usize::from(T::REACH).max(1);
    check_degree(DEGREE)?;
    if band(0, num_frames, num_positions, reach).is_empty() {
        // Not even the start is on a complete path: there are more positions
        // than the frames can advance through.
        return Ok(None);
    }

    let mut trace = Traceback::new(num_frames, num_positions, DEGREE)?;
    // Padded in front by `reach`, so that `position - advance` is in range for
    // every step a trellis may legally return and the inner loop needs no
    // guard. The padding holds infinity for the whole run.
    let mut cur = vec![f32::INFINITY; num_positions + 1 + reach];
    let mut next = cur.clone();
    cur[reach] = 0.0;

    for t in 0..num_frames {
        let frame = trellis.frame(t);
        let cells = band(t + 1, num_frames, num_positions, reach);
        let mut row = trace.row(t);

        for i in cells {
            let steps = trellis.steps_into(frame, i);
            let (mut best, mut code) = (f32::INFINITY, 0u8);
            for (candidate, step) in steps.iter().enumerate() {
                debug_assert!(
                    !step.cost.is_finite() || usize::from(step.advance) <= i.min(reach),
                    "a step advancing {} into position {i} is outside REACH or before the start",
                    step.advance
                );
                let total = cur[reach + i - usize::from(step.advance)] + step.cost;
                if total < best {
                    (best, code) = (total, candidate as u8);
                }
            }
            next[reach + i] = best;
            row.put(i, code);
        }

        row.finish();
        // Cells outside the band keep the infinity they were built with. The
        // band never writes them, and the next frame's never reads below its
        // own predecessor's or above a cell no frame has reached yet.
        std::mem::swap(&mut cur, &mut next);
    }

    let cost = cur[reach + num_positions];
    if !cost.is_finite() {
        return Ok(None);
    }

    let mut codes = vec![0u8; num_frames];
    let mut positions = vec![0u32; num_frames];
    let mut at = num_positions;
    for t in (0..num_frames).rev() {
        let code = trace.get(t, at);
        codes[t] = code;
        positions[t] = at as u32;
        let advance = trellis.steps_into(trellis.frame(t), at)[usize::from(code)].advance;
        at -= usize::from(advance);
    }
    debug_assert_eq!(at, 0, "the traceback left the trellis");

    Ok(Some(Path {
        codes,
        positions,
        cost,
    }))
}

/// Forward-backward over `trellis` in the log semiring: the posterior of every
/// transition, and the total.
///
/// `visit` is called once per transition carrying appreciable probability, with
/// that probability. What to do with it is the caller's: sum it by column for a
/// label prior, by position for an expected duration, by code for how often
/// each transition is taken. Transitions below [`NEGLIGIBLE`] are not visited,
/// so a caller must not count visits, only weigh them.
///
/// Returns the total cost, meaning `-log` of the probability of *all* paths,
/// which is at most [`Path::cost`] and equal to it only when there is one path.
/// Returns `None` in the same cases [`best_path`] does.
///
/// # Errors
///
/// As [`best_path`], with the forward plane in place of the traceback. It is
/// the larger of the two: `(T + 1) × (N + 1)` floats, against two bits a cell.
pub fn posteriors<const DEGREE: usize, T>(
    trellis: &T,
    mut visit: impl FnMut(Transition, f64),
) -> Result<Option<f32>, OpenFstError>
where
    T: ReversibleTrellis<DEGREE> + ?Sized,
{
    let num_frames = trellis.num_frames();
    let num_positions = trellis.num_positions();
    let reach = usize::from(T::REACH).max(1);
    check_degree(DEGREE)?;
    if band(0, num_frames, num_positions, reach).is_empty() {
        return Ok(None);
    }

    // Rows are padded in front exactly as `best_path`'s are.
    let width = num_positions + 1 + reach;
    let plane = (num_frames + 1).checked_mul(width).ok_or_else(|| {
        OpenFstError::InvalidOperation(format!(
            "posteriors: a forward plane for {num_frames} frames of {num_positions} positions \
             does not fit"
        ))
    })?;
    let mut alpha = vec![f32::INFINITY; plane];
    alpha[reach] = LogWeight::one().0;

    // Forward. The aligner's recurrence but for the ⊕: every transition into a
    // cell contributes, rather than the best one winning.
    for t in 0..num_frames {
        let frame = trellis.frame(t);
        let (done, rest) = alpha.split_at_mut((t + 1) * width);
        let prev = &done[t * width..];
        for i in band(t + 1, num_frames, num_positions, reach) {
            let steps = trellis.steps_into(frame, i);
            let mut terms = [f32::INFINITY; DEGREE];
            for (term, step) in terms.iter_mut().zip(steps.iter()) {
                *term = prev[reach + i - usize::from(step.advance)] + step.cost;
            }
            rest[reach + i] = log_sum(&terms).sum;
        }
    }

    let total = alpha[num_frames * width + reach + num_positions];
    if !total.is_finite() {
        return Ok(None);
    }

    // Backward, visiting as it goes. A transition's posterior is
    // `alpha[t][from] ⊗ weight ⊗ beta[t + 1][to] ⊘ total`, and those are the
    // terms the backward recurrence forms anyway, so the two share a loop.
    let mut beta_next = vec![f32::INFINITY; num_positions + 1];
    let mut beta_cur = beta_next.clone();
    beta_next[num_positions] = LogWeight::one().0;

    for t in (0..num_frames).rev() {
        let frame = trellis.frame(t);
        let alpha_row = &alpha[t * width..(t + 1) * width];
        // A cell outside the next frame's band is on no complete path, so
        // `beta_next` holds nothing meaningful there. Bounding the targets is
        // what lets the two rows be reused without being cleared.
        let reachable = band(t + 1, num_frames, num_positions, reach);

        for i in band(t, num_frames, num_positions, reach) {
            let steps = trellis.steps_out_of(frame, i);
            let mut terms = [f32::INFINITY; DEGREE];
            for (term, step) in terms.iter_mut().zip(steps.iter()) {
                let to = i + usize::from(step.advance);
                if reachable.contains(&to) {
                    *term = step.cost + beta_next[to];
                }
            }
            let folded = log_sum(&terms);
            beta_cur[i] = folded.sum;

            // Everything through this cell, over everything at all. The
            // transitions share the factor that does not depend on which one is
            // taken, so it is exponentiated once and the shares, which the ⊕ has
            // already formed, carry the rest. When the whole cell is too far
            // down to move an `f32`, none of its transitions can be.
            let cell = alpha_row[reach + i] + folded.pivot - total;
            if cell > NEGLIGIBLE {
                continue;
            }
            let scale = f64::from(-cell).exp();
            for (code, &share) in folded.shares.iter().enumerate() {
                if share == 0.0 {
                    continue;
                }
                visit(
                    Transition {
                        frame: t,
                        position: i + usize::from(steps[code].advance),
                        code: code as u8,
                    },
                    scale * share,
                );
            }
        }

        std::mem::swap(&mut beta_cur, &mut beta_next);
    }

    debug_assert!(
        (beta_next[0] - total).abs() < 1e-2 * total.abs().max(1.0),
        "the backward pass ended at {} where the forward one ended at {total}",
        beta_next[0]
    );

    Ok(Some(total))
}

/// Checks that a trellis obeys the contract the solvers rely on.
///
/// This is `sicada::weight::axioms` for [`Trellis`]: the laws
/// an implementation asserts by existing, written as something that can be run.
/// Put `check` in a test for every trellis you write. The conditions it checks
/// are ones the solvers *assume*, so breaking one does not produce an error, it
/// produces an answer.
///
/// It is not behind a feature. A contract you have to switch on is not one, and
/// the cost of leaving it available is nothing: it is generic, so a program
/// that never calls it never compiles it.
pub mod axioms {
    use super::*;

    /// Checks `trellis` over every frame, position and code.
    ///
    /// Walks the whole trellis, so give it a small one; a few frames against a
    /// few positions exercises every branch an implementation has.
    ///
    /// What it checks:
    ///
    /// 1. **[`REACH`](Trellis::REACH) is honest.** No transition advances
    ///    further than it says. The band is built from `REACH`, so a transition
    ///    that outruns it lands in a cell the search never considered.
    /// 2. **Nothing reaches back past the start.** A transition into position
    ///    `j` advancing more than `j` must be [`Step::ABSENT`]; [`best_path`]
    ///    subtracts without checking.
    /// 3. **No code is ambiguous.** One code names at most one transition out
    ///    of a cell, or there is no backward reading to have.
    /// 4. **The two readings are one graph.**
    ///    [`steps_out_of`](ReversibleTrellis::steps_out_of) agrees with
    ///    [`derive_steps_out_of`] everywhere. This is the one that matters and
    ///    the one that cannot be seen from the outside: [`posteriors`] over a
    ///    forward and a backward graph that differ returns numbers rather than
    ///    an error.
    ///
    /// # Panics
    ///
    /// On the first violation, naming the frame, position and code.
    pub fn check<const DEGREE: usize, T>(trellis: &T)
    where
        T: ReversibleTrellis<DEGREE> + ?Sized,
    {
        let reach = usize::from(T::REACH);
        assert!(reach >= 1, "a trellis whose REACH is zero can never finish");
        let last = trellis.num_positions();

        for f in 0..trellis.num_frames() {
            let frame = trellis.frame(f);

            for position in 0..=last {
                for (code, step) in trellis.steps_into(frame, position).iter().enumerate() {
                    if !step.cost.is_finite() {
                        continue;
                    }
                    let advance = usize::from(step.advance);
                    assert!(
                        advance <= reach,
                        "frame {f}: transition {code} into position {position} advances \
                         {advance}, past a REACH of {reach}"
                    );
                    assert!(
                        advance <= position,
                        "frame {f}: transition {code} into position {position} advances \
                         {advance}, from before the start, so it has to be Step::ABSENT there"
                    );
                }

                // One code, one transition out of a cell.
                for code in 0..DEGREE {
                    let leaving: Vec<usize> = (0..=reach)
                        .filter(|advance| position + advance <= last)
                        .filter(|&advance| {
                            let step = trellis.steps_into(frame, position + advance)[code];
                            usize::from(step.advance) == advance && step.cost.is_finite()
                        })
                        .collect();
                    assert!(
                        leaving.len() <= 1,
                        "frame {f}: transition {code} leaves position {position} by advances \
                         {leaving:?}, so it names more than one transition"
                    );
                }

                // And the reading a caller wrote is the one the forward
                // direction implies.
                let written = trellis.steps_out_of(frame, position);
                let derived = derive_steps_out_of(trellis, frame, position);
                for code in 0..DEGREE {
                    let (written, derived) = (written[code], derived[code]);
                    let agree = written == derived
                        || (!written.cost.is_finite() && !derived.cost.is_finite());
                    assert!(
                        agree,
                        "frame {f}: transition {code} out of position {position} reads \
                         {written:?} backwards but {derived:?} forwards"
                    );
                }
            }
        }
    }
}

fn check_degree(degree: usize) -> Result<(), OpenFstError> {
    if degree == 0 || degree > 256 {
        return Err(OpenFstError::InvalidOperation(format!(
            "trellis: a degree of {degree} cannot be named by a code"
        )));
    }
    Ok(())
}

/// Where a term stops being able to change an `f32`.
///
/// `e^-40` is 4e-18. Added to a total of one it moves nothing an `f32` holds,
/// since the type resolves 6e-8, and even the `DEGREE · T · N` such terms a
/// ten-minute utterance can produce come to 3e-9 between them. In the log
/// domain the same gap makes ⊕ its own smaller argument to within 4e-18 nats,
/// against an `f32` step of 8e-6 at the magnitudes these costs reach.
///
/// This is emphatically not a beam. It decides nothing, no answer moves if it
/// is changed, and it cannot be tuned into being wrong; it is the point past
/// which the arithmetic has already stopped. Everything above it is summed.
pub const NEGLIGIBLE: f32 = 40.0;

/// The log semiring's ⊕ over the transitions into one cell, and the share each
/// of them takes of the result.
#[derive(Debug, Clone, Copy)]
struct Folded<const DEGREE: usize> {
    /// `-log Σ e^-term`: ⊕ over all of them.
    sum: f32,
    /// The smallest term, which the shares are measured against. Keeping it is
    /// what lets a caller rebuild any one term's contribution as
    /// `e^-pivot × share` rather than exponentiating a second time.
    pivot: f32,
    /// `e^-(term - pivot)` for each term, or zero for one that was dropped.
    shares: [f64; DEGREE],
}

/// ⊕ over `terms`, folded on the smallest rather than in pairs.
///
/// This is [`LogWeight::plus`] across them, the same semiring and the same
/// value, but not folded pairwise. Pairwise costs a logarithm per pair to get
/// back into the log domain, only to leave it again for the next one; pivoting
/// on the smallest instead exponentiates each term once and takes a single
/// logarithm at the end. Both passes do this per cell of a `T × N` plane, so it is most
/// of the arithmetic there.
///
/// SICADA-OPT: a term more than [`NEGLIGIBLE`] above the pivot is dropped
/// rather than exponentiated, and so is the pivot's own `e^0`. Upstream's
/// `LogWeight::plus`, which sees two arguments and knows nothing of the fold it
/// is part of, can do neither.
#[inline(always)]
fn log_sum<const DEGREE: usize>(terms: &[f32; DEGREE]) -> Folded<DEGREE> {
    let mut pivot = f32::INFINITY;
    for &term in terms {
        if term < pivot {
            pivot = term;
        }
    }
    if !pivot.is_finite() {
        return Folded {
            sum: f32::INFINITY,
            pivot: f32::INFINITY,
            shares: [0.0; DEGREE],
        };
    }

    let mut shares = [0f64; DEGREE];
    // The pivot contributes exactly one, so the remainder is the argument the
    // logarithm needs, and `ln_1p` of it stays accurate when there is no
    // remainder at all.
    // Only one term may claim that one: a second sitting at the same cost is an
    // ordinary term whose share happens to be `e^0`.
    let mut rest = 0f64;
    let mut claimed = false;
    for (share, &term) in shares.iter_mut().zip(terms) {
        let above = term - pivot;
        if above == 0.0 && !claimed {
            claimed = true;
            *share = 1.0;
        } else if above <= NEGLIGIBLE {
            *share = f64::from(-above).exp();
            rest += *share;
        }
    }
    Folded {
        sum: pivot - rest.ln_1p() as f32,
        pivot,
        shares,
    }
}

/// Which transition each cell took, packed to the bits a code needs.
///
/// SICADA-OPT: a byte a cell is the obvious layout and what k2 spends. Four
/// transitions fit in two bits, so a ten-minute utterance against a
/// 5 385-phone reference costs 41 MB rather than 163 MB. The plane is written
/// once per cell and read once, so this is the memory traffic an alignment
/// costs.
struct Traceback {
    plane: Vec<u8>,
    stride: usize,
    bits: u32,
    per_byte: usize,
}

/// The widths a code packs into. Only divisors of 8, so a cell never straddles
/// two bytes.
const fn code_bits(degree: usize) -> u32 {
    match degree {
        0..=2 => 1,
        3..=4 => 2,
        5..=16 => 4,
        _ => 8,
    }
}

impl Traceback {
    fn new(num_frames: usize, num_positions: usize, degree: usize) -> Result<Self, OpenFstError> {
        let bits = code_bits(degree);
        let per_byte = 8 / bits as usize;
        let stride = num_positions / per_byte + 1;
        let cells = num_frames.checked_mul(stride).ok_or_else(|| {
            OpenFstError::InvalidOperation(format!(
                "trellis: a traceback for {num_frames} frames of {num_positions} positions does \
                 not fit"
            ))
        })?;
        Ok(Self {
            plane: vec![0u8; cells],
            stride,
            bits,
            per_byte,
        })
    }

    /// Writes one frame's codes, which the solver produces in position order.
    #[inline(always)]
    fn row(&mut self, frame: usize) -> RowWriter<'_> {
        RowWriter {
            row: &mut self.plane[frame * self.stride..(frame + 1) * self.stride],
            bits: self.bits,
            per_byte: self.per_byte,
            packed: 0,
            at: 0,
            pending: false,
        }
    }

    #[inline(always)]
    fn get(&self, frame: usize, position: usize) -> u8 {
        let byte = self.plane[frame * self.stride + position / self.per_byte];
        let mask = (1u16 << self.bits) - 1;
        (byte >> (self.bits * (position % self.per_byte) as u32)) & mask as u8
    }
}

/// Accumulates a byte of codes before storing it.
///
/// Cells arrive in order, so a byte is filled and written once rather than read
/// back out of the plane to have one cell updated. Codes below the band's start
/// are left zero, which the traceback never reads.
struct RowWriter<'a> {
    row: &'a mut [u8],
    bits: u32,
    per_byte: usize,
    packed: u8,
    at: usize,
    pending: bool,
}

impl RowWriter<'_> {
    #[inline(always)]
    fn put(&mut self, position: usize, code: u8) {
        let within = position % self.per_byte;
        self.packed |= code << (self.bits * within as u32);
        self.at = position / self.per_byte;
        self.pending = true;
        if within == self.per_byte - 1 {
            self.row[self.at] = self.packed;
            self.packed = 0;
            self.pending = false;
        }
    }

    #[inline(always)]
    fn finish(self) {
        if self.pending {
            self.row[self.at] = self.packed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chain of `align`, written out again against a plain matrix, so that
    /// this module's tests do not depend on that one's.
    ///
    /// Codes, best-first: hold the blank, hold the phone, commit, skip.
    struct Chain<'a> {
        scores: &'a [f32],
        symbols: usize,
        phones: &'a [u32],
        skip: f32,
    }

    const HOLD_BLANK: u8 = 0;
    const HOLD_PHONE: u8 = 1;
    const COMMIT: u8 = 2;
    const SKIP: u8 = 3;

    impl Trellis<4> for Chain<'_> {
        type Frame<'a>
            = &'a [f32]
        where
            Self: 'a;

        fn num_frames(&self) -> usize {
            self.scores.len() / self.symbols
        }
        fn num_positions(&self) -> usize {
            self.phones.len()
        }
        fn frame(&self, frame: usize) -> &[f32] {
            &self.scores[frame * self.symbols..(frame + 1) * self.symbols]
        }

        fn steps_into(&self, frame: &[f32], position: usize) -> [Step; 4] {
            let blank = Step::new(0, frame[0]);
            if position == 0 {
                return [blank, Step::ABSENT, Step::ABSENT, Step::ABSENT];
            }
            let phone = frame[self.phones[position - 1] as usize];
            [
                blank,
                Step::new(0, phone),
                Step::new(1, phone),
                Step::new(1, self.skip + frame[0]),
            ]
        }
    }

    impl ReversibleTrellis<4> for Chain<'_> {
        fn steps_out_of(&self, frame: &[f32], position: usize) -> [Step; 4] {
            let blank = Step::new(0, frame[0]);
            let hold = if position > 0 {
                Step::new(0, frame[self.phones[position - 1] as usize])
            } else {
                Step::ABSENT
            };
            let (commit, skip) = if position < self.phones.len() {
                (
                    Step::new(1, frame[self.phones[position] as usize]),
                    Step::new(1, self.skip + frame[0]),
                )
            } else {
                (Step::ABSENT, Step::ABSENT)
            };
            [blank, hold, commit, skip]
        }
    }

    fn chain<'a>(scores: &'a [f32], phones: &'a [u32]) -> Chain<'a> {
        Chain {
            scores,
            symbols: 4,
            phones,
            skip: f32::INFINITY,
        }
    }

    #[test]
    fn it_finds_the_path_the_scores_ask_for() {
        // phone 1, phone 1, blank, phone 2.
        let scores = [
            9.0, 0.0, 9.0, 9.0, //
            9.0, 0.0, 9.0, 9.0, //
            0.0, 9.0, 9.0, 9.0, //
            9.0, 9.0, 0.0, 9.0,
        ];
        let path = best_path(&chain(&scores, &[1, 2]))
            .unwrap()
            .expect("a path");
        assert_eq!(path.positions(), [1, 1, 1, 2]);
        assert_eq!(path.codes(), [COMMIT, HOLD_PHONE, HOLD_BLANK, COMMIT]);
        assert!(path.cost().abs() < 1e-6);
        assert_eq!(path.num_frames(), 4);
    }

    #[test]
    fn a_reference_the_frames_cannot_carry_has_no_path() {
        let scores = [0.0; 8];
        assert_eq!(best_path(&chain(&scores, &[1, 2, 3])).unwrap(), None);
        assert_eq!(
            posteriors(&chain(&scores, &[1, 2, 3]), |_, _| {}).unwrap(),
            None
        );
    }

    /// The order the transitions are listed in is the tie-break, and callers
    /// depend on it: a skip listed last never wins one.
    #[test]
    fn the_order_transitions_are_listed_in_is_the_tie_break() {
        // Every column costs the same, so all four transitions tie wherever
        // they are all available, including the skip, which is free here.
        let scores = [1.0; 12];
        let mut chain = chain(&scores, &[1]);
        chain.skip = 0.0;
        let path = best_path(&chain).unwrap().expect("a path");

        assert!(
            !path.codes().contains(&SKIP),
            "a tie must not give up a phone"
        );
        // The preference is read at the cell being computed, so what it prefers
        // is the path that was *already there*. On a tie that resolves to
        // arriving as early as the band allows and waiting, not to putting the
        // arrival off.
        assert_eq!(path.codes(), [COMMIT, HOLD_BLANK, HOLD_BLANK]);
        assert_eq!(path.positions(), [1, 1, 1]);
        // And among ways of standing still, silence beats sounding, which is
        // what keeps a phone's span down to the frames that argue for it.
        assert!(!path.codes().contains(&HOLD_PHONE));
    }

    /// What the module is for: a caller's own topology, solved without its own
    /// solver. This one gives up a *word*, three positions at once, which needs
    /// a reach of more than one.
    #[test]
    fn a_trellis_that_advances_more_than_one_position() {
        struct Words<'a> {
            scores: &'a [f32],
            phones: &'a [u32],
            word: usize,
            give_up: f32,
        }

        impl Trellis<3> for Words<'_> {
            const REACH: u8 = 3;
            type Frame<'a>
                = &'a [f32]
            where
                Self: 'a;

            fn num_frames(&self) -> usize {
                self.scores.len() / 4
            }
            fn num_positions(&self) -> usize {
                self.phones.len()
            }
            fn frame(&self, frame: usize) -> &[f32] {
                &self.scores[frame * 4..(frame + 1) * 4]
            }

            fn steps_into(&self, frame: &[f32], position: usize) -> [Step; 3] {
                let blank = Step::new(0, frame[0]);
                if position == 0 {
                    return [blank, Step::ABSENT, Step::ABSENT];
                }
                let phone = Step::new(1, frame[self.phones[position - 1] as usize]);
                // A whole word given up at once, landing on a word boundary.
                let word = if position >= self.word && position.is_multiple_of(self.word) {
                    Step::new(self.word as u8, self.give_up + frame[0])
                } else {
                    Step::ABSENT
                };
                [blank, phone, word]
            }
        }

        // Six frames, two words of three phones. The audio says only the first
        // word; the second has no evidence anywhere.
        let mut scores = vec![9.0f32; 6 * 4];
        for (frame, column) in [1usize, 2, 3, 0, 0, 0].into_iter().enumerate() {
            scores[frame * 4 + column] = 0.0;
        }
        let words = Words {
            scores: &scores,
            phones: &[1, 2, 3, 1, 2, 3],
            word: 3,
            give_up: 1.0,
        };

        let path = best_path(&words).unwrap().expect("a path");
        assert_eq!(path.num_frames(), 6);
        // The first word is sounded, then the second is given up in one frame
        // and the rest is silence.
        assert_eq!(path.positions(), [1, 2, 3, 6, 6, 6]);
        assert_eq!(path.codes(), [1, 1, 1, 2, 0, 0]);
        assert!((path.cost() - 1.0).abs() < 1e-6, "{}", path.cost());
    }

    #[test]
    fn a_degree_no_code_could_name_is_reported() {
        struct Nothing;
        impl Trellis<0> for Nothing {
            type Frame<'a> = ();
            fn num_frames(&self) -> usize {
                1
            }
            fn num_positions(&self) -> usize {
                0
            }
            fn frame(&self, _: usize) {}
            fn steps_into(&self, _: (), _: usize) -> [Step; 0] {
                []
            }
        }
        let err = best_path(&Nothing).unwrap_err();
        assert!(format!("{err}").contains("cannot be named"), "{err}");
    }

    #[test]
    fn the_band_is_the_cells_a_complete_path_can_stand_in() {
        // Four frames, two positions, reach one.
        assert_eq!(band(0, 4, 2, 1), 0..1);
        assert_eq!(band(1, 4, 2, 1), 0..2);
        assert_eq!(band(3, 4, 2, 1), 1..3);
        assert_eq!(band(4, 4, 2, 1), 2..3);
        // A reference as long as the frames leaves no slack anywhere.
        assert_eq!(band(2, 4, 4, 1), 2..3);
        // More positions than the frames can advance through: nothing at all.
        assert!(band(0, 2, 3, 1).is_empty());
        // A wider reach opens the band at both ends.
        assert_eq!(band(1, 4, 6, 2), 0..3);
        assert_eq!(band(0, 2, 3, 2), 0..1);
    }

    #[test]
    fn a_code_packs_into_the_bits_it_needs() {
        assert_eq!(
            (code_bits(2), code_bits(4), code_bits(16), code_bits(17)),
            (1, 2, 4, 8)
        );

        for degree in [2usize, 4, 16, 256] {
            let mut trace = Traceback::new(3, 20, degree).unwrap();
            let codes: Vec<u8> = (0..21).map(|i| (i % degree) as u8).collect();
            for frame in 0..3 {
                let mut row = trace.row(frame);
                for (position, &code) in codes.iter().enumerate() {
                    row.put(position, code);
                }
                row.finish();
            }
            for frame in 0..3 {
                for (position, &code) in codes.iter().enumerate() {
                    assert_eq!(trace.get(frame, position), code, "degree {degree}");
                }
            }
        }
    }

    /// A row written only over part of its width, as a band writes it, still
    /// reads back where it was written.
    #[test]
    fn a_partial_row_reads_back() {
        let mut trace = Traceback::new(1, 20, 4).unwrap();
        let mut row = trace.row(0);
        for position in 6..=13 {
            row.put(position, (position % 4) as u8);
        }
        row.finish();
        for position in 6..=13 {
            assert_eq!(trace.get(0, position), (position % 4) as u8);
        }
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
        fn cost(&mut self) -> f32 {
            self.below(1 << 14) as f32 / 4096.0
        }
    }

    /// Every path of the chain, enumerated and weighed, against both solvers.
    #[test]
    fn both_solvers_agree_with_enumerating_every_path() {
        fn walk(chain: &Chain<'_>, frame: usize, position: usize, cost: f32, paths: &mut Vec<f32>) {
            if frame == chain.num_frames() {
                if position == chain.num_positions() {
                    paths.push(cost);
                }
                return;
            }
            // Walked forwards, so the transitions out are what is wanted.
            let scores = chain.frame(frame);
            for step in chain.steps_out_of(scores, position) {
                if step.cost.is_finite() {
                    walk(
                        chain,
                        frame + 1,
                        position + usize::from(step.advance),
                        cost + step.cost,
                        paths,
                    );
                }
            }
        }

        let mut rng = Rng(0x7A17_1CE0_1234_5678);
        let mut compared = 0;

        for round in 0..200 {
            let num_frames = 1 + rng.below(6);
            let num_positions = rng.below(num_frames.min(3) + 1);
            let phones: Vec<u32> = (0..num_positions)
                .map(|_| 1 + rng.below(3) as u32)
                .collect();
            let scores: Vec<f32> = (0..num_frames * 4).map(|_| rng.cost()).collect();
            let mut under_test = chain(&scores, &phones);
            if rng.below(2) == 0 {
                under_test.skip = rng.cost();
            }

            let mut paths = Vec::new();
            walk(&under_test, 0, 0, 0.0, &mut paths);

            let best = best_path(&under_test).unwrap();
            let total = posteriors(&under_test, |_, _| {}).unwrap();

            if paths.is_empty() {
                assert_eq!(best, None, "round {round}");
                assert_eq!(total, None, "round {round}");
                continue;
            }
            compared += 1;

            let cheapest = paths.iter().copied().fold(f32::INFINITY, f32::min);
            let best = best.expect("a path");
            assert!(
                (best.cost() - cheapest).abs() < 1e-3,
                "round {round}: best_path {} against {cheapest}",
                best.cost()
            );

            let mass: f64 = paths.iter().map(|&cost| (-cost as f64).exp()).sum();
            let total = total.expect("a total");
            assert!(
                (total - -(mass.ln() as f32)).abs() < 1e-3,
                "round {round}: posteriors {total} against {}",
                -(mass.ln() as f32)
            );
        }

        assert!(compared > 150, "only {compared} rounds had a path");
    }

    /// What a visitor is handed has to name a transition that is actually
    /// there, and the mass of one frame has to come to one.
    #[test]
    fn every_frames_visits_come_to_one() {
        let mut rng = Rng(0x1DEA_5EED_9876_4321);
        for _ in 0..40 {
            let num_frames = 2 + rng.below(10);
            let num_positions = rng.below(num_frames.min(4) + 1);
            let phones: Vec<u32> = (0..num_positions)
                .map(|_| 1 + rng.below(3) as u32)
                .collect();
            let scores: Vec<f32> = (0..num_frames * 4).map(|_| rng.cost()).collect();
            let mut under_test = chain(&scores, &phones);
            under_test.skip = 2.0;

            let mut per_frame = vec![0f64; num_frames];
            let Some(_) = posteriors(&under_test, |seen, mass| {
                assert!(seen.position <= num_positions);
                assert!(seen.code < 4);
                per_frame[seen.frame] += mass;
            })
            .unwrap() else {
                continue;
            };
            for (frame, mass) in per_frame.iter().enumerate() {
                assert!((mass - 1.0).abs() < 1e-4, "frame {frame} carries {mass}");
            }
        }
    }

    /// The contract, run as the checker a caller is told to run.
    #[test]
    fn the_chain_obeys_the_contract() {
        let scores: Vec<f32> = (0..5 * 4).map(|i| i as f32 / 3.0).collect();
        let mut under_test = chain(&scores, &[1, 2, 3]);
        under_test.skip = 1.5;
        axioms::check(&under_test);

        // Including with the skips forbidden, which is a different set of
        // absent transitions.
        axioms::check(&chain(&scores, &[1, 2, 3]));
        axioms::check(&chain(&scores, &[]));
    }

    /// The checker has to fail on a trellis that is actually wrong, or it is
    /// only decorative.
    #[test]
    #[should_panic(expected = "backwards but")]
    fn it_catches_a_backward_reading_that_disagrees() {
        struct Crooked<'a>(Chain<'a>);

        impl Trellis<4> for Crooked<'_> {
            type Frame<'f>
                = &'f [f32]
            where
                Self: 'f;
            fn num_frames(&self) -> usize {
                self.0.num_frames()
            }
            fn num_positions(&self) -> usize {
                self.0.num_positions()
            }
            fn frame(&self, frame: usize) -> &[f32] {
                self.0.frame(frame)
            }
            fn steps_into(&self, frame: &[f32], position: usize) -> [Step; 4] {
                self.0.steps_into(frame, position)
            }
        }

        impl ReversibleTrellis<4> for Crooked<'_> {
            fn steps_out_of(&self, frame: &[f32], position: usize) -> [Step; 4] {
                // The mistake that costs nothing to make: the commit is priced
                // by the phone being left rather than the one being reached.
                let mut out = derive_steps_out_of(&self.0, frame, position);
                if position > 0 && position < self.0.num_positions() {
                    out[COMMIT as usize] =
                        Step::new(1, frame[self.0.phones[position - 1] as usize]);
                }
                out
            }
        }

        let scores: Vec<f32> = (0..5 * 4).map(|i| i as f32 / 3.0).collect();
        axioms::check(&Crooked(chain(&scores, &[1, 2, 3])));
    }

    #[test]
    #[should_panic(expected = "past a REACH")]
    fn it_catches_a_transition_that_outruns_its_reach() {
        struct TooFar;
        impl Trellis<2> for TooFar {
            type Frame<'a> = ();
            fn num_frames(&self) -> usize {
                4
            }
            fn num_positions(&self) -> usize {
                3
            }
            fn frame(&self, _: usize) {}
            fn steps_into(&self, _: (), position: usize) -> [Step; 2] {
                if position >= 2 {
                    // REACH is the default 1, and this advances 2.
                    [Step::new(0, 1.0), Step::new(2, 1.0)]
                } else {
                    [Step::new(0, 1.0), Step::ABSENT]
                }
            }
        }
        impl ReversibleTrellis<2> for TooFar {}
        axioms::check(&TooFar);
    }

    #[test]
    #[should_panic(expected = "from before the start")]
    fn it_catches_a_transition_that_reaches_back_past_the_start() {
        struct OffTheFront;
        impl Trellis<2> for OffTheFront {
            type Frame<'a> = ();
            fn num_frames(&self) -> usize {
                3
            }
            fn num_positions(&self) -> usize {
                2
            }
            fn frame(&self, _: usize) {}
            fn steps_into(&self, _: (), _: usize) -> [Step; 2] {
                // Position 0 has no cell before it, so the advancing one has to
                // be absent there and is not.
                [Step::new(0, 1.0), Step::new(1, 1.0)]
            }
        }
        impl ReversibleTrellis<2> for OffTheFront {}
        axioms::check(&OffTheFront);
    }

    /// The derived reading is the point of the default, so it has to be the one
    /// a careful implementation would have written.
    #[test]
    fn the_derived_backward_reading_is_the_written_one() {
        let scores: Vec<f32> = (0..6 * 4).map(|i| (i % 7) as f32 / 2.0).collect();
        let mut under_test = chain(&scores, &[1, 2, 3, 1]);
        under_test.skip = 0.75;

        // `Chain` writes `steps_out_of` out; deriving it must give the same
        // thing, and solving with either must give the same answer.
        for frame in 0..under_test.num_frames() {
            let scores = under_test.frame(frame);
            for position in 0..=under_test.num_positions() {
                assert_eq!(
                    under_test.steps_out_of(scores, position),
                    derive_steps_out_of(&under_test, scores, position),
                    "frame {frame}, position {position}"
                );
            }
        }
    }
}
