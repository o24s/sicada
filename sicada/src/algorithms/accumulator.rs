//! Summing the weights of a run of arcs.
//!
//! Port of OpenFst's `accumulator.h`. A lookahead matcher asks, over and over,
//! what the arcs in some range of a state weigh together, and for a state with
//! many arcs, walking the range each time is the cost that matters. The
//! accumulators here trade space for that: precomputing a cumulative sum every
//! so many arcs turns a range sum into a subtraction.
//!
//! Subtraction is why these are stated in the log semiring rather than in an
//! arbitrary one: `log(e^-a - e^-b)` is the inverse of the log-sum, and a
//! general semiring has no inverse for ⊕ at all.

use rustc_hash::FxHashMap as HashMap;
use std::cell::RefCell;

use crate::arc::{Arc, ArcStateId};
use crate::error::OpenFstError;
use crate::fst::Fst;
use crate::weight::{Adder, Weight};
use crate::weights::float_weight::Log64Weight;

/// Sums the weights of arcs.
///
/// SICADA-DIVERGE: upstream's interface takes an arc *iterator* and calls
/// `Seek(begin)` on it, which only a random-access iterator supports; a delayed
/// FST's iterator answers `Seek` by walking. Taking the iterator and skipping
/// says the same thing, and skipping a slice iterator is the O(1) `Seek` was.
pub trait WeightAccumulator<A: Arc> {
    /// Precomputes whatever this accumulator keeps about `fst`.
    fn init<F: Fst<A>>(&mut self, fst: &F) -> Result<(), OpenFstError> {
        let _ = fst;
        Ok(())
    }

    /// Says which state the ranges passed to [`sum_range`](Self::sum_range)
    /// come from.
    fn set_state(&mut self, state: A::StateId) {
        let _ = state;
    }

    /// The sum of two weights.
    fn sum(&self, w: &A::Weight, v: &A::Weight) -> A::Weight;

    /// `w` plus the weights of arcs `begin..end`.
    ///
    /// `arcs` has to start at the state's first arc; the range is taken from
    /// there.
    fn sum_range<I>(&mut self, w: &A::Weight, arcs: I, begin: usize, end: usize) -> A::Weight
    where
        I: Iterator<Item = A>;
}

/// Sums with the semiring's own ⊕, walking the range.
///
/// Nothing is precomputed, so a range of `n` arcs costs `n` additions. This is
/// the accumulator for a semiring that is not the log one, where the trick the
/// others use does not apply.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultAccumulator;

impl<A: Arc> WeightAccumulator<A> for DefaultAccumulator {
    fn sum(&self, w: &A::Weight, v: &A::Weight) -> A::Weight {
        w.plus(v)
    }

    fn sum_range<I>(&mut self, w: &A::Weight, arcs: I, begin: usize, end: usize) -> A::Weight
    where
        I: Iterator<Item = A>,
    {
        let mut adder = Adder::new();
        adder.reset(w.clone());
        for arc in arcs.skip(begin).take(end.saturating_sub(begin)) {
            adder.add(arc.weight());
        }
        adder.sum()
    }
}

/// `log(1 + e^-x)`, guarding the infinity that stands for the semiring's zero.
#[inline]
fn log_pos_exp(x: f64) -> f64 {
    if x == f64::INFINITY {
        0.0
    } else {
        (-x).exp().ln_1p()
    }
}

/// `log(1 - e^-x)`, the inverse of [`log_pos_exp`].
#[inline]
fn log_minus_exp(x: f64) -> f64 {
    if x == f64::INFINITY {
        0.0
    } else {
        (1.0 - (-x).exp()).ln()
    }
}

/// The ⊕ of the log semiring, on the raw values.
#[inline]
fn log_plus_raw(f1: f64, f2: f64) -> f64 {
    if f1 == f64::INFINITY {
        return f2;
    }
    if f1 > f2 {
        f2 - log_pos_exp(f1 - f2)
    } else {
        f1 - log_pos_exp(f2 - f1)
    }
}

/// `f1 ⊖ f2` in the log semiring, where `f1 < f2`. This is the arithmetic that
/// turns a pair of cumulative sums into the sum of the range between them.
#[inline]
fn log_minus_raw(f1: f64, f2: f64) -> f64 {
    if f2 == f64::INFINITY {
        f1
    } else {
        f1 - log_minus_exp(f2 - f1)
    }
}

/// The value a weight has when read as a log weight.
#[inline]
fn as_log<W>(w: &W) -> f64
where
    W: Weight + Clone,
    Log64Weight: From<W>,
{
    Log64Weight::from(w.clone()).value()
}

/// A weight built from a log value.
#[inline]
fn from_log<W>(value: f64) -> W
where
    W: Weight + From<Log64Weight>,
{
    W::from(Log64Weight(value))
}

/// The log semiring's ⊕, on any weight that converts to and from a log weight.
fn log_plus<W>(w: &W, v: &W) -> W
where
    W: Weight + Clone + From<Log64Weight>,
    Log64Weight: From<W>,
{
    if *w == W::zero() {
        return v.clone();
    }
    from_log(log_plus_raw(as_log(w), as_log(v)))
}

/// Sums in the log semiring, walking the range.
///
/// The weights are read as log weights whatever semiring they belong to, which
/// is what a lookahead matcher asks for: how much probability mass lies in a
/// range, rather than what the semiring says the best path is.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogAccumulator;

impl<A: Arc> WeightAccumulator<A> for LogAccumulator
where
    A::Weight: From<Log64Weight>,
    Log64Weight: From<A::Weight>,
{
    fn sum(&self, w: &A::Weight, v: &A::Weight) -> A::Weight {
        log_plus(w, v)
    }

    fn sum_range<I>(&mut self, w: &A::Weight, arcs: I, begin: usize, end: usize) -> A::Weight
    where
        I: Iterator<Item = A>,
    {
        let mut sum = w.clone();
        for arc in arcs.skip(begin).take(end.saturating_sub(begin)) {
            sum = log_plus(&sum, arc.weight());
        }
        sum
    }
}

/// Sums in the log semiring, with cumulative sums precomputed every
/// `arc_period` arcs for every state that has at least `arc_limit` of them.
///
/// A range then costs one subtraction plus whatever falls outside the stored
/// points, at most `arc_period` arcs at each end. Space is one `f64` per
/// `arc_period` arcs of the states that qualify, plus one index per state.
pub struct FastLogAccumulator {
    /// The fewest arcs a state must have to be worth precomputing.
    arc_limit: usize,
    /// How many arcs lie between stored sums.
    arc_period: usize,
    /// The stored sums, all states' runs laid end to end.
    weights: Vec<f64>,
    /// Where each state's run starts in `weights`, or `None` if it has none.
    positions: Vec<Option<usize>>,
    /// The run for the state [`set_state`](WeightAccumulator::set_state) named.
    current: Option<usize>,
}

impl FastLogAccumulator {
    /// Precomputes for states with at least `arc_limit` arcs, storing a sum
    /// every `arc_period` of them.
    ///
    /// SICADA-DIVERGE: upstream reports `arc_limit < arc_period`, which would
    /// store no useful point for a state right at the limit, by setting an
    /// error flag that makes every later `Sum` return `NoWeight`. It is refused
    /// at construction here.
    pub fn new(arc_limit: usize, arc_period: usize) -> Result<Self, OpenFstError> {
        if arc_period == 0 || arc_limit < arc_period {
            return Err(OpenFstError::InvalidOperation(format!(
                "FastLogAccumulator: arc_period {arc_period} must be positive and no larger \
                 than arc_limit {arc_limit}"
            )));
        }
        Ok(Self {
            arc_limit,
            arc_period,
            weights: Vec::new(),
            positions: Vec::new(),
            current: None,
        })
    }

    /// How many stored sums there are, over every state.
    pub fn stored(&self) -> usize {
        self.weights.len()
    }
}

impl Default for FastLogAccumulator {
    fn default() -> Self {
        Self::new(20, 10).expect("20 and 10 satisfy the constraint")
    }
}

impl<A: Arc> WeightAccumulator<A> for FastLogAccumulator
where
    A::Weight: From<Log64Weight>,
    Log64Weight: From<A::Weight>,
{
    fn init<F: Fst<A>>(&mut self, fst: &F) -> Result<(), OpenFstError> {
        self.weights.clear();
        self.positions.clear();
        self.current = None;
        for state in fst.states() {
            let narcs = fst.num_arcs(state);
            if narcs < self.arc_limit {
                continue;
            }
            let index = state.as_usize();
            if self.positions.len() <= index {
                self.positions.resize(index + 1, None);
            }
            self.positions[index] = Some(self.weights.len());
            // The run starts at the semiring's zero, so that the sum of an
            // empty prefix is zero.
            let mut sum = f64::INFINITY;
            self.weights.push(sum);
            for (seen, arc) in fst.arcs(state).enumerate() {
                sum = log_plus_raw(sum, as_log(arc.weight()));
                if (seen + 1) % self.arc_period == 0 {
                    self.weights.push(sum);
                }
            }
        }
        Ok(())
    }

    fn set_state(&mut self, state: A::StateId) {
        self.current = self.positions.get(state.as_usize()).copied().flatten();
    }

    fn sum(&self, w: &A::Weight, v: &A::Weight) -> A::Weight {
        log_plus(w, v)
    }

    fn sum_range<I>(&mut self, w: &A::Weight, arcs: I, begin: usize, end: usize) -> A::Weight
    where
        I: Iterator<Item = A>,
    {
        let mut sum = w.clone();
        if end <= begin {
            return sum;
        }
        // Which stored points lie inside the range. Without any, the whole
        // range is walked.
        let (index_begin, index_end, stored_begin, stored_end) = match self.current {
            Some(_) => {
                let index_begin = if begin > 0 {
                    (begin - 1) / self.arc_period + 1
                } else {
                    0
                };
                let index_end = end / self.arc_period;
                (
                    index_begin,
                    index_end,
                    index_begin * self.arc_period,
                    index_end * self.arc_period,
                )
            }
            None => (0, 0, end, end),
        };

        let mut arcs = arcs.skip(begin);
        let mut position = begin;

        // Before the first stored point.
        if begin < stored_begin {
            let stop = stored_begin.min(end);
            for arc in arcs.by_ref().take(stop - position) {
                sum = log_plus(&sum, arc.weight());
            }
            position = stop;
        }

        // Between two stored points, which is where the work is saved.
        if stored_begin < stored_end
            && let Some(base) = self.current
        {
            let f1 = self.weights[base + index_end];
            let f2 = self.weights[base + index_begin];
            // A cumulative sum that has not grown means the arcs between them
            // weigh zero, and adding zero changes nothing.
            if f1 < f2 {
                sum = log_plus(&sum, &from_log::<A::Weight>(log_minus_raw(f1, f2)));
            }
            if position < stored_end {
                // Skip past what the stored points covered.
                let _ = arcs.by_ref().take(stored_end - position).count();
                position = stored_end;
            }
        }

        // After the last stored point.
        if position < end {
            for arc in arcs.by_ref().take(end - position) {
                sum = log_plus(&sum, arc.weight());
            }
        }
        sum
    }
}

/// Sums in the log semiring, working the cumulative sums out as states are
/// asked about rather than up front.
///
/// For a delayed FST, or one where only a few states are ever looked at, the
/// up-front pass [`FastLogAccumulator`] makes is wasted. This fills a state's
/// run the first time the state is seen, and then extends it only as far as it
/// is asked about.
pub struct CacheLogAccumulator<A: Arc> {
    /// The fewest arcs a state must have to be worth remembering.
    arc_limit: usize,
    /// The cumulative sums of the states seen so far.
    ///
    /// Behind a `RefCell` because a range sum extends a state's run, and the
    /// caller is only reading.
    cached: RefCell<HashMap<usize, Vec<f64>>>,
    /// The state currently being asked about, if it is being remembered.
    current: Option<usize>,
    _marker: std::marker::PhantomData<fn(A)>,
}

impl<A: Arc> CacheLogAccumulator<A> {
    /// Remembers states with at least `arc_limit` arcs.
    pub fn new(arc_limit: usize) -> Self {
        Self {
            arc_limit,
            cached: RefCell::new(HashMap::default()),
            current: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// How many states are being remembered.
    pub fn cached_states(&self) -> usize {
        self.cached.borrow().len()
    }

    /// Fills the current state's run up to `end` arcs.
    fn extend<I>(&self, state: usize, arcs: I, end: usize)
    where
        I: Iterator<Item = A>,
        A::Weight: Clone,
        Log64Weight: From<A::Weight>,
    {
        let mut cached = self.cached.borrow_mut();
        let run = cached.entry(state).or_insert_with(|| vec![f64::INFINITY]);
        if run.len() > end {
            return;
        }
        let from = run.len() - 1;
        let mut sum = run[from];
        for arc in arcs.skip(from).take(end + 1 - run.len()) {
            sum = log_plus_raw(sum, as_log(arc.weight()));
            run.push(sum);
        }
    }
}

impl<A: Arc> Default for CacheLogAccumulator<A> {
    fn default() -> Self {
        Self::new(10)
    }
}

impl<A: Arc> CacheLogAccumulator<A>
where
    A::Weight: From<Log64Weight>,
    Log64Weight: From<A::Weight>,
{
    /// The first position at or after `from` whose cumulative weight reaches
    /// `w`, reading the semiring's zero as the lightest.
    ///
    /// A random generator over an FST needs exactly this: draw a weight, then
    /// find the arc whose share of the total it falls in.
    pub fn lower_bound<I>(&self, w: &A::Weight, arcs: I, from: usize, narcs: usize) -> usize
    where
        I: Iterator<Item = A> + Clone,
    {
        let target = as_log(w);
        if let Some(state) = self.current {
            self.extend(state, arcs, narcs);
            let cached = self.cached.borrow();
            let run = &cached[&state];
            // The run descends, so the search is for the first entry at or
            // below the target.
            let mut lo = from + 1;
            let mut hi = run.len();
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if run[mid] > target {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            return lo - 1;
        }
        let mut n = 0;
        let mut x = f64::INFINITY;
        for arc in arcs {
            x = log_plus_raw(x, as_log(arc.weight()));
            if n >= from && x <= target {
                break;
            }
            n += 1;
        }
        n
    }
}

impl<A: Arc> WeightAccumulator<A> for CacheLogAccumulator<A>
where
    A::Weight: From<Log64Weight>,
    Log64Weight: From<A::Weight>,
{
    fn set_state(&mut self, state: A::StateId) {
        self.current = Some(state.as_usize());
    }

    fn sum(&self, w: &A::Weight, v: &A::Weight) -> A::Weight {
        log_plus(w, v)
    }

    fn sum_range<I>(&mut self, w: &A::Weight, arcs: I, begin: usize, end: usize) -> A::Weight
    where
        I: Iterator<Item = A>,
    {
        let Some(state) = self.current else {
            let mut sum = w.clone();
            for arc in arcs.skip(begin).take(end.saturating_sub(begin)) {
                sum = log_plus(&sum, arc.weight());
            }
            return sum;
        };
        // A state with few arcs is not worth remembering: walking it costs less
        // than the run would.
        let known = self.cached.borrow().contains_key(&state);
        if !known && end < self.arc_limit {
            let mut sum = w.clone();
            for arc in arcs.skip(begin).take(end.saturating_sub(begin)) {
                sum = log_plus(&sum, arc.weight());
            }
            return sum;
        }
        self.extend(state, arcs, end);
        let cached = self.cached.borrow();
        let run = &cached[&state];
        let f1 = run[end.min(run.len() - 1)];
        let f2 = run[begin.min(run.len() - 1)];
        if f1 < f2 {
            log_plus(w, &from_log::<A::Weight>(log_minus_raw(f1, f2)))
        } else {
            w.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::Rng;
    use crate::arc::{LogArc, StdArc};
    use crate::fst::MutableFst;
    use crate::fsts::vector_fst::VectorFst;
    use crate::weights::float_weight::{LogWeight, TropicalWeight};

    /// One state with `weights` arcs leaving it.
    fn fan(weights: &[f64]) -> VectorFst<LogArc> {
        let mut fst = VectorFst::new();
        let start = fst.add_state();
        let end = fst.add_state();
        fst.set_start(start);
        fst.set_final(end, LogWeight::one());
        for (index, weight) in weights.iter().enumerate() {
            fst.add_arc(
                start,
                LogArc::new(
                    index as i32 + 1,
                    index as i32 + 1,
                    LogWeight(*weight as f32),
                    end,
                ),
            );
        }
        fst
    }

    /// The log-semiring sum of a range, worked out directly.
    fn reference(weights: &[f64], begin: usize, end: usize) -> f64 {
        let mut sum = f64::INFINITY;
        for weight in &weights[begin..end] {
            sum = log_plus_raw(sum, *weight);
        }
        sum
    }

    /// The plain accumulator is the semiring's own sum.
    #[test]
    fn the_default_accumulator_is_the_semirings_own_sum() {
        let mut fst: VectorFst<StdArc> = VectorFst::new();
        let start = fst.add_state();
        let end = fst.add_state();
        fst.set_start(start);
        fst.set_final(end, TropicalWeight::one());
        for weight in [3.0f32, 1.0, 4.0, 1.0, 5.0] {
            fst.add_arc(start, StdArc::new(1, 1, TropicalWeight(weight), end));
        }

        let mut acc = DefaultAccumulator;
        // Over the tropical semiring the sum of a range is its minimum.
        assert_eq!(
            acc.sum_range(&TropicalWeight::zero(), fst.arcs(start), 1, 4),
            TropicalWeight(1.0)
        );
        assert_eq!(
            acc.sum_range(&TropicalWeight::zero(), fst.arcs(start), 0, 5),
            TropicalWeight(1.0)
        );
        assert_eq!(
            acc.sum_range(&TropicalWeight::zero(), fst.arcs(start), 2, 2),
            TropicalWeight::zero(),
            "an empty range adds nothing"
        );
    }

    /// The log accumulator sums in the log semiring whatever the arcs' own
    /// semiring is.
    #[test]
    fn the_log_accumulator_sums_in_the_log_semiring() {
        let weights = [1.0f64, 2.0, 3.0];
        let fst = fan(&weights);
        let mut acc = LogAccumulator;
        let got = acc.sum_range(&LogWeight::zero(), fst.arcs(0), 0, 3);
        let want = reference(&weights, 0, 3);
        assert!(
            (got.value() as f64 - want).abs() < 1e-5,
            "{got:?} against {want}"
        );
    }

    /// Precomputing has to give the same answer as walking, for every range.
    #[test]
    fn the_fast_accumulator_agrees_with_walking_for_every_range() {
        let mut rng = Rng::new(0x00AC_C000_u64);
        for round in 0..40 {
            let narcs = 1 + rng.below(40);
            let weights: Vec<f64> = (0..narcs).map(|_| rng.below(20) as f64 / 4.0).collect();
            let fst = fan(&weights);

            let mut fast = FastLogAccumulator::new(8, 4).unwrap();
            fast.init(&fst).unwrap();
            WeightAccumulator::<LogArc>::set_state(&mut fast, 0);

            let mut cache = CacheLogAccumulator::<LogArc>::new(8);
            WeightAccumulator::<LogArc>::set_state(&mut cache, 0);

            for begin in 0..=narcs {
                for end in begin..=narcs {
                    let want = reference(&weights, begin, end);
                    for (name, got) in [
                        (
                            "fast",
                            fast.sum_range(&LogWeight::zero(), fst.arcs(0), begin, end),
                        ),
                        (
                            "cache",
                            cache.sum_range(&LogWeight::zero(), fst.arcs(0), begin, end),
                        ),
                        (
                            "plain",
                            LogAccumulator.sum_range(&LogWeight::zero(), fst.arcs(0), begin, end),
                        ),
                    ] {
                        let got = got.value() as f64;
                        let close = if want.is_infinite() {
                            got.is_infinite()
                        } else {
                            (got - want).abs() < 1e-4
                        };
                        assert!(
                            close,
                            "round {round}, {name}, {begin}..{end}: {got} against {want}"
                        );
                    }
                }
            }
        }
    }

    /// A state below the limit is not precomputed, and one above it is.
    #[test]
    fn only_states_with_enough_arcs_are_precomputed() {
        let small = fan(&[1.0, 2.0]);
        let mut acc = FastLogAccumulator::new(8, 4).unwrap();
        acc.init(&small).unwrap();
        assert_eq!(acc.stored(), 0, "two arcs is not worth a table");

        let big = fan(&[1.0; 20]);
        let mut acc = FastLogAccumulator::new(8, 4).unwrap();
        acc.init(&big).unwrap();
        // One entry for the empty prefix, then one every four arcs.
        assert_eq!(acc.stored(), 1 + 20 / 4);
    }

    /// A period larger than the limit would store no point for a state right at
    /// the limit, so it is refused rather than flagged.
    #[test]
    fn a_period_larger_than_the_limit_is_refused() {
        assert!(FastLogAccumulator::new(4, 8).is_err());
        assert!(FastLogAccumulator::new(8, 0).is_err());
        assert!(FastLogAccumulator::new(8, 8).is_ok());
    }

    /// The cache fills in only the states it is asked about.
    #[test]
    fn the_cache_fills_in_only_what_is_asked_for() {
        let mut fst: VectorFst<LogArc> = VectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.set_final(2, LogWeight::one());
        for state in [0i32, 1] {
            for index in 0..12 {
                fst.add_arc(state, LogArc::new(index + 1, index + 1, LogWeight(1.0), 2));
            }
        }

        let mut acc = CacheLogAccumulator::<LogArc>::new(4);
        assert_eq!(acc.cached_states(), 0);
        WeightAccumulator::<LogArc>::set_state(&mut acc, 0);
        acc.sum_range(&LogWeight::zero(), fst.arcs(0), 0, 12);
        assert_eq!(acc.cached_states(), 1, "only state 0 was asked about");
    }

    /// Drawing a weight and finding where it falls.
    ///
    /// The answer is the last position whose running total has not yet passed
    /// the weight: with the running total descending as arcs are added, the
    /// position returned for a target strictly between two consecutive totals
    /// is the earlier of the two.
    #[test]
    fn the_lower_bound_is_where_the_cumulative_sum_reaches_the_weight() {
        let weights = vec![1.0f64; 8];
        let fst = fan(&weights);
        let mut acc = CacheLogAccumulator::<LogArc>::new(4);
        WeightAccumulator::<LogArc>::set_state(&mut acc, 0);

        // The running totals descend, and the gaps between them are far larger
        // than the precision of the weight the target is carried in.
        let total = |k: usize| reference(&weights, 0, k);
        for k in 1..8 {
            let midpoint = (total(k) + total(k + 1)) / 2.0;
            assert!(total(k + 1) < midpoint && midpoint < total(k), "{k}");
            let at = acc.lower_bound(&LogWeight(midpoint as f32), fst.arcs(0), 0, 8);
            assert_eq!(at, k, "a target between totals {k} and {k} + 1");
        }
    }

    /// Without a state to remember, the search walks instead, and has to give
    /// the same answer.
    #[test]
    fn the_lower_bound_agrees_whether_or_not_the_state_is_remembered() {
        let weights = vec![1.0f64, 2.0, 0.5, 3.0, 1.5, 0.25];
        let fst = fan(&weights);
        let total = |k: usize| reference(&weights, 0, k);

        let mut remembered = CacheLogAccumulator::<LogArc>::new(1);
        WeightAccumulator::<LogArc>::set_state(&mut remembered, 0);
        // Never told a state, so it walks.
        let walking = CacheLogAccumulator::<LogArc>::new(1);

        for k in 1..weights.len() {
            let midpoint = (total(k) + total(k + 1)) / 2.0;
            let target = LogWeight(midpoint as f32);
            assert_eq!(
                remembered.lower_bound(&target, fst.arcs(0), 0, weights.len()),
                walking.lower_bound(&target, fst.arcs(0), 0, weights.len()),
                "position {k}"
            );
        }
    }
}
