//! Sets of half-open integer intervals, and the algebra over them.
//!
//! Port of OpenFst's `interval-set.h`. Used by the reachability tables that
//! composition's lookahead filters consult, where a state's reachable label set
//! is almost always a handful of contiguous ranges, so storing ranges rather
//! than members is both smaller and faster to intersect.
//!
//! Every operation apart from [`IntervalSet::normalize`] requires its inputs to
//! be normalized: sorted, with overlapping and adjacent intervals merged, and
//! `count` equal to the number of members.

use std::cmp::Ordering;
use std::fmt;

/// A half-open integral interval `[begin, end)` of integers of type `T`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntInterval<T> {
    pub begin: T,
    pub end: T,
}

impl<T> IntInterval<T> {
    #[inline(always)]
    pub fn new(begin: T, end: T) -> Self {
        Self { begin, end }
    }
}

// Implement partial ordering.
// Sorted first by `begin` ascending, then by `end` descending.
impl<T: Ord> PartialOrd for IntInterval<T> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord> Ord for IntInterval<T> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.begin
            .cmp(&other.begin)
            .then_with(|| other.end.cmp(&self.end))
    }
}

/// Stores and operates on a set of half-open integral intervals `[a, b)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntervalSet<T> {
    intervals: Vec<IntInterval<T>>,
    count: usize,
}

impl<T> Default for IntervalSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> IntervalSet<T> {
    #[inline]
    pub fn new() -> Self {
        Self {
            intervals: Vec::new(),
            count: 0,
        }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            intervals: Vec::with_capacity(capacity),
            count: 0,
        }
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    /// Number of distinct intervals.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.intervals.len()
    }

    /// Number of points in all intervals (undefined if not normalized).
    #[inline(always)]
    pub fn count(&self) -> usize {
        self.count
    }

    #[inline(always)]
    pub fn intervals(&self) -> &[IntInterval<T>] {
        &self.intervals
    }

    /// The intervals, to be changed in place.
    ///
    /// Corresponds to upstream's `MutableIntervals`. The set's member count is
    /// only recomputed by [`normalize`](Self::normalize), so call that before
    /// asking how many members the set has.
    #[inline]
    pub fn intervals_mut(&mut self) -> &mut Vec<IntInterval<T>> {
        &mut self.intervals
    }

    #[inline]
    pub fn clear(&mut self) {
        self.intervals.clear();
        self.count = 0;
    }

    /// Adds an interval to the set. The result may not be normalized.
    #[inline]
    pub fn push(&mut self, interval: IntInterval<T>) {
        self.intervals.push(interval);
    }

    /// Appends another interval set. The result may not be normalized.
    #[inline]
    pub fn union(&mut self, other: &IntervalSet<T>)
    where
        T: Copy,
    {
        self.intervals.extend_from_slice(&other.intervals);
    }

    /// True if the set contains exactly one interval of size 1.
    #[inline]
    pub fn is_singleton(&self) -> bool
    where
        T: TryInto<usize> + Copy + PartialOrd,
        usize: TryFrom<T>,
    {
        if self.len() == 1 {
            let iv = &self.intervals[0];
            if let (Ok(b), Ok(e)) = (usize::try_from(iv.begin), usize::try_from(iv.end)) {
                return b + 1 == e;
            }
        }
        false
    }
}

// Methods requiring specific type bounds (e.g. integer arithmetic capabilities).
// We use `usize` and standard castability as typical label types in Rust OpenFst
// fall into integer domains.
impl<T> IntervalSet<T>
where
    T: Copy + Ord + Into<i64>,
{
    /// Requires intervals be normalized.
    #[inline]
    pub fn member(&self, value: T) -> bool {
        // Fast path for empty.
        if self.is_empty() {
            return false;
        }

        // Binary search. We search for an interval starting with `value`.
        let target = IntInterval::new(value, value);
        let pos = self.intervals.partition_point(|iv| iv < &target);

        if pos == 0 {
            false
        } else {
            // Check the interval just before the partition point.
            self.intervals[pos - 1].end > value
        }
    }

    /// Sorts, collapses overlapping and adjacent intervals, and sets count.
    pub fn normalize(&mut self) {
        self.intervals.sort_unstable();

        // Sweep once, merging each interval into the one being built at
        // `write_idx - 1` whenever they touch or overlap. Empty intervals are
        // dropped. Sorting first means an interval can only ever extend the last
        // one kept, never an earlier one.
        let mut write_idx = 0;
        for read_idx in 0..self.intervals.len() {
            let interval = self.intervals[read_idx];
            if interval.begin == interval.end {
                continue;
            }
            match self.intervals[..write_idx].last_mut() {
                Some(last) if last.end >= interval.begin => {
                    if last.end < interval.end {
                        last.end = interval.end;
                    }
                }
                _ => {
                    self.intervals[write_idx] = interval;
                    write_idx += 1;
                }
            }
        }
        self.intervals.truncate(write_idx);

        self.count = self
            .intervals
            .iter()
            .map(|interval| (interval.end.into() - interval.begin.into()) as usize)
            .sum();
    }

    /// Intersects an interval set with the set. Requires intervals be normalized.
    /// The result is placed into `oset` and normalized.
    pub fn intersect(&self, iset: &IntervalSet<T>, oset: &mut IntervalSet<T>) {
        oset.clear();
        let mut count = 0;

        let mut it1 = self.intervals.iter();
        let mut it2 = iset.intervals.iter();

        let mut iv1_opt = it1.next();
        let mut iv2_opt = it2.next();

        while let (Some(iv1), Some(iv2)) = (iv1_opt, iv2_opt) {
            if iv1.end <= iv2.begin {
                iv1_opt = it1.next();
            } else if iv2.end <= iv1.begin {
                iv2_opt = it2.next();
            } else {
                let begin = std::cmp::max(iv1.begin, iv2.begin);
                let end = std::cmp::min(iv1.end, iv2.end);

                oset.push(IntInterval::new(begin, end));
                count += (end.into() - begin.into()) as usize;

                if iv1.end < iv2.end {
                    iv1_opt = it1.next();
                } else {
                    iv2_opt = it2.next();
                }
            }
        }

        oset.count = count;
    }

    /// Complements the set w.r.t `[0, maxval)`. Requires intervals be normalized.
    /// The result is placed into `oset` and normalized.
    pub fn complement(&self, maxval: T, oset: &mut IntervalSet<T>)
    where
        T: Default,
    {
        oset.clear();
        let mut count = 0;
        let mut current_begin: T = Default::default(); // Assumes T::default() == 0

        for current_interval in &self.intervals {
            let end = std::cmp::min(current_interval.begin, maxval);
            if current_begin < end {
                oset.push(IntInterval::new(current_begin, end));
                count += (end.into() - current_begin.into()) as usize;
            }
            current_begin = current_interval.end;
        }

        if current_begin < maxval {
            oset.push(IntInterval::new(current_begin, maxval));
            count += (maxval.into() - current_begin.into()) as usize;
        }

        oset.count = count;
    }

    /// Subtracts an interval set from the set. Requires intervals be normalized.
    /// The result is placed into `oset` and normalized.
    pub fn difference(&self, iset: &IntervalSet<T>, oset: &mut IntervalSet<T>)
    where
        T: Default,
    {
        if self.is_empty() {
            oset.clear();
        } else {
            let maxval = self.intervals.last().unwrap().end;
            let mut cset = IntervalSet::with_capacity(iset.len() + 1);
            iset.complement(maxval, &mut cset);
            self.intersect(&cset, oset);
        }
    }

    /// Determines if an interval set overlaps with the set. Requires intervals be normalized.
    pub fn overlaps(&self, iset: &IntervalSet<T>) -> bool {
        let mut it1 = self.intervals.iter();
        let mut it2 = iset.intervals.iter();

        let mut iv1_opt = it1.next();
        let mut iv2_opt = it2.next();

        while let (Some(iv1), Some(iv2)) = (iv1_opt, iv2_opt) {
            if iv1.end <= iv2.begin {
                iv1_opt = it1.next();
            } else if iv2.end <= iv1.begin {
                iv2_opt = it2.next();
            } else {
                return true;
            }
        }
        false
    }

    /// Determines if an interval set overlaps with the set but neither is contained in the other.
    /// Requires intervals be normalized.
    pub fn strictly_overlaps(&self, iset: &IntervalSet<T>) -> bool {
        let mut it1 = self.intervals.iter();
        let mut it2 = iset.intervals.iter();

        let mut iv1_opt = it1.next();
        let mut iv2_opt = it2.next();

        let mut only1 = false;
        let mut only2 = false;
        let mut overlap = false;

        while let (Some(iv1), Some(iv2)) = (iv1_opt, iv2_opt) {
            if iv1.end <= iv2.begin {
                only1 = true;
                iv1_opt = it1.next();
            } else if iv2.end <= iv1.begin {
                only2 = true;
                iv2_opt = it2.next();
            } else if iv2.begin == iv1.begin && iv2.end == iv1.end {
                overlap = true;
                iv1_opt = it1.next();
                iv2_opt = it2.next();
            } else if iv2.begin <= iv1.begin && iv2.end >= iv1.end {
                only2 = true;
                overlap = true;
                iv1_opt = it1.next();
            } else if iv1.begin <= iv2.begin && iv1.end >= iv2.end {
                only1 = true;
                overlap = true;
                iv2_opt = it2.next();
            } else {
                only1 = true;
                only2 = true;
                overlap = true;
            }

            if only1 && only2 && overlap {
                return true;
            }
        }

        if iv1_opt.is_some() {
            only1 = true;
        }
        if iv2_opt.is_some() {
            only2 = true;
        }

        only1 && only2 && overlap
    }

    /// Determines if an interval set is contained within the set.
    /// Requires intervals be normalized.
    pub fn contains(&self, iset: &IntervalSet<T>) -> bool {
        if iset.count() > self.count() {
            return false;
        }

        let mut it1 = self.intervals.iter();
        let mut it2 = iset.intervals.iter();

        let mut iv1_opt = it1.next();
        let mut iv2_opt = it2.next();

        while let (Some(iv1), Some(iv2)) = (iv1_opt, iv2_opt) {
            if iv1.end <= iv2.begin {
                iv1_opt = it1.next();
            } else if iv2.begin < iv1.begin || iv2.end > iv1.end {
                return false;
            } else if iv2.end == iv1.end {
                iv1_opt = it1.next();
                iv2_opt = it2.next();
            } else {
                iv2_opt = it2.next();
            }
        }

        iv2_opt.is_none()
    }
}

impl<T: fmt::Display> fmt::Display for IntervalSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        let mut first = true;
        for iv in &self.intervals {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "[{},{})", iv.begin, iv.end)?;
            first = false;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn test_normalize() {
        let mut set = IntervalSet::new();
        set.push(IntInterval::new(10, 20));
        set.push(IntInterval::new(5, 12)); // Overlaps with [10, 20) -> [5, 20)
        set.push(IntInterval::new(30, 40));
        set.push(IntInterval::new(20, 25)); // Adjacent to [5, 20) -> [5, 25)

        set.normalize();

        assert_eq!(set.len(), 2);
        assert_eq!(set.intervals()[0], IntInterval::new(5, 25));
        assert_eq!(set.intervals()[1], IntInterval::new(30, 40));
        assert_eq!(set.count(), 20 + 10);
    }

    #[test]
    fn test_intersect() {
        let mut set1 = IntervalSet::new();
        set1.push(IntInterval::new(0, 10));
        set1.push(IntInterval::new(20, 30));
        set1.normalize();

        let mut set2 = IntervalSet::new();
        set2.push(IntInterval::new(5, 25));
        set2.normalize();

        let mut oset = IntervalSet::new();
        set1.intersect(&set2, &mut oset);

        assert_eq!(oset.len(), 2);
        assert_eq!(oset.intervals()[0], IntInterval::new(5, 10));
        assert_eq!(oset.intervals()[1], IntInterval::new(20, 25));
    }

    #[test]
    fn test_member() {
        let mut set = IntervalSet::new();
        set.push(IntInterval::new(10, 20));
        set.normalize();

        assert!(!set.member(9));
        assert!(set.member(10));
        assert!(set.member(15));
        assert!(set.member(19));
        assert!(!set.member(20)); // Half-open
    }
    /// Brute-force model of an interval set: the members themselves.
    ///
    /// Every operation is checked against this rather than against another
    /// interval computation, so a shared misunderstanding of the algebra cannot
    /// hide a bug.
    fn members(set: &IntervalSet<i32>) -> BTreeSet<i32> {
        set.intervals()
            .iter()
            .flat_map(|interval| interval.begin..interval.end)
            .collect()
    }

    fn set_of(members: &BTreeSet<i32>) -> IntervalSet<i32> {
        let mut set = IntervalSet::new();
        for &member in members {
            set.push(IntInterval::new(member, member + 1));
        }
        set.normalize();
        set
    }

    /// A normalized set must be sorted, gap-separated, and carry the right count.
    fn assert_normalized(set: &IntervalSet<i32>) {
        let intervals = set.intervals();
        for interval in intervals {
            assert!(interval.begin < interval.end, "empty interval {interval:?}");
        }
        for pair in intervals.windows(2) {
            assert!(
                pair[0].end < pair[1].begin,
                "intervals {:?} and {:?} touch or overlap and should have merged",
                pair[0],
                pair[1]
            );
        }
        let expected: usize = intervals.iter().map(|i| (i.end - i.begin) as usize).sum();
        assert_eq!(set.count(), expected, "count disagrees with the intervals");
    }

    fn random_members(rng: &mut impl FnMut() -> u64, universe: i32) -> BTreeSet<i32> {
        let mut members = BTreeSet::new();
        for value in 0..universe {
            // Bias towards runs so the sets actually look like intervals.
            if !rng().is_multiple_of(3) {
                members.insert(value);
            }
        }
        members
    }

    #[test]
    fn normalize_merges_overlapping_and_adjacent_intervals() {
        let mut set = IntervalSet::new();
        for interval in [(5, 7), (0, 2), (2, 4), (6, 10), (12, 12)] {
            set.push(IntInterval::new(interval.0, interval.1));
        }
        set.normalize();

        // [0,2) and [2,4) are adjacent and merge; [5,7) and [6,10) overlap and
        // merge; [12,12) is empty and disappears.
        assert_eq!(
            set.intervals(),
            &[IntInterval::new(0, 4), IntInterval::new(5, 10)]
        );
        assert_eq!(set.count(), 9);
        assert_normalized(&set);
    }

    #[test]
    fn normalize_is_idempotent() {
        let mut set = IntervalSet::new();
        for interval in [(3, 9), (1, 4), (20, 21)] {
            set.push(IntInterval::new(interval.0, interval.1));
        }
        set.normalize();
        let once = set.clone();
        set.normalize();
        assert_eq!(set, once);
    }

    /// The whole algebra, against the brute-force model.
    #[test]
    fn the_set_algebra_matches_a_brute_force_model() {
        const UNIVERSE: i32 = 24;
        let mut state = 0x1357_9BDF_2468_ACE0u64;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for round in 0..300 {
            let left_members = random_members(&mut rng, UNIVERSE);
            let right_members = random_members(&mut rng, UNIVERSE);
            let left = set_of(&left_members);
            let right = set_of(&right_members);
            assert_normalized(&left);
            assert_normalized(&right);
            assert_eq!(members(&left), left_members, "round {round}");

            for value in -2..UNIVERSE + 2 {
                assert_eq!(
                    left.member(value),
                    left_members.contains(&value),
                    "member({value}) in round {round}"
                );
            }
            assert_eq!(left.count(), left_members.len(), "round {round}");
            assert_eq!(left.is_empty(), left_members.is_empty());
            assert_eq!(left.is_singleton(), left_members.len() == 1);

            let mut intersection = IntervalSet::new();
            left.intersect(&right, &mut intersection);
            assert_normalized(&intersection);
            assert_eq!(
                members(&intersection),
                left_members.intersection(&right_members).copied().collect(),
                "intersect in round {round}"
            );

            let mut difference = IntervalSet::new();
            left.difference(&right, &mut difference);
            assert_normalized(&difference);
            assert_eq!(
                members(&difference),
                left_members.difference(&right_members).copied().collect(),
                "difference in round {round}"
            );

            let mut complement = IntervalSet::new();
            left.complement(UNIVERSE, &mut complement);
            assert_normalized(&complement);
            assert_eq!(
                members(&complement),
                (0..UNIVERSE)
                    .filter(|v| !left_members.contains(v))
                    .collect(),
                "complement in round {round}"
            );

            let mut united = left.clone();
            united.union(&right);
            united.normalize();
            assert_normalized(&united);
            assert_eq!(
                members(&united),
                left_members.union(&right_members).copied().collect(),
                "union in round {round}"
            );

            assert_eq!(
                left.overlaps(&right),
                !left_members.is_disjoint(&right_members),
                "overlaps in round {round}"
            );
            assert_eq!(
                left.contains(&right),
                right_members.is_subset(&left_members),
                "contains in round {round}"
            );
            // Strictly overlapping: they meet, but neither side is contained in
            // the other.
            let strictly = !left_members.is_disjoint(&right_members)
                && !right_members.is_subset(&left_members)
                && !left_members.is_subset(&right_members);
            assert_eq!(
                left.strictly_overlaps(&right),
                strictly,
                "strictly_overlaps in round {round}: {left_members:?} vs {right_members:?}"
            );
        }
    }

    #[test]
    fn an_empty_set_behaves() {
        let empty = IntervalSet::<i32>::new();
        let other = set_of(&(3..7).collect());

        assert!(empty.is_empty());
        assert_eq!(empty.count(), 0);
        assert!(!empty.member(0));
        assert!(!empty.is_singleton());
        assert!(!empty.overlaps(&other));
        assert!(!empty.strictly_overlaps(&other));
        assert!(empty.contains(&IntervalSet::new()));
        assert!(other.contains(&empty), "every set contains the empty set");

        let mut out = IntervalSet::new();
        empty.intersect(&other, &mut out);
        assert!(out.is_empty());
        empty.complement(5, &mut out);
        assert_eq!(members(&out), (0..5).collect());
    }
}
