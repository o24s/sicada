//! A set specialised for keys confined to a narrow interval.
//!
//! Port of the `CompactSet` in OpenFst's `util.h`. Membership is answered by two
//! O(1) checks before the container is consulted at all: keys outside the
//! tracked `[lower_bound, upper_bound]` interval are rejected immediately, and a
//! key inside a *dense* interval is accepted immediately. Only a sparse interval
//! falls through to a real lookup.

/// Key value marking an unset bound.
pub const K_NO_KEY: usize = usize::MAX;

/// An integer key a [`CompactSet`] can hold.
///
/// Upstream parameterises the set on both the key type and the sentinel value
/// (`CompactSet<Label, kNoLabel>`); here the sentinel travels with the type. A
/// key equal to [`NO_KEY`](CompactSetKey::NO_KEY) can never be a member, which
/// is why label sets use `-1`: that is `kNoLabel`, which is not a real label.
pub trait CompactSetKey: Copy + Ord {
    /// Value marking an unset bound, and therefore not storable.
    const NO_KEY: Self;

    /// Number of integers in the inclusive range `[min, max]`, or `None` if that
    /// does not fit a `usize`.
    fn span(min: Self, max: Self) -> Option<usize>;
}

macro_rules! impl_compact_set_key {
    ($($ty:ty => $no_key:expr),* $(,)?) => {
        $(
            impl CompactSetKey for $ty {
                const NO_KEY: Self = $no_key;

                #[inline(always)]
                fn span(min: Self, max: Self) -> Option<usize> {
                    let width = (max as i128) - (min as i128) + 1;
                    usize::try_from(width).ok()
                }
            }
        )*
    };
}

impl_compact_set_key! {
    usize => usize::MAX,
    u32 => u32::MAX,
    u64 => u64::MAX,
    i32 => -1,
    i64 => -1,
}

/// A set of integer keys, fast to query when the keys cluster in an interval.
///
/// SICADA-OPT: upstream backs this with `std::set`, a red-black tree with a node
/// allocation per key. The sparse fallback here is a sorted `Vec` searched
/// binary: contiguous, allocation-free per key, and cache-friendly. It also
/// keeps iteration in key order, which upstream gets from `std::set` and which
/// `MultiEpsMatcher` depends on for the order it emits multi-epsilon arcs. A
/// hash set would make that order arbitrary and the composed output
/// irreproducible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactSet<K: CompactSetKey = usize> {
    /// Keys in ascending order.
    keys: Vec<K>,
    min_key: K,
    max_key: K,
}

impl<K: CompactSetKey> Default for CompactSet<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: CompactSetKey> CompactSet<K> {
    /// Creates an empty set.
    #[inline]
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            min_key: K::NO_KEY,
            max_key: K::NO_KEY,
        }
    }

    /// Inserts a key, widening the tracked interval.
    #[inline]
    pub fn insert(&mut self, key: K) {
        if let Err(index) = self.keys.binary_search(&key) {
            self.keys.insert(index, key);
        }
        if self.min_key == K::NO_KEY || key < self.min_key {
            self.min_key = key;
        }
        if self.max_key == K::NO_KEY || key > self.max_key {
            self.max_key = key;
        }
    }

    /// Removes a key.
    ///
    /// SICADA-OPT: upstream only relaxes the bound by one when an endpoint is
    /// erased, leaving the interval looser than it needs to be and so weakening
    /// the O(1) reject that the whole structure exists for. Because the keys are
    /// held sorted here, the true endpoint is one index away, so the bounds stay
    /// tight for the same cost.
    #[inline]
    pub fn erase(&mut self, key: K) {
        let Ok(index) = self.keys.binary_search(&key) else {
            return;
        };
        self.keys.remove(index);
        if self.keys.is_empty() {
            self.min_key = K::NO_KEY;
            self.max_key = K::NO_KEY;
        } else if key == self.min_key {
            // The true minimum is now at least the next key along, and the list
            // is sorted, so read it off rather than doing arithmetic on `K`.
            self.min_key = self.keys[0];
        } else if key == self.max_key {
            self.max_key = self.keys[self.keys.len() - 1];
        }
    }

    /// Removes every key.
    #[inline]
    pub fn clear(&mut self) {
        self.keys.clear();
        self.min_key = K::NO_KEY;
        self.max_key = K::NO_KEY;
    }

    /// Whether `key` is in the set.
    #[inline]
    pub fn is_member(&self, key: K) -> bool {
        if self.min_key == K::NO_KEY || key < self.min_key || key > self.max_key {
            // Outside the tracked interval.
            false
        } else if K::span(self.min_key, self.max_key) == Some(self.keys.len()) {
            // The interval is dense, so every key in it is present.
            true
        } else {
            self.keys.binary_search(&key).is_ok()
        }
    }

    /// Whether the set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Number of keys.
    #[inline]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Every key is greater than or equal to this, or [`K_NO_KEY`] if empty.
    #[inline]
    pub fn lower_bound(&self) -> K {
        self.min_key
    }

    /// Every key is less than or equal to this, or [`K_NO_KEY`] if empty.
    #[inline]
    pub fn upper_bound(&self) -> K {
        self.max_key
    }

    /// The keys, in ascending order.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, K> {
        self.keys.iter()
    }

    /// The keys as a sorted slice.
    #[inline]
    pub fn as_slice(&self) -> &[K] {
        &self.keys
    }
}

impl<'a, K: CompactSetKey> IntoIterator for &'a CompactSet<K> {
    type Item = &'a K;
    type IntoIter = std::slice::Iter<'a, K>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K: CompactSetKey> FromIterator<K> for CompactSet<K> {
    fn from_iter<I: IntoIterator<Item = K>>(iter: I) -> Self {
        let mut set = Self::new();
        for key in iter {
            set.insert(key);
        }
        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn membership_and_bounds_track_the_contents() {
        let mut set = CompactSet::new();
        assert!(set.is_empty());
        assert_eq!(set.lower_bound(), K_NO_KEY);
        assert_eq!(set.upper_bound(), K_NO_KEY);
        assert!(!set.is_member(0));

        for key in [5, 7, 6] {
            set.insert(key);
        }
        assert_eq!(set.len(), 3);
        assert_eq!(set.lower_bound(), 5);
        assert_eq!(set.upper_bound(), 7);
        for key in 5..=7 {
            assert!(set.is_member(key));
        }
        assert!(!set.is_member(4));
        assert!(!set.is_member(8));
    }

    #[test]
    fn inserting_a_key_twice_changes_nothing() {
        let mut set = CompactSet::new();
        set.insert(3);
        set.insert(3);
        assert_eq!(set.len(), 1);
        assert_eq!(set.as_slice(), &[3]);
    }

    /// Iteration is in key order, which `MultiEpsMatcher` relies on and which a
    /// hash-backed set would not give.
    #[test]
    fn iteration_is_sorted_regardless_of_insertion_order() {
        let set: CompactSet = [9, 1, 5, 3, 7].into_iter().collect();
        assert_eq!(set.iter().copied().collect::<Vec<_>>(), vec![1, 3, 5, 7, 9]);

        let reversed: CompactSet = [7, 3, 5, 1, 9].into_iter().collect();
        assert_eq!(set, reversed, "order of insertion must not be observable");
    }

    /// The dense fast path must agree with an actual lookup, both while the
    /// interval is dense and after a hole appears in it.
    #[test]
    fn the_dense_fast_path_agrees_with_a_real_lookup() {
        let mut set: CompactSet = (10..20).collect();
        assert_eq!(set.upper_bound() - set.lower_bound() + 1, set.len());
        for key in 0..30 {
            assert_eq!(set.is_member(key), (10..20).contains(&key), "dense, {key}");
        }

        set.erase(15);
        for key in 0..30 {
            let expected = (10..20).contains(&key) && key != 15;
            assert_eq!(set.is_member(key), expected, "sparse, {key}");
        }
    }

    /// Erasing an endpoint tightens the bound onto the new endpoint rather than
    /// stepping it by one, which keeps the interval reject as strong as it can
    /// be. Upstream steps by one and accepts a looser interval.
    #[test]
    fn erasing_an_endpoint_tightens_the_bound() {
        let mut set: CompactSet = (1..=3).collect();

        // Erasing from the middle leaves the bounds alone.
        set.erase(2);
        assert_eq!(set.lower_bound(), 1);
        assert_eq!(set.upper_bound(), 3);
        assert!(!set.is_member(2));

        // Erasing the low endpoint moves the bound to what is now the lowest key.
        set.erase(1);
        assert_eq!(set.lower_bound(), 3, "3 is the only key left");
        assert_eq!(set.upper_bound(), 3);
        assert!(set.is_member(3));
        assert!(!set.is_member(1));
    }

    /// The set has to work with signed keys too, since label sets use `-1` as
    /// the sentinel and real labels can be negative.
    #[test]
    fn signed_keys_work_and_exclude_the_sentinel() {
        let mut set: CompactSet<i32> = CompactSet::new();
        assert_eq!(set.lower_bound(), <i32 as CompactSetKey>::NO_KEY);
        for key in [-5, 0, 3] {
            set.insert(key);
        }
        assert_eq!(set.as_slice(), &[-5, 0, 3]);
        assert!(set.is_member(-5));
        assert!(set.is_member(0));
        assert!(!set.is_member(-4));
        assert_eq!(set.lower_bound(), -5);
        assert_eq!(set.upper_bound(), 3);
    }

    #[test]
    fn erasing_the_last_key_resets_the_bounds() {
        let mut set = CompactSet::new();
        set.insert(4);
        set.erase(4);
        assert!(set.is_empty());
        assert_eq!(set.lower_bound(), K_NO_KEY);
        assert_eq!(set.upper_bound(), K_NO_KEY);
        assert!(!set.is_member(4));
    }

    #[test]
    fn erasing_an_absent_key_leaves_the_set_alone() {
        let mut set: CompactSet = [2, 4].into_iter().collect();
        set.erase(3);
        set.erase(100);
        assert_eq!(set.as_slice(), &[2, 4]);
        assert_eq!(set.lower_bound(), 2);
        assert_eq!(set.upper_bound(), 4);
    }

    #[test]
    fn erasing_zero_does_not_underflow_the_upper_bound() {
        let mut set = CompactSet::new();
        set.insert(0);
        set.erase(0);
        assert_eq!(set.upper_bound(), K_NO_KEY);
    }

    #[test]
    fn clearing_resets_everything() {
        let mut set: CompactSet = (0..5).collect();
        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.lower_bound(), K_NO_KEY);
        assert!(!set.is_member(0));
        set.insert(9);
        assert_eq!(set.lower_bound(), 9);
    }

    /// Randomized comparison against a plain reference set: every query must
    /// agree, whichever fast path answered it.
    #[test]
    fn matches_a_reference_set_under_random_operations() {
        use std::collections::BTreeSet;

        let mut state = 0x5DEE_CE66_D1E5_1234u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let mut set = CompactSet::new();
        let mut reference = BTreeSet::new();
        for _ in 0..2000 {
            let key = (next() % 64) as usize;
            if next() % 3 == 0 {
                set.erase(key);
                reference.remove(&key);
            } else {
                set.insert(key);
                reference.insert(key);
            }

            assert_eq!(set.len(), reference.len());
            assert_eq!(
                set.iter().copied().collect::<Vec<_>>(),
                reference.iter().copied().collect::<Vec<_>>()
            );
            for probe in 0..70usize {
                assert_eq!(
                    set.is_member(probe),
                    reference.contains(&probe),
                    "probe {probe}"
                );
            }
            if let Some(&min) = reference.iter().next() {
                assert!(set.lower_bound() <= min, "lower bound must stay valid");
                assert!(
                    set.upper_bound() >= *reference.iter().next_back().unwrap(),
                    "upper bound must stay valid"
                );
            }
        }
    }
}
