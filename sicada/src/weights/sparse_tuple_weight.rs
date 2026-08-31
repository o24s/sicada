use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

use crate::error::ParseError;
use crate::utils::split_composite_weight;
use crate::weight::Weight;

pub const K_NO_KEY: i64 = -1;

/// Arbitrary dimension tuple weight, stored as a sorted vector.
///
/// `W` is any weight class, and `K` is the key value type (usually `i64`).
/// `K_NO_KEY` (`-1`) is reserved for internal/invalid use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseTupleWeight<W, K = i64> {
    default: W,
    /// Elements are strictly guaranteed to be sorted by `key` in ascending order,
    /// and no element in `rest` has a value equal to `default`.
    rest: Vec<(K, W)>,
}

impl<W: Weight, K: Copy + Ord + Hash + FromStr + fmt::Display> SparseTupleWeight<W, K> {
    /// Initializes an empty sparse tuple weight with a default value.
    #[inline(always)]
    pub fn new(default_value: W) -> Self {
        Self {
            default: default_value,
            rest: Vec::new(),
        }
    }

    /// Initializes from an iterator of `(K, W)`.
    /// The resulting `SparseTupleWeight` is automatically sorted and deduplicated.
    pub fn from_iter<I>(default_value: W, iter: I) -> Self
    where
        I: IntoIterator<Item = (K, W)>,
    {
        let mut rest: Vec<_> = iter
            .into_iter()
            .filter(|(_, w)| *w != default_value)
            .collect();
        rest.sort_unstable_by_key(|a| a.0);
        rest.dedup_by(|a, b| a.0 == b.0);

        Self {
            default: default_value,
            rest,
        }
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self::new(W::zero())
    }

    #[inline(always)]
    pub fn one() -> Self {
        Self::new(W::one())
    }

    #[inline(always)]
    pub fn no_weight() -> Self {
        Self::new(W::no_weight())
    }

    #[inline(always)]
    pub fn is_member(&self) -> bool {
        if !self.default.is_member() {
            return false;
        }
        self.rest.iter().all(|(_, w)| w.is_member())
    }

    pub fn quantize(&self, delta: f32) -> Self {
        let quantized_def = W::quantize(&self.default, delta);
        let quantized_rest: Vec<_> = self
            .rest
            .iter()
            .map(|(k, w)| (*k, W::quantize(w, delta)))
            .filter(|(_, w)| *w != quantized_def)
            .collect();

        Self {
            default: quantized_def,
            rest: quantized_rest,
        }
    }

    pub fn reverse(&self) -> SparseTupleWeight<W::ReverseWeight, K> {
        let reversed_def = W::reverse(&self.default);
        let reversed_rest: Vec<_> = self.rest.iter().map(|(k, w)| (*k, W::reverse(w))).collect();

        SparseTupleWeight {
            default: reversed_def,
            rest: reversed_rest,
        }
    }

    #[inline(always)]
    pub fn size(&self) -> usize {
        self.rest.len()
    }

    /// Returns the `key`-th component, or the default value if not set.
    #[inline]
    pub fn value(&self, key: K) -> &W {
        match self.rest.binary_search_by(|(k, _)| k.cmp(&key)) {
            Ok(idx) => &self.rest[idx].1,
            Err(_) => &self.default,
        }
    }

    #[inline(always)]
    pub fn default_value(&self) -> &W {
        &self.default
    }

    /// Appends a key/weight pair. Assumes the caller maintains sort order!
    /// Use `set_value` for safe unordered insertion.
    #[inline]
    pub fn push_back(&mut self, key: K, weight: W) {
        if weight != self.default {
            self.rest.push((key, weight));
        }
    }

    pub fn set_value(&mut self, key: K, weight: W) {
        match self.rest.binary_search_by(|(k, _)| k.cmp(&key)) {
            Ok(idx) => {
                if weight == self.default {
                    self.rest.remove(idx);
                } else {
                    self.rest[idx].1 = weight;
                }
            }
            Err(idx) => {
                if weight != self.default {
                    self.rest.insert(idx, (key, weight));
                }
            }
        }
    }

    pub fn set_default_value(&mut self, value: W) {
        self.default = value;
        // Purge any existing elements that happen to match the new default.
        self.rest.retain(|(_, w)| *w != self.default);
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, (K, W)> {
        self.rest.iter()
    }

    /// Maps two SparseTupleWeights into a new one by applying an `operator_mapper` closure.
    /// `M` must be a function: `Fn(Option<K>, &W, &W) -> W`.
    /// `Option<K>` will be `None` when calculating the new default value.
    pub fn map<M>(&self, other: &Self, mut operator_mapper: M) -> Self
    where
        M: FnMut(Option<K>, &W, &W) -> W,
    {
        let new_def = operator_mapper(None, &self.default, &other.default);
        let mut result = Self::new(new_def);

        let mut i = 0;
        let mut j = 0;

        while i < self.rest.len() && j < other.rest.len() {
            let (k1, w1) = &self.rest[i];
            let (k2, w2) = &other.rest[j];

            match k1.cmp(k2) {
                Ordering::Equal => {
                    let w = operator_mapper(Some(*k1), w1, w2);
                    result.push_back(*k1, w);
                    i += 1;
                    j += 1;
                }
                Ordering::Less => {
                    let w = operator_mapper(Some(*k1), w1, &other.default);
                    result.push_back(*k1, w);
                    i += 1;
                }
                Ordering::Greater => {
                    let w = operator_mapper(Some(*k2), &self.default, w2);
                    result.push_back(*k2, w);
                    j += 1;
                }
            }
        }

        while i < self.rest.len() {
            let (k1, w1) = &self.rest[i];
            let w = operator_mapper(Some(*k1), w1, &other.default);
            result.push_back(*k1, w);
            i += 1;
        }

        while j < other.rest.len() {
            let (k2, w2) = &other.rest[j];
            let w = operator_mapper(Some(*k2), &self.default, w2);
            result.push_back(*k2, w);
            j += 1;
        }

        result
    }
}

impl<W: Hash, K: Hash> Hash for SparseTupleWeight<W, K> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Implement custom hashing closely mirroring OpenFst to avoid discrepancy if needed,
        // or just use Rust's derived hash. Rust's derived Hash is robust and fast.
        self.default.hash(state);
        for pair in &self.rest {
            pair.0.hash(state);
            pair.1.hash(state);
        }
    }
}

impl<W: fmt::Display, K: fmt::Display> fmt::Display for SparseTupleWeight<W, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.default)?;
        for (k, w) in &self.rest {
            write!(f, ",{},{}", k, w)?;
        }
        Ok(())
    }
}

impl<W: FromStr + Weight, K: Copy + Ord + Hash + FromStr + fmt::Display> FromStr
    for SparseTupleWeight<W, K>
where
    <W as FromStr>::Err: Into<ParseError>,
    <K as FromStr>::Err: Into<ParseError>,
{
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts = split_composite_weight(s, ',', '(', ')')?;
        if parts.is_empty() {
            return Err(ParseError::InvalidElementCount {
                expected: 1,
                found: 0,
            });
        }

        let def = parts[0].parse::<W>().map_err(Into::into)?;
        let mut weight = Self::new(def);

        let mut i = 1;
        while i + 1 < parts.len() {
            let key = parts[i].parse::<K>().map_err(Into::into)?;
            let val = parts[i + 1].parse::<W>().map_err(Into::into)?;
            weight.push_back(key, val);
            i += 2;
        }

        if i < parts.len() {
            // Dangling key without a value
            return Err(ParseError::InvalidElementCount {
                expected: parts.len() + 1,
                found: parts.len(),
            });
        }

        Ok(weight)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    type Sparse = SparseTupleWeight<TropicalWeight, i64>;

    /// The representation only works if two things hold: the explicit entries
    /// are sorted by key, and none of them equals the default. A stored default
    /// would make two representations of the same weight compare unequal.
    fn assert_invariants(weight: &Sparse) {
        let keys: Vec<_> = weight.iter().map(|(key, _)| *key).collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(keys, sorted, "entries are not sorted by key");
        assert_eq!(
            keys.len(),
            keys.iter().collect::<std::collections::BTreeSet<_>>().len(),
            "duplicate keys"
        );
        for (key, value) in weight.iter() {
            assert_ne!(
                value,
                weight.default_value(),
                "key {key} stores the default explicitly"
            );
        }
    }

    /// Reads the sparse weight densely over a small key universe.
    fn dense(weight: &Sparse, universe: std::ops::Range<i64>) -> BTreeMap<i64, TropicalWeight> {
        universe.map(|key| (key, *weight.value(key))).collect()
    }

    #[test]
    fn an_unset_key_reads_as_the_default() {
        let weight = Sparse::new(TropicalWeight(7.0));
        assert_eq!(weight.value(0), &TropicalWeight(7.0));
        assert_eq!(weight.value(-100), &TropicalWeight(7.0));
        assert_eq!(weight.size(), 0);
    }

    #[test]
    fn setting_a_key_to_the_default_stores_nothing() {
        let mut weight = Sparse::new(TropicalWeight(7.0));
        weight.set_value(3, TropicalWeight(1.0));
        assert_eq!(weight.size(), 1);

        weight.set_value(3, TropicalWeight(7.0));
        assert_eq!(weight.size(), 0, "the default must not be stored");
        assert_eq!(weight.value(3), &TropicalWeight(7.0));
        assert_invariants(&weight);
    }

    #[test]
    fn changing_the_default_purges_entries_that_now_match_it() {
        let mut weight = Sparse::new(TropicalWeight::zero());
        weight.set_value(1, TropicalWeight(1.0));
        weight.set_value(2, TropicalWeight(2.0));

        weight.set_default_value(TropicalWeight(2.0));
        assert_eq!(weight.size(), 1, "the entry equal to the new default goes");
        assert_eq!(weight.value(1), &TropicalWeight(1.0));
        assert_eq!(weight.value(2), &TropicalWeight(2.0));
        assert_eq!(weight.value(99), &TropicalWeight(2.0));
        assert_invariants(&weight);
    }

    #[test]
    fn keys_stay_sorted_however_they_arrive() {
        let mut weight = Sparse::new(TropicalWeight::zero());
        for key in [5i64, 1, 9, 3, 7, 1] {
            weight.set_value(key, TropicalWeight(key as f32));
        }
        assert_invariants(&weight);
        assert_eq!(
            weight.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
            vec![1, 3, 5, 7, 9]
        );
    }

    #[test]
    fn membership_needs_the_default_and_every_entry() {
        let mut weight = Sparse::new(TropicalWeight::zero());
        weight.set_value(1, TropicalWeight(1.0));
        assert!(weight.is_member());

        weight.set_value(1, TropicalWeight::no_weight());
        assert!(!weight.is_member(), "a bad entry spoils it");

        let bad_default = Sparse::new(TropicalWeight::no_weight());
        assert!(!bad_default.is_member(), "a bad default spoils it");
    }

    #[test]
    fn quantize_and_reverse_reach_the_default_too() {
        let mut weight = Sparse::new(TropicalWeight(1.26));
        weight.set_value(4, TropicalWeight(1.24));

        let quantized = weight.quantize(0.5);
        assert_eq!(quantized.default_value(), &TropicalWeight(1.5));
        assert_eq!(quantized.value(4), &TropicalWeight(1.0));

        let reversed = weight.reverse();
        assert_eq!(reversed.default_value(), &TropicalWeight(1.26).reverse());
    }

    /// `map` merges two sorted entry lists while filling in each side's default
    /// where the other has an entry. Checked against evaluating the operation
    /// densely over the whole key universe.
    #[test]
    fn map_matches_a_dense_evaluation() {
        const UNIVERSE: std::ops::Range<i64> = 0..12;
        let mut state = 0x0BAD_C0FF_EE00_1234u64;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for round in 0..200 {
            let mut build = || {
                let mut weight = Sparse::new(TropicalWeight((rng() % 4) as f32));
                for key in UNIVERSE {
                    if rng() % 2 == 0 {
                        weight.set_value(key, TropicalWeight((rng() % 8) as f32));
                    }
                }
                weight
            };
            let left = build();
            let right = build();
            assert_invariants(&left);
            assert_invariants(&right);

            let product = left.map(&right, |_, a, b| a.times(b));
            assert_invariants(&product);

            let left_dense = dense(&left, UNIVERSE);
            let right_dense = dense(&right, UNIVERSE);
            let product_dense = dense(&product, UNIVERSE);
            for key in UNIVERSE {
                assert_eq!(
                    product_dense[&key],
                    left_dense[&key].times(&right_dense[&key]),
                    "round {round}, key {key}"
                );
            }
            // The default of the result is the operation on the two defaults, so
            // keys outside the universe agree as well.
            assert_eq!(
                product.default_value(),
                &left.default_value().times(right.default_value())
            );
        }
    }

    use super::*;
    use crate::float_weight::TropicalWeight;

    type SparseTropical = SparseTupleWeight<TropicalWeight, i64>;

    #[test]
    fn test_sparse_tuple_weight_parse_display() {
        let text = "0,1,5,3,10";
        let w = text.parse::<SparseTropical>().unwrap();

        assert_eq!(w.default_value().value(), 0.0);
        assert_eq!(w.size(), 2);
        assert_eq!(w.value(1).value(), 5.0);
        assert_eq!(w.value(2).value(), 0.0); // missing falls back to default
        assert_eq!(w.value(3).value(), 10.0);

        assert_eq!(w.to_string(), text);
    }

    #[test]
    fn test_sparse_tuple_weight_map() {
        let mut w1 = SparseTropical::new(TropicalWeight::zero());
        w1.set_value(1, TropicalWeight(2.0));
        w1.set_value(2, TropicalWeight(4.0));

        let mut w2 = SparseTropical::new(TropicalWeight::zero());
        w2.set_value(2, TropicalWeight(5.0));
        w2.set_value(3, TropicalWeight(6.0));

        // Let's implement Plus for TropicalWeight over the map
        let w3 = w1.map(&w2, |_, a, b| TropicalWeight::plus(a, b));

        assert_eq!(w3.default_value(), &TropicalWeight::zero());
        assert_eq!(w3.value(1), &TropicalWeight(2.0));
        assert_eq!(w3.value(2), &TropicalWeight(4.0)); // min(4, 5) = 4
        assert_eq!(w3.value(3), &TropicalWeight(6.0));
    }

    #[test]
    fn test_set_default_value_purges_matching() {
        let mut w = SparseTropical::new(TropicalWeight(1.0));
        w.set_value(1, TropicalWeight(5.0));
        w.set_value(2, TropicalWeight(2.0));

        assert_eq!(w.size(), 2);

        // Change default to 2.0. The element at key=2 should be purged.
        w.set_default_value(TropicalWeight(2.0));
        assert_eq!(w.size(), 1);
        assert_eq!(w.value(2).value(), 2.0); // returns default
    }
}
