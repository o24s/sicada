// TODO(porting-iteration): drop this once the porting iteration reaches this file.
#![allow(unused)]

use std::iter;
use std::slice;

type Word = u64;
const WORD_BYTES: usize = size_of::<Word>();
const WORD_BITS: usize = WORD_BYTES * 8;

#[inline(always)]
fn num_words(domain_size: usize) -> usize {
    domain_size.div_ceil(WORD_BITS)
}

#[inline(always)]
fn word_index_and_mask(index: usize) -> (usize, Word) {
    (index / WORD_BITS, 1 << (index % WORD_BITS))
}

/// Clears any bits outside the domain in the final word.
/// This ensures operations like `count` or equality checks evaluate correctly.
#[inline]
fn clear_excess_bits_in_final_word(domain_size: usize, words: &mut [Word]) {
    let num_bits_in_final_word = domain_size % WORD_BITS;
    if num_bits_in_final_word > 0
        && let Some(last) = words.last_mut()
    {
        let mask = (1 << num_bits_in_final_word) - 1;
        *last &= mask;
    }
}

/// Applies a bitwise operation to two word slices.
/// Returns `true` if `lhs` was modified.
///
/// Panics if `lhs` and `rhs` have different lengths.
#[inline]
fn update_words<Op>(lhs: &mut [Word], rhs: &[Word], op: Op) -> bool
where
    Op: Fn(Word, Word) -> Word,
{
    assert_eq!(lhs.len(), rhs.len());
    let mut changed = 0;
    for (lhs_word, &rhs_word) in iter::zip(lhs, rhs) {
        let old = *lhs_word;
        let new = op(old, rhs_word);
        *lhs_word = new;
        changed |= old ^ new;
    }
    changed != 0
}

/// A trait for performing set operations on bitsets.
pub trait BitRelations<Rhs> {
    /// Sets `self = self | other`. Returns `true` if `self` was modified.
    fn union(&mut self, other: &Rhs) -> bool;
    /// Sets `self = self - other`. Returns `true` if `self` was modified.
    fn subtract(&mut self, other: &Rhs) -> bool;
    /// Sets `self = self & other`. Returns `true` if `self` was modified.
    fn intersect(&mut self, other: &Rhs) -> bool;
}

macro_rules! bit_relations_inherent_impls {
    () => {
        #[inline]
        pub fn union<Rhs>(&mut self, other: &Rhs) -> bool
        where
            Self: BitRelations<Rhs>,
        {
            <Self as BitRelations<Rhs>>::union(self, other)
        }

        #[inline]
        pub fn subtract<Rhs>(&mut self, other: &Rhs) -> bool
        where
            Self: BitRelations<Rhs>,
        {
            <Self as BitRelations<Rhs>>::subtract(self, other)
        }

        #[inline]
        pub fn intersect<Rhs>(&mut self, other: &Rhs) -> bool
        where
            Self: BitRelations<Rhs>,
        {
            <Self as BitRelations<Rhs>>::intersect(self, other)
        }
    };
}

/// A fixed-capacity bitset backed by a dense array of words.
///
/// Memory is allocated once at creation. Operations that attempt to access
/// indices greater than or equal to the domain size will panic.
#[derive(Default, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DenseBitSet {
    domain_size: usize,
    words: Vec<Word>,
}

impl DenseBitSet {
    /// Creates a new, empty bitset with the specified fixed capacity.
    #[inline]
    pub fn new_empty(domain_size: usize) -> Self {
        Self {
            domain_size,
            words: vec![0; num_words(domain_size)],
        }
    }

    /// Creates a new bitset with all bits up to `domain_size` set to 1.
    #[inline]
    pub fn new_filled(domain_size: usize) -> Self {
        let mut result = Self {
            domain_size,
            words: vec![!0; num_words(domain_size)],
        };
        clear_excess_bits_in_final_word(domain_size, &mut result.words);
        result
    }

    /// Returns the maximum capacity of the bitset.
    #[inline]
    pub fn domain_size(&self) -> usize {
        self.domain_size
    }

    /// Returns the number of set bits.
    #[inline]
    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Returns `true` if the bitset contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Clears all elements in the bitset.
    #[inline]
    pub fn clear(&mut self) {
        self.words.fill(0);
    }

    /// Returns `true` if the bitset contains `index`.
    ///
    /// Panics if `index` is greater than or equal to the domain size.
    #[inline]
    pub fn contains(&self, index: usize) -> bool {
        assert!(index < self.domain_size, "Index out of bounds");
        let (word_index, mask) = word_index_and_mask(index);
        (self.words[word_index] & mask) != 0
    }

    /// Inserts `index` into the bitset. Returns `true` if the bit was newly inserted.
    ///
    /// Panics if `index` is greater than or equal to the domain size.
    #[inline]
    pub fn insert(&mut self, index: usize) -> bool {
        assert!(index < self.domain_size, "Index out of bounds");
        let (word_index, mask) = word_index_and_mask(index);
        let word = &mut self.words[word_index];
        let old = *word;
        *word |= mask;
        old != *word
    }

    /// Removes `index` from the bitset. Returns `true` if the bit was present.
    ///
    /// Panics if `index` is greater than or equal to the domain size.
    #[inline]
    pub fn remove(&mut self, index: usize) -> bool {
        assert!(index < self.domain_size, "Index out of bounds");
        let (word_index, mask) = word_index_and_mask(index);
        let word = &mut self.words[word_index];
        let old = *word;
        *word &= !mask;
        old != *word
    }

    /// Returns an iterator over the indices of set bits in ascending order.
    #[inline]
    pub fn iter(&self) -> BitIter<'_> {
        BitIter::new(&self.words)
    }

    bit_relations_inherent_impls! {}
}

impl BitRelations<DenseBitSet> for DenseBitSet {
    fn union(&mut self, other: &DenseBitSet) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        update_words(&mut self.words, &other.words, |a, b| a | b)
    }

    fn subtract(&mut self, other: &DenseBitSet) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        update_words(&mut self.words, &other.words, |a, b| a & !b)
    }

    fn intersect(&mut self, other: &DenseBitSet) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        update_words(&mut self.words, &other.words, |a, b| a & b)
    }
}

/// A dynamic bitset that automatically expands its capacity.
///
/// It wraps a `DenseBitSet` and resizes the underlying storage when an
/// operation requires an index outside the current domain size.
#[derive(Default, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GrowableBitSet {
    bit_set: DenseBitSet,
}

impl GrowableBitSet {
    /// Creates a new, empty growable bitset.
    #[inline]
    pub fn new() -> Self {
        Self {
            bit_set: DenseBitSet::new_empty(0),
        }
    }

    /// Creates a new, empty growable bitset with the specified initial capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bit_set: DenseBitSet::new_empty(capacity),
        }
    }

    /// Ensures that the bitset can hold at least `min_domain_size` elements.
    #[inline]
    pub fn ensure(&mut self, min_domain_size: usize) {
        if self.bit_set.domain_size < min_domain_size {
            self.bit_set.domain_size = min_domain_size;
            let required_words = num_words(min_domain_size);
            if self.bit_set.words.len() < required_words {
                self.bit_set.words.resize(required_words, 0);
            }
        }
    }

    /// Inserts `index` into the bitset, resizing if necessary.
    /// Returns `true` if the bit was newly inserted.
    #[inline]
    pub fn insert(&mut self, index: usize) -> bool {
        self.ensure(index + 1);
        self.bit_set.insert(index)
    }

    /// Removes `index` from the bitset. Returns `true` if the bit was present.
    #[inline]
    pub fn remove(&mut self, index: usize) -> bool {
        if index >= self.bit_set.domain_size {
            return false;
        }
        self.bit_set.remove(index)
    }

    /// Returns `true` if the bitset contains `index`.
    #[inline]
    pub fn contains(&self, index: usize) -> bool {
        if index >= self.bit_set.domain_size {
            return false;
        }
        self.bit_set.contains(index)
    }

    /// Clears all elements in the bitset.
    #[inline]
    pub fn clear(&mut self) {
        self.bit_set.clear();
    }

    /// Returns the number of set bits.
    #[inline]
    pub fn count(&self) -> usize {
        self.bit_set.count()
    }

    /// Returns `true` if the bitset contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bit_set.is_empty()
    }

    /// Returns the maximum capacity currently tracking.
    #[inline]
    pub fn domain_size(&self) -> usize {
        self.bit_set.domain_size
    }

    /// Returns an iterator over the indices of set bits in ascending order.
    #[inline]
    pub fn iter(&self) -> BitIter<'_> {
        self.bit_set.iter()
    }
}

impl BitRelations<GrowableBitSet> for GrowableBitSet {
    fn union(&mut self, other: &GrowableBitSet) -> bool {
        self.ensure(other.domain_size());

        let min_words = other.bit_set.words.len();

        update_words(
            &mut self.bit_set.words[..min_words],
            &other.bit_set.words[..min_words],
            |a, b| a | b,
        )
    }

    fn subtract(&mut self, other: &GrowableBitSet) -> bool {
        let min_words = std::cmp::min(self.bit_set.words.len(), other.bit_set.words.len());
        update_words(
            &mut self.bit_set.words[..min_words],
            &other.bit_set.words[..min_words],
            |a, b| a & !b,
        )
    }

    fn intersect(&mut self, other: &GrowableBitSet) -> bool {
        let min_words = std::cmp::min(self.bit_set.words.len(), other.bit_set.words.len());
        let changed = update_words(
            &mut self.bit_set.words[..min_words],
            &other.bit_set.words[..min_words],
            |a, b| a & b,
        );

        if self.bit_set.words.len() > min_words {
            self.bit_set.words[min_words..].fill(0);
        }

        changed || (self.bit_set.words.len() > min_words)
    }
}

/// An iterator over the set bits of a bitset.
///
/// Uses CPU trailing zero instructions to locate the next set bit in O(1) time
/// per word traversed.
pub struct BitIter<'a> {
    word: Word,
    offset: usize,
    iter: slice::Iter<'a, Word>,
}

impl<'a> BitIter<'a> {
    #[inline]
    fn new(words: &'a [Word]) -> Self {
        Self {
            word: 0,
            offset: usize::MAX.wrapping_sub(WORD_BITS - 1),
            iter: words.iter(),
        }
    }
}

impl<'a> Iterator for BitIter<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.word != 0 {
                let bit_pos = self.word.trailing_zeros() as usize;
                self.word ^= 1 << bit_pos;
                return Some(bit_pos + self.offset);
            }

            self.word = *self.iter.next()?;
            self.offset = self.offset.wrapping_add(WORD_BITS);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_bitset() {
        let mut bs = DenseBitSet::new_empty(100);
        assert!(bs.is_empty());

        assert!(bs.insert(10));
        assert!(!bs.insert(10));
        assert!(bs.contains(10));

        let mut bs2 = DenseBitSet::new_empty(100);
        bs2.insert(10);
        bs2.insert(20);

        assert!(bs.union(&bs2));
        assert!(bs.contains(20));

        assert_eq!(bs.iter().collect::<Vec<_>>(), vec![10, 20]);
    }

    #[test]
    fn test_growable_bitset() {
        let mut bs = GrowableBitSet::new();
        assert!(bs.insert(5));
        assert!(bs.insert(1024)); // Auto-resizes
        assert_eq!(bs.count(), 2);

        let mut bs2 = GrowableBitSet::new();
        bs2.insert(1024);
        bs2.insert(2048);

        bs.union(&bs2); // Auto-resizes self to match bs2
        assert!(bs.contains(2048));
        assert_eq!(bs.count(), 3);
    }
}
