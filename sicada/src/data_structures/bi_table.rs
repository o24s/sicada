//! Bijections between entries of an arbitrary type and a dense integer ID.
//!
//! Port of OpenFst's `bi-table.h`. Every table assigns IDs from 0 upwards and
//! supports looking up an ID from an entry and an entry from an ID. They differ
//! in how the entry-to-ID direction is stored, which is what suits one or
//! another to a given state table.
//!
//! Absence is expressed with `Option` throughout. Upstream encodes it either as
//! `-1` or as a stored `id + 1` with `0` meaning absent, an off-by-one that has
//! to be undone at every use site.

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};

use hashbrown::HashTable;
use rustc_hash::{FxHashMap, FxHasher};

/// A trait for integer types that can be used as BiTable IDs.
pub trait BiTableId: Copy + Eq {
    fn from_usize(val: usize) -> Self;
    fn as_usize(self) -> usize;
}

impl BiTableId for usize {
    #[inline(always)]
    fn from_usize(val: usize) -> Self {
        val
    }
    #[inline(always)]
    fn as_usize(self) -> usize {
        self
    }
}

impl BiTableId for i32 {
    #[inline(always)]
    fn from_usize(val: usize) -> Self {
        val as i32
    }
    #[inline(always)]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl BiTableId for u32 {
    #[inline(always)]
    fn from_usize(val: usize) -> Self {
        val as u32
    }
    #[inline(always)]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl BiTableId for i64 {
    #[inline(always)]
    fn from_usize(val: usize) -> Self {
        val as i64
    }
    #[inline(always)]
    fn as_usize(self) -> usize {
        self as usize
    }
}

/// An implementation using a hash map for the entry to ID mapping.
#[derive(Debug, Clone)]
pub struct HashBiTable<I, T> {
    entry2id: FxHashMap<T, I>,
    id2entry: Vec<T>,
}

impl<I, T> HashBiTable<I, T>
where
    T: Clone + Hash + Eq,
    I: BiTableId,
{
    pub fn new(table_size: usize) -> Self {
        Self {
            entry2id: FxHashMap::with_capacity_and_hasher(table_size, Default::default()),
            id2entry: Vec::with_capacity(table_size),
        }
    }

    #[inline]
    pub fn find_id(&mut self, entry: &T, insert: bool) -> Option<I> {
        if let Some(&id) = self.entry2id.get(entry) {
            return Some(id);
        }
        if insert {
            let id = I::from_usize(self.id2entry.len());
            self.id2entry.push(entry.clone());
            self.entry2id.insert(entry.clone(), id);
            Some(id)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn find_entry(&self, s: I) -> Option<&T> {
        self.id2entry.get(s.as_usize())
    }

    #[inline(always)]
    pub fn size(&self) -> usize {
        self.id2entry.len()
    }

    pub fn clear(&mut self) {
        self.entry2id.clear();
        self.id2entry.clear();
    }
}

impl<I, T> Default for HashBiTable<I, T>
where
    T: Clone + Hash + Eq,
    I: BiTableId,
{
    fn default() -> Self {
        Self::new(0)
    }
}

/// An implementation using a raw hash table for the entry to ID mapping.
///
/// By utilizing `hashbrown::HashTable`, we avoid storing the `T` elements
/// twice (unlike `HashBiTable`). It stores only the `usize` index, and looks
/// up the actual element in `id2entry` during hashing and equality checks.
/// This offers perfect cache locality and minimal memory footprint.
#[derive(Clone)]
pub struct CompactHashBiTable<I, T> {
    id2entry: Vec<T>,
    keys: HashTable<usize>,
    _phantom: std::marker::PhantomData<I>,
}

impl<I, T> CompactHashBiTable<I, T>
where
    T: Clone + Hash + Eq,
    I: BiTableId,
{
    pub fn new(table_size: usize) -> Self {
        Self {
            id2entry: Vec::with_capacity(table_size),
            keys: HashTable::with_capacity(table_size),
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn find_id(&mut self, entry: &T, insert: bool) -> Option<I> {
        let mut hasher = FxHasher::default();
        entry.hash(&mut hasher);
        let hash = hasher.finish();

        if let Some(&id) = self.keys.find(hash, |&id| self.id2entry[id] == *entry) {
            return Some(I::from_usize(id));
        }

        if insert {
            let id = self.id2entry.len();
            self.id2entry.push(entry.clone());

            self.keys.insert_unique(hash, id, |&i| {
                let mut h = FxHasher::default();
                self.id2entry[i].hash(&mut h);
                h.finish()
            });
            Some(I::from_usize(id))
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn find_entry(&self, s: I) -> Option<&T> {
        self.id2entry.get(s.as_usize())
    }

    #[inline(always)]
    pub fn size(&self) -> usize {
        self.id2entry.len()
    }

    pub fn clear(&mut self) {
        self.keys.clear();
        self.id2entry.clear();
    }
}

impl<I, T> Default for CompactHashBiTable<I, T>
where
    T: Clone + Hash + Eq,
    I: BiTableId,
{
    fn default() -> Self {
        Self::new(0)
    }
}

/// Maps an entry to the integer a [`VectorBiTable`] indexes it by.
///
/// SICADA-DIVERGE: upstream takes a functor object, which a closure cannot be
/// in Rust without boxing it, and a boxed fingerprint is an indirect call on
/// every lookup, which is the hottest path composition has. A trait lets a
/// named type carry its own state and be called directly; the blanket
/// implementation keeps closures working for the cases that need no state.
pub trait Fingerprint<T> {
    /// The index for `entry`. Must be injective over the entries a table will
    /// see, or two of them share a slot.
    fn fingerprint(&self, entry: &T) -> usize;
}

impl<T, F: Fn(&T) -> usize> Fingerprint<T> for F {
    #[inline(always)]
    fn fingerprint(&self, entry: &T) -> usize {
        self(entry)
    }
}

/// An implementation using a vector for the entry to ID mapping.
///
/// It requires a functor `fp` that uniquely fingerprints entries to an integer
/// (used as a direct vector index). This is the fastest approach when `T` maps
/// perfectly to a dense index space.
#[derive(Clone)]
pub struct VectorBiTable<I, T, FP> {
    fp2id: Vec<Option<I>>,
    id2entry: Vec<T>,
    fp: FP,
}

impl<I, T, FP> VectorBiTable<I, T, FP>
where
    T: Clone,
    I: BiTableId,
    FP: Fingerprint<T>,
{
    pub fn new(fp: FP, table_size: usize) -> Self {
        Self {
            fp2id: Vec::new(),
            id2entry: Vec::with_capacity(table_size),
            fp,
        }
    }

    /// SICADA-DIVERGE: upstream grows `fp2id_` to cover the fingerprint before
    /// checking `insert`, so a lookup that is not allowed to insert still
    /// enlarges the table permanently, and a sparse fingerprint makes that
    /// arbitrarily large. A non-inserting lookup returns without touching it
    /// here.
    #[inline]
    pub fn find_id(&mut self, entry: &T, insert: bool) -> Option<I> {
        let f = self.fp.fingerprint(entry);
        if f >= self.fp2id.len() {
            if insert {
                self.fp2id.resize(f + 1, None);
            } else {
                return None;
            }
        }

        if let Some(id) = self.fp2id[f] {
            Some(id)
        } else if insert {
            let id = I::from_usize(self.id2entry.len());
            self.id2entry.push(entry.clone());
            self.fp2id[f] = Some(id);
            Some(id)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn find_entry(&self, s: I) -> Option<&T> {
        self.id2entry.get(s.as_usize())
    }

    #[inline(always)]
    pub fn size(&self) -> usize {
        self.id2entry.len()
    }

    #[inline(always)]
    pub fn fingerprint(&self) -> &FP {
        &self.fp
    }
}

/// A hybrid table that conditionally uses a dense Vector index or a Compact Hash Table.
#[derive(Clone)]
pub struct VectorHashBiTable<I, T, S, FP> {
    id2entry: Vec<T>,
    fp2id: Vec<Option<I>>,
    keys: HashTable<usize>,
    selector: S,
    fp: FP,
    _phantom: std::marker::PhantomData<I>,
}

impl<I, T, S, FP> VectorHashBiTable<I, T, S, FP>
where
    T: Clone + Hash + Eq,
    I: BiTableId,
    S: Fn(&T) -> bool,
    FP: Fingerprint<T>,
{
    pub fn new(selector: S, fp: FP, vector_size: usize, entry_size: usize) -> Self {
        Self {
            id2entry: Vec::with_capacity(entry_size),
            fp2id: vec![None; vector_size],
            keys: HashTable::with_capacity(entry_size),
            selector,
            fp,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn find_id(&mut self, entry: &T, insert: bool) -> Option<I> {
        if (self.selector)(entry) {
            let f = self.fp.fingerprint(entry);
            if f >= self.fp2id.len() {
                if insert {
                    self.fp2id.resize(f + 1, None);
                } else {
                    return None;
                }
            }

            if let Some(id) = self.fp2id[f] {
                Some(id)
            } else if insert {
                let id = I::from_usize(self.id2entry.len());
                self.id2entry.push(entry.clone());
                self.fp2id[f] = Some(id);
                Some(id)
            } else {
                None
            }
        } else {
            let mut hasher = FxHasher::default();
            entry.hash(&mut hasher);
            let hash = hasher.finish();

            if let Some(&id_usize) = self.keys.find(hash, |&id| self.id2entry[id] == *entry) {
                return Some(I::from_usize(id_usize));
            }

            if insert {
                let id_usize = self.id2entry.len();
                self.id2entry.push(entry.clone());
                self.keys.insert_unique(hash, id_usize, |&i| {
                    let mut h = FxHasher::default();
                    self.id2entry[i].hash(&mut h);
                    h.finish()
                });
                Some(I::from_usize(id_usize))
            } else {
                None
            }
        }
    }

    #[inline(always)]
    pub fn find_entry(&self, s: I) -> Option<&T> {
        self.id2entry.get(s.as_usize())
    }

    #[inline(always)]
    pub fn size(&self) -> usize {
        self.id2entry.len()
    }

    #[inline(always)]
    pub fn selector(&self) -> &S {
        &self.selector
    }

    #[inline(always)]
    pub fn fingerprint(&self) -> &FP {
        &self.fp
    }
}

/// A BiTable implementation that permits the erasure of arbitrary IDs.
#[derive(Debug, Clone)]
pub struct ErasableBiTable<I, T> {
    entry2id: FxHashMap<T, I>,
    id2entry: VecDeque<T>,
    empty_entry: T,
    first: usize,
}

impl<I, T> ErasableBiTable<I, T>
where
    T: Clone + Hash + Eq,
    I: BiTableId,
{
    pub fn new(empty_entry: T) -> Self {
        Self {
            entry2id: FxHashMap::default(),
            id2entry: VecDeque::new(),
            empty_entry,
            first: 0,
        }
    }

    /// SICADA-DIVERGE: upstream reaches the map with `operator[]`, which inserts
    /// a zero for an entry that is merely being queried, so a non-inserting
    /// lookup still leaves a permanent record of it. This only reads.
    #[inline]
    pub fn find_id(&mut self, entry: &T, insert: bool) -> Option<I> {
        if let Some(&id) = self.entry2id.get(entry) {
            return Some(id);
        }

        if insert {
            let id = I::from_usize(self.id2entry.len() + self.first);
            self.id2entry.push_back(entry.clone());
            self.entry2id.insert(entry.clone(), id);
            Some(id)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn find_entry(&self, s: I) -> Option<&T> {
        let s_usize = s.as_usize();
        if s_usize >= self.first {
            self.id2entry.get(s_usize - self.first)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn size(&self) -> usize {
        self.id2entry.len()
    }

    pub fn erase(&mut self, s: I) {
        let s_usize = s.as_usize();
        if s_usize >= self.first && s_usize - self.first < self.id2entry.len() {
            let idx = s_usize - self.first;
            self.entry2id.remove(&self.id2entry[idx]);
            self.id2entry[idx] = self.empty_entry.clone();

            while let Some(front) = self.id2entry.front() {
                if *front == self.empty_entry {
                    self.id2entry.pop_front();
                    self.first += 1;
                } else {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compact_hash_bi_table() {
        let mut table = CompactHashBiTable::<i32, String>::new(0);

        // Find non-existent without insert
        assert_eq!(table.find_id(&"foo".to_string(), false), None);

        // Find with insert
        let id_foo = table.find_id(&"foo".to_string(), true).unwrap();
        assert_eq!(id_foo, 0);

        let id_bar = table.find_id(&"bar".to_string(), true).unwrap();
        assert_eq!(id_bar, 1);

        // Find existing without insert
        assert_eq!(table.find_id(&"foo".to_string(), false), Some(0));

        // Look up entry
        assert_eq!(table.find_entry(0), Some(&"foo".to_string()));
        assert_eq!(table.find_entry(1), Some(&"bar".to_string()));
        assert_eq!(table.find_entry(2), None);

        assert_eq!(table.size(), 2);
    }

    #[test]
    fn test_erasable_bi_table() {
        let mut table = ErasableBiTable::<i32, String>::new("EMPTY".to_string());

        let id0 = table.find_id(&"a".to_string(), true).unwrap();
        let id1 = table.find_id(&"b".to_string(), true).unwrap();

        assert_eq!(id0, 0);
        assert_eq!(id1, 1);

        table.erase(0);

        // Now 'a' is erased, first valid element shifts or is marked empty
        assert_eq!(table.find_entry(0), None);
        assert_eq!(table.find_entry(1), Some(&"b".to_string()));

        // Re-inserting 'a' should give a new ID
        let id_a2 = table.find_id(&"a".to_string(), true).unwrap();
        assert_eq!(id_a2, 2);
    }
    /// Every table type must satisfy the same contract: IDs are handed out from
    /// zero in order, the same entry always maps back to the same ID, and a
    /// non-inserting lookup of an unknown entry answers `None` without
    /// allocating an ID.
    macro_rules! bitable_contract {
        ($name:ident, $make:expr, $entry:expr) => {
            #[test]
            fn $name() {
                let mut table = $make;
                let entries: Vec<_> = (0..8).map($entry).collect();

                for (expected_id, entry) in entries.iter().enumerate() {
                    assert_eq!(table.find_id(entry, false), None, "not inserted yet");
                    assert_eq!(table.find_id(entry, true), Some(expected_id));
                    assert_eq!(table.size(), expected_id + 1);
                }

                // Lookups are stable and do not allocate new IDs.
                for (expected_id, entry) in entries.iter().enumerate() {
                    assert_eq!(table.find_id(entry, true), Some(expected_id));
                    assert_eq!(table.find_id(entry, false), Some(expected_id));
                    assert_eq!(table.find_entry(expected_id), Some(entry));
                }
                assert_eq!(table.size(), entries.len());
                assert_eq!(table.find_entry(entries.len()), None);
            }
        };
    }

    bitable_contract!(
        hash_bi_table_follows_the_contract,
        HashBiTable::<usize, String>::new(0),
        |i: usize| format!("entry-{i}")
    );
    bitable_contract!(
        compact_hash_bi_table_follows_the_contract,
        CompactHashBiTable::<usize, String>::new(0),
        |i: usize| format!("entry-{i}")
    );
    bitable_contract!(
        vector_bi_table_follows_the_contract,
        VectorBiTable::<usize, usize, _>::new(|entry: &usize| *entry * 3, 0),
        |i: usize| i
    );
    bitable_contract!(
        erasable_bi_table_follows_the_contract,
        ErasableBiTable::<usize, String>::new(String::new()),
        |i: usize| format!("entry-{i}")
    );

    /// The selector decides which half of the table an entry lands in; both
    /// halves must share one ID space.
    #[test]
    fn vector_hash_bi_table_shares_one_id_space_across_both_halves() {
        let mut table = VectorHashBiTable::<usize, usize, _, _>::new(
            |entry: &usize| entry.is_multiple_of(2), // even entries go to the vector
            |entry: &usize| *entry / 2,
            0,
            0,
        );

        assert_eq!(table.find_id(&0, true), Some(0)); // vector half
        assert_eq!(table.find_id(&1, true), Some(1)); // hash half
        assert_eq!(table.find_id(&2, true), Some(2)); // vector half
        assert_eq!(table.find_id(&3, true), Some(3)); // hash half

        for entry in 0..4usize {
            assert_eq!(table.find_id(&entry, false), Some(entry));
            assert_eq!(table.find_entry(entry), Some(&entry));
        }
        assert_eq!(table.size(), 4);
        assert_eq!(table.find_id(&9, false), None);
    }

    /// A non-inserting lookup must not enlarge the fingerprint vector; upstream
    /// resizes before it checks `insert`, so a sparse fingerprint grows the
    /// table without ever storing anything.
    #[test]
    fn vector_bi_table_does_not_grow_on_a_failed_lookup() {
        let mut table = VectorBiTable::<usize, usize, _>::new(|entry: &usize| *entry, 0);
        assert_eq!(table.find_id(&1_000_000, false), None);
        assert_eq!(table.size(), 0);
        // The first real insertion still starts at ID 0.
        assert_eq!(table.find_id(&5, true), Some(0));
    }

    #[test]
    fn clearing_restarts_the_id_space() {
        let mut table = CompactHashBiTable::<usize, String>::new(0);
        assert_eq!(table.find_id(&"a".to_string(), true), Some(0));
        assert_eq!(table.find_id(&"b".to_string(), true), Some(1));
        table.clear();
        assert_eq!(table.size(), 0);
        assert_eq!(table.find_id(&"b".to_string(), true), Some(0));
        assert_eq!(table.find_entry(0), Some(&"b".to_string()));
    }

    /// Erasing releases the entry but keeps the IDs of everything else, and the
    /// front of the deque is trimmed only as far as the erased prefix reaches.
    #[test]
    fn erasing_keeps_the_remaining_ids_stable() {
        let mut table = ErasableBiTable::<usize, String>::new(String::new());
        let ids: Vec<_> = ["a", "b", "c", "d"]
            .iter()
            .map(|e| table.find_id(&e.to_string(), true).unwrap())
            .collect();
        assert_eq!(ids, vec![0, 1, 2, 3]);

        // Erase from the middle: the front cannot be trimmed yet.
        table.erase(2);
        assert_eq!(table.find_id(&"c".to_string(), false), None);
        assert_eq!(table.find_entry(3), Some(&"d".to_string()));
        assert_eq!(table.find_entry(0), Some(&"a".to_string()));

        // Erasing the front trims it, and everything else keeps its ID.
        table.erase(0);
        assert_eq!(table.find_entry(0), None, "0 was trimmed off the front");
        assert_eq!(table.find_entry(1), Some(&"b".to_string()));
        assert_eq!(table.find_entry(3), Some(&"d".to_string()));
        assert_eq!(table.find_id(&"d".to_string(), false), Some(3));

        // A new entry continues after the highest ID handed out so far.
        assert_eq!(table.find_id(&"e".to_string(), true), Some(4));
    }

    /// Entries that hash the same but are not equal must get distinct IDs.
    #[test]
    fn compact_hash_bi_table_separates_colliding_entries() {
        // A deliberately terrible fingerprint: every entry hashes identically.
        #[derive(Clone, PartialEq, Eq, Debug)]
        struct Collides(u32);
        impl Hash for Collides {
            fn hash<H: Hasher>(&self, state: &mut H) {
                0u8.hash(state);
            }
        }

        let mut table = CompactHashBiTable::<usize, Collides>::new(0);
        for value in 0..64u32 {
            assert_eq!(table.find_id(&Collides(value), true), Some(value as usize));
        }
        for value in 0..64u32 {
            assert_eq!(table.find_id(&Collides(value), false), Some(value as usize));
            assert_eq!(table.find_entry(value as usize), Some(&Collides(value)));
        }
        assert_eq!(table.size(), 64);
    }
}
