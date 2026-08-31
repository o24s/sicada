//! Binary heap that supports in-place update of values through a key.
//!
//! Port of OpenFst's `heap.h`. Unlike `std::collections::BinaryHeap`, every
//! insertion returns a `key` that stays valid while the element is in the heap
//! and can be used to update its value in place: the decrease-key operation
//! that Dijkstra-style shortest-distance algorithms need.
//!
//! Positions are tracked in dense vectors rather than a hash map, so a key
//! lookup is a single indexed load.

/// Key value that never identifies an element in the heap.
pub const K_NO_KEY: usize = usize::MAX;

/// Set in a `pos` entry whose key is on the free list; the remaining bits hold
/// the next free key.
const FREE_FLAG: Key = 1 << 31;

/// End of the free list, and the exclusive upper bound on a heap index.
const NO_FREE: Key = FREE_FLAG - 1;

/// Internal key width.
///
/// SICADA-OPT: a key names a slot in a live heap, so `u32` covers more entries
/// than any queue will hold while halving the `pos` table against upstream's
/// `std::vector<int>`-sized-but-`usize`-in-Rust alternative. `pos` is walked on
/// every swap, so its footprint is cache traffic in the hot loop.
type Key = u32;

#[derive(Debug, Clone)]
struct Node<T> {
    value: T,
    key: Key,
}

/// A min-heap with respect to `comp`, supporting in-place updates via keys.
///
/// `C` is a comparison functor `Fn(&T, &T) -> bool`; `comp(a, b)` is true when
/// `a` has higher priority than `b` and must therefore sit above it.
///
/// # Storage
///
/// SICADA-OPT: upstream keeps popped elements constructed in its backing vector
/// and copies the value out of `Pop()`, which forces a copy of the weight on
/// every pop and makes the heap's live set implicit. Here `elements` holds
/// exactly the live entries and `pop` moves the value out, while freed keys are
/// recycled through `free_keys` so the `pos` table still stops growing. Pops
/// cost a move instead of a copy, which matters once `T` carries an owning
/// weight (`StringWeight`, `SparseTupleWeight`, ...), and no element outlives
/// its pop.
#[derive(Clone)]
pub struct IndexedHeap<T, C> {
    /// Live elements, in heap order.
    elements: Vec<Node<T>>,
    /// Maps a live key to its index in `elements`. A freed key instead holds
    /// [`FREE_FLAG`] set over the next free key, threading the free list through
    /// this table.
    ///
    /// SICADA-OPT: an explicit `Vec` of freed keys costs a push per pop, a
    /// separate allocation, and its own growth. The table already has a slot per
    /// key doing nothing while that key is free, so the list lives there.
    pos: Vec<Key>,
    /// Head of the free list, or [`NO_FREE`] when every key is in use.
    free_head: Key,
    /// Comparison function determining heap order.
    comp: C,
}

impl<T, C> IndexedHeap<T, C>
where
    C: Fn(&T, &T) -> bool,
{
    /// Creates an empty heap ordered by `comp`.
    pub fn new(comp: C) -> Self {
        Self {
            elements: Vec::new(),
            pos: Vec::new(),
            free_head: NO_FREE,
            comp,
        }
    }

    /// Reserves room for `capacity` elements.
    pub fn reserve(&mut self, capacity: usize) {
        self.elements.reserve(capacity);
        self.pos.reserve(capacity);
    }

    /// Inserts a value and returns the key identifying it.
    ///
    /// The key stays valid until the value is popped or the heap is cleared.
    #[inline]
    pub fn insert(&mut self, value: T) -> usize {
        let index = self.elements.len();
        debug_assert!(index < NO_FREE as usize, "heap larger than the key space");
        let index = index as Key;
        let key = if self.free_head != NO_FREE {
            let key = self.free_head;
            self.free_head = self.pos[key as usize] & !FREE_FLAG;
            self.pos[key as usize] = index;
            key
        } else {
            self.pos.push(index);
            (self.pos.len() - 1) as Key
        };
        self.elements.push(Node { value, key });
        self.sift_up(index as usize);
        key as usize
    }

    /// Replaces the value stored under `key`, restoring the heap order.
    ///
    /// # Panics
    ///
    /// Panics if `key` does not currently identify an element in the heap.
    #[inline]
    pub fn update(&mut self, key: usize, value: T) {
        // One bounds check on `pos`, then one range check that also rejects a
        // freed key: its entry has FREE_FLAG set, which puts it past any live
        // index. Cheaper than an `Option` dance on a path shortest-distance runs
        // per arc, and a stale key still panics rather than reaching a bad access.
        let index = self.pos[key] as usize;
        assert!(
            index < self.elements.len(),
            "update on a key not in the heap"
        );
        // SAFETY: `index_of` only answers with a position recorded in `pos`, and
        // `pos` is kept in step with `elements` by `swap_unchecked`, `insert` and
        // `pop`.
        unsafe { self.elements.get_unchecked_mut(index).value = value };
        // Sifting up is only possible if the new value outranks its parent;
        // otherwise the element can only have moved down.
        if index > 0 {
            let parent = Self::parent(index);
            // SAFETY: as above; `parent < index < elements.len()`.
            let outranks_parent = unsafe {
                (self.comp)(
                    &self.elements.get_unchecked(index).value,
                    &self.elements.get_unchecked(parent).value,
                )
            };
            if outranks_parent {
                self.sift_up(index);
                return;
            }
        }
        self.sift_down(index);
    }

    /// Removes and returns the highest-priority value, or `None` if empty.
    ///
    /// SICADA-OPT: upstream swaps the root with the last element and then
    /// shrinks, which writes `pos` for an element that is about to be discarded.
    /// Taking the last element out first and dropping it into the root writes
    /// `pos` only for the element that actually moved.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        let last = self.elements.pop()?;
        if self.elements.is_empty() {
            // It was the only element, so it is also the top.
            self.release(last.key);
            return Some(last.value);
        }

        let moved_key = last.key;
        // SAFETY: `elements` is non-empty, checked immediately above.
        let root = std::mem::replace(unsafe { self.elements.get_unchecked_mut(0) }, last);
        // SAFETY: a key held by a live element always indexes `pos`.
        unsafe { *self.pos.get_unchecked_mut(moved_key as usize) = 0 };
        self.release(root.key);
        self.sift_down(0);
        Some(root.value)
    }

    /// Returns the highest-priority value without removing it.
    #[inline]
    pub fn top(&self) -> Option<&T> {
        self.elements.first().map(|node| &node.value)
    }

    /// Returns the value stored under `key`, or `None` if the key is not live.
    #[inline]
    pub fn get(&self, key: usize) -> Option<&T> {
        let index = self.index_of(key)?;
        Some(&self.elements[index].value)
    }

    /// Whether the heap holds no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Number of elements in the heap.
    #[inline]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Removes every element, keeping the allocated capacity and recycling all
    /// keys for reuse.
    pub fn clear(&mut self) {
        while let Some(node) = self.elements.pop() {
            self.release(node.key);
        }
    }

    /// Returns the comparator.
    #[inline]
    pub fn comparator(&self) -> &C {
        &self.comp
    }

    #[inline(always)]
    fn index_of(&self, key: usize) -> Option<usize> {
        match self.pos.get(key) {
            Some(&index) if index & FREE_FLAG == 0 => Some(index as usize),
            _ => None,
        }
    }

    /// Parent of `i`, which must be non-zero, unlike the C++ original, where
    /// `Parent(0)` relies on `(-1) / 2` truncating to 0.
    #[inline(always)]
    fn parent(i: usize) -> usize {
        debug_assert!(i > 0, "the root has no parent");
        (i - 1) / 2
    }

    /// Returns `key` to the free list.
    #[inline(always)]
    fn release(&mut self, key: Key) {
        // SAFETY: a key taken off a live element always indexes `pos`.
        unsafe { *self.pos.get_unchecked_mut(key as usize) = FREE_FLAG | self.free_head };
        self.free_head = key;
    }

    /// Restores the heap order upwards from `index`.
    fn sift_up(&mut self, index: usize) {
        if index == 0 {
            return;
        }
        let Self {
            elements,
            pos,
            comp,
            ..
        } = self;
        // SAFETY: callers only pass an index of a live element.
        let mut hole = unsafe { Hole::new(elements, pos.as_mut_slice(), index) };
        while hole.index > 0 {
            let parent = (hole.index - 1) / 2;
            // SAFETY: `parent` is strictly below `hole.index`, so it is in range
            // and is not the hole itself.
            let outranks = unsafe { comp(hole.value(), hole.element(parent)) };
            if !outranks {
                break;
            }
            // SAFETY: as above.
            unsafe { hole.move_to(parent) };
        }
    }

    /// Restores the heap order downwards from `index`.
    fn sift_down(&mut self, index: usize) {
        let len = self.elements.len();
        if index >= len {
            return;
        }
        let Self {
            elements,
            pos,
            comp,
            ..
        } = self;
        // SAFETY: `index < len`, checked above.
        let mut hole = unsafe { Hole::new(elements, pos.as_mut_slice(), index) };
        loop {
            let left = 2 * hole.index + 1;
            if left >= len {
                break;
            }
            let right = left + 1;
            // SAFETY: both children are strictly above the hole and below `len`.
            let best = unsafe {
                if right < len && comp(hole.element(right), hole.element(left)) {
                    right
                } else {
                    left
                }
            };
            // SAFETY: `best` is in range, as just established.
            if !unsafe { comp(hole.element(best), hole.value()) } {
                break;
            }
            // SAFETY: as above.
            unsafe { hole.move_to(best) };
        }
    }

    /// Whether the heap order holds everywhere and the key table agrees with the
    /// element positions. Test-only.
    #[cfg(test)]
    fn is_consistent(&self) -> bool {
        for i in 1..self.elements.len() {
            if (self.comp)(
                &self.elements[i].value,
                &self.elements[Self::parent(i)].value,
            ) {
                return false;
            }
        }
        for (index, node) in self.elements.iter().enumerate() {
            if self.pos[node.key as usize] != index as Key {
                return false;
            }
        }
        true
    }
}

/// Lifts one element out of the heap, leaving a hole that is filled again when
/// the hole is dropped, including if the comparator panics part way through.
///
/// SICADA-OPT: sifting by swapping writes both elements at every level. Sifting
/// with a hole writes one per level and writes the lifted element exactly once,
/// at the end. Upstream swaps; `std::collections::BinaryHeap` uses this shape.
struct Hole<'a, T> {
    elements: &'a mut [Node<T>],
    pos: &'a mut [Key],
    /// The element that was lifted out. Owned by the hole until it is dropped.
    node: std::mem::ManuallyDrop<Node<T>>,
    /// Index of the slot that is currently logically uninitialized.
    index: usize,
}

impl<'a, T> Hole<'a, T> {
    /// # Safety
    ///
    /// `index` must be a valid index into `elements`, and every key held by an
    /// element must index `pos`.
    #[inline(always)]
    unsafe fn new(elements: &'a mut Vec<Node<T>>, pos: &'a mut [Key], index: usize) -> Self {
        debug_assert!(index < elements.len());
        // SAFETY: `index` is in range, so the slot holds a valid `Node`. It is
        // not read again until `Drop` puts one back.
        let node = unsafe { std::ptr::read(elements.as_ptr().add(index)) };
        Self {
            elements: elements.as_mut_slice(),
            pos,
            node: std::mem::ManuallyDrop::new(node),
            index,
        }
    }

    /// The value that was lifted out.
    #[inline(always)]
    fn value(&self) -> &T {
        &self.node.value
    }

    /// # Safety
    ///
    /// `index` must be in range and must not be the hole itself.
    #[inline(always)]
    unsafe fn element(&self, index: usize) -> &T {
        debug_assert!(index < self.elements.len() && index != self.index);
        // SAFETY: guaranteed by the caller; the slot is initialized because it
        // is not the hole.
        unsafe { &self.elements.get_unchecked(index).value }
    }

    /// Moves the element at `target` into the hole, and the hole to `target`.
    ///
    /// # Safety
    ///
    /// `target` must be in range and must not be the hole itself.
    #[inline(always)]
    unsafe fn move_to(&mut self, target: usize) {
        debug_assert!(target < self.elements.len() && target != self.index);
        // SAFETY: both indices are in range and distinct, so the copy does not
        // overlap; the moved element's key indexes `pos` by the invariant stated
        // on `new`.
        unsafe {
            let ptr = self.elements.as_mut_ptr();
            std::ptr::copy_nonoverlapping(ptr.add(target), ptr.add(self.index), 1);
            let key = (*ptr.add(self.index)).key as usize;
            *self.pos.get_unchecked_mut(key) = self.index as Key;
        }
        self.index = target;
    }
}

impl<T> Drop for Hole<'_, T> {
    #[inline(always)]
    fn drop(&mut self) {
        // SAFETY: `index` is in range, the slot is the hole and so logically
        // uninitialized, and `node` is not used again after this take.
        unsafe {
            let node = std::mem::ManuallyDrop::take(&mut self.node);
            let key = node.key as usize;
            std::ptr::write(self.elements.as_mut_ptr().add(self.index), node);
            *self.pos.get_unchecked_mut(key) = self.index as Key;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    fn min_heap() -> IndexedHeap<i32, fn(&i32, &i32) -> bool> {
        IndexedHeap::new(|a: &i32, b: &i32| a < b)
    }

    #[test]
    fn pops_in_priority_order() {
        let mut heap = min_heap();
        for value in [10, 5, 20, 1, 7] {
            heap.insert(value);
        }
        assert_eq!(heap.len(), 5);
        assert_eq!(heap.top(), Some(&1));

        let mut popped = Vec::new();
        while let Some(value) = heap.pop() {
            popped.push(value);
        }
        assert_eq!(popped, vec![1, 5, 7, 10, 20]);
        assert!(heap.is_empty());
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn update_moves_an_element_both_ways() {
        let mut heap = min_heap();
        let a = heap.insert(10);
        let b = heap.insert(5);
        let c = heap.insert(20);

        // Decrease-key: c becomes the new minimum.
        heap.update(c, 2);
        assert_eq!(heap.top(), Some(&2));
        assert!(heap.is_consistent());

        // Increase-key: the minimum sinks.
        heap.update(c, 50);
        assert_eq!(heap.top(), Some(&5));
        assert!(heap.is_consistent());

        assert_eq!(heap.get(a), Some(&10));
        assert_eq!(heap.get(b), Some(&5));
        assert_eq!(heap.get(c), Some(&50));
    }

    #[test]
    fn update_of_the_root_does_not_underflow() {
        // `parent(0)` is unrepresentable with unsigned indices; upstream leans on
        // C++ integer division making `Parent(0) == 0`.
        let mut heap = min_heap();
        let root = heap.insert(5);
        heap.update(root, 1);
        assert_eq!(heap.top(), Some(&1));
        heap.update(root, 100);
        assert_eq!(heap.top(), Some(&100));
    }

    #[test]
    fn keys_are_recycled_rather_than_growing_without_bound() {
        let mut heap = min_heap();
        for round in 0..8 {
            let keys: Vec<_> = (0..4).map(|i| heap.insert(round * 4 + i)).collect();
            assert!(keys.iter().all(|&key| key < 4));
            for _ in 0..4 {
                heap.pop();
            }
        }
        // Four slots served all thirty-two insertions.
        assert_eq!(heap.pos.len(), 4);
    }

    #[test]
    fn clear_releases_keys_and_keeps_capacity() {
        let mut heap = min_heap();
        let key = heap.insert(10);
        heap.insert(20);
        heap.clear();

        assert!(heap.is_empty());
        assert_eq!(heap.get(key), None);
        assert_eq!(heap.pos.len(), 2, "clear must not allocate new keys");

        heap.insert(30);
        assert_eq!(heap.top(), Some(&30));
        assert_eq!(heap.pos.len(), 2);
    }

    #[test]
    fn get_returns_none_for_a_popped_key() {
        let mut heap = min_heap();
        let key = heap.insert(1);
        assert_eq!(heap.get(key), Some(&1));
        heap.pop();
        assert_eq!(heap.get(key), None);
        assert_eq!(heap.get(K_NO_KEY), None);
        assert_eq!(heap.get(12345), None);
    }

    #[test]
    #[should_panic(expected = "update on a key not in the heap")]
    fn update_of_a_popped_key_panics() {
        let mut heap = min_heap();
        let key = heap.insert(1);
        heap.pop();
        heap.update(key, 2);
    }

    #[test]
    fn works_as_a_max_heap() {
        let mut heap = IndexedHeap::new(|a: &i32, b: &i32| a > b);
        for value in [3, 9, 4, 1] {
            heap.insert(value);
        }
        assert_eq!(heap.pop(), Some(9));
        assert_eq!(heap.pop(), Some(4));
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(1));
    }

    /// Regression test: `pop` used to hand out a bitwise copy of the value while
    /// leaving the original in the backing vector, so any `T` owning memory was
    /// freed twice: once by the caller and once when the slot was overwritten or
    /// the heap dropped.
    #[test]
    fn every_value_is_dropped_exactly_once() {
        struct Counted(#[allow(dead_code)] usize, Rc<Cell<usize>>);
        impl Drop for Counted {
            fn drop(&mut self) {
                self.1.set(self.1.get() + 1);
            }
        }

        let drops = Rc::new(Cell::new(0));
        {
            let mut heap = IndexedHeap::new(|a: &Counted, b: &Counted| a.0 < b.0);
            for i in [4, 1, 3, 2] {
                heap.insert(Counted(i, Rc::clone(&drops)));
            }

            // Popped values are owned by the caller and dropped here.
            drop(heap.pop());
            drop(heap.pop());
            assert_eq!(drops.get(), 2);

            // Reusing a slot must not free the value that already left it.
            heap.insert(Counted(9, Rc::clone(&drops)));
            assert_eq!(drops.get(), 2);

            // Replacing a value drops the old one.
            let key = heap.insert(Counted(8, Rc::clone(&drops)));
            heap.update(key, Counted(7, Rc::clone(&drops)));
            assert_eq!(drops.get(), 3);

            // Four elements are still in the heap at this point.
            heap.clear();
            assert_eq!(drops.get(), 7);

            heap.insert(Counted(0, Rc::clone(&drops)));
        }
        assert_eq!(drops.get(), 8, "the heap must drop what it still holds");
    }

    /// Randomized check against a straightforward reference implementation:
    /// interleaved inserts, updates and pops must keep the heap invariant and
    /// always yield the current minimum.
    #[test]
    fn matches_a_reference_implementation_under_random_operations() {
        // Deterministic xorshift; no rand dependency and reproducible failures.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let mut heap = min_heap();
        // Reference: key -> value for everything currently in the heap.
        let mut reference: Vec<(usize, i32)> = Vec::new();

        for step in 0..4000 {
            let roll = next() % 100;
            if roll < 45 || reference.is_empty() {
                let value = (next() % 1000) as i32;
                let key = heap.insert(value);
                reference.push((key, value));
            } else if roll < 70 {
                let victim = (next() as usize) % reference.len();
                let value = (next() % 1000) as i32;
                let key = reference[victim].0;
                heap.update(key, value);
                reference[victim].1 = value;
            } else {
                let expected = reference.iter().map(|&(_, v)| v).min().unwrap();
                assert_eq!(heap.top(), Some(&expected), "step {step}");
                let popped = heap.pop().unwrap();
                assert_eq!(popped, expected, "step {step}");
                let victim = reference
                    .iter()
                    .position(|&(_, v)| v == popped)
                    .expect("popped value must be in the reference");
                reference.swap_remove(victim);
            }

            assert!(heap.is_consistent(), "step {step}");
            assert_eq!(heap.len(), reference.len(), "step {step}");
        }
    }
}
