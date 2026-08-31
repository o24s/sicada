//! Allocation helpers.
//!
//! Port of OpenFst's `memory.h` and `arc-arena.h`.
//!
//! # What `memory.h` is for, and what carries over
//!
//! Upstream's `memory.h` exists to keep node-based STL containers from calling
//! the global allocator once per element. It provides a bump arena
//! (`MemoryArena`), a free-list pool (`MemoryPool`), collections of both keyed
//! by `sizeof(T)`, and two STL allocators built on them (`BlockAllocator`,
//! `PoolAllocator`).
//!
//! | Upstream | sicada |
//! | --- | --- |
//! | `MemoryPool<T>` | [`MemoryPool<T>`], a [`slab::Slab`] |
//! | `MemoryArena<T>` | superseded; see below |
//! | `BlockAllocator<T>` | not applicable |
//! | `PoolAllocator<T>` | not applicable |
//! | `MemoryArenaCollection`, `MemoryPoolCollection` | not applicable |
//!
//! `MemoryPool` is a real dependency where objects are freed out of order:
//! `cache.rs` evicts arbitrary states, so its slab's freed slots are genuinely
//! reused. A slab gives the same O(1) allocate/free with slot reuse, keyed by
//! an index rather than a pointer, without any unsafe code.
//!
//! It is *not* the answer where upstream reaches for it out of habit. Upstream
//! pools `DfsState` in `dfs-visit.h` and `ArcIterator` in `visit.h`, but both
//! are pushed and popped in strict last-in-first-out order: a pool standing in
//! for a stack, because C++ has no way to hold a non-movable iterator in a
//! `std::vector`. sicada's `dfs_visit.rs` uses a `Vec` directly, which is the
//! same allocation behaviour with none of the indirection.
//!
//! SICADA-OPT: the two STL allocators have no sicada counterpart because the
//! containers that needed them do not exist here. `PoolAllocator` is used
//! upstream by `bi-table.h`'s `std::unordered_set`, by `cache.h`'s
//! `std::list<StateId>`, and by its state hash map, all node-per-element
//! structures whose allocation traffic the pool is there to absorb. sicada uses
//! `hashbrown`/`rustc-hash` open-addressed tables and `Vec`, which allocate in
//! blocks rather than per element, so the pool has nothing to amortize. Adding
//! an allocator layer would be pure overhead, and custom allocators are not
//! available on stable Rust anyway (`allocator_api` is unstable).
//!
//! `MemoryArena` and `BlockAllocator` have no users at all in `openfst/lib`:
//! `BlockAllocator` is the arena's only consumer, and nothing constructs one.
//! The one arena sicada actually needs is the contiguous arc allocator, which is
//! [`ArcArena`], ported from `arc-arena.h`.

pub mod arc_arena;

pub use arc_arena::{ArcArena, ArcRun};

use slab::Slab;

/// Pool of same-typed objects with O(1) allocation and free-slot reuse.
///
/// Corresponds to OpenFst's `MemoryPool<T>`. Objects are addressed by an index
/// rather than a pointer: `insert` returns a key, `remove` returns the value and
/// releases the slot for the next `insert`.
pub type MemoryPool<T> = Slab<T>;

#[cfg(test)]
mod tests {
    use super::*;

    /// The property that makes this a pool rather than a growing vector: a freed
    /// slot is handed straight back out, so a churn of short-lived objects does
    /// not keep growing the backing storage.
    #[test]
    fn freed_slots_are_reused() {
        let mut pool: MemoryPool<String> = MemoryPool::new();
        let first = pool.insert("a".to_string());
        let second = pool.insert("b".to_string());

        assert_eq!(pool.remove(first), "a");
        let third = pool.insert("c".to_string());
        assert_eq!(third, first, "the freed slot should be reused");
        assert_eq!(pool.len(), 2);
        assert_eq!(pool[second], "b");
        assert_eq!(pool[third], "c");
    }

    #[test]
    fn capacity_survives_a_full_churn() {
        let mut pool: MemoryPool<u32> = MemoryPool::with_capacity(64);
        let capacity = pool.capacity();
        for round in 0..100u32 {
            let keys: Vec<_> = (0..64).map(|i| pool.insert(round * 64 + i)).collect();
            for key in keys {
                pool.remove(key);
            }
        }
        assert!(pool.is_empty());
        assert_eq!(pool.capacity(), capacity, "the pool should not have grown");
    }
}
