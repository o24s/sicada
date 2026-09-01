//! Container types the FST algorithms are built on.

pub mod bi_table;
pub mod bit_set;
pub mod compact_set;
pub(crate) mod fast_cell;
pub mod indexed_heap;
pub mod interval_set;
pub mod partition;
pub mod state_table;
pub mod union_find;

pub use indexed_heap::{IndexedHeap, K_NO_KEY};
pub use partition::{Partition, PartitionClassIter};

pub use compact_set::{CompactSet, K_NO_KEY as COMPACT_SET_NO_KEY};

pub use bi_table::{
    BiTableId, CompactHashBiTable, ErasableBiTable, HashBiTable, VectorBiTable, VectorHashBiTable,
};

// TODO(porting-iteration): these re-exports have no consumer yet; the allow goes
// away as the iteration reaches the modules and their users.
#[allow(unused_imports)]
pub(crate) use bit_set::*;
#[allow(unused_imports)]
pub(crate) use fast_cell::*;
#[allow(unused_imports)]
pub use state_table::*;
pub use union_find::UnionFind;
