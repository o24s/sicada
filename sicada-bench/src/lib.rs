//! Benchmark harness for `sicada`.
//!
//! This crate exists to keep `criterion` and `rustfst` out of `sicada`'s own
//! dependency graph: they are needed solely to measure against, not to build or
//! test the library. The benchmarks themselves live in `benches/`.
//!
//! It also carries the OpenFst side of the comparison. `cpp/openfst_shim.cc`
//! extracts the upstream data structures verbatim and exposes one entry point
//! per benchmark; [`openfst`] declares them. Measuring against upstream's real
//! code rather than a paraphrase is the whole point, so the shim must stay a
//! copy of the headers, not a reimplementation.

/// Entry points into the OpenFst reference implementations.
///
/// Each runs a fixed operation sequence and returns a checksum, so the Rust side
/// can assert both implementations computed the same thing before timing them.
pub mod openfst {
    unsafe extern "C" {
        fn openfst_bench_heap(n: u64, seed: u64) -> u64;
        fn openfst_bench_heap_insert(n: u64, seed: u64) -> u64;
        fn openfst_bench_heap_insert_pop(n: u64, seed: u64) -> u64;
        fn openfst_bench_union_find(n: u64, seed: u64) -> u64;
        fn openfst_bench_arc_arena(states: u64, arcs_per_state: u64) -> u64;
        fn openfst_bench_arc_arena_build(states: u64, arcs_per_state: u64) -> u64;
        fn openfst_bench_compact_set(n: u64, probes: u64, seed: u64) -> u64;
    }

    /// `n` inserts, a decrease-key on every fourth element, then `n` pops.
    pub fn heap(n: u64, seed: u64) -> u64 {
        // SAFETY: the shim takes only integers, allocates and frees everything
        // it uses within the call, and returns a plain u64.
        unsafe { openfst_bench_heap(n, seed) }
    }

    /// `n` inserts and nothing else.
    pub fn heap_insert(n: u64, seed: u64) -> u64 {
        // SAFETY: as above.
        unsafe { openfst_bench_heap_insert(n, seed) }
    }

    /// `n` inserts followed by `n` pops, with no updates in between.
    pub fn heap_insert_pop(n: u64, seed: u64) -> u64 {
        // SAFETY: as above.
        unsafe { openfst_bench_heap_insert_pop(n, seed) }
    }

    /// `n` singletons, `n - 1` random unions, then a lookup per element.
    pub fn union_find(n: u64, seed: u64) -> u64 {
        // SAFETY: as above.
        unsafe { openfst_bench_union_find(n, seed) }
    }

    /// `states` runs of `arcs_per_state` arcs, then a walk over every run.
    pub fn arc_arena(states: u64, arcs_per_state: u64) -> u64 {
        // SAFETY: as above.
        unsafe { openfst_bench_arc_arena(states, arcs_per_state) }
    }

    /// The same, without the walk: the building half alone.
    pub fn arc_arena_build(states: u64, arcs_per_state: u64) -> u64 {
        // SAFETY: as above.
        unsafe { openfst_bench_arc_arena_build(states, arcs_per_state) }
    }

    /// Builds a set of `n` keys in a narrow interval and probes it `probes` times.
    pub fn compact_set(n: u64, probes: u64, seed: u64) -> u64 {
        // SAFETY: as above.
        unsafe { openfst_bench_compact_set(n, probes, seed) }
    }
}

/// The pseudo-random generator both sides of every benchmark use, so the two
/// implementations see an identical input sequence.
pub struct Xorshift(pub u64);

impl Xorshift {
    /// Creates a generator from a seed.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Returns the next value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// The algorithm comparisons, which link against a real build of OpenFst.
///
/// Present only when `OPENFST_BUILD_DIR` pointed at one when this crate was
/// built. The data-structure comparisons need no such build, since
/// `cpp/openfst_shim.cc` carries the structures themselves.
#[cfg(openfst_algorithms)]
pub mod openfst_algorithms {
    use std::ffi::c_void;

    unsafe extern "C" {
        fn openfst_bench_fst_new(
            states: u64,
            arcs_per_state: u64,
            seed: u64,
            acyclic: i32,
        ) -> *mut c_void;
        fn openfst_bench_acceptor_new(states: u64, arcs_per_state: u64, seed: u64) -> *mut c_void;
        fn openfst_bench_dense_acceptor_new(
            states: u64,
            arcs_per_state: u64,
            seed: u64,
        ) -> *mut c_void;
        fn openfst_bench_fst_delete(fst: *mut c_void);
        fn openfst_bench_fst_checksum(fst: *const c_void) -> u64;
        fn openfst_bench_shortest_distance(fst: *const c_void, verify: i32) -> u64;
        fn openfst_bench_shortest_path(fst: *const c_void, verify: i32) -> u64;
        fn openfst_bench_connect(fst: *const c_void, verify: i32) -> u64;
        fn openfst_bench_arcsort(fst: *const c_void, verify: i32) -> u64;
        fn openfst_bench_topsort(fst: *const c_void, verify: i32) -> u64;
        fn openfst_bench_rmepsilon(fst: *const c_void, verify: i32) -> u64;
        fn openfst_bench_determinize(fst: *const c_void, verify: i32) -> u64;
        fn openfst_bench_minimize(fst: *const c_void, verify: i32) -> u64;
        fn openfst_bench_compose(lhs: *const c_void, rhs: *const c_void, verify: i32) -> u64;
    }

    /// An OpenFst `StdVectorFst`, built once and measured against many times.
    pub struct Fst(*mut c_void);

    impl Fst {
        /// Builds the same random graph the other three sides build.
        pub fn graph(states: u64, arcs_per_state: u64, seed: u64, acyclic: bool) -> Self {
            // SAFETY: the shim allocates a `StdVectorFst` and hands back the
            // pointer; `Drop` gives it back to the same allocator.
            Self(unsafe { openfst_bench_fst_new(states, arcs_per_state, seed, i32::from(acyclic)) })
        }

        /// Builds the same acyclic acceptor with epsilons.
        pub fn acceptor(states: u64, arcs_per_state: u64, seed: u64) -> Self {
            // SAFETY: as `graph`.
            Self(unsafe { openfst_bench_acceptor_new(states, arcs_per_state, seed) })
        }

        /// The same without epsilons.
        pub fn dense_acceptor(states: u64, arcs_per_state: u64, seed: u64) -> Self {
            // SAFETY: as `graph`.
            Self(unsafe { openfst_bench_dense_acceptor_new(states, arcs_per_state, seed) })
        }

        /// The input's checksum, to confirm all four sides built one FST.
        pub fn checksum(&self) -> u64 {
            // SAFETY: `self.0` came from the shim and is still alive.
            unsafe { openfst_bench_fst_checksum(self.0) }
        }

        /// `ShortestDistance` from the start state.
        pub fn shortest_distance(&self, verify: bool) -> u64 {
            // SAFETY: as above.
            unsafe { openfst_bench_shortest_distance(self.0, i32::from(verify)) }
        }

        /// `ShortestPath`, one best path.
        pub fn shortest_path(&self, verify: bool) -> u64 {
            // SAFETY: as above.
            unsafe { openfst_bench_shortest_path(self.0, i32::from(verify)) }
        }

        /// A copy, then `Connect`.
        pub fn connect(&self, verify: bool) -> u64 {
            // SAFETY: as above.
            unsafe { openfst_bench_connect(self.0, i32::from(verify)) }
        }

        /// A copy, then `ArcSort` on input labels.
        pub fn arcsort(&self, verify: bool) -> u64 {
            // SAFETY: as above.
            unsafe { openfst_bench_arcsort(self.0, i32::from(verify)) }
        }

        /// A copy, then `TopSort`.
        pub fn topsort(&self, verify: bool) -> u64 {
            // SAFETY: as above.
            unsafe { openfst_bench_topsort(self.0, i32::from(verify)) }
        }

        /// A copy, then `RmEpsilon`.
        pub fn rmepsilon(&self, verify: bool) -> u64 {
            // SAFETY: as above.
            unsafe { openfst_bench_rmepsilon(self.0, i32::from(verify)) }
        }

        /// `Determinize`.
        pub fn determinize(&self, verify: bool) -> u64 {
            // SAFETY: as above.
            unsafe { openfst_bench_determinize(self.0, i32::from(verify)) }
        }

        /// `Determinize`, then `Minimize`.
        pub fn minimize(&self, verify: bool) -> u64 {
            // SAFETY: as above.
            unsafe { openfst_bench_minimize(self.0, i32::from(verify)) }
        }

        /// Sorts copies of both sides, then `Compose`.
        pub fn compose(&self, rhs: &Self, verify: bool) -> u64 {
            // SAFETY: both pointers came from the shim and are still alive.
            unsafe { openfst_bench_compose(self.0, rhs.0, i32::from(verify)) }
        }
    }

    impl Drop for Fst {
        fn drop(&mut self) {
            // SAFETY: `self.0` came from `openfst_bench_fst_new` and is given
            // back exactly once.
            unsafe { openfst_bench_fst_delete(self.0) }
        }
    }
}

pub mod align_impl;
pub mod arcweight_impl;
pub mod decode_impl;
pub mod rustfst_impl;
pub mod sicada_impl;

/// The checksum every algorithm result is compared through, across all four
/// implementations.
///
/// Determinization, minimization and composition each number their output
/// states as they please, and no two libraries agree on that, nor do they need
/// to. What they must agree on is how big the result is and what it accepts.
/// `total` is the ⊕-sum over every path, or `None` where there is no path at
/// all.
///
/// The order of the arcs leaving a state is deliberately *not* in here: no
/// algorithm but a sort promises anything about it, and arcweight's
/// determinization leaves them in a different order from everyone else's while
/// producing the same automaton. The sorting benchmark uses
/// [`sorted_shape`] instead.
pub fn shape(states: usize, arcs: usize, total: Option<f32>) -> u64 {
    (states as u64)
        .wrapping_mul(1_000_003)
        .wrapping_add(arcs as u64)
        .wrapping_mul(31)
        .wrapping_add(tick(total))
}

/// As [`shape`], and whether the arcs came out sorted on input labels.
///
/// Only the sorting benchmark uses this; for everything else the arc order is
/// nobody's promise.
pub fn sorted_shape(states: usize, arcs: usize, total: Option<f32>, sorted: bool) -> u64 {
    shape(states, arcs, total)
        .wrapping_mul(2)
        .wrapping_add(u64::from(sorted))
}

/// A weight as a whole number of quarters, so that a checksum says "the same
/// answer" rather than "the same float".
///
/// Every weight in the generated FSTs is a multiple of 1/4, so every path sum
/// is exact in binary and no two implementations can be a rounding step apart.
/// `None` is Zero: no path at all.
pub fn tick(weight: Option<f32>) -> u64 {
    match weight {
        Some(w) => (w * 4.0 + 0.5) as i64 as u64,
        None => 0,
    }
}

/// A checksum of a whole distance vector, `None` standing for Zero.
pub fn distance_checksum(distance: impl Iterator<Item = Option<f32>>) -> u64 {
    let mut acc = 0u64;
    let mut count = 0u64;
    for weight in distance {
        acc = acc.wrapping_mul(31).wrapping_add(tick(weight));
        count += 1;
    }
    acc.wrapping_mul(31).wrapping_add(count)
}

/// Seed shared by every benchmark.
pub const SEED: u64 = 0x2545_F491_4F6C_DD1D;
