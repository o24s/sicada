//! sicada's data structures against the OpenFst originals.
//!
//! The C++ side is the upstream code itself, extracted verbatim into
//! `cpp/openfst_shim.cc` and compiled at `-O3`; the Rust side runs the same
//! operation sequence over the same pseudo-random stream, so both do equal work.
//! Nothing here is tuned per implementation: if sicada loses, the number stands
//! and gets written up rather than the benchmark adjusted.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use sicada::data_structures::compact_set::CompactSet;
use sicada::data_structures::indexed_heap::IndexedHeap;
use sicada::data_structures::union_find::UnionFind;
use sicada::memory::ArcArena;

use sicada_bench::{SEED, Xorshift, openfst};

fn sicada_heap(n: u64, seed: u64) -> u64 {
    let mut heap = IndexedHeap::new(|a: &i64, b: &i64| a < b);
    let mut rng = Xorshift::new(seed);
    let mut keys = Vec::with_capacity(n as usize);
    for _ in 0..n {
        keys.push(heap.insert((rng.next_u64() % 1_000_000) as i64));
    }
    for i in (0..n).step_by(4) {
        heap.update(keys[i as usize], (rng.next_u64() % 1000) as i64);
    }
    let mut checksum = 0u64;
    while let Some(value) = heap.pop() {
        checksum = checksum.wrapping_add(value as u64);
    }
    checksum
}

fn sicada_union_find(n: u64, seed: u64) -> u64 {
    let mut uf = UnionFind::new(n as usize);
    uf.make_all_set(n as usize);
    let mut rng = Xorshift::new(seed);
    for _ in 0..n.saturating_sub(1) {
        let a = (rng.next_u64() % n) as usize;
        let b = (rng.next_u64() % n) as usize;
        uf.union(a, b);
    }
    let mut checksum = 0u64;
    for i in 0..n as usize {
        checksum = checksum.wrapping_add(uf.find_set(i).unwrap_or(0) as u64);
    }
    checksum
}

fn sicada_arc_arena(states: u64, arcs_per_state: u64) -> u64 {
    #[derive(Clone, Copy)]
    struct BenchArc {
        ilabel: i32,
        _olabel: i32,
        _weight: f32,
        _nextstate: i32,
    }

    let mut arena = ArcArena::with_block_size(256);
    let mut runs = Vec::with_capacity(states as usize);
    for state in 0..states {
        for arc in 0..arcs_per_state {
            arena.push_arc(BenchArc {
                ilabel: arc as i32,
                _olabel: arc as i32,
                _weight: 1.0,
                _nextstate: state as i32,
            });
        }
        runs.push(arena.commit_arcs());
    }
    let mut checksum = 0u64;
    for run in &runs {
        for arc in arena.arcs(*run) {
            checksum = checksum.wrapping_add(arc.ilabel as u64);
        }
    }
    checksum
}

fn sicada_compact_set(n: u64, probes: u64, seed: u64) -> u64 {
    let mut set = CompactSet::new();
    let mut rng = Xorshift::new(seed);
    for _ in 0..n {
        set.insert((rng.next_u64() % (n * 2)) as usize);
    }
    let mut hits = 0u64;
    for _ in 0..probes {
        if set.is_member((rng.next_u64() % (n * 4)) as usize) {
            hits += 1;
        }
    }
    hits
}

fn bench_heap(c: &mut Criterion) {
    let mut group = c.benchmark_group("heap/insert-update-pop");
    for &n in &[1_000u64, 100_000] {
        group.throughput(Throughput::Elements(n));
        // Both implementations must agree on the answer, or the comparison is
        // meaningless.
        assert_eq!(sicada_heap(n, SEED), openfst::heap(n, SEED));

        group.bench_with_input(BenchmarkId::new("sicada", n), &n, |b, &n| {
            b.iter(|| black_box(sicada_heap(n, SEED)))
        });
        group.bench_with_input(BenchmarkId::new("openfst", n), &n, |b, &n| {
            b.iter(|| black_box(openfst::heap(n, SEED)))
        });
    }
    group.finish();
}

fn bench_union_find(c: &mut Criterion) {
    let mut group = c.benchmark_group("union-find/union-then-find");
    for &n in &[1_000u64, 100_000] {
        group.throughput(Throughput::Elements(n));
        assert_eq!(sicada_union_find(n, SEED), openfst::union_find(n, SEED));

        group.bench_with_input(BenchmarkId::new("sicada", n), &n, |b, &n| {
            b.iter(|| black_box(sicada_union_find(n, SEED)))
        });
        group.bench_with_input(BenchmarkId::new("openfst", n), &n, |b, &n| {
            b.iter(|| black_box(openfst::union_find(n, SEED)))
        });
    }
    group.finish();
}

fn bench_arc_arena(c: &mut Criterion) {
    let mut group = c.benchmark_group("arc-arena/build-runs");
    for &(states, arcs) in &[(10_000u64, 4u64), (1_000, 64)] {
        group.throughput(Throughput::Elements(states * arcs));
        assert_eq!(
            sicada_arc_arena(states, arcs),
            openfst::arc_arena(states, arcs)
        );

        let label = format!("{states}x{arcs}");
        group.bench_with_input(
            BenchmarkId::new("sicada", &label),
            &(states, arcs),
            |b, &(s, a)| b.iter(|| black_box(sicada_arc_arena(s, a))),
        );
        group.bench_with_input(
            BenchmarkId::new("openfst", &label),
            &(states, arcs),
            |b, &(s, a)| b.iter(|| black_box(openfst::arc_arena(s, a))),
        );
    }
    group.finish();
}

fn bench_compact_set(c: &mut Criterion) {
    let mut group = c.benchmark_group("compact-set/build-then-probe");
    for &n in &[64u64, 4_096] {
        let probes = 10_000;
        group.throughput(Throughput::Elements(probes));
        assert_eq!(
            sicada_compact_set(n, probes, SEED),
            openfst::compact_set(n, probes, SEED)
        );

        group.bench_with_input(BenchmarkId::new("sicada", n), &n, |b, &n| {
            b.iter(|| black_box(sicada_compact_set(n, probes, SEED)))
        });
        group.bench_with_input(BenchmarkId::new("openfst", n), &n, |b, &n| {
            b.iter(|| black_box(openfst::compact_set(n, probes, SEED)))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_heap,
    bench_union_find,
    bench_arc_arena,
    bench_compact_set
);
criterion_main!(benches);
