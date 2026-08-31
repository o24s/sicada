//! Paired A/B comparison against OpenFst.
//!
//! Criterion measures each benchmark to completion before moving to the next, so
//! any drift in CPU frequency over a run lands entirely on whichever
//! implementation happened to be measured during it. On a shared or thermally
//! throttled machine that drift is larger than the differences being measured:
//! it showed up here as the same unchanged C++ benchmark reporting 14 µs in one
//! run and 31 µs in the next.
//!
//! This harness alternates the two implementations round by round and reports
//! the best round of each, so drift affects both equally. Its ratios are the ones
//! worth quoting; criterion is kept for its statistics on a single implementation
//! over time.
//!
//! Run with `cargo run --release -p sicada-bench --bin ab`.

use std::time::Instant;

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
    let mut acc = 0u64;
    while let Some(value) = heap.pop() {
        acc = acc.wrapping_add(value as u64);
    }
    acc
}

fn sicada_heap_insert(n: u64, seed: u64) -> u64 {
    let mut heap = IndexedHeap::new(|a: &i64, b: &i64| a < b);
    let mut rng = Xorshift::new(seed);
    for _ in 0..n {
        heap.insert((rng.next_u64() % 1_000_000) as i64);
    }
    *heap.top().expect("non-empty") as u64
}

fn sicada_heap_insert_pop(n: u64, seed: u64) -> u64 {
    let mut heap = IndexedHeap::new(|a: &i64, b: &i64| a < b);
    let mut rng = Xorshift::new(seed);
    for _ in 0..n {
        heap.insert((rng.next_u64() % 1_000_000) as i64);
    }
    let mut acc = 0u64;
    while let Some(value) = heap.pop() {
        acc = acc.wrapping_add(value as u64);
    }
    acc
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
    let mut acc = 0u64;
    for i in 0..n as usize {
        acc = acc.wrapping_add(uf.find_set(i).unwrap_or(0) as u64);
    }
    acc
}

#[derive(Clone, Copy)]
struct BenchArc {
    ilabel: i32,
    _olabel: i32,
    _weight: f32,
    _nextstate: i32,
}

fn sicada_arc_arena(states: u64, arcs_per_state: u64) -> u64 {
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
    let mut acc = 0u64;
    for run in &runs {
        for arc in arena.arcs(*run) {
            acc = acc.wrapping_add(arc.ilabel as u64);
        }
    }
    acc
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

/// The fastest round, not the median.
///
/// On a machine with other work on it, every sample is the true cost plus some
/// non-negative interference, so the minimum is the best estimate of the cost
/// itself. Medians drifted by more than the differences being measured here.
fn best(samples: &[u128]) -> u128 {
    *samples.iter().min().expect("at least one round")
}

/// One implementation entered into a comparison.
struct Contender<'a> {
    name: &'static str,
    /// The work being measured, returning something cheap so that it cannot be
    /// optimised away.
    run: Box<dyn FnMut() -> u64 + 'a>,
    /// The same work, returning a checksum of the *result*. Run once, outside
    /// the timing, so that computing the checksum does not inflate the answer.
    verify: Box<dyn FnMut() -> u64 + 'a>,
}

impl<'a> Contender<'a> {
    fn new(
        name: &'static str,
        run: impl FnMut() -> u64 + 'a,
        verify: impl FnMut() -> u64 + 'a,
    ) -> Self {
        Self {
            name,
            run: Box::new(run),
            verify: Box::new(verify),
        }
    }

    /// A contender whose result is its own checksum, which is how the
    /// data-structure comparisons work.
    fn simple(name: &'static str, run: impl FnMut() -> u64 + Clone + 'a) -> Self {
        let verify = run.clone();
        Self::new(name, run, verify)
    }
}

/// Runs every contender alternately and reports the best round of each.
///
/// The first is the baseline; the ratios are relative to it. A contender whose
/// checksum differs from the baseline's is reported and then dropped: a
/// comparison between two functions computing different things is worthless,
/// but the rest of the row is still worth having.
fn race(label: &str, rounds: usize, inner: u64, mut contenders: Vec<Contender>) {
    assert!(!contenders.is_empty(), "{label}: nothing to race");
    let expected = (contenders[0].verify)();
    let mut disagreed = Vec::new();
    contenders.retain_mut(|contender| {
        if (contender.verify)() == expected {
            return true;
        }
        disagreed.push(contender.name);
        false
    });

    let mut times: Vec<Vec<u128>> = vec![Vec::with_capacity(rounds); contenders.len()];
    for _ in 0..rounds {
        for (index, contender) in contenders.iter_mut().enumerate() {
            let start = Instant::now();
            let mut acc = 0u64;
            for _ in 0..inner {
                acc = acc.wrapping_add((contender.run)());
            }
            times[index].push(start.elapsed().as_nanos() / u128::from(inner));
            std::hint::black_box(acc);
        }
    }

    let baseline = best(&times[0]) as f64;
    print!("{label:<34}");
    for (index, contender) in contenders.iter().enumerate() {
        let ns = best(&times[index]);
        print!(" | {:<9} {:>10} ns", contender.name, ns);
        if index > 0 {
            print!(" {:>6.2}x", ns as f64 / baseline);
        } else {
            print!("        ");
        }
    }
    println!();
    for name in disagreed {
        println!("{:<34} | {name}: different answer, left out", "");
    }
}

/// Two contenders, the second measured against the first.
fn compare(
    label: &str,
    rounds: usize,
    inner: u64,
    lhs: impl FnMut() -> u64 + Clone,
    rhs: impl FnMut() -> u64 + Clone,
) {
    race(
        label,
        rounds,
        inner,
        vec![
            Contender::simple("sicada", lhs),
            Contender::simple("openfst", rhs),
        ],
    );
}

/// The four-way algorithm comparisons.
///
/// OpenFst is present only when this crate was built against a real build of
/// it; rustfst and arcweight are ordinary Cargo dependencies and always are.
fn algorithms() {
    use sicada_bench::{arcweight_impl as aw, rustfst_impl as rf, sicada_impl as si};

    macro_rules! four {
        ($label:expr, $rounds:expr, $inner:expr, $bench:ident, $s:expr, $r:expr, $a:expr, $o:expr) => {{
            let mut contenders = vec![
                Contender::new("sicada", || si::$bench::run($s), || si::$bench::verify($s)),
                #[cfg(openfst_algorithms)]
                Contender::new("openfst", || $o(false), || $o(true)),
                Contender::new("rustfst", || rf::$bench::run($r), || rf::$bench::verify($r)),
                Contender::new(
                    "arcweight",
                    || aw::$bench::run($a),
                    || aw::$bench::verify($a),
                ),
            ];
            // `cfg` on an element of a `vec!` is stable, but keep the shape
            // obvious for a reader: nothing else is conditional.
            contenders.shrink_to_fit();
            race($label, $rounds, $inner, contenders);
        }};
    }

    println!();
    for &(label, states, arcs, acyclic) in &[
        ("10000x4", 10_000u64, 4u64, false),
        ("2000x16", 2_000, 16, false),
        ("10000x4-acyclic", 10_000, 4, true),
    ] {
        let s = si::graph(states, arcs, SEED, acyclic);
        let r = rf::graph(states, arcs, SEED, acyclic);
        let a = aw::graph(states, arcs, SEED, acyclic);
        assert_eq!(
            si::checksum(&s),
            rf::checksum(&r),
            "{label}: sicada and rustfst did not build the same FST"
        );
        assert_eq!(
            si::checksum(&s),
            aw::checksum(&a),
            "{label}: sicada and arcweight did not build the same FST"
        );
        #[cfg(openfst_algorithms)]
        let o = {
            let o = sicada_bench::openfst_algorithms::Fst::graph(states, arcs, SEED, acyclic);
            assert_eq!(
                si::checksum(&s),
                o.checksum(),
                "{label}: sicada and openfst did not build the same FST"
            );
            o
        };

        four!(
            &format!("shortest-distance/{label}"),
            9,
            3,
            shortest_distance_bench,
            &s,
            &r,
            &a,
            |v| o.shortest_distance(v)
        );
        four!(
            &format!("shortest-path/{label}"),
            9,
            3,
            shortest_path_bench,
            &s,
            &r,
            &a,
            |v| o.shortest_path(v)
        );
        four!(
            &format!("connect/{label}"),
            9,
            10,
            connect_bench,
            &s,
            &r,
            &a,
            |v| o.connect(v)
        );
        four!(
            &format!("arcsort/{label}"),
            9,
            10,
            arcsort_bench,
            &s,
            &r,
            &a,
            |v| o.arcsort(v)
        );
        // Only where the FST is acyclic: on a cyclic one this measures how
        // fast each library gives up, and they give up at different points --
        // arcweight before it has copied anything, the others after. That is
        // not the same work.
        if acyclic {
            four!(
                &format!("topsort/{label}"),
                9,
                10,
                topsort_bench,
                &s,
                &r,
                &a,
                |v| o.topsort(v)
            );
        }
    }

    println!();
    for &(label, states, arcs) in &[("1000x4", 1_000u64, 4u64), ("3000x4", 3_000, 4)] {
        let s = si::acceptor(states, arcs, SEED);
        let r = rf::acceptor(states, arcs, SEED);
        let a = aw::acceptor(states, arcs, SEED);
        assert_eq!(si::checksum(&s), rf::checksum(&r), "{label}: rustfst");
        assert_eq!(si::checksum(&s), aw::checksum(&a), "{label}: arcweight");
        #[cfg(openfst_algorithms)]
        let o = sicada_bench::openfst_algorithms::Fst::acceptor(states, arcs, SEED);

        four!(
            &format!("rmepsilon/{label}"),
            9,
            3,
            rmepsilon_bench,
            &s,
            &r,
            &a,
            |v| o.rmepsilon(v)
        );
        four!(
            &format!("determinize/{label}"),
            9,
            3,
            determinize_bench,
            &s,
            &r,
            &a,
            |v| o.determinize(v)
        );
        four!(
            &format!("minimize/{label}"),
            9,
            3,
            minimize_bench,
            &s,
            &r,
            &a,
            |v| o.minimize(v)
        );

        let s2 = si::acceptor(states, arcs, SEED ^ 1);
        let r2 = rf::acceptor(states, arcs, SEED ^ 1);
        let a2 = aw::acceptor(states, arcs, SEED ^ 1);
        #[cfg(openfst_algorithms)]
        let o2 = sicada_bench::openfst_algorithms::Fst::acceptor(states, arcs, SEED ^ 1);
        let mut contenders = vec![
            Contender::new(
                "sicada",
                || si::compose_bench::run(&s, &s2),
                || si::compose_bench::verify(&s, &s2),
            ),
            #[cfg(openfst_algorithms)]
            Contender::new("openfst", || o.compose(&o2, false), || o.compose(&o2, true)),
            Contender::new(
                "rustfst",
                || rf::compose_bench::run(&r, &r2),
                || rf::compose_bench::verify(&r, &r2),
            ),
            Contender::new(
                "arcweight",
                || aw::compose_bench::run(&a, &a2),
                || aw::compose_bench::verify(&a, &a2),
            ),
        ];
        contenders.shrink_to_fit();
        race(&format!("compose/{label}"), 9, 3, contenders);

        // Look-ahead composition needs its second argument free of input
        // epsilons, because the index says which label the first FST can read
        // next and an epsilon is not a label, so it gets a row of its own over
        // acceptors built without any.
        let ds = si::dense_acceptor(states, arcs, SEED);
        // Built once, as a saved index would be: the row beside it pays for the
        // index on every composition instead.
        let prepared = si::compose_indexed_bench::prepare(&ds);
        let ds2 = si::dense_acceptor(states, arcs, SEED ^ 1);
        let dr = rf::dense_acceptor(states, arcs, SEED);
        let dr2 = rf::dense_acceptor(states, arcs, SEED ^ 1);
        let da = aw::dense_acceptor(states, arcs, SEED);
        let da2 = aw::dense_acceptor(states, arcs, SEED ^ 1);
        #[cfg(openfst_algorithms)]
        let (dof, dof2) = (
            sicada_bench::openfst_algorithms::Fst::dense_acceptor(states, arcs, SEED),
            sicada_bench::openfst_algorithms::Fst::dense_acceptor(states, arcs, SEED ^ 1),
        );
        let mut contenders = vec![
            Contender::new(
                "sicada",
                || si::compose_bench::run(&ds, &ds2),
                || si::compose_bench::verify(&ds, &ds2),
            ),
            Contender::new(
                "sicada-la",
                || si::compose_lookahead_bench::run(&ds, &ds2),
                || si::compose_lookahead_bench::verify(&ds, &ds2),
            ),
            Contender::new(
                "sicada-idx",
                || si::compose_indexed_bench::run(&prepared, &ds2),
                || si::compose_indexed_bench::verify(&prepared, &ds2),
            ),
            #[cfg(openfst_algorithms)]
            Contender::new(
                "openfst",
                || dof.compose(&dof2, false),
                || dof.compose(&dof2, true),
            ),
            Contender::new(
                "rustfst",
                || rf::compose_bench::run(&dr, &dr2),
                || rf::compose_bench::verify(&dr, &dr2),
            ),
            Contender::new(
                "arcweight",
                || aw::compose_bench::run(&da, &da2),
                || aw::compose_bench::verify(&da, &da2),
            ),
        ];
        contenders.shrink_to_fit();
        race(&format!("compose/dense-{label}"), 9, 3, contenders);
    }
}

fn main() {
    println!("Paired A/B, best of alternating rounds; ratios are against sicada\n");

    compare(
        "heap/1k",
        15,
        200,
        || sicada_heap(1_000, SEED),
        || openfst::heap(1_000, SEED),
    );
    compare(
        "heap/100k",
        15,
        5,
        || sicada_heap(100_000, SEED),
        || openfst::heap(100_000, SEED),
    );
    compare(
        "heap-insert/1k",
        15,
        400,
        || sicada_heap_insert(1_000, SEED),
        || openfst::heap_insert(1_000, SEED),
    );
    compare(
        "heap-insert-pop/1k",
        15,
        200,
        || sicada_heap_insert_pop(1_000, SEED),
        || openfst::heap_insert_pop(1_000, SEED),
    );
    compare(
        "union-find/1k",
        15,
        500,
        || sicada_union_find(1_000, SEED),
        || openfst::union_find(1_000, SEED),
    );
    compare(
        "union-find/100k",
        15,
        20,
        || sicada_union_find(100_000, SEED),
        || openfst::union_find(100_000, SEED),
    );
    compare(
        "arc-arena/10000x4",
        15,
        100,
        || sicada_arc_arena(10_000, 4),
        || openfst::arc_arena(10_000, 4),
    );
    compare(
        "arc-arena/1000x64",
        15,
        100,
        || sicada_arc_arena(1_000, 64),
        || openfst::arc_arena(1_000, 64),
    );
    compare(
        "compact-set/64",
        15,
        100,
        || sicada_compact_set(64, 10_000, SEED),
        || openfst::compact_set(64, 10_000, SEED),
    );
    compare(
        "compact-set/4096",
        15,
        50,
        || sicada_compact_set(4_096, 10_000, SEED),
        || openfst::compact_set(4_096, 10_000, SEED),
    );

    #[cfg(not(openfst_algorithms))]
    println!(
        "\nOpenFst left out of the algorithm comparisons: set OPENFST_BUILD_DIR to a build of \
         vendor/openfst"
    );
    algorithms();
    decoding();
    alignment();
}

/// The exact aligner against the general decoder on the same chain.
///
/// Both compute the best path of the same `N + 1`-state chain against the same
/// scores with no beam, and the harness checks they agree on its cost before
/// timing either. The difference is entirely in how: a banded plane of `f32`
/// with a two-bit traceback, against a hash-map frontier and a link arena that
/// grows with every frame.
///
/// The head-to-head is at a size the decoder can still hold. Above it the limit
/// is the link arena rather than the clock: at ten minutes of audio it wants
/// gigabytes for what the aligner does in 41 MB. The realistic sizes are
/// therefore aligner-only rows, with the forward-backward alongside to show what
/// the second answer costs.
fn alignment() {
    use sicada_bench::align_impl as alignment;

    println!();
    let symbols = 48;
    for &(label, frames, phones) in &[
        ("align/2000x400", 2_000usize, 400usize),
        ("align/6000x1200", 6_000, 1_200),
    ] {
        let (chain, scores) = alignment::utterance(frames, phones, symbols, SEED);
        race(
            label,
            7,
            1,
            vec![
                Contender::simple("exact", {
                    let chain = chain.clone();
                    let scores = scores.clone();
                    move || alignment::exact(&chain, &scores, frames, symbols)
                }),
                Contender::simple("decoder", {
                    let chain = chain.clone();
                    let scores = scores.clone();
                    move || alignment::by_decoder(&chain, &scores, frames, symbols)
                }),
            ],
        );
    }

    // The sizes the reference material actually runs at: a four-minute rap and
    // a ten-minute stretch of speech and song.
    for &(label, frames, phones) in &[
        ("align/12115x2411", 12_115usize, 2_411usize),
        ("align/30241x5385", 30_241, 5_385),
    ] {
        let (chain, scores) = alignment::utterance(frames, phones, symbols, SEED);
        race(
            label,
            5,
            1,
            vec![Contender::simple("exact", {
                let chain = chain.clone();
                let scores = scores.clone();
                move || alignment::exact(&chain, &scores, frames, symbols)
            })],
        );
        race(
            &format!("{label}/soft"),
            3,
            1,
            vec![Contender::simple("fwd-bwd", {
                let chain = chain.clone();
                let scores = scores.clone();
                move || alignment::soft(&chain, &scores, frames, symbols)
            })],
        );
    }
}

/// The decoder against the composition it exists to avoid.
///
/// Both compute the same best path from the same scores through the same CTC
/// topology, and the harness checks that before timing either. There is no
/// third-party decoder here yet: k2's is a GPU one, and comparing against it
/// honestly means reporting a real-time factor at one stream *and* a throughput
/// at the batch size where a GPU pays off, which is a separate piece of work.
///
/// The beams differ by row on purpose. 16 nats is Kaldi's default and prunes
/// nothing on this input, because the runner-up in every frame is only about
/// 2 nats behind, so a lattice at that beam is the whole composition and costs
/// what building it costs. 2 nats is where the beam starts doing something.
fn decoding() {
    use sicada_bench::decode_impl as decode;

    println!();
    for &(label, frames, symbols) in &[
        ("ctc-decode/1000x32", 1_000usize, 32usize),
        ("ctc-decode/250x500", 250, 500),
    ] {
        let graph = decode::graph(symbols);
        let scores = decode::scores(frames, symbols, SEED);
        let (states, arcs) = decode::composition_size(&graph, &scores, frames, symbols);

        race(
            label,
            9,
            1,
            vec![
                Contender::simple("frames", {
                    let graph = graph.clone();
                    let scores = scores.clone();
                    move || decode::viterbi(&graph, &scores, frames, symbols)
                }),
                Contender::simple("compose", {
                    let graph = graph.clone();
                    let scores = scores.clone();
                    move || decode::via_composition(&graph, &scores, frames, symbols)
                }),
            ],
        );
        println!(
            "{:<34} | the composition skipped: {states} states, {arcs} arcs",
            ""
        );

        race(
            &format!("{label}/beam"),
            9,
            1,
            vec![Contender::simple("viterbi", {
                let graph = graph.clone();
                let scores = scores.clone();
                move || decode::viterbi_pruned(&graph, &scores, frames, symbols, 2.0)
            })],
        );
        race(
            &format!("{label}/lattice"),
            9,
            1,
            vec![Contender::simple("beam 2", {
                let graph = graph.clone();
                let scores = scores.clone();
                move || decode::lattice(&graph, &scores, frames, symbols, 2.0)
            })],
        );
        // Two beams, because the contrast is the finding. At 2 nats the
        // lattice is small and the first determinization succeeds. At 16, which
        // is Kaldi's default and prunes nothing on this input, the first attempt
        // runs to the state cap before failing, and the retry at 4 nats is the
        // one that finishes. The row costs what that wasted attempt costs.
        for beam in [2.0f32, 16.0] {
            let (settled, narrowed, states) =
                decode::collapse_narrowing(&graph, &scores, frames, symbols, beam);
            race(
                &format!("{label}/collapse@{beam}"),
                5,
                1,
                vec![Contender::simple("narrowing", {
                    let graph = graph.clone();
                    let scores = scores.clone();
                    move || decode::lattice_collapsed_pruned(&graph, &scores, frames, symbols, beam)
                })],
            );
            println!(
                "{:<34} | settled at beam {settled}, after {narrowed} narrowings, {states} states",
                ""
            );
        }
    }

    // Collapsing the alignments is measured on short utterances, because on a
    // long one it does not finish. It is its own row rather than a contender
    // against the lattice: the two produce different objects, so there is no
    // answer for the harness to check them against.
    for &(label, frames, symbols, beam) in &[
        ("ctc-collapse/50x8", 50usize, 8usize, 4.0f32),
        ("ctc-collapse/100x8", 100, 8, 2.0),
    ] {
        let graph = decode::graph(symbols);
        let scores = decode::scores(frames, symbols, SEED);
        race(
            label,
            9,
            1,
            vec![Contender::simple("lattice", {
                let graph = graph.clone();
                let scores = scores.clone();
                move || decode::lattice(&graph, &scores, frames, symbols, beam)
            })],
        );
        race(
            &format!("{label}/collapsed"),
            9,
            1,
            vec![Contender::simple("+collapse", {
                let graph = graph.clone();
                let scores = scores.clone();
                move || decode::lattice_determinized(&graph, &scores, frames, symbols, beam)
            })],
        );
    }
}
