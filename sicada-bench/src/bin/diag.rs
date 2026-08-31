//! Prints each library's answer component by component, for when the A/B
//! harness reports that two of them disagree.
//!
//! Run with `cargo run --release -p sicada-bench --bin diag`.

use sicada_bench::{SEED, arcweight_impl as aw, rustfst_impl as rf, sicada_impl as si};

/// One library's answer as `states/arcs/total-weight`.
fn shape(parts: Option<(usize, usize, Option<f32>)>) -> String {
    match parts {
        Some((states, arcs, Some(total))) => format!("{states}/{arcs}/{total}"),
        Some((states, arcs, None)) => format!("{states}/{arcs}/-"),
        None => "failed".to_owned(),
    }
}

fn main() {
    if std::env::var("DIAG_EPS").is_ok() {
        rmepsilon_arc_diff();
        return;
    }
    if std::env::var("DIAG_SORT").is_ok() {
        arcsort_breakdown();
        return;
    }
    if std::env::var("DIAG_ARENA").is_ok() {
        arena_breakdown();
        return;
    }
    if std::env::var("DIAG_QUEUE").is_ok() {
        auto_queue_breakdown();
        return;
    }
    if std::env::var("DIAG_REACH").is_ok() {
        label_reachable_breakdown();
        return;
    }
    for n in [8u64, 40, 200, 1000, 3000] {
        println!("== acceptor {n}x4, as states/arcs/total-weight");
        let s = si::acceptor(n, 4, SEED);
        let r = rf::acceptor(n, 4, SEED);
        let a = aw::acceptor(n, 4, SEED);
        let s2 = si::acceptor(n, 4, SEED ^ 1);
        let r2 = rf::acceptor(n, 4, SEED ^ 1);
        let a2 = aw::acceptor(n, 4, SEED ^ 1);
        for (name, sv, rv, av) in [
            (
                "input",
                shape(Some(si::parts(&s))),
                shape(Some(rf::parts(&r))),
                shape(Some(aw::parts(&a))),
            ),
            (
                "rmepsilon",
                shape(Some(si::parts(&si::rmepsilon_bench::result(&s)))),
                shape(Some(rf::parts(&rf::rmepsilon_bench::result(&r)))),
                shape(aw::rmepsilon_bench::result(&a).map(|f| aw::parts(&f))),
            ),
            (
                "determinize",
                shape(Some(si::parts(&si::determinize_bench::result(&s)))),
                shape(rf::determinize_bench::result(&r).map(|f| rf::parts(&f))),
                shape(aw::determinize_bench::result(&a).map(|f| aw::parts(&f))),
            ),
            (
                "minimize",
                shape(Some(si::parts(&si::minimize_bench::result(&s)))),
                shape(rf::minimize_bench::result(&r).map(|f| rf::parts(&f))),
                shape(aw::minimize_bench::result(&a).map(|f| aw::parts(&f))),
            ),
            (
                "compose",
                shape(Some(si::parts(&si::compose_bench::result(&s, &s2)))),
                shape(rf::compose_bench::result(&r, &r2).map(|f| rf::parts(&f))),
                shape(aw::compose_bench::result(&a, &a2).map(|f| aw::parts(&f))),
            ),
        ] {
            println!("  {name:<12} sicada {sv:>16} rustfst {rv:>16} arcweight {av:>16}");
        }
    }

    println!("\n== graph 200x4 cyclic (topsort)");
    let s = si::graph(200, 4, SEED, false);
    let r = rf::graph(200, 4, SEED, false);
    let a = aw::graph(200, 4, SEED, false);
    println!(
        "  topsort   sicada {} rustfst {} arcweight {}",
        si::topsort_bench::verify(&s),
        rf::topsort_bench::verify(&r),
        aw::topsort_bench::verify(&a)
    );
}

/// How much of the sorting benchmark is the copy and how much is the sort, in
/// each of the three Rust libraries.
///
/// Run with `DIAG_SORT=1`.
fn arcsort_breakdown() {
    use arcweight::Fst as _;
    use arcweight::prelude as awp;
    use rustfst::fst_traits::ExpandedFst as _;
    use rustfst::prelude as rfp;
    use sicada::prelude::*;
    use std::time::Instant;

    let rounds = 40;
    let best = |name: &str, f: &mut dyn FnMut() -> u64| {
        let mut least = u128::MAX;
        for _ in 0..rounds {
            let start = Instant::now();
            let value = f();
            least = least.min(start.elapsed().as_nanos());
            std::hint::black_box(value);
        }
        println!("  {name:<34} {least:>10}");
    };

    for (label, states, arcs) in [("10000x4", 10_000u64, 4u64), ("2000x16", 2_000, 16)] {
        println!("\n== arcsort breakdown, {label} (ns, best of {rounds})");
        let s = si::graph(states, arcs, SEED, false);
        let r = rf::graph(states, arcs, SEED, false);
        let a = aw::graph(states, arcs, SEED, false);

        best("sicada    clone only", &mut || {
            s.clone().num_states() as u64
        });
        best("sicada    clone + sort", &mut || {
            let mut out = s.clone();
            arc_sort(&mut out, &ILabelCompare);
            out.num_states() as u64
        });
        best("rustfst   clone only", &mut || {
            r.clone().num_states() as u64
        });
        best("rustfst   clone + sort", &mut || {
            let mut out = r.clone();
            rfp::tr_sort(&mut out, rfp::ILabelCompare {});
            out.num_states() as u64
        });
        best("arcweight clone only", &mut || {
            a.clone().num_states() as u64
        });
        best("arcweight clone + sort", &mut || {
            let mut out = a.clone();
            awp::arc_sort(&mut out, awp::ArcSortType::ByInput).unwrap();
            out.num_states() as u64
        });
    }
}

/// What building the label-reachability index costs, and how big its interval
/// sets are.
///
/// Run with `DIAG_REACH=1`.
fn label_reachable_breakdown() {
    use sicada::algorithms::accumulator::DefaultAccumulator;
    use sicada::algorithms::label_reachable::LabelReachable;
    use sicada::arc::StdArc;
    use std::time::Instant;

    for (label, states, arcs) in [("5000x4", 5_000u64, 4u64), ("2000x16", 2_000, 16)] {
        let fst = si::acceptor(states, arcs, SEED);
        let rounds = 20;
        let mut least = u128::MAX;
        for _ in 0..rounds {
            let start = Instant::now();
            let reachable =
                LabelReachable::<StdArc, _>::with_accumulator(&fst, true, DefaultAccumulator)
                    .expect("index");
            least = least.min(start.elapsed().as_nanos());
            std::hint::black_box(reachable.data().len());
        }
        let reachable =
            LabelReachable::<StdArc, _>::with_accumulator(&fst, true, DefaultAccumulator)
                .expect("index");
        let sets = reachable.data().interval_sets();
        let total: usize = sets.iter().map(|s| s.intervals().len()).sum();
        let most = sets.iter().map(|s| s.intervals().len()).max().unwrap_or(0);
        println!(
            "  LabelReachable::new {label:<10} {least:>10} ns   {} sets, {total} intervals, \
             {:.2} each, most {most}",
            sets.len(),
            total as f64 / sets.len().max(1) as f64
        );
    }
}

/// Where `AutoQueue::new` spends its time on a cyclic graph.
///
/// Run with `DIAG_QUEUE=1`.
fn auto_queue_breakdown() {
    use sicada::arc::{Arc as _, StdArc};
    use sicada::fst::{ExpandedFst as _, Fst as _};
    use sicada::queue::{AutoQueue, components, scc_queue_types};
    use sicada::weights::float_weight::TropicalWeight;
    use std::time::Instant;

    let rounds = 20;
    let best = |name: &str, f: &mut dyn FnMut() -> usize| {
        let mut least = u128::MAX;
        let mut value = 0;
        for _ in 0..rounds {
            let start = Instant::now();
            value = f();
            least = least.min(start.elapsed().as_nanos());
        }
        println!("  {name:<34} {least:>10} ns   ({value})");
        least
    };

    for (label, states, arcs) in [("10000x4", 10_000u64, 4u64), ("2000x16", 2_000, 16)] {
        println!("\n== AutoQueue::new breakdown, {label} cyclic (ns, best of {rounds})");
        let fst = si::graph(states, arcs, SEED, false);

        best("a bare walk over every arc", &mut || {
            let mut acc = 0usize;
            for state in 0..fst.num_states() as i32 {
                for arc in fst.arcs(state) {
                    acc += arc.nextstate() as usize;
                }
            }
            acc
        });
        best("a bare DFS, visitor doing nothing", &mut || {
            struct Nothing;
            impl sicada::algorithms::dfs_visit::DfsVisitor<StdArc> for Nothing {
                fn init_visit<F2: sicada::prelude::Fst<StdArc>>(&mut self, _fst: &F2) {}
                fn init_state(&mut self, _s: i32, _root: i32) -> bool {
                    true
                }
                fn tree_arc(&mut self, _s: i32, _arc: &StdArc) -> bool {
                    true
                }
                fn back_arc(&mut self, _s: i32, _arc: &StdArc) -> bool {
                    true
                }
                fn forward_or_cross_arc(&mut self, _s: i32, _arc: &StdArc) -> bool {
                    true
                }
                fn finish_state(&mut self, _s: i32, _p: Option<i32>, _arc: Option<&StdArc>) {}
                fn finish_visit(&mut self) {}
            }
            let mut visitor = Nothing;
            sicada::algorithms::dfs_visit::dfs_visit_any(&fst, &mut visitor);
            0
        });
        best("components (the SCC DFS)", &mut || components(&fst).len());

        let scc = components(&fst);
        let nscc = scc.iter().map(|s| *s as usize + 1).max().unwrap_or(0);
        println!("  {:<34} {:>10}      ({nscc} components)", "", "");
        let less = |a: &TropicalWeight, b: &TropicalWeight| a.0 < b.0;
        best("scc_queue_types (the arc scan)", &mut || {
            scc_queue_types(&fst, &scc, nscc, Some(&less))
                .queue_types
                .len()
        });

        best("the whole AutoQueue::new", &mut || {
            let q = AutoQueue::<i32, fn(&i32, &i32) -> bool>::new(&fst, None);
            std::hint::black_box(&q);
            0
        });
    }
}

/// How much of the arena benchmark is building the runs and how much is walking
/// them back, on each side.
///
/// Run with `DIAG_ARENA=1`.
fn arena_breakdown() {
    use sicada::memory::ArcArena;
    use sicada_bench::openfst;
    use std::time::Instant;

    /// The same four-field arc both sides push.
    #[derive(Clone, Copy)]
    struct BenchArc {
        ilabel: i32,
        _olabel: i32,
        _weight: f32,
        _nextstate: i32,
    }

    fn build(
        states: u64,
        arcs_per_state: u64,
    ) -> (ArcArena<BenchArc>, Vec<sicada::memory::ArcRun>) {
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
        (arena, runs)
    }

    /// One thing to time, and what to call it in the table.
    type Measured<'a> = (&'a str, Box<dyn FnMut() -> u64>);

    let rounds = 60;
    for (label, states, arcs) in [("10000x4", 10_000u64, 4u64), ("1000x64", 1_000, 64)] {
        // Four measurements, alternated round by round and warmed up first, so
        // that neither allocator state nor frequency drift lands on one of them.
        let mut work: Vec<Measured<'_>> = vec![
            (
                "sicada  build only",
                Box::new(move || {
                    let (arena, runs) = build(states, arcs);
                    std::hint::black_box(&arena);
                    runs.len() as u64
                }),
            ),
            (
                "sicada  build + walk",
                Box::new(move || {
                    let (arena, runs) = build(states, arcs);
                    let mut acc = 0u64;
                    for run in &runs {
                        for arc in arena.arcs(*run) {
                            acc = acc.wrapping_add(arc.ilabel as u64);
                        }
                    }
                    acc
                }),
            ),
            (
                "openfst build only",
                Box::new(move || openfst::arc_arena_build(states, arcs)),
            ),
            (
                "openfst build + walk",
                Box::new(move || openfst::arc_arena(states, arcs)),
            ),
        ];
        for (_, f) in work.iter_mut() {
            for _ in 0..5 {
                std::hint::black_box(f());
            }
        }
        let mut least = vec![u128::MAX; work.len()];
        for _ in 0..rounds {
            for (index, (_, f)) in work.iter_mut().enumerate() {
                let start = Instant::now();
                let value = f();
                least[index] = least[index].min(start.elapsed().as_nanos());
                std::hint::black_box(value);
            }
        }
        println!("\n== arc-arena breakdown, {label} (ns, best of {rounds}, alternated)");
        for (index, (name, _)) in work.iter().enumerate() {
            println!("  {name:<34} {:>10} ns", least[index]);
        }
        println!(
            "  {:<34} {:>10} ns   (sicada walk)",
            "",
            least[1].saturating_sub(least[0])
        );
        println!(
            "  {:<34} {:>10} ns   (openfst walk)",
            "",
            least[3].saturating_sub(least[2])
        );
    }
}

/// The arcs sicada and arcweight disagree about after epsilon removal, lined up
/// by state, with the numbering left alone on both sides.
///
/// Run with `DIAG_EPS=1`.
fn rmepsilon_arc_diff() {
    for n in [200u64, 400, 600, 800, 1000] {
        let s = si::rmepsilon_bench::result_unconnected(&si::acceptor(n, 4, SEED));
        let a = aw::rmepsilon_bench::result_unconnected(&aw::acceptor(n, 4, SEED))
            .expect("arcweight rmepsilon");
        let mut states = 0usize;
        let mut parallel_worse = 0usize;
        let mut other = 0usize;
        for state in 0..n as usize {
            let mut mine = si_arcs(&s, state);
            let mut theirs = aw_arcs(&a, state);
            mine.sort_unstable();
            theirs.sort_unstable();
            if mine == theirs {
                continue;
            }
            states += 1;
            let mut left = mine.clone();
            for arc in theirs {
                if let Some(at) = left.iter().position(|m| *m == arc) {
                    left.remove(at);
                    continue;
                }
                // An arc the other side does not have. Is it a second copy of
                // one it does have, along the same label to the same state,
                // carrying a weight that ⊕ would have discarded?
                let (ilabel, olabel, next, weight) = arc;
                let twin = mine.iter().any(|(i, o, t, w)| {
                    (*i, *o, *t) == (ilabel, olabel, next)
                        && f32::from_bits(*w) < f32::from_bits(weight)
                });
                if twin {
                    parallel_worse += 1;
                } else {
                    other += 1;
                }
            }
        }
        println!(
            "{n}x4: {states} states differ; arcweight has {parallel_worse} extra arcs that \
             repeat one of sicada's along the same label to the same state with a heavier \
             weight, and {other} that do not"
        );
    }
}

/// `(ilabel, olabel, nextstate, weight bits)` for one sicada state.
fn si_arcs(fst: &si::Sfst, state: usize) -> Vec<(u32, u32, u32, u32)> {
    use sicada::prelude::Fst;
    fst.arcs(state as i32)
        .map(|arc| {
            (
                arc.ilabel as u32,
                arc.olabel as u32,
                arc.nextstate as u32,
                arc.weight.0.to_bits(),
            )
        })
        .collect()
}

/// `(ilabel, olabel, nextstate, weight bits)` for one arcweight state.
fn aw_arcs(fst: &aw::Afst, state: usize) -> Vec<(u32, u32, u32, u32)> {
    use arcweight::prelude::Fst;
    fst.arcs(state as u32)
        .map(|arc| {
            (
                arc.ilabel,
                arc.olabel,
                arc.nextstate,
                (*arcweight::prelude::Semiring::value(&arc.weight)).to_bits(),
            )
        })
        .collect()
}
