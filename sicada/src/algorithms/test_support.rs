//! Shared machinery for the algorithm tests.
//!
//! An algorithm over FSTs is usually easiest to check against its own
//! definition, meaning what the FST accepts and with what weight, rather than
//! against the shape of what comes out. Enumerating accepting paths is how that
//! is done here, so it lives in one place.

use crate::arc::{Arc, StdArc};
use crate::fst::{Fst, MutableFst};
use crate::fsts::vector_fst::StdVectorFst;
use crate::weight::Weight;
use crate::weights::float_weight::TropicalWeight;

/// A reproducible generator. A failing case has to be the same one next run.
pub struct Rng(u64);

impl Rng {
    /// Seeds the generator.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// A number below `bound`.
    pub fn below(&mut self, bound: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as usize) % bound.max(1)
    }
}

/// One accepting path: its input labels, its output labels, and the product of
/// every weight along it including the final one.
pub type Path = (Vec<i32>, Vec<i32>, TropicalWeight);

/// Every accepting path of `fst` reachable in at most `max_len` arcs.
pub fn paths<A, F>(fst: &F, max_len: usize) -> Vec<Path>
where
    A: Arc<Label = i32, StateId = i32, Weight = TropicalWeight>,
    F: Fst<A>,
{
    let mut out = Vec::new();
    if let Some(start) = fst.start() {
        walk(
            fst,
            start,
            &mut Vec::new(),
            &mut Vec::new(),
            TropicalWeight::one(),
            max_len,
            &mut out,
        );
    }
    out
}

fn walk<A, F>(
    fst: &F,
    state: A::StateId,
    ilabels: &mut Vec<i32>,
    olabels: &mut Vec<i32>,
    weight: TropicalWeight,
    left: usize,
    out: &mut Vec<Path>,
) where
    A: Arc<Label = i32, StateId = i32, Weight = TropicalWeight>,
    F: Fst<A>,
{
    let final_weight = fst.final_weight(state);
    if final_weight != TropicalWeight::zero() {
        out.push((
            ilabels.clone(),
            olabels.clone(),
            weight.times(&final_weight),
        ));
    }
    if left == 0 {
        return;
    }
    for arc in fst.arcs(state) {
        ilabels.push(arc.ilabel());
        olabels.push(arc.olabel());
        walk(
            fst,
            arc.nextstate(),
            ilabels,
            olabels,
            weight.times(arc.weight()),
            left - 1,
            out,
        );
        ilabels.pop();
        olabels.pop();
    }
}

/// Paths in a comparable order, with the weights rendered so that they compare
/// by value rather than by `TropicalWeight`'s ordering.
pub fn sorted(mut paths: Vec<Path>) -> Vec<(Vec<i32>, Vec<i32>, String)> {
    paths.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.cmp(&b.1))
            .then(a.2.value().total_cmp(&b.2.value()))
    });
    paths
        .into_iter()
        .map(|(i, o, w)| (i, o, format!("{:.4}", w.value())))
        .collect()
}

/// A random FST whose arcs only ever go forwards, so its set of accepting paths
/// is finite and can be enumerated in full.
pub fn random_acyclic_fst(rng: &mut Rng, max_states: usize) -> StdVectorFst {
    let nstates = 1 + rng.below(max_states);
    let mut fst = StdVectorFst::new();
    for _ in 0..nstates {
        fst.add_state();
    }
    fst.set_start(rng.below(nstates) as i32);
    for s in 0..nstates {
        for _ in 0..rng.below(3) {
            let to = s + 1 + rng.below(nstates - s.min(nstates - 1));
            if to >= nstates {
                continue;
            }
            let label = 1 + rng.below(3) as i32;
            fst.add_arc(
                s as i32,
                StdArc::new(label, label, TropicalWeight(rng.below(4) as f32), to as i32),
            );
        }
        if rng.below(3) == 0 {
            fst.set_final(s as i32, TropicalWeight(rng.below(4) as f32));
        }
    }
    fst
}

/// The weight each string pair comes to, which is the quantity an FST actually
/// defines: the semiring sum over every path carrying that pair.
///
/// Comparing whole paths is stronger than comparing this, and wrong for any
/// operation allowed to merge two paths for the same string, which in the
/// tropical semiring means keeping the lighter of the two.
pub fn string_weights(paths: Vec<Path>) -> Vec<(Vec<i32>, Vec<i32>, String)> {
    let mut totals: std::collections::BTreeMap<(Vec<i32>, Vec<i32>), TropicalWeight> =
        std::collections::BTreeMap::new();
    for (ilabels, olabels, weight) in paths {
        totals
            .entry((ilabels, olabels))
            .and_modify(|total| *total = total.plus(&weight))
            .or_insert(weight);
    }
    totals
        .into_iter()
        .map(|((i, o), w)| (i, o, format!("{:.4}", w.value())))
        .collect()
}

/// The paths of `fst` with epsilon labels dropped, since an operation may add
/// or remove arcs that consume nothing.
pub fn visible_paths<A, F>(fst: &F, max_len: usize) -> Vec<Path>
where
    A: Arc<Label = i32, StateId = i32, Weight = TropicalWeight>,
    F: Fst<A>,
{
    paths(fst, max_len)
        .into_iter()
        .map(|(i, o, w)| {
            (
                i.into_iter().filter(|&l| l != 0).collect(),
                o.into_iter().filter(|&l| l != 0).collect(),
                w,
            )
        })
        .collect()
}
