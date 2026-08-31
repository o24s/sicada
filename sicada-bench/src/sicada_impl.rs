//! The sicada side of the comparison.
//!
//! Built from the same generator as every other side; see [`crate::shape`] for
//! what the checksum means.

use sicada::algorithms::minimize::minimize;
use sicada::algorithms::shortest_distance::{
    SHORTEST_DELTA, shortest_distance, shortest_distance_forward,
};
use sicada::prelude::*;

use crate::Xorshift;

/// The FST type measured against.
pub type Sfst = StdVectorFst;

fn weight(rng: &mut Xorshift) -> TropicalWeight {
    TropicalWeight((rng.next_u64() % 400) as f32 / 4.0)
}

/// A random graph, as `openfst_algo_shim.cc`'s `Build` makes it.
pub fn graph(states: u64, arcs_per_state: u64, seed: u64, acyclic: bool) -> Sfst {
    let mut fst = Sfst::new();
    for _ in 0..states {
        fst.add_state();
    }
    fst.set_start(0);
    let mut rng = Xorshift::new(seed);
    for s in 0..states {
        for _ in 0..arcs_per_state {
            let label = 1 + (rng.next_u64() % 64) as i32;
            let w = weight(&mut rng);
            let next = if acyclic {
                let room = states - s - 1;
                if room == 0 {
                    continue;
                }
                s + 1 + rng.next_u64() % room
            } else {
                rng.next_u64() % states
            };
            fst.add_arc(s as i32, StdArc::new(label, label, w, next as i32));
        }
        if s % 8 == 0 {
            let w = weight(&mut rng);
            fst.set_final(s as i32, w);
        }
    }
    fst.properties(sicada::properties::K_FST_PROPERTIES, true);
    fst
}

/// An acyclic acceptor with epsilons, as the shim's `BuildAcceptor` makes it.
pub fn acceptor(states: u64, arcs_per_state: u64, seed: u64) -> Sfst {
    build_acceptor(states, arcs_per_state, seed, true)
}

/// The same without epsilons, as look-ahead composition requires of its second
/// argument.
pub fn dense_acceptor(states: u64, arcs_per_state: u64, seed: u64) -> Sfst {
    build_acceptor(states, arcs_per_state, seed, false)
}

fn build_acceptor(states: u64, arcs_per_state: u64, seed: u64, epsilons: bool) -> Sfst {
    let mut fst = Sfst::new();
    for _ in 0..states {
        fst.add_state();
    }
    fst.set_start(0);
    let mut rng = Xorshift::new(seed);
    for s in 0..states {
        for _ in 0..arcs_per_state {
            let draw = rng.next_u64() % 8;
            let label = if epsilons && draw == 0 {
                0
            } else {
                (1 + draw % 7) as i32
            };
            let w = weight(&mut rng);
            let room = states - s - 1;
            if room == 0 {
                continue;
            }
            let next = s + 1 + rng.next_u64() % room;
            fst.add_arc(s as i32, StdArc::new(label, label, w, next as i32));
        }
        if s % 8 == 0 {
            let w = weight(&mut rng);
            fst.set_final(s as i32, w);
        }
    }
    fst.properties(sicada::properties::K_FST_PROPERTIES, true);
    fst
}

/// The checksum every library's results are compared through.
pub fn checksum(fst: &Sfst) -> u64 {
    let (states, arcs, total) = parts(fst);
    crate::shape(states, arcs, total)
}

/// The three numbers the checksum is made of, for reporting a disagreement.
pub fn parts(fst: &Sfst) -> (usize, usize, Option<f32>) {
    let mut arcs = 0usize;
    for s in 0..fst.num_states() as i32 {
        arcs += fst.num_arcs(s);
    }
    (fst.num_states(), arcs, total_weight(fst))
}

/// The ⊕-sum over every path.
fn total_weight(fst: &Sfst) -> Option<f32> {
    let total = shortest_distance(fst, SHORTEST_DELTA).expect("shortest distance");
    (total != TropicalWeight::zero()).then_some(total.0)
}

/// As [`checksum`], and whether the arcs came out sorted on input labels.
pub fn sorted_checksum(fst: &Sfst) -> u64 {
    let mut arcs = 0usize;
    let mut sorted = true;
    for s in 0..fst.num_states() as i32 {
        let mut previous = 0i32;
        for arc in fst.arcs(s) {
            arcs += 1;
            if arc.ilabel() < previous {
                sorted = false;
            }
            previous = arc.ilabel();
        }
    }
    crate::sorted_shape(fst.num_states(), arcs, total_weight(fst), sorted)
}

/// A cheap value standing for "the result", so that timing cannot elide it.
fn size(fst: &Sfst) -> u64 {
    fst.num_states() as u64
}

/// `shortest_distance` from the start state.
pub mod shortest_distance_bench {
    use super::*;

    /// A checksum of the distances.
    pub fn verify(fst: &Sfst) -> u64 {
        let distance = shortest_distance_forward(fst, SHORTEST_DELTA).expect("shortest distance");
        crate::distance_checksum(
            distance
                .iter()
                .map(|w| (*w != TropicalWeight::zero()).then_some(w.0)),
        )
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Sfst) -> u64 {
        shortest_distance_forward(fst, SHORTEST_DELTA)
            .expect("shortest distance")
            .len() as u64
    }
}

/// One best path.
pub mod shortest_path_bench {
    use super::*;

    pub fn result(fst: &Sfst) -> Sfst {
        let mut out = Sfst::new();
        shortest_path(fst, &mut out, &ShortestPathOptions::default()).expect("shortest path");
        out
    }

    /// The result's checksum.
    pub fn verify(fst: &Sfst) -> u64 {
        checksum(&result(fst))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Sfst) -> u64 {
        size(&result(fst))
    }
}

/// A copy, then `connect`.
pub mod connect_bench {
    use super::*;

    pub fn result(fst: &Sfst) -> Sfst {
        let mut out = fst.clone();
        connect(&mut out);
        out
    }

    /// The result's checksum.
    pub fn verify(fst: &Sfst) -> u64 {
        checksum(&result(fst))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Sfst) -> u64 {
        size(&result(fst))
    }
}

/// A copy, then sorting on input labels.
pub mod arcsort_bench {
    use super::*;

    pub fn result(fst: &Sfst) -> Sfst {
        let mut out = fst.clone();
        arc_sort(&mut out, &ILabelCompare);
        out
    }

    /// The result's checksum, including whether it came out sorted.
    pub fn verify(fst: &Sfst) -> u64 {
        sorted_checksum(&result(fst))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Sfst) -> u64 {
        size(&result(fst))
    }
}

/// A copy, then a topological sort.
pub mod topsort_bench {
    use super::*;

    pub fn result(fst: &Sfst) -> Option<Sfst> {
        let mut out = fst.clone();
        match top_sort(&mut out) {
            Ok(true) => Some(out),
            _ => None,
        }
    }

    /// The result's checksum, or zero if the FST is cyclic.
    pub fn verify(fst: &Sfst) -> u64 {
        result(fst).map_or(0, |out| checksum(&out))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Sfst) -> u64 {
        result(fst).map_or(0, |out| size(&out))
    }
}

/// A copy, then epsilon removal.
pub mod rmepsilon_bench {
    use super::*;

    pub fn result(fst: &Sfst) -> Sfst {
        let mut out = fst.clone();
        rm_epsilon(&mut out, true).expect("rmepsilon");
        out
    }

    /// The same, with the state numbering left alone, so the arcs can be lined
    /// up against another library's.
    pub fn result_unconnected(fst: &Sfst) -> Sfst {
        let mut out = fst.clone();
        rm_epsilon(&mut out, false).expect("rmepsilon");
        out
    }

    /// The result's checksum.
    pub fn verify(fst: &Sfst) -> u64 {
        checksum(&result(fst))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Sfst) -> u64 {
        size(&result(fst))
    }
}

/// Determinization.
pub mod determinize_bench {
    use super::*;

    pub fn result(fst: &Sfst) -> Sfst {
        let mut out = Sfst::new();
        determinize(fst, &mut out, &DeterminizeOptions::default()).expect("determinize");
        // Determinization leaves states the result cannot finish from, and the
        // four libraries leave different ones; trimming makes the answers
        // comparable, and all four pay for it.
        connect(&mut out);
        out
    }

    /// The result's checksum.
    pub fn verify(fst: &Sfst) -> u64 {
        checksum(&result(fst))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Sfst) -> u64 {
        size(&result(fst))
    }
}

/// Determinization, then minimization.
pub mod minimize_bench {
    use super::*;

    pub fn result(fst: &Sfst) -> Sfst {
        let mut out = determinize_bench::result(fst);
        minimize(&mut out, sicada::algorithms::minimize::DELTA, false).expect("minimize");
        out
    }

    /// The result's checksum.
    pub fn verify(fst: &Sfst) -> u64 {
        checksum(&result(fst))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Sfst) -> u64 {
        size(&result(fst))
    }
}

/// Composition with a look-ahead index built once and reused, which is the
/// shape the index is for: one lexicon against many inputs.
pub mod compose_indexed_bench {
    use super::*;
    use sicada::algorithms::compose::{compose_lookahead_indexed, lookahead_index, sorted_copy};
    use sicada::algorithms::label_reachable::LabelReachableData;

    /// The sorted first argument and its index, built once.
    pub struct Prepared {
        left: Sfst,
        index: std::sync::Arc<LabelReachableData>,
    }

    /// Sorts `lhs` and builds its index.
    pub fn prepare(lhs: &Sfst) -> Prepared {
        let left = sorted_copy(lhs, true);
        let index = lookahead_index(&left).expect("an index");
        Prepared { left, index }
    }

    pub fn result(prepared: &Prepared, rhs: &Sfst) -> Sfst {
        let right = sorted_copy(rhs, false);
        let mut out = Sfst::new();
        compose_lookahead_indexed(&prepared.left, &prepared.index, &right, &mut out)
            .expect("compose with a prebuilt index");
        out
    }

    /// The result's checksum.
    pub fn verify(prepared: &Prepared, rhs: &Sfst) -> u64 {
        checksum(&result(prepared, rhs))
    }

    /// The same work, returning something cheap.
    pub fn run(prepared: &Prepared, rhs: &Sfst) -> u64 {
        size(&result(prepared, rhs))
    }
}

/// Composition of two acceptors, with a look-ahead matcher over the first.
pub mod compose_lookahead_bench {
    use super::*;

    pub fn result(lhs: &Sfst, rhs: &Sfst) -> Sfst {
        let mut out = Sfst::new();
        sicada::algorithms::compose::compose_lookahead(lhs, rhs, &mut out)
            .expect("compose with look-ahead");
        out
    }

    /// The result's checksum.
    pub fn verify(lhs: &Sfst, rhs: &Sfst) -> u64 {
        checksum(&result(lhs, rhs))
    }

    /// The same work, returning something cheap.
    pub fn run(lhs: &Sfst, rhs: &Sfst) -> u64 {
        size(&result(lhs, rhs))
    }
}

/// Composition of two acceptors.
pub mod compose_bench {
    use super::*;

    pub fn result(lhs: &Sfst, rhs: &Sfst) -> Sfst {
        let mut out = Sfst::new();
        compose(lhs, rhs, &mut out).expect("compose");
        out
    }

    /// The result's checksum.
    pub fn verify(lhs: &Sfst, rhs: &Sfst) -> u64 {
        checksum(&result(lhs, rhs))
    }

    /// The same work, returning something cheap.
    pub fn run(lhs: &Sfst, rhs: &Sfst) -> u64 {
        size(&result(lhs, rhs))
    }
}
