//! The arcweight side of the comparison.
//!
//! Built from the same generator as every other side, so all four libraries are
//! handed the same graph; see [`crate::shape`] for what the checksum means.

use arcweight::prelude::*;

use crate::Xorshift;

/// The FST type measured against.
pub type Afst = VectorFst<TropicalWeight>;

fn weight(rng: &mut Xorshift) -> TropicalWeight {
    TropicalWeight::new((rng.next_u64() % 400) as f32 / 4.0)
}

/// A random graph, as `openfst_algo_shim.cc`'s `Build` makes it.
pub fn graph(states: u64, arcs_per_state: u64, seed: u64, acyclic: bool) -> Afst {
    let mut fst = Afst::new();
    for _ in 0..states {
        fst.add_state();
    }
    fst.set_start(0);
    let mut rng = Xorshift::new(seed);
    for s in 0..states {
        for _ in 0..arcs_per_state {
            let label = 1 + (rng.next_u64() % 64) as u32;
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
            fst.add_arc(s as u32, Arc::new(label, label, w, next as u32));
        }
        if s % 8 == 0 {
            let w = weight(&mut rng);
            fst.set_final(s as u32, w);
        }
    }
    fst
}

/// An acyclic acceptor with epsilons, as the shim's `BuildAcceptor` makes it.
pub fn acceptor(states: u64, arcs_per_state: u64, seed: u64) -> Afst {
    build_acceptor(states, arcs_per_state, seed, true)
}

/// The same without epsilons, as look-ahead composition requires of its second
/// argument.
pub fn dense_acceptor(states: u64, arcs_per_state: u64, seed: u64) -> Afst {
    build_acceptor(states, arcs_per_state, seed, false)
}

fn build_acceptor(states: u64, arcs_per_state: u64, seed: u64, epsilons: bool) -> Afst {
    let mut fst = Afst::new();
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
                (1 + draw % 7) as u32
            };
            let w = weight(&mut rng);
            let room = states - s - 1;
            if room == 0 {
                continue;
            }
            let next = s + 1 + rng.next_u64() % room;
            fst.add_arc(s as u32, Arc::new(label, label, w, next as u32));
        }
        if s % 8 == 0 {
            let w = weight(&mut rng);
            fst.set_final(s as u32, w);
        }
    }
    fst
}

/// The checksum every library's results are compared through.
pub fn checksum(fst: &Afst) -> u64 {
    let (states, arcs, total) = parts(fst);
    crate::shape(states, arcs, total)
}

/// The three numbers the checksum is made of, for reporting a disagreement.
pub fn parts(fst: &Afst) -> (usize, usize, Option<f32>) {
    let mut arcs = 0usize;
    for s in 0..fst.num_states() as u32 {
        arcs += fst.num_arcs(s);
    }
    (fst.num_states(), arcs, total_weight(fst))
}

/// The ⊕-sum over every path.
fn total_weight(fst: &Afst) -> Option<f32> {
    let distance = shortest_distance(fst).ok()?;
    let mut total = TropicalWeight::zero();
    for (state, d) in distance.iter().enumerate() {
        if let Some(f) = fst.final_weight(state as u32) {
            total = total.plus(&d.times(f));
        }
    }
    let value = *total.value();
    value.is_finite().then_some(value)
}

/// As [`checksum`], and whether the arcs came out sorted on input labels.
pub fn sorted_checksum(fst: &Afst) -> u64 {
    let mut arcs = 0usize;
    let mut sorted = true;
    for s in 0..fst.num_states() as u32 {
        let mut previous = 0u32;
        for arc in fst.arcs(s) {
            arcs += 1;
            if arc.ilabel < previous {
                sorted = false;
            }
            previous = arc.ilabel;
        }
    }
    crate::sorted_shape(fst.num_states(), arcs, total_weight(fst), sorted)
}

/// A cheap value standing for "the result", so that timing cannot elide it.
fn size(fst: &Afst) -> u64 {
    fst.num_states() as u64
}

/// `shortest_distance` from the start state.
pub mod shortest_distance_bench {
    use super::*;

    /// A checksum of the distances.
    pub fn verify(fst: &Afst) -> u64 {
        let Ok(distance) = shortest_distance(fst) else {
            return u64::MAX;
        };
        crate::distance_checksum(distance.iter().map(|w| {
            let v = *w.value();
            v.is_finite().then_some(v)
        }))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Afst) -> u64 {
        shortest_distance(fst).map_or(0, |d| d.len() as u64)
    }
}

/// One best path.
pub mod shortest_path_bench {
    use super::*;

    pub fn result(fst: &Afst) -> Option<Afst> {
        shortest_path_single(fst).ok()
    }

    /// The result's checksum.
    pub fn verify(fst: &Afst) -> u64 {
        result(fst).map_or(u64::MAX, |out| checksum(&out))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Afst) -> u64 {
        result(fst).map_or(0, |out| size(&out))
    }
}

/// A copy, then `connect`.
pub mod connect_bench {
    use super::*;

    pub fn result(fst: &Afst) -> Option<Afst> {
        connect(fst).ok()
    }

    /// The result's checksum.
    pub fn verify(fst: &Afst) -> u64 {
        result(fst).map_or(u64::MAX, |out| checksum(&out))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Afst) -> u64 {
        result(fst).map_or(0, |out| size(&out))
    }
}

/// A copy, then sorting on input labels.
pub mod arcsort_bench {
    use super::*;

    pub fn result(fst: &Afst) -> Afst {
        let mut out = fst.clone();
        arc_sort(&mut out, ArcSortType::ByInput).expect("arc_sort");
        out
    }

    /// The result's checksum, including whether it came out sorted.
    pub fn verify(fst: &Afst) -> u64 {
        sorted_checksum(&result(fst))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Afst) -> u64 {
        size(&result(fst))
    }
}

/// A copy, then a topological sort.
pub mod topsort_bench {
    use super::*;

    pub fn result(fst: &Afst) -> Option<Afst> {
        topsort(fst).ok()
    }

    /// The result's checksum, or zero if the FST is cyclic.
    pub fn verify(fst: &Afst) -> u64 {
        result(fst).map_or(0, |out| checksum(&out))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Afst) -> u64 {
        result(fst).map_or(0, |out| size(&out))
    }
}

/// A copy, then epsilon removal.
pub mod rmepsilon_bench {
    use super::*;

    pub fn result(fst: &Afst) -> Option<Afst> {
        let out: Afst = remove_epsilons(fst).ok()?;
        connect(&out).ok()
    }

    /// The same, with the state numbering left alone, so the arcs can be lined
    /// up against another library's.
    pub fn result_unconnected(fst: &Afst) -> Option<Afst> {
        remove_epsilons(fst).ok()
    }

    /// The result's checksum.
    pub fn verify(fst: &Afst) -> u64 {
        result(fst).map_or(u64::MAX, |out| checksum(&out))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Afst) -> u64 {
        result(fst).map_or(0, |out| size(&out))
    }
}

/// Determinization.
pub mod determinize_bench {
    use super::*;

    pub fn result(fst: &Afst) -> Option<Afst> {
        let out: Afst = determinize(fst).ok()?;
        // See the note in the other three: trimmed so the answers compare.
        connect(&out).ok()
    }

    /// The result's checksum.
    pub fn verify(fst: &Afst) -> u64 {
        result(fst).map_or(u64::MAX, |out| checksum(&out))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Afst) -> u64 {
        result(fst).map_or(0, |out| size(&out))
    }
}

/// Determinization, then minimization.
pub mod minimize_bench {
    use super::*;

    pub fn result(fst: &Afst) -> Option<Afst> {
        let out: Afst = determinize(fst).ok()?;
        minimize(&out).ok()
    }

    /// The result's checksum.
    pub fn verify(fst: &Afst) -> u64 {
        result(fst).map_or(u64::MAX, |out| checksum(&out))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Afst) -> u64 {
        result(fst).map_or(0, |out| size(&out))
    }
}

/// Composition of two acceptors.
pub mod compose_bench {
    use super::*;

    pub fn result(lhs: &Afst, rhs: &Afst) -> Option<Afst> {
        let mut left = lhs.clone();
        let mut right = rhs.clone();
        arc_sort(&mut left, ArcSortType::ByOutput).ok()?;
        arc_sort(&mut right, ArcSortType::ByInput).ok()?;
        let out: Afst = compose_default(&left, &right).ok()?;
        connect(&out).ok()
    }

    /// The result's checksum.
    pub fn verify(lhs: &Afst, rhs: &Afst) -> u64 {
        result(lhs, rhs).map_or(u64::MAX, |out| checksum(&out))
    }

    /// The same work, returning something cheap.
    pub fn run(lhs: &Afst, rhs: &Afst) -> u64 {
        result(lhs, rhs).map_or(0, |out| size(&out))
    }
}
