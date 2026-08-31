//! The rustfst side of the comparison.
//!
//! Built from the same generator as every other side, so all four libraries are
//! handed the same graph; see [`crate::shape`] for what the checksum means.

use rustfst::prelude::*;

use crate::Xorshift;

/// The FST type measured against.
pub type Fst = VectorFst<TropicalWeight>;

fn weight(rng: &mut Xorshift) -> TropicalWeight {
    TropicalWeight::new((rng.next_u64() % 400) as f32 / 4.0)
}

/// A random graph, as `openfst_algo_shim.cc`'s `Build` makes it.
pub fn graph(states: u64, arcs_per_state: u64, seed: u64, acyclic: bool) -> Fst {
    let mut fst = Fst::new();
    fst.add_states(states as usize);
    fst.set_start(0).expect("a state to start from");
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
            fst.add_tr(s as u32, Tr::new(label, label, w, next as u32))
                .expect("the state exists");
        }
        if s % 8 == 0 {
            let w = weight(&mut rng);
            fst.set_final(s as u32, w).expect("the state exists");
        }
    }
    fst.compute_and_update_properties_all()
        .expect("properties are computable");
    fst
}

/// An acyclic acceptor with epsilons, as the shim's `BuildAcceptor` makes it.
pub fn acceptor(states: u64, arcs_per_state: u64, seed: u64) -> Fst {
    build_acceptor(states, arcs_per_state, seed, true)
}

/// The same without epsilons, as look-ahead composition requires of its second
/// argument.
pub fn dense_acceptor(states: u64, arcs_per_state: u64, seed: u64) -> Fst {
    build_acceptor(states, arcs_per_state, seed, false)
}

fn build_acceptor(states: u64, arcs_per_state: u64, seed: u64, epsilons: bool) -> Fst {
    let mut fst = Fst::new();
    fst.add_states(states as usize);
    fst.set_start(0).expect("a state to start from");
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
            fst.add_tr(s as u32, Tr::new(label, label, w, next as u32))
                .expect("the state exists");
        }
        if s % 8 == 0 {
            let w = weight(&mut rng);
            fst.set_final(s as u32, w).expect("the state exists");
        }
    }
    fst.compute_and_update_properties_all()
        .expect("properties are computable");
    fst
}

/// The checksum every library's results are compared through.
pub fn checksum(fst: &Fst) -> u64 {
    let (states, arcs, total) = parts(fst);
    crate::shape(states, arcs, total)
}

/// The three numbers the checksum is made of, for reporting a disagreement.
pub fn parts(fst: &Fst) -> (usize, usize, Option<f32>) {
    let mut arcs = 0usize;
    for s in 0..fst.num_states() as u32 {
        arcs += fst.get_trs(s).expect("the state exists").trs().len();
    }
    (fst.num_states(), arcs, shortest_distance_total(fst))
}

/// The ⊕-sum over every path, as a whole number of quarters.
fn shortest_distance_total(fst: &Fst) -> Option<f32> {
    let distance = shortest_distance(fst, false).ok()?;
    let mut total = TropicalWeight::zero();
    for (state, d) in distance.iter().enumerate() {
        if let Ok(Some(f)) = fst.final_weight(state as u32) {
            total.plus_assign(d.times(f).ok()?).ok()?;
        }
    }
    (!total.is_zero()).then(|| *total.value())
}

/// As [`checksum`], and whether the arcs came out sorted on input labels.
pub fn sorted_checksum(fst: &Fst) -> u64 {
    let mut arcs = 0usize;
    let mut sorted = true;
    for s in 0..fst.num_states() as u32 {
        let trs = fst.get_trs(s).expect("the state exists");
        arcs += trs.trs().len();
        let mut previous = 0u32;
        for tr in trs.trs() {
            if tr.ilabel < previous {
                sorted = false;
            }
            previous = tr.ilabel;
        }
    }
    crate::sorted_shape(fst.num_states(), arcs, shortest_distance_total(fst), sorted)
}

/// A cheap value standing for "the result", so that timing cannot elide it.
fn size<F: ExpandedFst<TropicalWeight>>(fst: &F) -> u64 {
    fst.num_states() as u64
}

/// `shortest_distance` from the start state.
pub mod shortest_distance_bench {
    use super::*;

    /// A checksum of the distances.
    pub fn verify(fst: &Fst) -> u64 {
        let Ok(distance) = shortest_distance(fst, false) else {
            return u64::MAX;
        };
        crate::distance_checksum(distance.iter().map(|w| {
            let v = *w.value();
            v.is_finite().then_some(v)
        }))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Fst) -> u64 {
        shortest_distance(fst, false).map_or(0, |d| d.len() as u64)
    }
}

/// One best path.
pub mod shortest_path_bench {
    use super::*;

    pub fn result(fst: &Fst) -> Option<Fst> {
        shortest_path(fst).ok()
    }

    /// The result's checksum.
    pub fn verify(fst: &Fst) -> u64 {
        result(fst).map_or(u64::MAX, |out| checksum(&out))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Fst) -> u64 {
        result(fst).map_or(0, |out| size(&out))
    }
}

/// A copy, then `connect`.
pub mod connect_bench {
    use super::*;

    pub fn result(fst: &Fst) -> Fst {
        let mut out = fst.clone();
        connect(&mut out).expect("connect");
        out
    }

    /// The result's checksum.
    pub fn verify(fst: &Fst) -> u64 {
        checksum(&result(fst))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Fst) -> u64 {
        size(&result(fst))
    }
}

/// A copy, then sorting on input labels.
pub mod arcsort_bench {
    use super::*;

    pub fn result(fst: &Fst) -> Fst {
        let mut out = fst.clone();
        tr_sort(&mut out, ILabelCompare {});
        out
    }

    /// The result's checksum, including whether it came out sorted.
    pub fn verify(fst: &Fst) -> u64 {
        sorted_checksum(&result(fst))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Fst) -> u64 {
        size(&result(fst))
    }
}

/// A copy, then a topological sort.
pub mod topsort_bench {
    use super::*;

    pub fn result(fst: &Fst) -> Option<Fst> {
        let mut out = fst.clone();
        top_sort(&mut out).ok()?;
        Some(out)
    }

    /// The result's checksum, or zero if the FST is cyclic.
    pub fn verify(fst: &Fst) -> u64 {
        result(fst).map_or(0, |out| checksum(&out))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Fst) -> u64 {
        result(fst).map_or(0, |out| size(&out))
    }
}

/// A copy, then epsilon removal.
pub mod rmepsilon_bench {
    use super::*;

    pub fn result(fst: &Fst) -> Fst {
        let mut out = fst.clone();
        rustfst::algorithms::rm_epsilon::rm_epsilon(&mut out).expect("rm_epsilon");
        connect(&mut out).expect("connect");
        out
    }

    /// The result's checksum.
    pub fn verify(fst: &Fst) -> u64 {
        checksum(&result(fst))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Fst) -> u64 {
        size(&result(fst))
    }
}

/// Determinization.
pub mod determinize_bench {
    use super::*;

    pub fn result(fst: &Fst) -> Option<Fst> {
        let mut out: Fst = rustfst::algorithms::determinize::determinize(fst).ok()?;
        // rustfst leaves states the result cannot finish from; the other three
        // do not, so they are trimmed here to make the answers comparable.
        connect(&mut out).ok()?;
        Some(out)
    }

    /// The result's checksum.
    pub fn verify(fst: &Fst) -> u64 {
        result(fst).map_or(u64::MAX, |out| checksum(&out))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Fst) -> u64 {
        result(fst).map_or(0, |out| size(&out))
    }
}

/// Determinization, then minimization.
pub mod minimize_bench {
    use super::*;

    pub fn result(fst: &Fst) -> Option<Fst> {
        let mut out: Fst = rustfst::algorithms::determinize::determinize(fst).ok()?;
        minimize(&mut out).ok()?;
        Some(out)
    }

    /// The result's checksum.
    pub fn verify(fst: &Fst) -> u64 {
        result(fst).map_or(u64::MAX, |out| checksum(&out))
    }

    /// The same work, returning something cheap.
    pub fn run(fst: &Fst) -> u64 {
        result(fst).map_or(0, |out| size(&out))
    }
}

/// Composition of two acceptors.
pub mod compose_bench {
    use super::*;

    pub fn result(lhs: &Fst, rhs: &Fst) -> Option<Fst> {
        let mut left = lhs.clone();
        let mut right = rhs.clone();
        tr_sort(&mut left, OLabelCompare {});
        tr_sort(&mut right, ILabelCompare {});
        let mut out: Fst = rustfst::algorithms::compose::compose(left, right).ok()?;
        connect(&mut out).ok()?;
        Some(out)
    }

    /// The result's checksum.
    pub fn verify(lhs: &Fst, rhs: &Fst) -> u64 {
        result(lhs, rhs).map_or(u64::MAX, |out| checksum(&out))
    }

    /// The same work, returning something cheap.
    pub fn run(lhs: &Fst, rhs: &Fst) -> u64 {
        result(lhs, rhs).map_or(0, |out| size(&out))
    }
}
