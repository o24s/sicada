//! Working out an FST's properties by looking at it.
//!
//! Port of OpenFst's `test-properties.h`. An FST carries a cache of property
//! bits, kept up to date by the operations that change it; see
//! [`properties`](crate::properties). This module is the other direction: given
//! an FST, scan it and settle the bits the cache does not know.
//!
//! That is what `fst.properties(mask, /*test=*/true)` means, and it is the only
//! caller these functions really have.

use rustc_hash::FxHashSet;

use crate::algorithms::cc_visitors::SccVisitor;
use crate::algorithms::dfs_visit::dfs_visit_any;
use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::fst::{Fst, PropertyCache};
use crate::properties::{
    K_ACCEPTOR, K_ACCESSIBLE, K_ACYCLIC, K_BINARY_PROPERTIES, K_CO_ACCESSIBLE, K_CYCLIC,
    K_EPSILONS, K_ERROR, K_FST_PROPERTIES, K_I_DETERMINISTIC, K_I_EPSILONS, K_I_LABEL_SORTED,
    K_INITIAL_ACYCLIC, K_INITIAL_CYCLIC, K_NO_EPSILONS, K_NO_I_EPSILONS, K_NO_O_EPSILONS,
    K_NON_I_DETERMINISTIC, K_NON_O_DETERMINISTIC, K_NOT_ACCEPTOR, K_NOT_ACCESSIBLE,
    K_NOT_CO_ACCESSIBLE, K_NOT_I_LABEL_SORTED, K_NOT_O_LABEL_SORTED, K_NOT_STRING,
    K_NOT_TOP_SORTED, K_O_DETERMINISTIC, K_O_EPSILONS, K_O_LABEL_SORTED, K_STRING, K_TOP_SORTED,
    K_UNWEIGHTED, K_UNWEIGHTED_CYCLES, K_WEIGHTED, K_WEIGHTED_CYCLES,
    internal::{compat_properties, known_properties},
};
use crate::weight::Weight;

/// What a scan settled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputedProperties {
    /// The property bits, the false ones as much as the true ones.
    pub props: u64,
    /// Which bits the scan settled either way.
    pub known: u64,
}

/// The properties only a depth-first search can settle.
///
/// Kept apart because the search costs a stack proportional to the FST's depth,
/// and most masks do not need it.
const K_DFS_PROPERTIES: u64 = K_CYCLIC
    | K_ACYCLIC
    | K_INITIAL_CYCLIC
    | K_INITIAL_ACYCLIC
    | K_ACCESSIBLE
    | K_NOT_ACCESSIBLE
    | K_CO_ACCESSIBLE
    | K_NOT_CO_ACCESSIBLE;

/// The properties whose computation needs to know each state's component.
const K_CYCLE_PROPERTIES: u64 = K_DFS_PROPERTIES | K_WEIGHTED_CYCLES | K_UNWEIGHTED_CYCLES;

/// Scans `fst` and returns the properties in `mask`, plus any others the scan
/// happened to settle on the way.
///
/// Nothing comes back unknown: a bit named in `mask` is either set or cleared.
pub fn compute_properties<A: Arc, F: Fst<A>>(fst: &F, mask: u64) -> ComputedProperties {
    // The binary properties are stored rather than derived, so they carry over
    // as they are.
    let mut props = fst.properties(K_FST_PROPERTIES, false) & K_BINARY_PROPERTIES;

    let mut scc: Vec<A::StateId> = Vec::new();
    if mask & K_CYCLE_PROPERTIES != 0 {
        let mut visitor = SccVisitor::new(fst, Some(&mut scc), None, None, &mut props);
        dfs_visit_any(fst, &mut visitor);
    }

    if mask & !(K_BINARY_PROPERTIES | K_DFS_PROPERTIES) != 0 {
        // Each of these holds until an arc or a state disproves it.
        props |= K_ACCEPTOR
            | K_NO_EPSILONS
            | K_NO_I_EPSILONS
            | K_NO_O_EPSILONS
            | K_I_LABEL_SORTED
            | K_O_LABEL_SORTED
            | K_UNWEIGHTED
            | K_TOP_SORTED
            | K_STRING;
        let check_ideterministic = mask & (K_I_DETERMINISTIC | K_NON_I_DETERMINISTIC) != 0;
        let check_odeterministic = mask & (K_O_DETERMINISTIC | K_NON_O_DETERMINISTIC) != 0;
        if check_ideterministic {
            props |= K_I_DETERMINISTIC;
        }
        if check_odeterministic {
            props |= K_O_DETERMINISTIC;
        }
        // Only claimed when the search above ran, since it is what filled `scc`.
        if mask & K_CYCLE_PROPERTIES != 0 {
            props |= K_UNWEIGHTED_CYCLES;
        }

        let epsilon = A::Label::epsilon();
        let zero = A::Weight::zero();
        let one = A::Weight::one();
        // SICADA-OPT: upstream builds a fresh hash set per state and per side.
        // Reusing one keeps the buckets a long FST has already paid for.
        let mut ilabels: FxHashSet<A::Label> = FxHashSet::default();
        let mut olabels: FxHashSet<A::Label> = FxHashSet::default();
        let mut nfinal = 0usize;

        for s in fst.states() {
            ilabels.clear();
            olabels.clear();
            let mut prev: Option<A> = None;
            let mut narcs = 0usize;

            for arc in fst.arcs(s) {
                narcs += 1;
                // SICADA-OPT: upstream looks the label up and then inserts it;
                // `insert` reports whether it was already there, so one probe
                // does both.
                if check_ideterministic && !ilabels.insert(arc.ilabel()) {
                    props |= K_NON_I_DETERMINISTIC;
                    props &= !K_I_DETERMINISTIC;
                }
                if check_odeterministic && !olabels.insert(arc.olabel()) {
                    props |= K_NON_O_DETERMINISTIC;
                    props &= !K_O_DETERMINISTIC;
                }
                if arc.ilabel() != arc.olabel() {
                    props |= K_NOT_ACCEPTOR;
                    props &= !K_ACCEPTOR;
                }
                if arc.ilabel() == epsilon && arc.olabel() == epsilon {
                    props |= K_EPSILONS;
                    props &= !K_NO_EPSILONS;
                }
                if arc.ilabel() == epsilon {
                    props |= K_I_EPSILONS;
                    props &= !K_NO_I_EPSILONS;
                }
                if arc.olabel() == epsilon {
                    props |= K_O_EPSILONS;
                    props &= !K_NO_O_EPSILONS;
                }
                if let Some(prev) = &prev {
                    if arc.ilabel() < prev.ilabel() {
                        props |= K_NOT_I_LABEL_SORTED;
                        props &= !K_I_LABEL_SORTED;
                    }
                    if arc.olabel() < prev.olabel() {
                        props |= K_NOT_O_LABEL_SORTED;
                        props &= !K_O_LABEL_SORTED;
                    }
                }
                if *arc.weight() != one && *arc.weight() != zero {
                    props |= K_WEIGHTED;
                    props &= !K_UNWEIGHTED;
                    // An arc within a component is an arc on a cycle.
                    //
                    // SICADA-BUGFIX: upstream indexes `scc` here whenever
                    // `kUnweightedCycles` is set, and sets it whenever the mask
                    // asks for a cycle property, but the search that fills
                    // `scc` returns without visiting anything when the FST has
                    // no start state, leaving it empty. Any weighted arc then
                    // reads off the end of it. Skipping the test where the
                    // search left no answer keeps the values upstream produces
                    // everywhere upstream is defined.
                    if props & K_UNWEIGHTED_CYCLES != 0
                        && let Some(&from) = scc.get(s.as_usize())
                        && let Some(&to) = scc.get(arc.nextstate().as_usize())
                        && from == to
                    {
                        props |= K_WEIGHTED_CYCLES;
                        props &= !K_UNWEIGHTED_CYCLES;
                    }
                }
                if arc.nextstate() <= s {
                    props |= K_NOT_TOP_SORTED;
                    props &= !K_TOP_SORTED;
                }
                if arc.nextstate().as_usize() != s.as_usize() + 1 {
                    props |= K_NOT_STRING;
                    props &= !K_STRING;
                }
                prev = Some(arc);
            }

            // A string has exactly one final state, and it comes last.
            if nfinal > 0 {
                props |= K_NOT_STRING;
                props &= !K_STRING;
            }
            let final_weight = fst.final_weight(s);
            if final_weight != zero {
                if final_weight != one {
                    props |= K_WEIGHTED;
                    props &= !K_UNWEIGHTED;
                }
                nfinal += 1;
            } else if narcs != 1 {
                // Every other state of a string has exactly one way out.
                props |= K_NOT_STRING;
                props &= !K_STRING;
            }
        }

        if let Some(start) = fst.start()
            && start.as_usize() != 0
        {
            props |= K_NOT_STRING;
            props &= !K_STRING;
        }
    }

    ComputedProperties {
        props,
        known: known_properties(props),
    }
}

/// Returns the properties in `mask`, scanning `fst` only if what it already
/// stores does not cover them.
pub fn compute_or_use_stored_properties<A: Arc, F: Fst<A>>(
    fst: &F,
    mask: u64,
) -> ComputedProperties {
    let props = fst.properties(K_FST_PROPERTIES, false);
    let known = known_properties(props);
    if known & mask == mask {
        return ComputedProperties { props, known };
    }
    compute_properties(fst, mask)
}

/// Returns the properties in `mask`, and with `verify` also checks the stored
/// ones against them.
///
/// This is what `fst.properties(mask, /*test=*/true)` runs. `verify` stands in
/// for upstream's `fst_verify_properties` flag: it forces the scan even when the
/// cache would have answered, so that a cache which has gone wrong is caught.
///
/// SICADA-DIVERGE: upstream logs the mismatch and returns the computed
/// properties regardless. Here [`K_ERROR`] is set in what comes back, which is
/// the bit that exists to carry exactly this news to a caller that never sees
/// the log.
pub fn test_properties<A: Arc, F: Fst<A>>(fst: &F, mask: u64, verify: bool) -> ComputedProperties {
    if !verify {
        return compute_or_use_stored_properties(fst, mask);
    }
    let stored = fst.properties(K_FST_PROPERTIES, false);
    let mut computed = compute_properties(fst, mask);
    if !compat_properties(stored, computed.props) {
        computed.props |= K_ERROR;
    }
    computed
}

/// Returns the properties in `check_mask | test_mask`, scanning only if the
/// stored properties do not already cover `check_mask`.
///
/// The split is for properties added to the library after a file was written:
/// `check_mask` is what an old file would know about, `test_mask` what it would
/// not, so a scan triggered by the former settles the latter for free.
pub fn check_properties<A: Arc, F: Fst<A>>(
    fst: &F,
    check_mask: u64,
    test_mask: u64,
    verify: bool,
) -> u64 {
    let mut props = fst.properties(K_FST_PROPERTIES, false);
    if verify {
        props = test_properties(fst, check_mask | test_mask, true).props;
    } else if known_properties(props) & check_mask != check_mask {
        props = compute_properties(fst, check_mask | test_mask).props;
    }
    props & (check_mask | test_mask)
}

/// The body of [`Fst::properties`] for an FST that caches its property bits.
///
/// Port of `ImplToFst::Properties`, which is the only part of `impl-to-fst.h`
/// that carries behaviour rather than C++ lifetime plumbing; the rest of that
/// header has no counterpart here. With `test`, whatever the cache does not know
/// is worked out by scanning and written back, so the next caller does not pay
/// for it again.
///
/// SICADA-DIVERGE: upstream reads `fst_verify_properties` here, a process-wide
/// flag that turns every such call into a check of the cache against the FST.
/// A caller who wants that calls [`test_properties`] with `verify` set, rather
/// than flipping a switch under every other library in the same binary.
pub fn cached_properties<A: Arc, F: Fst<A>>(
    fst: &F,
    cache: &PropertyCache,
    mask: u64,
    test: bool,
) -> u64 {
    if !test {
        return cache.get_masked(mask);
    }
    let computed = test_properties(fst, mask, false);
    cache.discover(computed.props, computed.known);
    computed.props & mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AtomicRc;
    use crate::arc::StdArc;
    use crate::fst::{ExpandedFst, MutableFst};
    use crate::fsts::vector_fst::VectorFst;
    use crate::properties::{K_ACYCLIC, K_MUTABLE, K_TRINARY_PROPERTIES};
    use crate::symbol_table::SymbolTable;
    use crate::weights::float_weight::TropicalWeight;
    use std::iter::Empty;

    /// A small reproducible generator; the FSTs it drives have to be the same
    /// from run to run for a failure to be worth anything.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self, bound: usize) -> usize {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 33) as usize) % bound
        }
    }

    fn random_fst(rng: &mut Rng, nstates: usize) -> VectorFst<StdArc> {
        let mut fst = VectorFst::new();
        for _ in 0..nstates {
            fst.add_state();
        }
        if nstates > 0 {
            fst.set_start(rng.next(nstates) as i32);
        }
        for s in 0..nstates {
            for _ in 0..rng.next(4) {
                // A small label alphabet, so that repeats, and with them
                // non-determinism and unsortedness, actually happen.
                let ilabel = rng.next(3) as i32;
                let olabel = rng.next(3) as i32;
                let weight = match rng.next(3) {
                    0 => TropicalWeight::one(),
                    1 => TropicalWeight::zero(),
                    _ => TropicalWeight(rng.next(5) as f32 + 1.0),
                };
                fst.add_arc(
                    s as i32,
                    StdArc::new(ilabel, olabel, weight, rng.next(nstates) as i32),
                );
            }
            match rng.next(4) {
                0 => fst.set_final(s as i32, TropicalWeight::one()),
                1 => fst.set_final(s as i32, TropicalWeight(rng.next(5) as f32 + 1.0)),
                _ => {}
            }
        }
        fst
    }

    /// `reaches[s][t]`: t is reachable from s along one or more arcs.
    fn reachability(fst: &VectorFst<StdArc>) -> Vec<Vec<bool>> {
        let n = fst.num_states();
        let mut reaches = vec![vec![false; n]; n];
        for (s, row) in reaches.iter_mut().enumerate() {
            for arc in fst.arcs(s as i32) {
                row[arc.nextstate() as usize] = true;
            }
        }
        for k in 0..n {
            let through_k = reaches[k].clone();
            for row in reaches.iter_mut() {
                if row[k] {
                    for (dest, &reached) in row.iter_mut().zip(&through_k) {
                        *dest |= reached;
                    }
                }
            }
        }
        reaches
    }

    /// Works out the same properties from their definitions, without reusing
    /// any of the code under test.
    fn expected(fst: &VectorFst<StdArc>) -> u64 {
        let n = fst.num_states();
        let zero = TropicalWeight::zero();
        let one = TropicalWeight::one();
        let trivial = |w: TropicalWeight| w == zero || w == one;
        let arcs = |s: usize| fst.arcs(s as i32).collect::<Vec<_>>();

        let all_arcs = || (0..n).flat_map(|s| arcs(s).into_iter().map(move |a| (s, a)));

        let mut props = 0;
        props |= if all_arcs().all(|(_, a)| a.ilabel() == a.olabel()) {
            K_ACCEPTOR
        } else {
            K_NOT_ACCEPTOR
        };
        props |= if all_arcs().any(|(_, a)| a.ilabel() == 0 && a.olabel() == 0) {
            K_EPSILONS
        } else {
            K_NO_EPSILONS
        };
        props |= if all_arcs().any(|(_, a)| a.ilabel() == 0) {
            K_I_EPSILONS
        } else {
            K_NO_I_EPSILONS
        };
        props |= if all_arcs().any(|(_, a)| a.olabel() == 0) {
            K_O_EPSILONS
        } else {
            K_NO_O_EPSILONS
        };

        let unique = |labels: Vec<i32>| {
            let mut sorted = labels;
            sorted.sort_unstable();
            let len = sorted.len();
            sorted.dedup();
            sorted.len() == len
        };
        props |= if (0..n).all(|s| unique(arcs(s).iter().map(|a| a.ilabel()).collect())) {
            K_I_DETERMINISTIC
        } else {
            K_NON_I_DETERMINISTIC
        };
        props |= if (0..n).all(|s| unique(arcs(s).iter().map(|a| a.olabel()).collect())) {
            K_O_DETERMINISTIC
        } else {
            K_NON_O_DETERMINISTIC
        };

        let sorted_by = |key: fn(&StdArc) -> i32| {
            (0..n).all(|s| arcs(s).windows(2).all(|w| key(&w[0]) <= key(&w[1])))
        };
        props |= if sorted_by(|a| a.ilabel()) {
            K_I_LABEL_SORTED
        } else {
            K_NOT_I_LABEL_SORTED
        };
        props |= if sorted_by(|a| a.olabel()) {
            K_O_LABEL_SORTED
        } else {
            K_NOT_O_LABEL_SORTED
        };

        let weighted = all_arcs().any(|(_, a)| !trivial(*a.weight()))
            || (0..n).any(|s| {
                let w = fst.final_weight(s as i32);
                w != zero && w != one
            });
        props |= if weighted { K_WEIGHTED } else { K_UNWEIGHTED };

        props |= if all_arcs().all(|(s, a)| a.nextstate() as usize > s) {
            K_TOP_SORTED
        } else {
            K_NOT_TOP_SORTED
        };

        // A string is a chain 0 → 1 → … → n-1 whose last state, and only its
        // last state, is final.
        let is_string = n == 0 || {
            let chain = (0..n - 1).all(|s| {
                let out = arcs(s);
                fst.final_weight(s as i32) == zero
                    && out.len() == 1
                    && out[0].nextstate() as usize == s + 1
            });
            chain
                && fst.final_weight((n - 1) as i32) != zero
                && arcs(n - 1).is_empty()
                && fst.start().is_none_or(|start| start == 0)
        };
        props |= if is_string { K_STRING } else { K_NOT_STRING };

        let reaches = reachability(fst);
        let cyclic = (0..n).any(|s| reaches[s][s]);
        props |= if cyclic { K_CYCLIC } else { K_ACYCLIC };

        // A weighted cycle is a non-trivially weighted arc whose head can get
        // back to its tail.
        let weighted_cycles =
            all_arcs().any(|(s, a)| !trivial(*a.weight()) && reaches[a.nextstate() as usize][s]);
        props |= if weighted_cycles {
            K_WEIGHTED_CYCLES
        } else {
            K_UNWEIGHTED_CYCLES
        };

        props
    }

    /// The bits both `expected` and `compute_properties` speak for.
    const CHECKED: u64 = K_ACCEPTOR
        | K_NOT_ACCEPTOR
        | K_EPSILONS
        | K_NO_EPSILONS
        | K_I_EPSILONS
        | K_NO_I_EPSILONS
        | K_O_EPSILONS
        | K_NO_O_EPSILONS
        | K_I_DETERMINISTIC
        | K_NON_I_DETERMINISTIC
        | K_O_DETERMINISTIC
        | K_NON_O_DETERMINISTIC
        | K_I_LABEL_SORTED
        | K_NOT_I_LABEL_SORTED
        | K_O_LABEL_SORTED
        | K_NOT_O_LABEL_SORTED
        | K_WEIGHTED
        | K_UNWEIGHTED
        | K_TOP_SORTED
        | K_NOT_TOP_SORTED
        | K_STRING
        | K_NOT_STRING
        | K_CYCLIC
        | K_ACYCLIC
        | K_WEIGHTED_CYCLES
        | K_UNWEIGHTED_CYCLES;

    /// Every property, worked out twice: once by the scan, once from the
    /// definition of the property itself.
    /// An FST with no start state: upstream reads off the end of an empty
    /// component vector here, given any weighted arc.
    ///
    /// The values are upstream's own for this input, minus the out-of-bounds
    /// read.
    #[test]
    fn an_fst_with_no_start_state_is_scanned_without_reading_past_the_components() {
        let mut fst = VectorFst::<StdArc>::new();
        fst.add_state();
        fst.add_state();
        // A weighted arc is what reaches the indexing, and a self-loop makes
        // the FST genuinely cyclic.
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(2.5), 1));
        fst.add_arc(1, StdArc::new(1, 1, TropicalWeight(2.5), 1));
        fst.set_final(1, TropicalWeight::one());
        assert_eq!(fst.start(), None);

        let got = compute_properties(&fst, K_FST_PROPERTIES);
        assert_ne!(got.props & K_WEIGHTED, 0);
        // With nothing reachable, the search reports the vacuous answers, and
        // cycle weighting is left at its default rather than guessed at.
        assert_ne!(got.props & K_ACYCLIC, 0);
        assert_ne!(got.props & K_ACCESSIBLE, 0);
        assert_ne!(got.props & K_UNWEIGHTED_CYCLES, 0);
    }

    #[test]
    fn a_scan_agrees_with_the_definitions_it_implements() {
        let mut rng = Rng(0x5EED);
        for round in 0..400 {
            let n = 1 + rng.next(6);
            let fst = random_fst(&mut rng, n);
            let got = compute_properties(&fst, K_FST_PROPERTIES);
            let want = expected(&fst);
            assert_eq!(
                got.props & CHECKED,
                want,
                "round {round}: {:#x} differs on {:#x}",
                got.props,
                (got.props & CHECKED) ^ want
            );
        }
    }

    /// The scan settles every bit it was asked about, and says so.
    #[test]
    fn everything_asked_for_comes_back_known() {
        let mut rng = Rng(7);
        for _ in 0..100 {
            let n = 1 + rng.next(5);
            let fst = random_fst(&mut rng, n);
            let got = compute_properties(&fst, K_FST_PROPERTIES);
            assert_eq!(got.known, known_properties(got.props));
            assert_eq!(
                got.known & K_TRINARY_PROPERTIES,
                K_TRINARY_PROPERTIES,
                "a full mask should leave nothing unknown"
            );
        }
    }

    /// The properties a mutable FST keeps up to date as it is built must agree
    /// with what a scan finds. They are maintained by an entirely different body
    /// of code, the incremental rules in `properties.rs`, so this is where the
    /// two meet.
    #[test]
    fn the_incrementally_maintained_cache_agrees_with_a_scan() {
        let mut rng = Rng(0xC0FFEE);
        for round in 0..400 {
            let n = 1 + rng.next(6);
            let fst = random_fst(&mut rng, n);
            let stored = fst.properties(K_FST_PROPERTIES, false);
            let computed = compute_properties(&fst, K_FST_PROPERTIES).props;
            assert!(
                compat_properties(stored, computed),
                "round {round}: stored {stored:#x} contradicts computed {computed:#x} \
                 on {:#x}",
                (stored ^ computed) & known_properties(stored) & known_properties(computed)
            );
        }
    }

    /// A narrow mask must not pay for a depth-first search.
    #[test]
    fn a_mask_that_needs_no_search_does_not_settle_search_properties() {
        let mut rng = Rng(11);
        let fst = random_fst(&mut rng, 5);
        let got = compute_properties(&fst, K_ACCEPTOR | K_NOT_ACCEPTOR);
        assert_ne!(got.known & (K_ACCEPTOR | K_NOT_ACCEPTOR), 0);
        assert_eq!(
            got.known & (K_ACYCLIC | K_CYCLIC),
            0,
            "the search ran for a mask that did not need it"
        );
        // Cycle weighting rests on the search too, so it stays unsettled.
        assert_eq!(got.known & (K_WEIGHTED_CYCLES | K_UNWEIGHTED_CYCLES), 0);
    }

    /// An FST that already knows everything asked of it is not scanned.
    struct Liar {
        claims: u64,
    }

    impl Fst<StdArc> for Liar {
        type StateIter<'a> = Empty<i32>;
        type ArcIter<'a> = Empty<StdArc>;

        fn start(&self) -> Option<i32> {
            None
        }

        fn final_weight(&self, _state: i32) -> TropicalWeight {
            TropicalWeight::zero()
        }

        fn num_arcs(&self, _state: i32) -> usize {
            0
        }

        fn num_input_epsilons(&self, _state: i32) -> usize {
            0
        }

        fn num_output_epsilons(&self, _state: i32) -> usize {
            0
        }

        fn num_states_if_known(&self) -> Option<usize> {
            Some(0)
        }

        fn properties(&self, mask: u64, _test: bool) -> u64 {
            self.claims & mask
        }

        fn fst_type(&self) -> &str {
            "liar"
        }

        fn input_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
            None
        }

        fn output_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
            None
        }

        fn states<'a>(&'a self) -> Self::StateIter<'a> {
            std::iter::empty()
        }

        fn arcs<'a>(&'a self, _state: i32) -> Self::ArcIter<'a> {
            std::iter::empty()
        }
    }

    #[test]
    fn stored_properties_are_used_when_they_cover_the_mask() {
        // An empty FST is an acceptor, but this one says it is not, and gets
        // believed, because it claims to know.
        let liar = Liar {
            claims: K_NOT_ACCEPTOR | K_MUTABLE,
        };
        let mask = K_ACCEPTOR | K_NOT_ACCEPTOR;
        assert_eq!(
            compute_or_use_stored_properties(&liar, mask).props & mask,
            K_NOT_ACCEPTOR
        );
        // Asked about something it does not claim, it gets scanned, and the
        // scan finds the truth.
        let wider = mask | K_STRING | K_NOT_STRING;
        assert_eq!(
            compute_or_use_stored_properties(&liar, wider).props & mask,
            K_ACCEPTOR
        );
    }

    /// Verifying turns a stale cache from a wrong answer into a reported
    /// error.
    #[test]
    fn verifying_reports_a_cache_that_contradicts_the_fst() {
        let mask = K_ACCEPTOR | K_NOT_ACCEPTOR;
        let liar = Liar {
            claims: K_NOT_ACCEPTOR,
        };
        assert_eq!(
            test_properties(&liar, mask, false).props & mask,
            K_NOT_ACCEPTOR
        );

        let verified = test_properties(&liar, mask, true);
        assert_ne!(verified.props & K_ERROR, 0, "the lie went unreported");
        assert_eq!(verified.props & mask, K_ACCEPTOR, "the truth is returned");

        // A cache that agrees is not flagged.
        let honest = Liar { claims: K_ACCEPTOR };
        let verified = test_properties(&honest, mask, true);
        assert_eq!(verified.props & K_ERROR, 0);
    }

    #[test]
    fn checking_settles_the_test_mask_only_when_the_check_mask_forces_a_scan() {
        let mut rng = Rng(3);
        let fst = random_fst(&mut rng, 4);

        // The FST knows about acceptor-ness, so nothing is scanned and the
        // extra mask comes back as whatever was stored.
        let stored = fst.properties(K_FST_PROPERTIES, false);
        assert_ne!(known_properties(stored) & K_ACCEPTOR, 0);
        let checked = check_properties(&fst, K_ACCEPTOR, K_STRING | K_NOT_STRING, false);
        assert_eq!(checked, stored & (K_ACCEPTOR | K_STRING | K_NOT_STRING));

        // Asked to check something it does not know, the scan runs and settles
        // both masks.
        let unknown = K_WEIGHTED_CYCLES | K_UNWEIGHTED_CYCLES;
        assert_eq!(known_properties(stored) & unknown, 0);
        let checked = check_properties(&fst, unknown, K_STRING | K_NOT_STRING, false);
        assert_ne!(checked & unknown, 0);
        assert_ne!(checked & (K_STRING | K_NOT_STRING), 0);
    }

    /// What `impl-to-fst.h` is for: asking with `test` settles what the cache
    /// did not know, and remembers the answer.
    #[test]
    fn testing_settles_unknown_properties_and_keeps_them() {
        let mut fst = VectorFst::<StdArc>::new();
        fst.add_state();
        fst.add_state();
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(1, StdArc::new(1, 1, TropicalWeight::one(), 0));
        fst.set_final(1, TropicalWeight::one());

        // Whether the FST has a cycle is not something the incremental rules
        // can keep up to date, so the cache does not know.
        let cycles = K_CYCLIC | K_ACYCLIC;
        assert_eq!(fst.properties(cycles, false), 0);

        assert_eq!(fst.properties(cycles, true), K_CYCLIC);
        // Asking again without testing now answers from the cache.
        assert_eq!(fst.properties(cycles, false), K_CYCLIC);
    }

    /// Only bits that were not settled before are taken, which keeps a cache
    /// that is merely wrong from becoming a cache that is contradictory.
    #[test]
    fn discovering_leaves_settled_bits_alone() {
        let cache = PropertyCache::new(K_NOT_ACCEPTOR);
        cache.discover(
            K_ACCEPTOR | K_CYCLIC,
            K_ACCEPTOR | K_NOT_ACCEPTOR | K_CYCLIC | K_ACYCLIC,
        );

        assert_eq!(
            cache.get() & (K_ACCEPTOR | K_NOT_ACCEPTOR),
            K_NOT_ACCEPTOR,
            "a settled bit was overwritten rather than left wrong"
        );
        assert_eq!(cache.get() & (K_CYCLIC | K_ACYCLIC), K_CYCLIC);
    }

    #[test]
    fn discovering_takes_nothing_outside_the_mask() {
        let cache = PropertyCache::new(0);
        cache.discover(K_ACCEPTOR | K_CYCLIC, K_ACCEPTOR | K_NOT_ACCEPTOR);
        assert_eq!(cache.get(), K_ACCEPTOR);
    }

    /// Setting is how a wrong bit gets corrected; discovering never does.
    #[test]
    fn setting_replaces_what_discovering_would_have_kept() {
        let mut cache = PropertyCache::new(K_NOT_ACCEPTOR);
        cache.set(K_ACCEPTOR);
        assert_eq!(cache.get(), K_ACCEPTOR);
        cache.modify(|props| props | K_CYCLIC);
        assert_eq!(cache.get(), K_ACCEPTOR | K_CYCLIC);
    }
}
