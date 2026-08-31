//! Pushing labels along paths so that the two sides stay in step.
//!
//! Port of OpenFst's `synchronize.h`.
//!
//! The *delay* along a path is how far ahead one side has got: the number of
//! non-epsilon output labels minus the number of input ones. Synchronizing
//! rewrites a transducer into an equivalent one where the delay along any path
//! is either zero or strictly growing, a form other algorithms can rely on.
//!
//! A state of the result is a state of the input plus whatever labels one side
//! is holding back, so the input must have *bounded delay*: every cycle must
//! come back to zero delay, or the labels held back grow without limit and this
//! does not terminate. See Mohri, M. 2003, "Edit-distance of weighted automata:
//! General definitions and algorithms", International Journal of Computer
//! Science 14(6): 957-982.

use rustc_hash::FxHashMap;

use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::fst::{Fst, MutableFst};
use crate::properties::{K_FST_PROPERTIES, synchronize_properties};
use crate::weight::Weight;

/// A residual string, interned so that comparing two is comparing two numbers.
///
/// SICADA-DIVERGE: upstream interns into a hash set and passes `string_view`s
/// that point into it, comparing them by the address of their data. An index
/// says the same thing without the states holding borrows of a table they are
/// stored beside.
type StringId = usize;

/// The interning table. Entry 0 is the empty string.
struct Strings<L> {
    strings: Vec<Vec<L>>,
    ids: FxHashMap<Vec<L>, StringId>,
}

impl<L: ArcLabel> Strings<L> {
    fn new() -> Self {
        let mut table = Self {
            strings: Vec::new(),
            ids: FxHashMap::default(),
        };
        table.intern(Vec::new());
        table
    }

    fn intern(&mut self, string: Vec<L>) -> StringId {
        if let Some(&id) = self.ids.get(&string) {
            return id;
        }
        let id = self.strings.len();
        self.ids.insert(string.clone(), id);
        self.strings.push(string);
        id
    }

    fn get(&self, id: StringId) -> &[L] {
        &self.strings[id]
    }

    /// The first label of `string` followed by `label`, which is `label` itself
    /// when the string is empty. Upstream calls this `Car`.
    fn head(&self, id: StringId, label: L) -> L {
        self.get(id).first().copied().unwrap_or(label)
    }

    /// Everything after the first label of `string` followed by `label`.
    /// Upstream calls this `Cdr`.
    fn tail(&mut self, id: StringId, label: L) -> StringId {
        if self.get(id).is_empty() {
            return self.intern(Vec::new());
        }
        let rest = self.get(id)[1..].to_vec();
        let rest = self.intern(rest);
        self.append(rest, label)
    }

    /// `string` followed by `label`, which appends nothing when `label` is
    /// epsilon. Upstream calls this `Concat`.
    fn append(&mut self, id: StringId, label: L) -> StringId {
        if label == L::epsilon() {
            return id;
        }
        let mut string = self.get(id).to_vec();
        string.push(label);
        self.intern(string)
    }

    /// Whether `string` followed by `label` is empty.
    fn is_empty(&self, id: StringId, label: L) -> bool {
        self.get(id).is_empty() && label == L::epsilon()
    }

    fn len(&self, id: StringId) -> usize {
        self.get(id).len()
    }
}

/// A state of the result: a state of the input, plus what each side is holding
/// back.
///
/// The input state is `None` for the states that only exist to emit labels that
/// were held back at the end of a path.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Element<S> {
    state: Option<S>,
    istring: StringId,
    ostring: StringId,
}

/// Rewrites `ifst` into `ofst` so that the delay along any path is zero or
/// strictly growing.
///
/// The input must have bounded delay, meaning every cycle must return to zero
/// delay, or this does not terminate. That is upstream's condition too; it is a
/// property of the input rather than something the algorithm can check
/// cheaply.
///
/// SICADA-DIVERGE: upstream builds a delayed `SynchronizeFst` and copies it,
/// with the cache told to keep only the last state. Building the result
/// directly does the same work without the cache in the middle. The delayed
/// form belongs with the other delayed wrappers, which are still outstanding.
pub fn synchronize<A: Arc, F1: Fst<A>, F2: MutableFst<A>>(ifst: &F1, ofst: &mut F2) {
    ofst.delete_all_states();
    ofst.set_input_symbols(ifst.input_symbols());
    ofst.set_output_symbols(ifst.output_symbols());

    let iprops = ifst.properties(K_FST_PROPERTIES, false);
    let Some(istart) = ifst.start() else {
        ofst.set_properties(synchronize_properties(iprops), K_FST_PROPERTIES);
        return;
    };

    let mut strings = Strings::<A::Label>::new();
    let empty = strings.intern(Vec::new());
    let mut elements: Vec<Element<A::StateId>> = Vec::new();
    let mut ids: FxHashMap<Element<A::StateId>, A::StateId> = FxHashMap::default();

    let mut find_state = |element: Element<A::StateId>,
                          elements: &mut Vec<Element<A::StateId>>,
                          ofst: &mut F2|
     -> A::StateId {
        *ids.entry(element).or_insert_with(|| {
            elements.push(element);
            ofst.add_state()
        })
    };

    let start = find_state(
        Element {
            state: Some(istart),
            istring: empty,
            ostring: empty,
        },
        &mut elements,
        ofst,
    );
    ofst.set_start(start);

    let zero = A::Weight::zero();
    let epsilon = A::Label::epsilon();
    let mut next = 0;
    while next < elements.len() {
        let element = elements[next];
        let state = A::StateId::from_usize(next);
        next += 1;

        if let Some(input_state) = element.state {
            for arc in ifst.arcs(input_state) {
                // Both sides have something to emit, so one label comes off
                // each and the rest is carried forward.
                let target = if !strings.is_empty(element.istring, arc.ilabel())
                    && !strings.is_empty(element.ostring, arc.olabel())
                {
                    let ilabel = strings.head(element.istring, arc.ilabel());
                    let olabel = strings.head(element.ostring, arc.olabel());
                    let istring = strings.tail(element.istring, arc.ilabel());
                    let ostring = strings.tail(element.ostring, arc.olabel());
                    let next_state = find_state(
                        Element {
                            state: Some(arc.nextstate()),
                            istring,
                            ostring,
                        },
                        &mut elements,
                        ofst,
                    );
                    A::new(ilabel, olabel, arc.weight().clone(), next_state)
                } else {
                    // One side has nothing, so neither emits and both labels
                    // are held back.
                    let istring = strings.append(element.istring, arc.ilabel());
                    let ostring = strings.append(element.ostring, arc.olabel());
                    let next_state = find_state(
                        Element {
                            state: Some(arc.nextstate()),
                            istring,
                            ostring,
                        },
                        &mut elements,
                        ofst,
                    );
                    A::new(epsilon, epsilon, arc.weight().clone(), next_state)
                };
                ofst.add_arc(state, target);
            }
        }

        let weight = match element.state {
            Some(input_state) => ifst.final_weight(input_state),
            None => A::Weight::one(),
        };
        let held = strings.len(element.istring) + strings.len(element.ostring);
        if weight != zero && held > 0 {
            // The path ends but labels are still held back, so they are emitted
            // through states that have no input state of their own.
            let ilabel = strings.head(element.istring, epsilon);
            let olabel = strings.head(element.ostring, epsilon);
            let istring = strings.tail(element.istring, epsilon);
            let ostring = strings.tail(element.ostring, epsilon);
            let next_state = find_state(
                Element {
                    state: None,
                    istring,
                    ostring,
                },
                &mut elements,
                ofst,
            );
            ofst.add_arc(state, A::new(ilabel, olabel, weight.clone(), next_state));
        }
        if weight != zero && held == 0 {
            ofst.set_final(state, weight);
        }
    }

    ofst.set_properties(synchronize_properties(iprops), K_FST_PROPERTIES);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{string_weights, visible_paths};
    use crate::arc::StdArc;
    use crate::fst::ExpandedFst as _;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::weights::float_weight::TropicalWeight;

    /// The delay along each accepting path, at every point along it.
    ///
    /// Synchronized means: never negative-then-positive, never returning to a
    /// smaller non-zero value. The delay is zero or strictly growing.
    fn delays<F: Fst<StdArc>>(fst: &F, max_len: usize) -> Vec<Vec<i32>> {
        fn walk<F: Fst<StdArc>>(
            fst: &F,
            state: i32,
            delay: i32,
            trace: &mut Vec<i32>,
            left: usize,
            out: &mut Vec<Vec<i32>>,
        ) {
            if fst.final_weight(state) != TropicalWeight::zero() {
                out.push(trace.clone());
            }
            if left == 0 {
                return;
            }
            for arc in fst.arcs(state) {
                let next = delay + i32::from(arc.olabel() != 0) - i32::from(arc.ilabel() != 0);
                trace.push(next);
                walk(fst, arc.nextstate(), next, trace, left - 1, out);
                trace.pop();
            }
        }

        let mut out = Vec::new();
        if let Some(start) = fst.start() {
            walk(fst, start, 0, &mut Vec::new(), max_len, &mut out);
        }
        out
    }

    /// 0 → 1 on a:ε, 1 → 2 on ε:x, 2 final. The output label lags one behind.
    fn lagging() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 0, TropicalWeight(1.0), 1));
        fst.add_arc(1, StdArc::new(0, 2, TropicalWeight(2.0), 2));
        fst.set_final(2, TropicalWeight(3.0));
        fst
    }

    /// The input has `a` on one arc and `x` on the next; the result carries
    /// them on one arc together.
    ///
    /// It cannot be the *first* arc: the output label is not known until the
    /// second arc of the input has been read, so the first arc of the result
    /// consumes nothing and holds the input label back. That delay of one is
    /// inherent, not a shortcoming.
    #[test]
    fn synchronizing_pairs_the_labels_up() {
        let ifst = lagging();
        let mut ofst = StdVectorFst::new();
        synchronize(&ifst, &mut ofst);

        let labelled: Vec<(i32, i32)> = (0..ofst.num_states() as i32)
            .flat_map(|s| {
                ofst.arcs(s)
                    .map(|a| (a.ilabel(), a.olabel()))
                    .collect::<Vec<_>>()
            })
            .filter(|&(i, o)| i != 0 || o != 0)
            .collect();
        assert_eq!(
            labelled,
            vec![(1, 2)],
            "the two labels travel on one arc, and there is only the one"
        );
    }

    /// The point of the operation: what the transducer maps does not change.
    #[test]
    fn synchronizing_preserves_what_the_transducer_maps() {
        let ifst = lagging();
        let mut ofst = StdVectorFst::new();
        synchronize(&ifst, &mut ofst);

        assert_eq!(
            string_weights(visible_paths(&ofst, 8)),
            string_weights(visible_paths(&ifst, 8))
        );
    }

    /// After synchronizing, the delay along every path is zero or growing.
    #[test]
    fn the_delay_never_comes_back_down() {
        let mut fst = StdVectorFst::new();
        for _ in 0..4 {
            fst.add_state();
        }
        fst.set_start(0);
        // Two output labels get ahead, then two input labels catch up.
        fst.add_arc(0, StdArc::new(0, 1, TropicalWeight::one(), 1));
        fst.add_arc(1, StdArc::new(0, 2, TropicalWeight::one(), 2));
        fst.add_arc(2, StdArc::new(3, 0, TropicalWeight::one(), 3));
        fst.set_final(3, TropicalWeight::one());

        // The input dips back down: +1, +2, +1.
        let before = delays(&fst, 8);
        assert!(
            before
                .iter()
                .any(|trace| trace.windows(2).any(|w| w[1] < w[0] && w[1] != 0)),
            "the input should not already be synchronized"
        );

        let mut ofst = StdVectorFst::new();
        synchronize(&fst, &mut ofst);
        for trace in delays(&ofst, 8) {
            for window in trace.windows(2) {
                assert!(
                    window[1] == 0 || window[1] > window[0],
                    "delay went from {} to {} in {trace:?}",
                    window[0],
                    window[1]
                );
            }
        }
        assert_eq!(
            string_weights(visible_paths(&ofst, 8)),
            string_weights(visible_paths(&fst, 8))
        );
    }

    /// An FST already in step is reproduced, not rearranged.
    #[test]
    fn an_already_synchronized_fst_keeps_its_paths() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(1.0), 1));
        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight(2.0), 2));
        fst.set_final(2, TropicalWeight::one());

        let mut ofst = StdVectorFst::new();
        synchronize(&fst, &mut ofst);
        assert_eq!(
            string_weights(visible_paths(&ofst, 8)),
            string_weights(visible_paths(&fst, 8))
        );
        assert_eq!(ofst.num_states(), 3);
    }

    /// A cycle with zero delay is bounded, so this terminates on it.
    #[test]
    fn a_zero_delay_cycle_terminates() {
        let mut fst = StdVectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 0, TropicalWeight::one(), 1));
        fst.add_arc(1, StdArc::new(0, 2, TropicalWeight::one(), 0));
        fst.set_final(0, TropicalWeight::one());

        let mut ofst = StdVectorFst::new();
        synchronize(&fst, &mut ofst);
        assert!(ofst.num_states() > 0);
        assert_eq!(
            string_weights(visible_paths(&ofst, 6)),
            string_weights(visible_paths(&fst, 6))
        );
    }

    #[test]
    fn an_fst_with_no_start_state_synchronizes_to_nothing() {
        let ifst = StdVectorFst::new();
        let mut ofst = StdVectorFst::new();
        ofst.add_state();
        synchronize(&ifst, &mut ofst);
        assert_eq!(ofst.num_states(), 0);
    }
}
