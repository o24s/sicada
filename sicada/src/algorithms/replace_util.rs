//! Working on a recursive transition network without expanding all of it.
//!
//! Port of OpenFst's `replace-util.h`. Expanding an RTN in full is often the
//! wrong thing to do: it may not terminate, and where it does it may be far
//! larger than the network. This holds the label/FST pairs and answers questions
//! about their call graph, and expands parts of the network chosen by rule,
//! leaving the rest as calls.

use hashbrown::{HashMap, HashSet};

use crate::algorithms::cc_visitors::SccVisitor;
use crate::algorithms::connect::connect;
use crate::algorithms::dfs_visit::dfs_visit_any;
use crate::algorithms::replace::{ReplaceOptions, called_fst, replace};
use crate::algorithms::topsort::TopOrderVisitor;
use crate::arc::{Arc, ArcStateId};
use crate::data_structures::bit_set::GrowableBitSet;
use crate::error::OpenFstError;
use crate::fst::{ExpandedFst, Fst, MutableFst};
use crate::fsts::vector_fst::VectorFst;
use crate::properties::{K_ACCESSIBLE, K_CO_ACCESSIBLE, K_CYCLIC};
use crate::weight::Weight;

/// Every non-terminal on a path is the first label on that path, in every FST
/// of this component, as a left-linear grammar rule gives.
pub const REPLACE_SCC_LEFT_LINEAR: u8 = 0x01;
/// Every non-terminal on a path is the last label on that path, in every FST of
/// this component, as a right-linear rule gives.
pub const REPLACE_SCC_RIGHT_LINEAR: u8 = 0x02;
/// The component has more than one FST in it, or an FST that calls itself.
pub const REPLACE_SCC_NON_TRIVIAL: u8 = 0x04;

/// How big an FST is and how it is referred to, for deciding whether expanding
/// it is worth it.
#[derive(Debug, Clone, Default)]
struct ReplaceStats {
    /// States in the FST.
    nstates: usize,
    /// Final states in it.
    nfinal: usize,
    /// Arcs in it.
    narcs: usize,
    /// Non-terminals it calls, counted with multiplicity.
    nnonterms: usize,
    /// Calls to it from anywhere, counted with multiplicity.
    nref: usize,
    /// Calls to it from each FST.
    inref: HashMap<usize, usize>,
    /// Calls from it to each FST.
    outref: HashMap<usize, usize>,
}

/// The call graph and what was worked out from it.
struct Dependencies<A: Arc> {
    /// One state per FST, one arc per call.
    graph: VectorFst<A>,
    /// The strongly connected component of each FST.
    scc: Vec<usize>,
    /// Whether each FST can be reached from the root.
    access: GrowableBitSet,
    /// The properties of the call graph, notably whether it is cyclic.
    props: u64,
    /// Per-FST statistics, when they were asked for.
    stats: Option<Vec<ReplaceStats>>,
    /// The per-component properties, once computed.
    scc_props: Option<Vec<u8>>,
}

/// A recursive transition network, held as label/FST pairs.
///
/// SICADA-DIVERGE: upstream takes either `MutableFst*` (taking ownership) or
/// `const Fst*` (copying), keeps both arrays in step by hand, and converts the
/// second kind into `VectorFst` the moment anything mutates, via a
/// `CheckMutableFsts` call at the top of every such method. Since every mutating
/// method needs the conversion anyway, the FSTs are owned `VectorFst`s from the
/// start and the duality goes away.
pub struct ReplaceUtil<A: Arc> {
    /// The FSTs by index; `None` once [`connect`](Self::connect) has dropped
    /// one as unreachable.
    fsts: Vec<Option<VectorFst<A>>>,
    /// The non-terminal naming each FST.
    labels: Vec<A::Label>,
    /// Which FST each non-terminal names.
    index_of: HashMap<A::Label, usize>,
    /// The FST to start from.
    root: usize,
    /// How call and return arcs are labelled when expanding.
    opts: ReplaceOptions<A::Label>,
    /// The call graph, recomputed whenever the network changes.
    deps: Option<Dependencies<A>>,
}

impl<A: Arc> ReplaceUtil<A> {
    /// Takes the label/FST pairs of a network.
    ///
    /// SICADA-DIVERGE: upstream looks the root label up with `operator[]`,
    /// which **inserts** a zero entry when it is missing. After that the label
    /// looks like a non-terminal naming the null FST slot, and every arc
    /// carrying it is deleted without a word. Here a root naming no FST is an
    /// error.
    pub fn new(
        pairs: Vec<(A::Label, VectorFst<A>)>,
        opts: ReplaceOptions<A::Label>,
    ) -> Result<Self, OpenFstError> {
        let mut index_of = HashMap::with_capacity(pairs.len());
        let mut labels = Vec::with_capacity(pairs.len());
        let mut fsts = Vec::with_capacity(pairs.len());
        for (index, (label, fst)) in pairs.into_iter().enumerate() {
            index_of.insert(label, index);
            labels.push(label);
            fsts.push(Some(fst));
        }
        let Some(&root) = index_of.get(&opts.root) else {
            return Err(OpenFstError::InvalidOperation(format!(
                "ReplaceUtil: no FST for the root label {}",
                opts.root
            )));
        };
        Ok(Self {
            fsts,
            labels,
            index_of,
            root,
            opts,
            deps: None,
        })
    }

    /// The FST each non-terminal names, in the order they were given, leaving
    /// out any that [`connect`](Self::connect) dropped.
    pub fn fst_pairs(&self) -> Vec<(A::Label, &VectorFst<A>)> {
        self.labels
            .iter()
            .zip(&self.fsts)
            .filter_map(|(label, fst)| fst.as_ref().map(|fst| (*label, fst)))
            .collect()
    }

    /// As [`fst_pairs`](Self::fst_pairs), handing the FSTs over.
    pub fn into_fst_pairs(self) -> Vec<(A::Label, VectorFst<A>)> {
        self.labels
            .into_iter()
            .zip(self.fsts)
            .filter_map(|(label, fst)| fst.map(|fst| (label, fst)))
            .collect()
    }

    /// The FST a non-terminal names.
    pub fn fst(&self, label: A::Label) -> Option<&VectorFst<A>> {
        self.fsts.get(*self.index_of.get(&label)?)?.as_ref()
    }

    /// Whether the calls recurse, which makes a network unexpandable.
    pub fn cyclic_dependencies(&mut self) -> bool {
        self.dependencies(false).props & K_CYCLIC != 0
    }

    /// The strongly connected component of the call graph a non-terminal
    /// belongs to.
    pub fn scc(&mut self, label: A::Label) -> Option<usize> {
        let index = *self.index_of.get(&label)?;
        Some(self.dependencies(false).scc[index])
    }

    /// What is known about a component of the call graph.
    ///
    /// A component that is [`REPLACE_SCC_LEFT_LINEAR`] or
    /// [`REPLACE_SCC_RIGHT_LINEAR`] describes a regular language even though its
    /// calls recurse, but not one that replacement can produce, since the call
    /// stack still deepens at every call. Recognizing it takes a pushdown
    /// transducer.
    pub fn scc_properties(&mut self, scc: usize) -> u8 {
        self.compute_scc_properties();
        self.deps
            .as_ref()
            .and_then(|deps| deps.scc_props.as_ref())
            .and_then(|props| props.get(scc).copied())
            .unwrap_or(0)
    }

    /// Whether every FST is reachable from the root and has no useless states.
    pub fn connected(&mut self) -> bool {
        let props = K_ACCESSIBLE | K_CO_ACCESSIBLE;
        let useful: Vec<bool> = self
            .fsts
            .iter()
            .map(|fst| {
                fst.as_ref()
                    .is_none_or(|fst| fst.properties(props, true) == props)
            })
            .collect();
        let present: Vec<bool> = self.fsts.iter().map(Option::is_some).collect();
        let deps = self.dependencies(false);
        for (index, ok) in useful.into_iter().enumerate() {
            if present[index] && (!ok || !deps.access.contains(index)) {
                return false;
            }
        }
        true
    }

    /// Removes the states, arcs and whole FSTs that no path through the network
    /// can use.
    pub fn connect(&mut self) {
        let props = K_ACCESSIBLE | K_CO_ACCESSIBLE;
        for fst in self.fsts.iter_mut().flatten() {
            if fst.properties(props, false) != props {
                connect(fst);
            }
        }
        let count = self.fsts.len();
        let unreachable: Vec<usize> = {
            let deps = self.dependencies(false);
            (0..count)
                .filter(|index| !deps.access.contains(*index))
                .collect()
        };
        for index in unreachable {
            self.fsts[index] = None;
        }
        self.deps = None;
    }

    /// Expands the named non-terminals wherever they are called, leaving the
    /// rest as calls.
    ///
    /// SICADA-DIVERGE: upstream logs a warning and returns having done nothing
    /// when the restricted call graph turns out to be cyclic, so the caller
    /// cannot tell the difference between "expanded" and "gave up". It is an
    /// error here.
    pub fn replace_labels(&mut self, labels: &[A::Label]) -> Result<(), OpenFstError> {
        // The root cannot be replaced: there would be nothing left to hold the
        // result.
        let wanted: HashSet<usize> = labels
            .iter()
            .filter(|label| **label != self.opts.root)
            .filter_map(|label| self.index_of.get(label).copied())
            .collect();

        // The call graph restricted to the calls that are to be expanded.
        let restricted = {
            let deps = self.dependencies(false);
            let mut restricted = VectorFst::<A>::new();
            for _ in 0..deps.graph.num_states() {
                restricted.add_state();
            }
            if let Some(start) = deps.graph.start() {
                restricted.set_start(start);
            }
            for state in deps.graph.states() {
                for arc in deps.graph.arcs(state) {
                    if wanted.contains(&arc.nextstate().as_usize()) {
                        restricted.add_arc(state, arc);
                    }
                }
            }
            restricted
        };

        let Some(order) = top_order(&restricted) else {
            return Err(OpenFstError::InvalidOperation(
                "ReplaceUtil: the non-terminals to expand call each other, so the expansion has \
                 no end"
                    .into(),
            ));
        };

        // Expanding bottom-up means an FST is expanded only once its callees
        // already have been, so one pass suffices.
        for &index in order.iter().rev() {
            let callees: Vec<usize> = {
                let mut seen = HashSet::new();
                restricted
                    .arcs(A::StateId::from_usize(index))
                    .map(|arc| arc.nextstate().as_usize())
                    .filter(|callee| seen.insert(*callee))
                    .collect()
            };
            if callees.is_empty() {
                continue;
            }
            let mut network: Vec<(A::Label, &VectorFst<A>)> = Vec::with_capacity(callees.len() + 1);
            for callee in &callees {
                let Some(fst) = self.fsts[*callee].as_ref() else {
                    continue;
                };
                network.push((self.labels[*callee], fst));
            }
            let Some(fst) = self.fsts[index].as_ref() else {
                continue;
            };
            network.push((self.labels[index], fst));

            let mut opts = self.opts.clone();
            opts.root = self.labels[index];
            let mut expanded = VectorFst::<A>::new();
            replace(&network, &mut expanded, &opts)?;
            self.fsts[index] = Some(expanded);
        }
        self.deps = None;
        Ok(())
    }

    /// Expands every non-terminal whose FST is at most this big, so that the
    /// small rules disappear into their callers.
    ///
    /// Sizes are counted as they will be *after* the expansions already chosen,
    /// so a rule that is small now but grows once its own callees are inlined
    /// is judged on what it will become.
    pub fn replace_by_size(
        &mut self,
        nstates: usize,
        narcs: usize,
        nnonterms: usize,
    ) -> Result<(), OpenFstError> {
        self.dependencies(true);
        let Some(order) = self.dependency_order() else {
            return Err(OpenFstError::InvalidOperation(
                "ReplaceUtil: the calls recurse, so nothing can be expanded".into(),
            ));
        };
        let mut labels = Vec::new();
        for &index in order.iter().rev() {
            let small = {
                let stats = self.stats();
                stats[index].nstates <= nstates
                    && stats[index].narcs <= narcs
                    && stats[index].nnonterms <= nnonterms
            };
            if small {
                labels.push(self.labels[index]);
                self.update_stats(index);
            }
        }
        self.replace_labels(&labels)
    }

    /// Expands the rules that are a single arc, which are pure indirection.
    pub fn replace_trivial(&mut self) -> Result<(), OpenFstError> {
        self.replace_by_size(2, 1, 1)
    }

    /// Expands every non-terminal called at most this many times, so that a
    /// rule used in few places is not worth keeping as a rule.
    pub fn replace_by_instances(&mut self, ninstances: usize) -> Result<(), OpenFstError> {
        self.dependencies(true);
        let Some(order) = self.dependency_order() else {
            return Err(OpenFstError::InvalidOperation(
                "ReplaceUtil: the calls recurse, so nothing can be expanded".into(),
            ));
        };
        let mut labels = Vec::new();
        // Top-down here, unlike `replace_by_size`: inlining a rule moves its
        // calls into its callers, so a callee's count is only settled once its
        // callers have been dealt with.
        for &index in order.iter() {
            if self.stats()[index].nref <= ninstances {
                labels.push(self.labels[index]);
                self.update_stats(index);
            }
        }
        self.replace_labels(&labels)
    }

    /// Expands the rules called from only one place.
    pub fn replace_unique(&mut self) -> Result<(), OpenFstError> {
        self.replace_by_instances(1)
    }

    // --- Internals.

    /// The call graph, built if it is not already there.
    fn dependencies(&mut self, stats: bool) -> &Dependencies<A> {
        let stale = match &self.deps {
            None => true,
            Some(deps) => stats && deps.stats.is_none(),
        };
        if stale {
            self.deps = Some(self.build_dependencies(stats));
        }
        self.deps.as_ref().expect("just built")
    }

    fn stats(&self) -> &[ReplaceStats] {
        self.deps
            .as_ref()
            .and_then(|deps| deps.stats.as_deref())
            .expect("statistics were asked for")
    }

    fn build_dependencies(&self, want_stats: bool) -> Dependencies<A> {
        let mut graph = VectorFst::<A>::new();
        let mut stats = want_stats.then(|| vec![ReplaceStats::default(); self.fsts.len()]);
        for _ in 0..self.fsts.len() {
            let state = graph.add_state();
            // Every FST is a place the walk may stop, so that reachability in
            // this graph is reachability in the network.
            graph.set_final(state, A::Weight::one());
        }
        graph.set_start(A::StateId::from_usize(self.root));

        let zero = A::Weight::zero();
        for (index, fst) in self.fsts.iter().enumerate() {
            let Some(fst) = fst else { continue };
            for state in fst.states() {
                if let Some(stats) = stats.as_mut() {
                    stats[index].nstates += 1;
                    if fst.final_weight(state) != zero {
                        stats[index].nfinal += 1;
                    }
                }
                for arc in fst.arcs(state) {
                    if let Some(stats) = stats.as_mut() {
                        stats[index].narcs += 1;
                    }
                    let Some(callee) = called_fst(&self.index_of, arc.olabel()) else {
                        continue;
                    };
                    graph.add_arc(
                        A::StateId::from_usize(index),
                        A::new(
                            arc.olabel(),
                            arc.olabel(),
                            A::Weight::one(),
                            A::StateId::from_usize(callee),
                        ),
                    );
                    if let Some(stats) = stats.as_mut() {
                        stats[index].nnonterms += 1;
                        stats[callee].nref += 1;
                        *stats[callee].inref.entry(index).or_default() += 1;
                        *stats[index].outref.entry(callee).or_default() += 1;
                    }
                }
            }
        }

        let mut scc_ids: Vec<A::StateId> = Vec::new();
        let mut access = GrowableBitSet::new();
        let mut props = 0;
        {
            let mut visitor = SccVisitor::new(
                &graph,
                Some(&mut scc_ids),
                Some(&mut access),
                None,
                &mut props,
            );
            dfs_visit_any(&graph, &mut visitor);
        }
        let scc = scc_ids.iter().map(|id| id.as_usize()).collect();

        Dependencies {
            graph,
            scc,
            access,
            props,
            stats,
            scc_props: None,
        }
    }

    /// The FSTs in topological order of their calls, or `None` if the calls
    /// recurse.
    fn dependency_order(&mut self) -> Option<Vec<usize>> {
        top_order(&self.dependencies(false).graph)
    }

    /// Folds the effect of expanding FST `j` into the statistics of the FSTs
    /// around it, so that a later decision sees the sizes as they will be.
    fn update_stats(&mut self, j: usize) {
        if j == self.root {
            return; // The root is never replaced.
        }
        let Some(deps) = self.deps.as_mut() else {
            return;
        };
        let Some(stats) = deps.stats.as_mut() else {
            return;
        };
        let target = stats[j].clone();

        // Each caller absorbs a copy of `j` per call: its states and arcs, plus
        // one arc for the call that is going away, and its non-terminals in
        // place of the one call.
        for (&caller, &count) in &target.inref {
            stats[caller].nstates += target.nstates * count;
            stats[caller].narcs += (target.narcs + 1) * count;
            stats[caller].nnonterms += target.nnonterms.saturating_sub(1) * count;
            stats[caller].outref.remove(&j);
            for (&callee, &out) in &target.outref {
                *stats[caller].outref.entry(callee).or_default() += count * out;
            }
        }
        // What `j` called is now called by `j`'s callers instead.
        for (&callee, &out) in &target.outref {
            stats[callee].nref -= out;
            stats[callee].inref.remove(&j);
            for (&caller, &count) in &target.inref {
                *stats[callee].inref.entry(caller).or_default() += count * out;
                stats[callee].nref += count * out;
            }
        }
    }

    /// Works out the per-component properties, once.
    ///
    /// SICADA-BUGFIX: upstream's self-loop pass indexes `depsccprops_`, an
    /// array over *components*, with a *state* of the dependency graph, which
    /// is an FST id. The rest of the same function indexes it correctly with
    /// `depscc_[i]`, so a network where an FST's id and its component number
    /// differ has `kReplaceSCCNonTrivial` recorded against the wrong component.
    fn compute_scc_properties(&mut self) {
        if self
            .deps
            .as_ref()
            .is_some_and(|deps| deps.scc_props.is_some())
        {
            return;
        }
        self.dependencies(false);
        let deps = self.deps.as_ref().expect("just built");
        if deps.scc.is_empty() {
            return;
        }
        let nscc = deps.scc.iter().copied().max().map_or(0, |max| max + 1);
        let mut scc_props = vec![REPLACE_SCC_LEFT_LINEAR | REPLACE_SCC_RIGHT_LINEAR; nscc];

        if deps.props & K_CYCLIC == 0 {
            // Without recursion every component is a single FST that does not
            // call itself, and both linearity claims hold vacuously.
            self.deps.as_mut().expect("just built").scc_props = Some(scc_props);
            return;
        }

        // An FST that calls itself makes its component non-trivial.
        for state in deps.graph.states() {
            for arc in deps.graph.arcs(state) {
                if arc.nextstate() == state {
                    scc_props[deps.scc[state.as_usize()]] |= REPLACE_SCC_NON_TRIVIAL;
                }
            }
        }

        let zero = A::Weight::zero();
        let mut seen = vec![false; nscc];
        for (index, fst) in self.fsts.iter().enumerate() {
            let Some(fst) = fst else { continue };
            let scc = deps.scc[index];
            if seen[scc] {
                // A second FST in the component: more than one state.
                scc_props[scc] |= REPLACE_SCC_NON_TRIVIAL;
            }
            seen[scc] = true;

            // The components of this FST's own states, to tell a non-terminal
            // on a cycle from one that is not.
            let mut inner: Vec<A::StateId> = Vec::new();
            let mut props = 0;
            {
                let mut visitor = SccVisitor::new(fst, Some(&mut inner), None, None, &mut props);
                dfs_visit_any(fst, &mut visitor);
            }

            for state in fst.states() {
                for arc in fst.arcs(state) {
                    let Some(callee) = called_fst(&self.index_of, arc.olabel()) else {
                        continue; // A terminal.
                    };
                    if deps.scc[callee] != scc {
                        continue; // A call out of this component.
                    }
                    let on_a_cycle = inner
                        .get(state.as_usize())
                        .zip(inner.get(arc.nextstate().as_usize()))
                        .is_some_and(|(from, to)| from == to);
                    // Left linear only if every non-terminal leaves the start.
                    if Some(state) != fst.start() || on_a_cycle {
                        scc_props[scc] &= !REPLACE_SCC_LEFT_LINEAR;
                    }
                    // Right linear only if every non-terminal arrives at a
                    // final state.
                    if fst.final_weight(arc.nextstate()) == zero || on_a_cycle {
                        scc_props[scc] &= !REPLACE_SCC_RIGHT_LINEAR;
                    }
                }
            }
        }

        self.deps.as_mut().expect("just built").scc_props = Some(scc_props);
    }
}

/// The states of `fst` in topological order, or `None` if it is cyclic.
///
/// SICADA-DIVERGE: upstream walks the order with `for (Label o = size - 1; o >=
/// 0; --o)`. `Label` is whatever the arc type says, and for an unsigned label
/// type `o >= 0` never fails, so the loop runs off the front of the vector.
/// Nothing here counts down through a signed index.
fn top_order<A: Arc>(fst: &VectorFst<A>) -> Option<Vec<usize>> {
    let mut visitor = TopOrderVisitor::<A>::new();
    dfs_visit_any(fst, &mut visitor);
    // `order[state]` is the position the state takes; the inverse is what a
    // caller walking in order wants.
    let order = visitor.order()?;
    let mut inverse = vec![0usize; order.len()];
    for (state, position) in order.iter().enumerate() {
        inverse[position.as_usize()] = state;
    }
    Some(inverse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_support::{paths, sorted};
    use crate::arc::StdArc;
    use crate::properties::K_FST_PROPERTIES;
    use crate::weights::float_weight::TropicalWeight;

    type Vf = VectorFst<StdArc>;

    /// A linear acceptor over `labels`, final at weight one.
    fn chain(labels: &[i32]) -> Vf {
        let mut fst = Vf::new();
        let mut state = fst.add_state();
        fst.set_start(state);
        for label in labels {
            let next = fst.add_state();
            fst.add_arc(
                state,
                StdArc::new(*label, *label, TropicalWeight::one(), next),
            );
            state = next;
        }
        fst.set_final(state, TropicalWeight::one());
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    /// The strings an FST spells, with epsilons dropped so that call and return
    /// arcs do not show.
    fn strings(fst: &Vf) -> Vec<Vec<i32>> {
        sorted(paths(fst, 16))
            .into_iter()
            .map(|(ilabels, _, _)| ilabels.into_iter().filter(|l| *l != 0).collect())
            .collect()
    }

    fn util(pairs: Vec<(i32, Vf)>, root: i32) -> ReplaceUtil<StdArc> {
        ReplaceUtil::new(pairs, ReplaceOptions::epsilon_calls(root)).unwrap()
    }

    /// root: 1 <-1> 4, where -1 spells 2 and calls nothing.
    fn simple() -> Vec<(i32, Vf)> {
        let mut root = Vf::new();
        for _ in 0..4 {
            root.add_state();
        }
        root.set_start(0);
        root.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        root.add_arc(1, StdArc::new(-1, -1, TropicalWeight::one(), 2));
        root.add_arc(2, StdArc::new(4, 4, TropicalWeight::one(), 3));
        root.set_final(3, TropicalWeight::one());
        root.properties(K_FST_PROPERTIES, true);
        vec![(0, root), (-1, chain(&[2]))]
    }

    #[test]
    fn a_root_naming_no_fst_is_refused() {
        let err = ReplaceUtil::<StdArc>::new(simple(), ReplaceOptions::epsilon_calls(7))
            .err()
            .expect("no FST is named 7");
        assert!(format!("{err}").contains("root label"), "{err}");
    }

    /// Expanding a non-terminal folds the FST it names into its caller, and the
    /// non-terminal is gone from the result.
    #[test]
    fn expanding_a_label_folds_it_into_its_caller() {
        let mut util = util(simple(), 0);
        util.replace_labels(&[-1]).unwrap();

        let root = util.fst(0).unwrap();
        assert_eq!(strings(root), vec![vec![1, 2, 4]]);
        assert!(
            root.states()
                .all(|s| root.arcs(s).all(|arc| arc.olabel() != -1)),
            "no call to -1 is left"
        );
        // The rule itself is still there, untouched.
        assert_eq!(strings(util.fst(-1).unwrap()), vec![vec![2]]);
    }

    /// The root is never expanded, since there would be nothing left to hold
    /// the result.
    #[test]
    fn the_root_is_never_expanded() {
        let mut util = util(simple(), 0);
        util.replace_labels(&[0]).unwrap();
        // The call to -1 survives, because -1 was not asked for.
        let root = util.fst(0).unwrap();
        assert!(
            root.states()
                .any(|s| root.arcs(s).any(|a| a.olabel() == -1))
        );
    }

    /// Expansion is bottom-up, so a rule that calls another comes out fully
    /// expanded in one pass.
    #[test]
    fn expansion_goes_bottom_up() {
        let mut root = Vf::new();
        for _ in 0..2 {
            root.add_state();
        }
        root.set_start(0);
        root.add_arc(0, StdArc::new(-1, -1, TropicalWeight::one(), 1));
        root.set_final(1, TropicalWeight::one());
        root.properties(K_FST_PROPERTIES, true);

        // -1 spells 5 then calls -2.
        let mut middle = Vf::new();
        for _ in 0..3 {
            middle.add_state();
        }
        middle.set_start(0);
        middle.add_arc(0, StdArc::new(5, 5, TropicalWeight::one(), 1));
        middle.add_arc(1, StdArc::new(-2, -2, TropicalWeight::one(), 2));
        middle.set_final(2, TropicalWeight::one());
        middle.properties(K_FST_PROPERTIES, true);

        let mut util = util(vec![(0, root), (-1, middle), (-2, chain(&[6]))], 0);
        util.replace_labels(&[-1, -2]).unwrap();
        assert_eq!(strings(util.fst(0).unwrap()), vec![vec![5, 6]]);
    }

    /// A call graph whose restriction to the chosen labels is cyclic has no
    /// finite expansion.
    #[test]
    fn expanding_labels_that_call_each_other_is_refused() {
        let mut a = Vf::new();
        for _ in 0..2 {
            a.add_state();
        }
        a.set_start(0);
        a.add_arc(0, StdArc::new(-2, -2, TropicalWeight::one(), 1));
        a.set_final(1, TropicalWeight::one());

        let mut b = Vf::new();
        for _ in 0..2 {
            b.add_state();
        }
        b.set_start(0);
        b.add_arc(0, StdArc::new(-1, -1, TropicalWeight::one(), 1));
        b.set_final(1, TropicalWeight::one());

        let mut root = Vf::new();
        for _ in 0..2 {
            root.add_state();
        }
        root.set_start(0);
        root.add_arc(0, StdArc::new(-1, -1, TropicalWeight::one(), 1));
        root.set_final(1, TropicalWeight::one());

        let mut util = util(vec![(0, root), (-1, a), (-2, b)], 0);
        assert!(util.cyclic_dependencies());
        let err = util.replace_labels(&[-1, -2]).unwrap_err();
        assert!(format!("{err}").contains("no end"), "{err}");
    }

    /// Whether the calls recurse is what says the network can be expanded at
    /// all.
    #[test]
    fn recursion_in_the_calls_is_reported() {
        let mut acyclic = util(simple(), 0);
        assert!(!acyclic.cyclic_dependencies());

        let mut recursive = Vf::new();
        for _ in 0..2 {
            recursive.add_state();
        }
        recursive.set_start(0);
        recursive.add_arc(0, StdArc::new(-1, -1, TropicalWeight::one(), 1));
        recursive.set_final(1, TropicalWeight::one());
        let mut util = util(vec![(-1, recursive)], -1);
        assert!(util.cyclic_dependencies());
    }

    /// Rules the root never calls, and states no path can use, are dropped.
    #[test]
    fn connecting_drops_what_no_path_can_use() {
        let mut pairs = simple();
        // A rule nothing calls.
        pairs.push((-9, chain(&[8])));
        // And a state in the root that leads nowhere.
        let dead = pairs[0].1.add_state();
        pairs[0]
            .1
            .add_arc(0, StdArc::new(7, 7, TropicalWeight::one(), dead));
        pairs[0].1.properties(K_FST_PROPERTIES, true);

        let mut util = util(pairs, 0);
        assert!(!util.connected());
        util.connect();
        assert!(util.connected());

        assert!(util.fst(-9).is_none(), "an uncalled rule is dropped");
        assert!(util.fst(-1).is_some(), "a called rule is kept");
        let root = util.fst(0).unwrap();
        assert!(
            root.states().all(|s| root.arcs(s).all(|a| a.ilabel() != 7)),
            "the arc into the dead state is gone"
        );
        assert_eq!(util.fst_pairs().len(), 2);
    }

    /// A rule that is one arc long is pure indirection, so it is folded away.
    #[test]
    fn trivial_rules_are_folded_away() {
        let mut util = util(simple(), 0);
        util.replace_trivial().unwrap();
        assert_eq!(strings(util.fst(0).unwrap()), vec![vec![1, 2, 4]]);
    }

    /// A rule too big for the limits given is left as a call.
    #[test]
    fn a_rule_over_the_size_limit_is_left_alone() {
        let mut pairs = simple();
        pairs[1].1 = chain(&[2, 3, 5]);
        let mut util = util(pairs, 0);
        util.replace_by_size(2, 1, 1).unwrap();
        let root = util.fst(0).unwrap();
        assert!(
            root.states()
                .any(|s| root.arcs(s).any(|a| a.olabel() == -1)),
            "the call is still there"
        );
    }

    /// A rule called from one place only is not worth keeping as a rule; one
    /// called from two is.
    #[test]
    fn rules_called_once_are_folded_away() {
        let mut root = Vf::new();
        for _ in 0..4 {
            root.add_state();
        }
        root.set_start(0);
        root.add_arc(0, StdArc::new(-1, -1, TropicalWeight::one(), 1));
        root.add_arc(1, StdArc::new(-2, -2, TropicalWeight::one(), 2));
        root.add_arc(2, StdArc::new(-2, -2, TropicalWeight::one(), 3));
        root.set_final(3, TropicalWeight::one());
        root.properties(K_FST_PROPERTIES, true);

        let mut util = util(vec![(0, root), (-1, chain(&[1])), (-2, chain(&[2]))], 0);
        util.replace_unique().unwrap();

        let expanded = util.fst(0).unwrap();
        assert!(
            expanded
                .states()
                .all(|s| expanded.arcs(s).all(|a| a.olabel() != -1)),
            "-1 is called once, so it is folded in"
        );
        assert!(
            expanded
                .states()
                .any(|s| expanded.arcs(s).any(|a| a.olabel() == -2)),
            "-2 is called twice, so it stays a rule"
        );
    }

    /// Expanding must not change what the network describes, whichever rules
    /// are chosen.
    #[test]
    fn expanding_some_rules_does_not_change_the_language() {
        let mut root = Vf::new();
        for _ in 0..4 {
            root.add_state();
        }
        root.set_start(0);
        root.add_arc(0, StdArc::new(-1, -1, TropicalWeight::one(), 1));
        root.add_arc(1, StdArc::new(-2, -2, TropicalWeight::one(), 2));
        root.add_arc(1, StdArc::new(9, 9, TropicalWeight::one(), 2));
        root.add_arc(2, StdArc::new(-1, -1, TropicalWeight::one(), 3));
        root.set_final(3, TropicalWeight::one());
        root.properties(K_FST_PROPERTIES, true);

        let mut middle = Vf::new();
        for _ in 0..3 {
            middle.add_state();
        }
        middle.set_start(0);
        middle.add_arc(0, StdArc::new(-3, -3, TropicalWeight::one(), 1));
        middle.add_arc(1, StdArc::new(7, 7, TropicalWeight::one(), 2));
        middle.set_final(2, TropicalWeight::one());
        middle.properties(K_FST_PROPERTIES, true);

        let pairs = || {
            vec![
                (0, root.clone()),
                (-1, chain(&[1])),
                (-2, middle.clone()),
                (-3, chain(&[3])),
            ]
        };

        // Everything expanded, as the answer to compare against.
        let mut all = util(pairs(), 0);
        all.replace_labels(&[-1, -2, -3]).unwrap();
        let want = strings(all.fst(0).unwrap());
        assert!(want.len() > 1, "the network describes more than one string");

        for chosen in [
            &[-1i32][..],
            &[-2][..],
            &[-3][..],
            &[-1, -3][..],
            &[-2, -3][..],
        ] {
            let mut some = util(pairs(), 0);
            some.replace_labels(chosen).unwrap();
            // Whatever is left as a call gets expanded now, and the result must
            // be the same.
            let mut rest = util(some.into_fst_pairs(), 0);
            rest.replace_labels(&[-1, -2, -3]).unwrap();
            assert_eq!(strings(rest.fst(0).unwrap()), want, "{chosen:?}");
        }
    }

    /// A component's properties are recorded against the component, not against
    /// whichever FST id happens to share its number. Upstream gets this
    /// wrong.
    #[test]
    fn component_properties_are_recorded_against_the_component() {
        // The root calls -2, -2 calls -1, and -1 calls itself. The FST ids are
        // 0, 1, 2 in the order the pairs are given, but the components come out
        // in topological order of the call graph, so -1, at id 1, does not sit
        // in component 1.
        let call = |label: i32| {
            let mut fst = Vf::new();
            for _ in 0..2 {
                fst.add_state();
            }
            fst.set_start(0);
            fst.add_arc(0, StdArc::new(label, label, TropicalWeight::one(), 1));
            fst.set_final(1, TropicalWeight::one());
            fst
        };

        let mut selfish = call(-1);
        selfish.add_arc(0, StdArc::new(5, 5, TropicalWeight::one(), 1));

        let mut util = util(vec![(0, call(-2)), (-1, selfish), (-2, call(-1))], 0);
        assert!(util.cyclic_dependencies());

        let selfish_scc = util.scc(-1).unwrap();
        assert_ne!(
            selfish_scc, 1,
            "the self-calling FST is at id 1; the point of this test is that its \
             component number is not 1"
        );

        assert_ne!(
            util.scc_properties(selfish_scc) & REPLACE_SCC_NON_TRIVIAL,
            0,
            "the component holding the self-calling rule is non-trivial"
        );
        for label in [0, -2] {
            let scc = util.scc(label).unwrap();
            assert_eq!(
                util.scc_properties(scc) & REPLACE_SCC_NON_TRIVIAL,
                0,
                "component of {label} calls nothing in its own component"
            );
        }
    }

    /// A right-linear component is recognized as such: the non-terminal is the
    /// last label on every path.
    #[test]
    fn a_right_linear_component_is_recognized() {
        // NT -> 5 | 5 NT. The call has to leave the state rather than loop on
        // it, or the non-terminal sits on a cycle of the FST and could appear
        // any number of times, which is neither kind of linearity.
        let mut right = Vf::new();
        for _ in 0..3 {
            right.add_state();
        }
        right.set_start(0);
        right.add_arc(0, StdArc::new(5, 5, TropicalWeight::one(), 1));
        right.add_arc(1, StdArc::new(-1, -1, TropicalWeight::one(), 2));
        right.set_final(1, TropicalWeight::one());
        right.set_final(2, TropicalWeight::one());

        let mut util = util(vec![(-1, right)], -1);
        let scc = util.scc(-1).unwrap();
        let props = util.scc_properties(scc);
        assert_ne!(props & REPLACE_SCC_RIGHT_LINEAR, 0, "{props:#x}");
        assert_ne!(props & REPLACE_SCC_NON_TRIVIAL, 0, "{props:#x}");
        assert_eq!(
            props & REPLACE_SCC_LEFT_LINEAR,
            0,
            "the call does not leave the start state: {props:#x}"
        );
    }

    /// And a left-linear one, where the non-terminal leaves the start state.
    #[test]
    fn a_left_linear_component_is_recognized() {
        // NT -> 5 | NT 5.
        let mut left = Vf::new();
        for _ in 0..3 {
            left.add_state();
        }
        left.set_start(0);
        left.add_arc(0, StdArc::new(-1, -1, TropicalWeight::one(), 1));
        left.add_arc(1, StdArc::new(5, 5, TropicalWeight::one(), 2));
        left.add_arc(0, StdArc::new(5, 5, TropicalWeight::one(), 2));
        left.set_final(2, TropicalWeight::one());

        let mut util = util(vec![(-1, left)], -1);
        let scc = util.scc(-1).unwrap();
        let props = util.scc_properties(scc);
        assert_ne!(props & REPLACE_SCC_LEFT_LINEAR, 0, "{props:#x}");
        assert_ne!(props & REPLACE_SCC_NON_TRIVIAL, 0, "{props:#x}");
        assert_eq!(
            props & REPLACE_SCC_RIGHT_LINEAR,
            0,
            "the call does not arrive at a final state: {props:#x}"
        );
    }

    /// A non-terminal on a cycle of its own FST can appear any number of times
    /// on a path, so it is neither the first label nor the last.
    #[test]
    fn a_non_terminal_on_a_cycle_is_neither_linear() {
        let mut looped = Vf::new();
        for _ in 0..2 {
            looped.add_state();
        }
        looped.set_start(0);
        looped.add_arc(0, StdArc::new(5, 5, TropicalWeight::one(), 1));
        looped.add_arc(1, StdArc::new(-1, -1, TropicalWeight::one(), 1));
        looped.set_final(1, TropicalWeight::one());

        let mut util = util(vec![(-1, looped)], -1);
        let scc = util.scc(-1).unwrap();
        let props = util.scc_properties(scc);
        assert_eq!(props, REPLACE_SCC_NON_TRIVIAL, "{props:#x}");
    }

    /// Without recursion there is one FST per component and nothing is
    /// non-trivial.
    #[test]
    fn an_acyclic_network_has_only_trivial_components() {
        let mut util = util(simple(), 0);
        assert!(!util.cyclic_dependencies());
        for label in [0, -1] {
            let scc = util.scc(label).unwrap();
            assert_eq!(util.scc_properties(scc) & REPLACE_SCC_NON_TRIVIAL, 0);
        }
    }
}
