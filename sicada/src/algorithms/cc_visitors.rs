use super::dfs_visit::DfsVisitor;
use crate::arc::{Arc, ArcStateId};
use crate::data_structures::bit_set::GrowableBitSet;
use crate::data_structures::union_find::UnionFind;
use crate::fst::Fst;
use crate::properties::{
    K_ACCESSIBLE, K_ACYCLIC, K_CO_ACCESSIBLE, K_CYCLIC, K_INITIAL_ACYCLIC, K_INITIAL_CYCLIC,
    K_NOT_ACCESSIBLE, K_NOT_CO_ACCESSIBLE,
};
use crate::weight::Weight;

pub enum UnionFindRef<'a> {
    Owned(UnionFind),
    Borrowed(&'a mut UnionFind),
}

impl<'a> UnionFindRef<'a> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut UnionFind {
        match self {
            Self::Owned(uf) => uf,
            Self::Borrowed(uf) => uf,
        }
    }
}

/// Finds and returns connected components. Use with DFS Visit.
pub struct CcVisitor<'a, A: Arc> {
    comps: UnionFindRef<'a>,
    cc: Option<&'a mut Vec<A::StateId>>,
    nstates: usize,
}

impl<'a, A: Arc> CcVisitor<'a, A> {
    /// `cc[i]`: connected component number for state `i`.
    #[inline]
    pub fn new_with_cc(cc: &'a mut Vec<A::StateId>) -> Self {
        Self {
            comps: UnionFindRef::Owned(UnionFind::new(0)),
            cc: Some(cc),
            nstates: 0,
        }
    }

    /// comps: connected components equiv classes.
    #[inline]
    pub fn new_with_comps(comps: &'a mut UnionFind) -> Self {
        Self {
            comps: UnionFindRef::Borrowed(comps),
            cc: None,
            nstates: 0,
        }
    }

    /// Returns number of components.
    /// `cc[i]`: connected component number for state `i`.
    pub fn get_cc_vector(&mut self, cc: &mut Vec<A::StateId>) -> usize {
        cc.clear();
        cc.resize(self.nstates, A::StateId::no_state());
        let mut ncomp = 0;
        for s in 0..self.nstates {
            let rep = self.comps.as_mut().find_set(s).unwrap_or(s);
            let mut comp = cc[rep];
            if comp == A::StateId::no_state() {
                comp = A::StateId::from_usize(ncomp);
                ncomp += 1;
            }
            cc[s] = comp;
            cc[rep] = comp;
        }
        ncomp
    }
}

impl<'a, A: Arc> DfsVisitor<A> for CcVisitor<'a, A> {
    #[inline(always)]
    fn init_visit<F: Fst<A>>(&mut self, _fst: &F) {}

    #[inline]
    fn init_state(&mut self, s: A::StateId, _root: A::StateId) -> bool {
        self.nstates += 1;
        self.comps.as_mut().make_set(s.as_usize());
        true
    }

    #[inline]
    fn tree_arc(&mut self, s: A::StateId, arc: &A) -> bool {
        let nextstate = arc.nextstate().as_usize();
        self.comps.as_mut().make_set(nextstate);
        self.comps.as_mut().union(s.as_usize(), nextstate);
        true
    }

    #[inline]
    fn back_arc(&mut self, s: A::StateId, arc: &A) -> bool {
        self.comps
            .as_mut()
            .union(s.as_usize(), arc.nextstate().as_usize());
        true
    }

    #[inline]
    fn forward_or_cross_arc(&mut self, s: A::StateId, arc: &A) -> bool {
        self.comps
            .as_mut()
            .union(s.as_usize(), arc.nextstate().as_usize());
        true
    }

    #[inline(always)]
    fn finish_state(&mut self, _s: A::StateId, _parent: Option<A::StateId>, _arc: Option<&A>) {}

    #[inline]
    fn finish_visit(&mut self) {
        if self.cc.is_some() {
            let cc_opt = self.cc.take().unwrap();
            self.get_cc_vector(cc_opt);
            self.cc = Some(cc_opt);
        }
    }
}

pub enum BitSetRef<'a> {
    Borrowed(&'a mut GrowableBitSet),
    Owned(GrowableBitSet),
}

impl<'a> BitSetRef<'a> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut GrowableBitSet {
        match self {
            Self::Borrowed(v) => v,
            Self::Owned(v) => v,
        }
    }
}

/// Finds and returns strongly-connected components, accessible and
/// coaccessible states and related properties. Uses Tarjan's single
/// DFS SCC algorithm.
pub struct SccVisitor<'a, 'f, A: Arc, F: Fst<A>> {
    fst: &'f F,
    scc: Option<&'a mut Vec<A::StateId>>,
    access: Option<&'a mut GrowableBitSet>,
    coaccess: BitSetRef<'a>,
    props: &'a mut u64,
    /// The bits being worked out, kept here rather than written through
    /// `props` on every back arc: that is a load and a store through a
    /// reference the optimiser cannot keep in a register. Flushed by
    /// `finish_visit`.
    building: u64,
    start: Option<A::StateId>,
    nstates: usize,
    nscc: usize,
    /// SICADA-OPT: upstream holds these as `StateId`-wide vectors of the
    /// search's own numbering. `u32` halves what a random probe pulls in, two
    /// 40 KB tables instead of two 80 KB ones for 10000 states, and the
    /// numbering cannot exceed the state count, which a 32-bit state id bounds
    /// already.
    dfnumber: Vec<u32>,
    lowlink: Vec<u32>,
    onstack: GrowableBitSet,
    scc_stack: Vec<A::StateId>,
    /// Whether to work out which states can still reach a final state.
    want_coaccess: bool,
}

impl<'a, 'f, A: Arc, F: Fst<A>> SccVisitor<'a, 'f, A, F> {
    /// Stops working out which states can still reach a final state.
    ///
    /// SICADA-OPT: the coaccess bookkeeping is two bitset operations per arc,
    /// plus a scan of every component when it closes. A caller that only wants
    /// the components, such as [`components`](crate::queue::components), from
    /// which [`AutoQueue`](crate::queue::AutoQueue) decides its discipline, pays
    /// for an answer it then drops. Upstream has no way to say so.
    ///
    /// The coaccess property bits are left *unset* rather than wrong: unknown,
    /// not false.
    pub fn without_coaccess(mut self) -> Self {
        self.want_coaccess = false;
        self
    }

    pub fn new(
        fst: &'f F,
        scc: Option<&'a mut Vec<A::StateId>>,
        access: Option<&'a mut GrowableBitSet>,
        coaccess: Option<&'a mut GrowableBitSet>,
        props: &'a mut u64,
    ) -> Self {
        let coaccess_ref = match coaccess {
            Some(v) => BitSetRef::Borrowed(v),
            None => BitSetRef::Owned(GrowableBitSet::new()),
        };

        Self {
            want_coaccess: true,
            building: 0,
            fst,
            scc,
            access,
            coaccess: coaccess_ref,
            props,
            start: None,
            nstates: 0,
            nscc: 0,
            dfnumber: Vec::new(),
            lowlink: Vec::new(),
            onstack: GrowableBitSet::new(),
            scc_stack: Vec::new(),
        }
    }
}

impl<'a, 'f, A: Arc, F: Fst<A>> DfsVisitor<A> for SccVisitor<'a, 'f, A, F> {
    #[inline]
    fn init_visit<F2: Fst<A>>(&mut self, _fst: &F2) {
        if let Some(scc) = &mut self.scc {
            scc.clear();
        }
        if let Some(access) = &mut self.access {
            access.clear();
        }
        self.coaccess.as_mut().clear();

        self.building = *self.props;
        self.building |= K_ACYCLIC | K_INITIAL_ACYCLIC | K_ACCESSIBLE;
        self.building &= !(K_CYCLIC | K_INITIAL_CYCLIC | K_NOT_ACCESSIBLE);
        if self.want_coaccess {
            self.building |= K_CO_ACCESSIBLE;
            self.building &= !K_NOT_CO_ACCESSIBLE;
        }

        self.start = self.fst.start();
        self.nstates = 0;
        self.nscc = 0;
        self.dfnumber.clear();
        self.lowlink.clear();
        self.onstack.clear();
        self.scc_stack.clear();
    }

    #[inline]
    fn init_state(&mut self, s: A::StateId, root: A::StateId) -> bool {
        self.scc_stack.push(s);
        let s_idx = s.as_usize();

        if self.dfnumber.len() <= s_idx {
            if let Some(scc) = &mut self.scc {
                scc.resize(s_idx + 1, A::StateId::no_state());
            }
            // Grown to cover `s` whether or not `s` turns out to be accessible
            // or coaccessible, as upstream's `resize(s + 1, false)` does. A
            // caller reads these by asking how far they reach, and `connect`
            // walks the whole range looking for the states that are neither, so
            // a set that stops short of the highest state visited hides it.
            if let Some(access) = &mut self.access {
                access.ensure(s_idx + 1);
            }
            self.coaccess.as_mut().ensure(s_idx + 1);
            self.dfnumber.resize(s_idx + 1, u32::MAX);
            self.lowlink.resize(s_idx + 1, u32::MAX);
        }

        let number = self.nstates as u32;
        self.dfnumber[s_idx] = number;
        self.lowlink[s_idx] = number;
        self.onstack.insert(s_idx);

        if Some(root) == self.start {
            if let Some(access) = &mut self.access {
                access.insert(s_idx);
            }
        } else {
            if let Some(access) = &mut self.access {
                access.remove(s_idx);
            }
            self.building |= K_NOT_ACCESSIBLE;
            self.building &= !K_ACCESSIBLE;
        }

        self.nstates += 1;
        true
    }

    #[inline(always)]
    fn tree_arc(&mut self, _s: A::StateId, _arc: &A) -> bool {
        true
    }

    #[inline]
    fn back_arc(&mut self, s: A::StateId, arc: &A) -> bool {
        let s_idx = s.as_usize();
        let t = arc.nextstate();
        let t_idx = t.as_usize();

        if self.dfnumber[t_idx] < self.lowlink[s_idx] {
            self.lowlink[s_idx] = self.dfnumber[t_idx];
        }
        if self.want_coaccess {
            let coaccess = self.coaccess.as_mut();
            if coaccess.contains(t_idx) {
                coaccess.insert(s_idx);
            }
        }

        self.building |= K_CYCLIC;
        self.building &= !K_ACYCLIC;
        if Some(t) == self.start {
            self.building |= K_INITIAL_CYCLIC;
            self.building &= !K_INITIAL_ACYCLIC;
        }
        true
    }

    #[inline]
    fn forward_or_cross_arc(&mut self, s: A::StateId, arc: &A) -> bool {
        let s_idx = s.as_usize();
        let t = arc.nextstate();
        let t_idx = t.as_usize();

        if self.dfnumber[t_idx] < self.dfnumber[s_idx]
            && self.onstack.contains(t_idx)
            && self.dfnumber[t_idx] < self.lowlink[s_idx]
        {
            self.lowlink[s_idx] = self.dfnumber[t_idx];
        }
        if self.want_coaccess {
            let coaccess = self.coaccess.as_mut();
            if coaccess.contains(t_idx) {
                coaccess.insert(s_idx);
            }
        }
        true
    }

    #[inline]
    fn finish_state(&mut self, s: A::StateId, p: Option<A::StateId>, _arc: Option<&A>) {
        let s_idx = s.as_usize();
        if self.want_coaccess {
            let w = self.fst.final_weight(s);
            if w.is_member() && w != A::Weight::zero() {
                self.coaccess.as_mut().insert(s_idx);
            }
        }

        // Root of new SCC
        if self.dfnumber[s_idx] == self.lowlink[s_idx] {
            let mut scc_coaccess = false;
            if self.want_coaccess {
                let mut i = self.scc_stack.len();
                loop {
                    i -= 1;
                    let t = self.scc_stack[i];
                    if self.coaccess.as_mut().contains(t.as_usize()) {
                        scc_coaccess = true;
                    }
                    if s == t {
                        break;
                    }
                }
            }
            let mut t;

            loop {
                t = self.scc_stack.pop().unwrap();
                let t_idx = t.as_usize();
                if let Some(scc) = &mut self.scc {
                    scc[t_idx] = A::StateId::from_usize(self.nscc);
                }
                if scc_coaccess {
                    self.coaccess.as_mut().insert(t_idx);
                }
                self.onstack.remove(t_idx);
                if s == t {
                    break;
                }
            }

            if self.want_coaccess && !scc_coaccess {
                self.building |= K_NOT_CO_ACCESSIBLE;
                self.building &= !K_CO_ACCESSIBLE;
            }
            self.nscc += 1;
        }

        if let Some(parent) = p {
            let p_idx = parent.as_usize();
            if self.want_coaccess {
                let coaccess = self.coaccess.as_mut();
                if coaccess.contains(s_idx) {
                    coaccess.insert(p_idx);
                }
            }
            if self.lowlink[s_idx] < self.lowlink[p_idx] {
                self.lowlink[p_idx] = self.lowlink[s_idx];
            }
        }
    }

    #[inline]
    fn finish_visit(&mut self) {
        *self.props = self.building;
        // Numbers SCCs in topological order when acyclic.
        if let Some(scc) = &mut self.scc {
            for s in scc.iter_mut() {
                if *s != A::StateId::no_state() {
                    let new_val = self.nscc - 1 - s.as_usize();
                    *s = A::StateId::from_usize(new_val);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AtomicRc;
    use crate::fst_type::ArcType;
    use crate::properties::{K_ACCESSIBLE, K_ACYCLIC, K_CYCLIC};
    use crate::symbol_table::SymbolTable;
    use crate::weight::Weight;
    use crate::weights::float_weight::TropicalWeight;

    use std::iter::Empty;

    #[derive(Clone, Debug, PartialEq)]
    struct TestArc {
        next: usize,
        w: TropicalWeight,
    }

    impl Arc for TestArc {
        type Weight = TropicalWeight;
        type Label = i32;
        type StateId = usize;
        type Reverse = Self;

        fn new(
            _ilabel: Self::Label,
            _olabel: Self::Label,
            weight: Self::Weight,
            nextstate: Self::StateId,
        ) -> Self {
            Self {
                next: nextstate,
                w: weight,
            }
        }

        fn type_name() -> ArcType {
            ArcType::new_dynamic(String::from("test_arc"))
        }

        fn ilabel(&self) -> Self::Label {
            0
        }
        fn olabel(&self) -> Self::Label {
            0
        }
        fn weight(&self) -> &Self::Weight {
            &self.w
        }
        fn nextstate(&self) -> Self::StateId {
            self.next
        }
    }

    struct TestFst {
        start_state: Option<usize>,
        finals: std::collections::HashMap<usize, TropicalWeight>,
    }

    impl Fst<TestArc> for TestFst {
        type StateIter<'a> = Empty<usize>;
        type ArcIter<'a> = Empty<TestArc>;

        fn start(&self) -> Option<usize> {
            self.start_state
        }

        fn final_weight(&self, state: usize) -> TropicalWeight {
            self.finals
                .get(&state)
                .cloned()
                .unwrap_or_else(TropicalWeight::zero)
        }

        fn num_arcs(&self, _state: usize) -> usize {
            0
        }
        fn num_input_epsilons(&self, _state: usize) -> usize {
            0
        }
        fn num_output_epsilons(&self, _state: usize) -> usize {
            0
        }
        fn num_states_if_known(&self) -> Option<usize> {
            None
        }
        fn properties(&self, _mask: u64, _test: bool) -> u64 {
            0
        }

        fn fst_type(&self) -> &str {
            "test_fst"
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

        fn arcs<'a>(&'a self, _state: usize) -> Self::ArcIter<'a> {
            std::iter::empty()
        }
    }

    #[test]
    fn test_cc_visitor_components() {
        let mut cc = Vec::new();
        let mut visitor = CcVisitor::<TestArc>::new_with_cc(&mut cc);
        let fst = TestFst {
            start_state: None,
            finals: std::collections::HashMap::new(),
        };

        visitor.init_visit(&fst);

        visitor.init_state(0, 0);
        visitor.init_state(1, 0);
        visitor.init_state(2, 0);
        visitor.init_state(3, 3);
        visitor.init_state(4, 3);
        visitor.init_state(5, 5);

        let arc_0_1 = TestArc {
            next: 1,
            w: TropicalWeight::one(),
        };
        let arc_1_2 = TestArc {
            next: 2,
            w: TropicalWeight::one(),
        };
        let arc_3_4 = TestArc {
            next: 4,
            w: TropicalWeight::one(),
        };

        visitor.tree_arc(0, &arc_0_1);
        visitor.tree_arc(1, &arc_1_2);
        visitor.tree_arc(3, &arc_3_4);

        visitor.finish_visit();

        let mut out_cc = Vec::new();
        let num_components = visitor.get_cc_vector(&mut out_cc);

        assert_eq!(num_components, 3);
        assert_eq!(out_cc.len(), 6);

        assert_eq!(out_cc[0], out_cc[1]);
        assert_eq!(out_cc[1], out_cc[2]);
        assert_eq!(out_cc[3], out_cc[4]);

        assert_ne!(out_cc[0], out_cc[3]);
        assert_ne!(out_cc[0], out_cc[5]);
        assert_ne!(out_cc[3], out_cc[5]);
    }

    #[test]
    fn test_scc_visitor_tarjan() {
        let mut finals = std::collections::HashMap::new();
        finals.insert(1, TropicalWeight::one());

        let fst = TestFst {
            start_state: Some(0),
            finals,
        };

        let mut scc = Vec::new();
        let mut access = GrowableBitSet::new();
        let mut coaccess = GrowableBitSet::new();
        let mut props = 0;

        let mut visitor = SccVisitor::new(
            &fst,
            Some(&mut scc),
            Some(&mut access),
            Some(&mut coaccess),
            &mut props,
        );

        visitor.init_visit(&fst);

        visitor.init_state(0, 0);

        let arc_0_1 = TestArc {
            next: 1,
            w: TropicalWeight::one(),
        };
        visitor.tree_arc(0, &arc_0_1);

        visitor.init_state(1, 0);

        let arc_1_0 = TestArc {
            next: 0,
            w: TropicalWeight::one(),
        };
        visitor.back_arc(1, &arc_1_0);

        visitor.finish_state(1, Some(0), Some(&arc_1_0));
        visitor.finish_state(0, None, None);

        visitor.finish_visit();

        assert_eq!(scc.len(), 2);
        assert_eq!(scc[0], scc[1]);

        assert!(access.contains(0));
        assert!(access.contains(1));
        assert!(coaccess.contains(0));
        assert!(coaccess.contains(1));

        assert_eq!(props & K_CYCLIC, K_CYCLIC);
        assert_eq!(props & K_ACYCLIC, 0);

        assert_eq!(props & K_ACCESSIBLE, K_ACCESSIBLE);
    }
}

#[cfg(test)]
mod real_fst_tests {
    use super::*;
    use crate::algorithms::dfs_visit::dfs_visit;
    use crate::arc::StdArc;
    use crate::arc_filter::AnyArcFilter;
    use crate::data_structures::bit_set::GrowableBitSet;
    use crate::fst::MutableFst;
    use crate::fsts::vector_fst::VectorFst;
    use crate::properties::{
        K_ACCESSIBLE, K_ACYCLIC, K_CO_ACCESSIBLE, K_CYCLIC, K_NOT_ACCESSIBLE, K_NOT_CO_ACCESSIBLE,
    };
    use crate::weight::Weight;
    use crate::weights::float_weight::TropicalWeight;

    /// Builds an FST from an edge list, with the given states final.
    fn build(states: usize, edges: &[(i32, i32)], finals: &[i32]) -> VectorFst<StdArc> {
        let mut fst = VectorFst::new();
        for _ in 0..states {
            fst.add_state();
        }
        fst.set_start(0);
        for &(from, to) in edges {
            fst.add_arc(from, StdArc::new(1, 1, TropicalWeight::one(), to));
        }
        for &state in finals {
            fst.set_final(state, TropicalWeight::one());
        }
        fst
    }

    /// Runs the SCC visitor and returns the component of each state, the access
    /// and coaccess flags, and the properties it derived.
    fn scc_of(fst: &VectorFst<StdArc>) -> (Vec<i32>, GrowableBitSet, GrowableBitSet, u64) {
        let mut scc = Vec::new();
        let mut access = GrowableBitSet::new();
        let mut coaccess = GrowableBitSet::new();
        let mut props = 0u64;
        {
            let mut visitor = SccVisitor::new(
                fst,
                Some(&mut scc),
                Some(&mut access),
                Some(&mut coaccess),
                &mut props,
            );
            dfs_visit(fst, &mut visitor, AnyArcFilter, false);
        }
        (scc, access, coaccess, props)
    }

    /// Two mutually reachable states share a component; a state reachable only
    /// one way does not.
    #[test]
    fn a_cycle_is_one_component_and_a_chain_is_not() {
        // 0 -> 1 -> 2 -> 1, with 2 final: {1, 2} is a cycle, 0 is alone.
        let fst = build(3, &[(0, 1), (1, 2), (2, 1)], &[2]);
        let (scc, _, _, props) = scc_of(&fst);

        assert_eq!(scc.len(), 3);
        assert_eq!(scc[1], scc[2], "1 and 2 are mutually reachable");
        assert_ne!(scc[0], scc[1], "0 is not reachable from 1");

        assert_ne!(props & K_CYCLIC, 0, "the FST has a cycle");
        assert_eq!(props & K_ACYCLIC, 0);
    }

    #[test]
    fn an_acyclic_fst_gives_one_component_per_state() {
        let fst = build(3, &[(0, 1), (1, 2)], &[2]);
        let (scc, _, _, props) = scc_of(&fst);

        assert_eq!(scc.len(), 3);
        assert_ne!(scc[0], scc[1]);
        assert_ne!(scc[1], scc[2]);
        assert_ne!(props & K_ACYCLIC, 0);
        assert_eq!(props & K_CYCLIC, 0);
    }

    /// Access means reachable from the start; coaccess means a final state is
    /// reachable from it. Both are what `Connect` prunes on.
    #[test]
    fn access_and_coaccess_are_reported_per_state() {
        // 0 -> 1 (final), and an isolated 2 -> 3 with 3 final.
        let fst = build(4, &[(0, 1), (2, 3)], &[1, 3]);
        let (_, access, coaccess, props) = scc_of(&fst);

        assert!(access.contains(0));
        assert!(access.contains(1));
        assert!(!access.contains(2), "2 is not reachable from the start");
        assert!(!access.contains(3));
        assert_ne!(props & K_NOT_ACCESSIBLE, 0);
        assert_eq!(props & K_ACCESSIBLE, 0);

        // Every state here can reach a final state.
        for state in 0..4 {
            assert!(
                coaccess.contains(state),
                "state {state} should be coaccessible"
            );
        }
        assert_ne!(props & K_CO_ACCESSIBLE, 0);
    }

    #[test]
    fn a_dead_end_is_not_coaccessible() {
        // 0 -> 1 (final), 0 -> 2 (dead end).
        let fst = build(3, &[(0, 1), (0, 2)], &[1]);
        let (_, _, coaccess, props) = scc_of(&fst);

        assert!(coaccess.contains(0));
        assert!(coaccess.contains(1));
        assert!(!coaccess.contains(2), "2 reaches no final state");
        assert_ne!(props & K_NOT_CO_ACCESSIBLE, 0);
        assert_eq!(props & K_CO_ACCESSIBLE, 0);
    }

    /// Cross-checks Tarjan's output against the definition: two states share a
    /// component exactly when each can reach the other.
    #[test]
    fn components_match_mutual_reachability() {
        let mut state = 0xF00D_BEEF_1234_5678u64;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        const N: usize = 8;
        for round in 0..80 {
            let mut edges = Vec::new();
            for from in 0..N as i32 {
                for to in 0..N as i32 {
                    if rng() % 5 == 0 {
                        edges.push((from, to));
                    }
                }
            }
            let fst = build(N, &edges, &[(N - 1) as i32]);
            let (scc, _, _, _) = scc_of(&fst);

            // Transitive closure of the edge relation.
            let mut reach = [[false; N]; N];
            for &(from, to) in &edges {
                reach[from as usize][to as usize] = true;
            }
            for k in 0..N {
                for i in 0..N {
                    for j in 0..N {
                        if reach[i][k] && reach[k][j] {
                            reach[i][j] = true;
                        }
                    }
                }
            }

            for i in 0..N {
                for j in 0..N {
                    let mutual = i == j || (reach[i][j] && reach[j][i]);
                    // Only states the search actually visited get a component.
                    if scc[i] < 0 || scc[j] < 0 {
                        continue;
                    }
                    assert_eq!(
                        scc[i] == scc[j],
                        mutual,
                        "round {round}: states {i} and {j}"
                    );
                }
            }
        }
    }
}
