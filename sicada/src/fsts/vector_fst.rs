//! The ordinary mutable FST.
//!
//! Port of OpenFst's `vector-fst.h`. States live in a vector and each holds its
//! own arcs, which makes it the one FST that can be built up a piece at a
//! time.

use std::io::{Read, Write};

use crate::AtomicRc;
use crate::algorithms::test_properties::cached_properties;
use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::error::OpenFstError;
use crate::fst::{
    ContiguousArcsFst, ExpandedFst, Fst, FstReadOptions, FstWriteOptions, MutableFst, NO_STATE_ID,
    PropertyCache, read_fst_header, write_fst_header,
};
use crate::fst_header::FstHeader;
use crate::fst_type::FstType;
use crate::properties::*;
use crate::symbol_table::SymbolTable;
use crate::utils::io::{FstScalar, read_scalar, write_scalar};
use crate::weight::{Weight, WeightIo};

pub type StdVectorFst = VectorFst<crate::arc::StdArc>;
pub type Log64VectorFst = VectorFst<crate::arc::Log64Arc>;

/// Represents a single state in a `VectorFst`.
/// Stores the final weight, outgoing arcs, and cached epsilon counts.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorState<A: Arc> {
    pub final_weight: A::Weight,
    pub arcs: ArcVec<A>,
    niepsilons: usize,
    noepsilons: usize,
}

/// How many arcs a state holds without reaching for the heap.
///
/// SICADA-OPT: upstream's `VectorState` holds a `std::vector<Arc>`, so every
/// state with any arc at all costs an allocation, and the FSTs this library
/// exists for are mostly states of small out-degree. Four `StdArc`s are 64
/// bytes, which is one cache line and the out-degree of a typical lexicon or
/// grammar state. Cloning a 10000-state FST of out-degree 4 stops being 10000
/// allocations and becomes one memcpy: 318 µs → 136 µs, measured by
/// `sicada-bench`'s `DIAG_SORT=1` breakdown.
///
/// The cost is that a state carries the inline space whether it uses it or
/// not, so an FST of large out-degree copies more bytes than it needs. That
/// shape is measured too (`arcsort/2000x16`): cloning 2000 states of
/// out-degree 16 goes from 38 µs to 56 µs, and the whole benchmark still comes
/// out ahead of both other Rust libraries.
pub const INLINE_ARCS: usize = 4;

/// The arcs of one state.
pub type ArcVec<A> = smallvec::SmallVec<[A; INLINE_ARCS]>;

impl<A: Arc> VectorState<A> {
    pub fn new() -> Self {
        Self {
            final_weight: A::Weight::zero(),
            arcs: ArcVec::new(),
            niepsilons: 0,
            noepsilons: 0,
        }
    }

    #[inline]
    fn increment_num_epsilons(&mut self, arc: &A) {
        if arc.ilabel() == A::Label::epsilon() {
            self.niepsilons += 1;
        }
        if arc.olabel() == A::Label::epsilon() {
            self.noepsilons += 1;
        }
    }

    pub fn add_arc(&mut self, arc: A) {
        self.increment_num_epsilons(&arc);
        self.arcs.push(arc);
    }

    pub fn delete_arcs_n(&mut self, n: usize) {
        let count = std::cmp::min(n, self.arcs.len());
        for _ in 0..count {
            let arc = self.arcs.pop().unwrap();
            if arc.ilabel() == A::Label::epsilon() {
                self.niepsilons -= 1;
            }
            if arc.olabel() == A::Label::epsilon() {
                self.noepsilons -= 1;
            }
        }
    }

    pub fn delete_all_arcs(&mut self) {
        self.arcs.clear();
        self.niepsilons = 0;
        self.noepsilons = 0;
    }

    pub fn reserve_arcs(&mut self, n: usize) {
        self.arcs.reserve(n);
    }
}

impl<A: Arc> Default for VectorState<A> {
    fn default() -> Self {
        Self::new()
    }
}

/// A simple, mutable FST whose states and arcs are stored in standard `Vec`s.
#[derive(Debug, Clone)]
pub struct VectorFst<A: Arc> {
    states: Vec<VectorState<A>>,
    start: Option<A::StateId>,
    properties: PropertyCache,
    isymbols: Option<AtomicRc<SymbolTable>>,
    osymbols: Option<AtomicRc<SymbolTable>>,
}

impl<A: Arc> Default for VectorFst<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Arc> VectorFst<A> {
    pub const STATIC_PROPERTIES: u64 = K_EXPANDED | K_MUTABLE;

    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            start: None,
            properties: PropertyCache::new(K_NULL_PROPERTIES | Self::STATIC_PROPERTIES),
            isymbols: None,
            osymbols: None,
        }
    }

    #[inline]
    fn update_properties_after_add_arc(&mut self, state: A::StateId) {
        let s_idx = state.as_usize();
        if let Some(vstate) = self.states.get(s_idx) {
            let num_arcs = vstate.arcs.len();
            if num_arcs > 0 {
                let arc = &vstate.arcs[num_arcs - 1];
                let prev_arc = if num_arcs >= 2 {
                    Some(&vstate.arcs[num_arcs - 2])
                } else {
                    None
                };
                self.properties
                    .modify(|props| add_arc_properties(props, state, arc, prev_arc));
            }
        }
    }
}

impl<A: Arc> Fst<A> for VectorFst<A> {
    type StateIter<'a>
        = VectorFstStateIter<'a, A>
    where
        A: 'a;
    type ArcIter<'a>
        = std::iter::Cloned<std::slice::Iter<'a, A>>
    where
        A: 'a;

    #[inline]
    fn start(&self) -> Option<A::StateId> {
        self.start
    }

    #[inline]
    fn final_weight(&self, state: A::StateId) -> A::Weight {
        self.states
            .get(state.as_usize())
            .map(|s| s.final_weight.clone())
            .unwrap_or_else(A::Weight::zero)
    }

    #[inline]
    fn num_arcs(&self, state: A::StateId) -> usize {
        self.states
            .get(state.as_usize())
            .map(|s| s.arcs.len())
            .unwrap_or(0)
    }

    #[inline]
    fn num_input_epsilons(&self, state: A::StateId) -> usize {
        self.states
            .get(state.as_usize())
            .map(|s| s.niepsilons)
            .unwrap_or(0)
    }

    #[inline]
    fn num_output_epsilons(&self, state: A::StateId) -> usize {
        self.states
            .get(state.as_usize())
            .map(|s| s.noepsilons)
            .unwrap_or(0)
    }

    #[inline]
    fn num_states_if_known(&self) -> Option<usize> {
        Some(self.states.len())
    }

    #[inline]
    fn properties(&self, mask: u64, test: bool) -> u64 {
        // The mutating methods keep the cache up to date, so without `test`
        // there is nothing to work out.
        cached_properties(self, &self.properties, mask, test)
    }

    #[inline]
    fn fst_type(&self) -> &str {
        FstType::VECTOR.as_str()
    }

    #[inline]
    fn input_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
        self.isymbols.clone()
    }

    #[inline]
    fn output_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
        self.osymbols.clone()
    }

    #[inline]
    fn states<'a>(&'a self) -> Self::StateIter<'a> {
        VectorFstStateIter {
            pos: 0,
            len: self.states.len(),
            _marker: std::marker::PhantomData,
        }
    }

    #[inline]
    fn arcs<'a>(&'a self, state: A::StateId) -> Self::ArcIter<'a> {
        if let Some(s) = self.states.get(state.as_usize()) {
            s.arcs.iter().cloned()
        } else {
            [].iter().cloned() // Return empty iterator safely
        }
    }
}

/// The oldest file version that can still be read.
const MIN_FILE_VERSION: i32 = 2;
/// The version written.
const FILE_VERSION: i32 = 2;
/// The properties every `VectorFst` has.
const K_STATIC_PROPERTIES: u64 = K_EXPANDED | K_MUTABLE;
/// Smallest an arc can be on the wire: two labels, a weight, a state.
const MIN_BYTES_PER_ARC: i64 = 16;
/// How much a reader will reserve up front on the strength of a count in the
/// file.
///
/// SICADA-DIVERGE: upstream reserves exactly what the file claims, so a file
/// whose arc count has been corrupted, and one flipped byte in the high half of
/// a 64-bit count is enough, asks the allocator for that many arcs and takes the
/// process down when it says no. Reserving is only a hint; anything past this
/// grows as the arcs are actually read, and a file that really is that large
/// will supply them.
const MAX_RESERVE: usize = 1 << 16;

impl<A: Arc> VectorFst<A>
where
    A::Label: FstScalar,
    A::StateId: FstScalar,
    A::Weight: WeightIo,
{
    /// Reads a `VectorFst` from a stream.
    ///
    /// Unlike [`ConstFst`](crate::fsts::const_fst::ConstFst), the states and
    /// arcs are written field by field rather than as a block of memory, so
    /// there is nothing here that could be mapped and nothing that depends on
    /// how a struct happens to be laid out.
    pub fn read<R: Read>(reader: &mut R, opts: &FstReadOptions) -> Result<Self, OpenFstError> {
        let read =
            read_fst_header::<A, _>(reader, opts, FstType::VECTOR.as_str(), MIN_FILE_VERSION)?;
        let header = read.header;
        let bad =
            |message: String| OpenFstError::InvalidFstHeader(format!("{}: {message}", opts.source));

        let mut fst = Self::new();
        fst.set_input_symbols(read.isymbols);
        fst.set_output_symbols(read.osymbols);

        // A header may say it does not know how many states there are, in
        // which case the states run to the end of the stream.
        let expected = (header.num_states != NO_STATE_ID as i64).then(|| {
            let n = header.num_states.max(0) as usize;
            fst.reserve_states(n.min(MAX_RESERVE));
            n
        });

        let mut nstates = 0usize;
        while expected.is_none_or(|n| nstates < n) {
            let Ok(final_weight) = A::Weight::read(reader) else {
                break;
            };
            let state = fst.add_state();
            fst.set_final(state, final_weight);
            nstates += 1;

            let narcs: i64 = read_scalar(reader)?;
            if narcs < 0 {
                return Err(bad(format!(
                    "invalid arc count ({narcs}) for state {}",
                    nstates - 1
                )));
            }
            if narcs > i64::MAX / MIN_BYTES_PER_ARC {
                return Err(bad(format!("arc count ({narcs}) causes integer overflow")));
            }
            fst.reserve_arcs(state, (narcs as usize).min(MAX_RESERVE));
            for _ in 0..narcs {
                let ilabel: A::Label = read_scalar(reader)?;
                let olabel: A::Label = read_scalar(reader)?;
                let weight = A::Weight::read(reader)?;
                let nextstate: A::StateId = read_scalar(reader)?;
                if nextstate == A::StateId::no_state() {
                    return Err(bad("disallowed next state".to_string()));
                }
                if nextstate < A::StateId::from_usize(0) {
                    return Err(bad("next state is negative".to_string()));
                }
                fst.add_arc(state, A::new(ilabel, olabel, weight, nextstate));
            }
        }

        if expected.is_some_and(|n| n != nstates) {
            return Err(bad("unexpected end of file".to_string()));
        }
        if header.start != NO_STATE_ID as i64 {
            if header.start < 0 || header.start >= nstates as i64 {
                return Err(bad(format!(
                    "start state {} out of range [0, {nstates})",
                    header.start
                )));
            }
            fst.set_start(A::StateId::from_usize(header.start as usize));
        }
        if opts.verify {
            for state in 0..nstates {
                for arc in fst.arcs(A::StateId::from_usize(state)) {
                    if arc.nextstate().as_usize() >= nstates {
                        return Err(bad(format!(
                            "next state {} is larger than the number of states {nstates}",
                            arc.nextstate().as_usize()
                        )));
                    }
                }
            }
        }
        fst.properties.set(header.properties | K_STATIC_PROPERTIES);
        Ok(fst)
    }

    /// Writes any FST out in this format.
    ///
    /// SICADA-DIVERGE: upstream writes a header saying it does not know the
    /// state count when the stream cannot seek, then seeks back to correct it
    /// when it can. This always counts first; see
    /// [`ConstFst::write_fst`](crate::fsts::const_fst::ConstFst::write_fst),
    /// which makes the same trade for the same reason.
    pub fn write_fst<F: Fst<A>, W: Write>(
        fst: &F,
        writer: &mut W,
        opts: &FstWriteOptions,
    ) -> Result<(), OpenFstError> {
        let header = FstHeader {
            fst_type: FstType::VECTOR.as_str().to_string(),
            arc_type: A::type_name().as_str().to_string(),
            version: FILE_VERSION,
            // Filled in by `write_fst_header`.
            flags: 0,
            properties: fst.properties(K_COPY_PROPERTIES, false) | K_STATIC_PROPERTIES,
            start: fst
                .start()
                .map_or(NO_STATE_ID as i64, |s| s.as_usize() as i64),
            num_states: fst.count_states() as i64,
            // Upstream never sets this for a vector FST, so the field keeps the
            // zero it was built with. Writing anything else would change the
            // bytes.
            num_arcs: 0,
        };
        let isymbols = fst.input_symbols();
        let osymbols = fst.output_symbols();
        write_fst_header(
            writer,
            opts,
            &header,
            isymbols.as_deref(),
            osymbols.as_deref(),
        )?;

        for state in fst.states() {
            fst.final_weight(state).write(writer)?;
            write_scalar(writer, fst.num_arcs(state) as i64)?;
            for arc in fst.arcs(state) {
                write_scalar(writer, arc.ilabel())?;
                write_scalar(writer, arc.olabel())?;
                arc.weight().write(writer)?;
                write_scalar(writer, arc.nextstate())?;
            }
        }
        Ok(())
    }

    /// Writes this FST out.
    pub fn write<W: Write>(
        &self,
        writer: &mut W,
        opts: &FstWriteOptions,
    ) -> Result<(), OpenFstError> {
        Self::write_fst(self, writer, opts)
    }
}

impl<A: Arc> ExpandedFst<A> for VectorFst<A> {
    #[inline]
    fn num_states(&self) -> usize {
        self.states.len()
    }
}

impl<A: Arc> MutableFst<A> for VectorFst<A> {
    fn set_start(&mut self, state: A::StateId) {
        // Upstream stores `kNoStateId` as it would any other state ID and lets
        // `Start()` hand it back; the `Option` here is what tells the two
        // apart, so the sentinel has to be turned into `None` on the way in.
        self.start = (state != A::StateId::no_state()).then_some(state);
        self.properties.modify(set_start_properties);
    }

    fn set_final(&mut self, state: A::StateId, weight: A::Weight) {
        if let Some(s) = self.states.get_mut(state.as_usize()) {
            self.properties
                .modify(|props| set_final_properties(props, &s.final_weight, &weight));
            s.final_weight = weight;
        }
    }

    fn set_properties(&mut self, props: u64, mask: u64) {
        // kError cannot be cleared.
        self.properties
            .modify(|current| (current & (!mask | K_ERROR)) | (props & mask));
    }

    fn add_state(&mut self) -> A::StateId {
        // Safely map usize to StateId (e.g. i32) internally.
        // This is safe because `states.len()` is guaranteed to fit before pushing.
        let new_id_usize = self.states.len();

        // Use a generic conversion trait if needed, but assuming StateId is `i32` or `usize`
        // we can transmute or use a TryFrom if implemented. For now, assume it's castable.
        // In the ArcStateId trait you can add `fn from_usize(n: usize) -> Self`.
        let new_id = A::StateId::from_usize(new_id_usize);

        self.states.push(VectorState::new());
        self.properties.modify(add_state_properties);
        new_id
    }

    fn add_states(&mut self, n: usize) {
        self.states
            .resize_with(self.states.len() + n, VectorState::new);
        self.properties.modify(add_state_properties);
    }

    fn add_arc(&mut self, state: A::StateId, arc: A) {
        if let Some(s) = self.states.get_mut(state.as_usize()) {
            s.add_arc(arc);
            self.update_properties_after_add_arc(state);
        }
    }

    #[inline]
    fn arcs_mut(&mut self, state: A::StateId) -> &mut [A] {
        match self.states.get_mut(state.as_usize()) {
            Some(s) => &mut s.arcs,
            None => &mut [],
        }
    }

    fn delete_arcs_n(&mut self, state: A::StateId, n: usize) {
        if let Some(s) = self.states.get_mut(state.as_usize()) {
            s.delete_arcs_n(n);
            self.properties.modify(delete_arcs_properties);
        }
    }

    fn delete_arcs(&mut self, state: A::StateId) {
        if let Some(s) = self.states.get_mut(state.as_usize()) {
            s.delete_all_arcs();
            self.properties.modify(delete_arcs_properties);
        }
    }

    fn delete_all_states(&mut self) {
        self.states.clear();
        self.start = None;
        self.properties
            .modify(|props| delete_all_states_properties(props, Self::STATIC_PROPERTIES));
    }

    fn delete_states(&mut self, dstates: &[A::StateId]) {
        if dstates.is_empty() {
            return;
        }

        let mut newid = vec![0; self.states.len()];
        for &dstate in dstates {
            if dstate.as_usize() < newid.len() {
                // Sentinel value indicating deletion
                newid[dstate.as_usize()] = usize::MAX;
            }
        }

        let mut nstates = 0;
        (0..self.states.len()).for_each(|i| {
            if newid[i] != usize::MAX {
                newid[i] = nstates;
                if i != nstates {
                    self.states.swap(i, nstates);
                }
                nstates += 1;
            }
        });
        self.states.truncate(nstates);

        // Remap arc nextstates and re-calculate epsilon counts
        for i in 0..self.states.len() {
            let mut valid_arcs = 0;
            let mut nieps = 0;
            let mut noeps = 0;

            for j in 0..self.states[i].arcs.len() {
                let mut arc = self.states[i].arcs[j].clone();
                let t = newid[arc.nextstate().as_usize()];

                if t != usize::MAX {
                    let new_nextstate = A::StateId::from_usize(t);
                    arc = A::new(
                        arc.ilabel(),
                        arc.olabel(),
                        arc.weight().clone(),
                        new_nextstate,
                    );

                    if arc.ilabel() == A::Label::epsilon() {
                        nieps += 1;
                    }
                    if arc.olabel() == A::Label::epsilon() {
                        noeps += 1;
                    }

                    if j != valid_arcs {
                        self.states[i].arcs[valid_arcs] = arc;
                    } else {
                        self.states[i].arcs[j] = arc;
                    }
                    valid_arcs += 1;
                }
            }

            self.states[i].arcs.truncate(valid_arcs);
            self.states[i].niepsilons = nieps;
            self.states[i].noepsilons = noeps;
        }

        if let Some(start_id) = self.start {
            let new_start = newid[start_id.as_usize()];
            if new_start != usize::MAX {
                self.start = Some(A::StateId::from_usize(new_start));
            } else {
                self.start = None;
            }
        }

        self.properties.modify(delete_states_properties);
    }

    fn reserve_states(&mut self, n: usize) {
        self.states.reserve(n);
    }

    fn reserve_arcs(&mut self, state: A::StateId, n: usize) {
        if let Some(s) = self.states.get_mut(state.as_usize()) {
            s.reserve_arcs(n);
        }
    }

    fn set_input_symbols(&mut self, syms: Option<AtomicRc<SymbolTable>>) {
        self.isymbols = syms;
    }

    fn set_output_symbols(&mut self, syms: Option<AtomicRc<SymbolTable>>) {
        self.osymbols = syms;
    }

    fn mutable_input_symbols(&mut self) -> Option<&mut SymbolTable> {
        self.isymbols.as_mut().map(AtomicRc::make_mut)
    }

    fn mutable_output_symbols(&mut self) -> Option<&mut SymbolTable> {
        self.osymbols.as_mut().map(AtomicRc::make_mut)
    }

    fn mutate_arcs<F>(&mut self, state: A::StateId, mut mutator: F)
    where
        F: FnMut(&mut A),
    {
        let Some(s) = self.states.get_mut(state.as_usize()) else {
            return;
        };
        let mut props = self.properties.get();
        for arc in &mut s.arcs {
            let old = arc.clone();
            mutator(arc);
            props = set_arc_properties(props, &old, arc);
        }
        self.properties.set(props);
    }
}

/// An iterator over the state IDs of a `VectorFst`.
pub struct VectorFstStateIter<'a, A: Arc> {
    pos: usize,
    len: usize,
    _marker: std::marker::PhantomData<&'a A>,
}

impl<'a, A: Arc> Iterator for VectorFstStateIter<'a, A> {
    type Item = A::StateId;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.len {
            let id = A::StateId::from_usize(self.pos);
            self.pos += 1;
            Some(id)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a, A: Arc> ExactSizeIterator for VectorFstStateIter<'a, A> {}

impl<A: Arc> ContiguousArcsFst<A> for VectorFst<A> {
    #[inline(always)]
    fn arcs_slice(&self, state: A::StateId) -> &[A] {
        self.states
            .get(state.as_usize())
            .map(|s| s.arcs.as_slice())
            .unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::test_properties::compute_properties;
    use crate::algorithms::test_support::{Rng, random_acyclic_fst};
    use crate::arc::StdArc;
    use crate::properties::{
        K_ACCEPTOR, K_EPSILONS, K_FST_PROPERTIES, K_I_EPSILONS, K_I_LABEL_SORTED, K_NO_I_EPSILONS,
        K_NOT_ACCEPTOR, K_NOT_I_LABEL_SORTED, K_UNWEIGHTED, K_WEIGHTED,
        internal::compat_properties,
    };
    use crate::weights::float_weight::TropicalWeight;
    use std::io::Cursor;

    /// `set_start(no_state())` is how a start state is taken away. Upstream
    /// stores the sentinel and hands it back from `Start()`; the `Option` here
    /// is what tells "no start" from "state -1", so it has to be `None`.
    #[test]
    fn setting_the_start_state_to_the_sentinel_clears_it() {
        let mut fst = StdVectorFst::new();
        fst.add_state();
        fst.add_state();
        fst.set_start(1);
        assert_eq!(fst.start(), Some(1));

        fst.set_start(<i32 as ArcStateId>::no_state());
        assert_eq!(fst.start(), None);
    }

    /// Rewriting arcs has to keep the property bits honest: what the old arcs
    /// justified stops being claimed, and what the new ones force is claimed.
    #[test]
    fn mutating_arcs_keeps_the_property_bits_honest() {
        let mut fst = StdVectorFst::new();
        fst.add_state();
        fst.add_state();
        fst.set_start(0);
        fst.set_final(1, TropicalWeight::one());
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(0, StdArc::new(2, 2, TropicalWeight::one(), 1));

        let props = fst.properties(K_FST_PROPERTIES, false);
        assert_ne!(props & K_ACCEPTOR, 0);
        assert_ne!(props & K_UNWEIGHTED, 0);
        assert_ne!(props & K_NO_I_EPSILONS, 0);

        // Turn the acceptor into a weighted transducer with an input epsilon.
        fst.mutate_arcs(0, |arc| {
            if arc.ilabel() == 1 {
                *arc = StdArc::new(0, 5, TropicalWeight(2.0), arc.nextstate());
            }
        });

        let arcs: Vec<_> = fst.arcs(0).collect();
        assert_eq!((arcs[0].ilabel(), arcs[0].olabel()), (0, 5));
        assert_eq!((arcs[1].ilabel(), arcs[1].olabel()), (2, 2));

        let props = fst.properties(K_FST_PROPERTIES, false);
        assert_eq!(props & K_ACCEPTOR, 0);
        assert_ne!(props & K_NOT_ACCEPTOR, 0);
        assert_ne!(props & K_I_EPSILONS, 0);
        assert_eq!(props & K_NO_I_EPSILONS, 0);
        assert_ne!(props & K_WEIGHTED, 0);
        assert_eq!(props & K_UNWEIGHTED, 0);
        // Neither arc has an epsilon on both sides.
        assert_eq!(props & K_EPSILONS, 0);

        // And what it now says is the truth.
        let scanned = compute_properties(&fst, K_FST_PROPERTIES);
        assert!(compat_properties(props, scanned.props));
    }

    /// Bits an arc rewrite could have broken are dropped to unknown rather than
    /// left claimed. Sortedness is the clearest case: the labels moved.
    #[test]
    fn mutating_arcs_stops_claiming_what_it_can_no_longer_vouch_for() {
        let mut fst = StdVectorFst::new();
        fst.add_state();
        fst.add_state();
        fst.set_start(0);
        fst.set_final(1, TropicalWeight::one());
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(0, StdArc::new(2, 2, TropicalWeight::one(), 1));
        fst.set_properties(K_I_LABEL_SORTED, K_I_LABEL_SORTED);
        assert_ne!(fst.properties(K_I_LABEL_SORTED, false), 0);

        // Reverse the labels, which unsorts them.
        fst.mutate_arcs(0, |arc| {
            *arc = StdArc::new(
                3 - arc.ilabel(),
                3 - arc.olabel(),
                *arc.weight(),
                arc.nextstate(),
            );
        });

        let props = fst.properties(K_FST_PROPERTIES, false);
        assert_eq!(
            props & (K_I_LABEL_SORTED | K_NOT_I_LABEL_SORTED),
            0,
            "sortedness should be unknown, not claimed either way"
        );
        assert!(compat_properties(
            props,
            compute_properties(&fst, K_FST_PROPERTIES).props
        ));
    }

    /// A shared symbol table is copied before it is changed, so the FST that
    /// shared it does not see the change.
    #[test]
    fn changing_a_shared_symbol_table_does_not_reach_the_other_holder() {
        let mut table = SymbolTable::new("input".to_string());
        table.add_symbol("<eps>", 0);
        let shared = AtomicRc::new(table);

        let mut fst = StdVectorFst::new();
        fst.set_input_symbols(Some(AtomicRc::clone(&shared)));
        fst.mutable_input_symbols()
            .expect("a table is attached")
            .add_symbol("a", 1);

        assert_eq!(fst.input_symbols().unwrap().find_symbol(1), Some("a"));
        assert!(shared.find_symbol(1).is_none(), "the original was changed");
    }

    #[test]
    fn there_is_no_mutable_symbol_table_when_there_is_no_table() {
        let mut fst = StdVectorFst::new();
        assert!(fst.mutable_input_symbols().is_none());
        assert!(fst.mutable_output_symbols().is_none());
    }

    /// The exact bytes OpenFst writes for the body of a vector FST file.
    ///
    /// From `tests/oracles/vector-fst-golden.cc`. This format writes each
    /// field in turn, so what is pinned is their order and width: three states,
    /// a final weight and an arc count apiece, then each arc as two labels, a
    /// weight and a destination.
    #[test]
    fn the_body_is_the_bytes_openfst_writes() {
        const GOLDEN: &str = concat!(
            "0000807f01000000000000000100000001000000",
            "0000003f010000000000807f0100000000000000",
            "02000000020000000000c03f0200000000000000",
            "0000000000000000",
        );

        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(0.5), 1));
        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight(1.5), 2));
        fst.set_final(2, TropicalWeight::one());

        let mut bytes = Vec::new();
        StdVectorFst::write_fst(&fst, &mut bytes, &FstWriteOptions::default()).unwrap();

        let header = FstHeader::read(&mut Cursor::new(bytes.clone())).unwrap();
        let header_len = {
            let mut probe = Vec::new();
            header.write(&mut probe).unwrap();
            probe.len()
        };
        let body: String = bytes[header_len..]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(body, GOLDEN);

        assert_eq!(header.fst_type, "vector");
        assert_eq!(header.arc_type, "standard");
        assert_eq!(header.version, FILE_VERSION);
        assert_eq!(header.num_states, 3);
        assert_eq!(header.start, 0);
        // Upstream never sets the arc count for this format.
        assert_eq!(header.num_arcs, 0);
    }

    fn round_trip(fst: &StdVectorFst) -> StdVectorFst {
        let mut bytes = Vec::new();
        StdVectorFst::write_fst(fst, &mut bytes, &FstWriteOptions::default()).unwrap();
        StdVectorFst::read(&mut Cursor::new(bytes), &FstReadOptions::default()).unwrap()
    }

    fn assert_same(got: &StdVectorFst, want: &StdVectorFst) {
        assert_eq!(got.num_states(), want.num_states());
        assert_eq!(got.start(), want.start());
        for s in 0..want.num_states() as i32 {
            assert_eq!(got.final_weight(s), want.final_weight(s), "state {s}");
            assert_eq!(
                got.arcs(s).collect::<Vec<_>>(),
                want.arcs(s).collect::<Vec<_>>(),
                "state {s}"
            );
        }
    }

    #[test]
    fn a_vector_fst_round_trips() {
        let mut rng = Rng::new(0xBEEF_CAFE);
        for _ in 0..100 {
            let fst = random_acyclic_fst(&mut rng, 6);
            assert_same(&round_trip(&fst), &fst);
        }
    }

    #[test]
    fn an_empty_fst_round_trips() {
        let fst = StdVectorFst::new();
        let read = round_trip(&fst);
        assert_eq!(read.num_states(), 0);
        assert_eq!(read.start(), None);
    }

    /// An FST with states but no start state is a normal thing to write; the
    /// start field is the sentinel and has to come back as no start state.
    #[test]
    fn an_fst_with_no_start_state_round_trips() {
        let mut fst = StdVectorFst::new();
        fst.add_state();
        fst.add_state();
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.set_final(1, TropicalWeight::one());

        let read = round_trip(&fst);
        assert_eq!(read.start(), None);
        assert_same(&read, &fst);
    }

    #[test]
    fn symbol_tables_survive_the_round_trip() {
        let mut table = SymbolTable::new("input".to_string());
        table.add_symbol("<eps>", 0);
        table.add_symbol("a", 1);

        let mut fst = StdVectorFst::new();
        fst.add_state();
        fst.set_start(0);
        fst.set_final(0, TropicalWeight::one());
        fst.set_output_symbols(Some(AtomicRc::new(table)));

        let read = round_trip(&fst);
        assert!(read.input_symbols().is_none());
        assert_eq!(read.output_symbols().unwrap().find_symbol(1), Some("a"));
    }

    #[test]
    fn a_file_that_lies_about_its_shape_is_refused() {
        let mut fst = StdVectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.set_final(1, TropicalWeight::one());

        let mut bytes = Vec::new();
        StdVectorFst::write_fst(&fst, &mut bytes, &FstWriteOptions::default()).unwrap();

        let rewrite = |edit: &dyn Fn(&mut FstHeader)| {
            let mut header = FstHeader::read(&mut Cursor::new(bytes.clone())).unwrap();
            let header_len = {
                let mut probe = Vec::new();
                header.write(&mut probe).unwrap();
                probe.len()
            };
            edit(&mut header);
            let mut out = Vec::new();
            header.write(&mut out).unwrap();
            out.extend_from_slice(&bytes[header_len..]);
            StdVectorFst::read(&mut Cursor::new(out), &FstReadOptions::default())
        };

        assert!(rewrite(&|_| {}).is_ok(), "the unedited file reads");
        assert!(
            rewrite(&|h| h.num_states = 5).is_err(),
            "too few states in the file"
        );
        assert!(rewrite(&|h| h.start = 9).is_err(), "start past the end");
        assert!(rewrite(&|h| h.start = -4).is_err(), "negative start");
        assert!(
            rewrite(&|h| h.fst_type = "const".to_string()).is_err(),
            "wrong type"
        );
        assert!(
            rewrite(&|h| h.version = MIN_FILE_VERSION - 1).is_err(),
            "old version"
        );
    }

    /// With `verify`, an arc leading nowhere is caught at read time.
    #[test]
    fn an_arc_leading_outside_the_fst_is_caught_when_verifying() {
        // The arc hangs off the last state, so its destination is the last four
        // bytes of the file.
        let mut fst = StdVectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.set_final(0, TropicalWeight::one());
        fst.add_arc(1, StdArc::new(1, 1, TropicalWeight::one(), 0));

        let mut bytes = Vec::new();
        StdVectorFst::write_fst(&fst, &mut bytes, &FstWriteOptions::default()).unwrap();
        // The last four bytes are the only arc's destination.
        let len = bytes.len();
        bytes[len - 4..].copy_from_slice(&9i32.to_le_bytes());

        assert!(
            StdVectorFst::read(&mut Cursor::new(bytes.clone()), &FstReadOptions::default())
                .is_err()
        );
        let opts = FstReadOptions {
            verify: false,
            ..Default::default()
        };
        assert!(StdVectorFst::read(&mut Cursor::new(bytes), &opts).is_ok());
    }

    /// A corrupt arc count must not be believed to the extent of asking the
    /// allocator for it. One flipped byte in the high half of a count turns 1
    /// into thirty-eight billion, which upstream reserves and dies on.
    #[test]
    fn an_absurd_arc_count_does_not_take_the_process_down() {
        let mut fst = StdVectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.set_final(0, TropicalWeight::one());
        fst.add_arc(1, StdArc::new(1, 1, TropicalWeight::one(), 0));

        let mut bytes = Vec::new();
        StdVectorFst::write_fst(&fst, &mut bytes, &FstWriteOptions::default()).unwrap();

        // State 1's arc count is the eight bytes before its single arc.
        let count_at = bytes.len() - 16 - 8;
        bytes[count_at..count_at + 8].copy_from_slice(&38_654_705_664i64.to_le_bytes());

        // Reading stops when the arcs run out, rather than on the reservation.
        assert!(StdVectorFst::read(&mut Cursor::new(bytes), &FstReadOptions::default()).is_err());
    }

    /// The sentinel is not a state, so an arc naming it is refused whether or
    /// not the reader was asked to verify.
    #[test]
    fn an_arc_to_the_sentinel_state_is_always_refused() {
        let mut fst = StdVectorFst::new();
        fst.add_state();
        fst.set_start(0);
        fst.set_final(0, TropicalWeight::one());
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 0));

        let mut bytes = Vec::new();
        StdVectorFst::write_fst(&fst, &mut bytes, &FstWriteOptions::default()).unwrap();
        let len = bytes.len();
        bytes[len - 4..].copy_from_slice(&(-1i32).to_le_bytes());

        let opts = FstReadOptions {
            verify: false,
            ..Default::default()
        };
        assert!(StdVectorFst::read(&mut Cursor::new(bytes), &opts).is_err());
    }
}
