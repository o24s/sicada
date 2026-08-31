//! An immutable FST whose arcs are stored in a packed form.
//!
//! Port of OpenFst's `compact-fst.h`. Where [`ConstFst`](super::const_fst) keeps
//! whole arcs, this keeps whatever an [`ArcCompactor`] can reduce them to: for
//! an unweighted acceptor, one label and a destination instead of two labels, a
//! weight and a destination. The arcs are expanded on demand and cached, which
//! is why this is a delayed FST rather than a plain array.

use std::fs::File;
use std::io::{Read, Seek, Write};
use std::marker::PhantomData;
use std::mem::size_of;
use std::path::Path;
use std::sync::Arc as StdArc;

use crate::algorithms::test_properties::cached_properties;
use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::cache::{CacheArcIter, CacheExpander, CacheImpl, CacheOptions};
use crate::error::OpenFstError;
use crate::fst::{
    ExpandedFst, FileReadMode, Fst, FstReadOptions, FstWriteOptions, NO_STATE_ID, PropertyCache,
    read_fst_header, write_fst_header,
};
use crate::fst_header::{FstHeader, flags};
use crate::fst_type::FstType;
use crate::properties::*;
use crate::symbol_table::SymbolTable;
use crate::utils::io::{CountingWriter, align_input};
use crate::utils::mapped_file::{ARCH_ALIGNMENT, MappedFile};
use crate::weight::Weight;

/// A trait for extracting numbers from states array for variable out-degree compactors.
pub trait Unsigned: Copy + Default + PartialEq + Eq + PartialOrd + Ord {
    fn from_usize(v: usize) -> Self;
    fn as_usize(&self) -> usize;
    /// Resolves the corresponding FstType statically based on the size of the unsigned integer.
    fn compact_fst_type<A: Arc, C: ArcCompactor<A>>() -> FstType;
}

impl Unsigned for u32 {
    #[inline(always)]
    fn from_usize(v: usize) -> Self {
        v as u32
    }
    #[inline(always)]
    fn as_usize(&self) -> usize {
        *self as usize
    }
    #[inline(always)]
    fn compact_fst_type<A: Arc, C: ArcCompactor<A>>() -> FstType {
        C::COMPACT_TYPE_32
    }
}

impl Unsigned for u64 {
    #[inline(always)]
    fn from_usize(v: usize) -> Self {
        v as u64
    }
    #[inline(always)]
    fn as_usize(&self) -> usize {
        *self as usize
    }
    #[inline(always)]
    fn compact_fst_type<A: Arc, C: ArcCompactor<A>>() -> FstType {
        C::COMPACT_TYPE_64
    }
}

/// The ArcCompactor trait determines how arcs and final weights are compacted and expanded.
pub trait ArcCompactor<A: Arc>: Clone + 'static {
    /// The packed element type in the memory-mapped storage.
    type Element: Copy + Clone;

    /// The static FST type names for 32-bit and 64-bit bounds.
    const COMPACT_TYPE_32: FstType;
    const COMPACT_TYPE_64: FstType;

    /// Compacts a transition or a final weight.
    /// Final weights are treated as transitions with `no_label()` and `no_state()`.
    fn compact(&self, state: A::StateId, arc: &A) -> Self::Element;

    /// Expands a packed element back into an Arc.
    fn expand(&self, state: A::StateId, element: &Self::Element) -> A;

    /// Returns -1 for variable out-degree compactors, and the mandatory
    /// out-degree otherwise (e.g., 1 for String FSTs).
    fn size(&self) -> isize;

    /// Returns the properties that are always true for an FST compacted using this compactor.
    fn properties(&self) -> u64;

    /// Whether `fst` can be stored in this form without losing anything.
    ///
    /// The properties a compactor claims for its output are exactly the ones
    /// the input has to have already: an acceptor compactor keeps one label per
    /// arc, so a transducer put through it comes back with its output side
    /// replaced by its input side. Every compactor here answers the question
    /// the same way, which is why the default is the whole implementation.
    fn is_compatible<F: Fst<A>>(&self, fst: &F) -> bool {
        let props = self.properties();
        fst.properties(props, true) == props
    }

    /// Writes whatever the compactor itself needs to record in a file.
    ///
    /// The compactors here are stateless, so this writes nothing; a compactor
    /// carrying a table of its own would put it here, as upstream's
    /// `ArcCompactor::Write` does.
    fn write<W: std::io::Write>(&self, _writer: &mut W) -> Result<(), OpenFstError> {
        Ok(())
    }

    /// Reads back what [`write`](Self::write) recorded.
    fn read<R: std::io::Read>(_reader: &mut R) -> Result<Self, OpenFstError>
    where
        Self: Default,
    {
        Ok(Self::default())
    }
}

/// Provides memory-efficient storage for compacted FSTs.
pub struct CompactArcStore<'a, A: Arc, C: ArcCompactor<A>, U: Unsigned> {
    states_region: Option<MappedFile<'a>>,
    compacts_region: MappedFile<'a>,
    nstates: usize,
    ncompacts: usize,
    narcs: usize,
    start: Option<A::StateId>,
    arc_compactor: C,
    _marker: PhantomData<(A, U)>,
}

impl<'a, A: Arc, C: ArcCompactor<A>, U: Unsigned> CompactArcStore<'a, A, C, U> {
    pub fn new<F: Fst<A>>(fst: &F, arc_compactor: C) -> Result<Self, OpenFstError> {
        let start = fst.start();
        let mut nstates = 0;
        let mut nfinals = 0;
        let mut narcs = 0;

        for s in fst.states() {
            nstates += 1;
            narcs += fst.num_arcs(s);
            if fst.final_weight(s) != A::Weight::zero() {
                nfinals += 1;
            }
        }

        let is_variable = arc_compactor.size() == -1;

        let (states_region, mut states_slice) = if is_variable {
            let mut region = MappedFile::allocate_type::<U>(nstates + 1)?;
            let slice = unsafe {
                std::slice::from_raw_parts_mut(
                    region.as_mut_slice().unwrap().as_mut_ptr() as *mut U,
                    nstates + 1,
                )
            };
            (Some(region), Some(slice))
        } else {
            (None, None)
        };

        let ncompacts = if is_variable {
            narcs + nfinals
        } else {
            let size = arc_compactor.size() as usize;
            let expected_compacts = nstates * size;
            if (narcs + nfinals) != expected_compacts {
                return Err(OpenFstError::InvalidOperation(
                    "CompactArcStore: ArcCompactor incompatible with FST".into(),
                ));
            }
            expected_compacts
        };

        let mut compacts_region = MappedFile::allocate_type::<C::Element>(ncompacts)?;
        let compacts_slice = unsafe {
            std::slice::from_raw_parts_mut(
                compacts_region.as_mut_slice().unwrap().as_mut_ptr() as *mut C::Element,
                ncompacts,
            )
        };

        if let Some(states) = states_slice.as_mut() {
            states[nstates] = U::from_usize(ncompacts);
        }

        let mut pos = 0;
        for (s_idx, s) in fst.states().enumerate() {
            let fpos = pos;
            if let Some(states) = states_slice.as_deref_mut() {
                states[s_idx] = U::from_usize(pos);
            }

            let fin = fst.final_weight(s);
            if fin != A::Weight::zero() {
                compacts_slice[pos] = arc_compactor.compact(
                    s,
                    &A::new(
                        A::Label::no_label(),
                        A::Label::no_label(),
                        fin,
                        A::StateId::no_state(),
                    ),
                );
                pos += 1;
            }

            for arc in fst.arcs(s) {
                compacts_slice[pos] = arc_compactor.compact(s, &arc);
                pos += 1;
            }

            if !is_variable && pos != fpos + arc_compactor.size() as usize {
                return Err(OpenFstError::InvalidOperation(
                    "CompactArcStore: ArcCompactor incompatible with FST".into(),
                ));
            }
        }

        if pos != ncompacts {
            return Err(OpenFstError::InvalidOperation(
                "CompactArcStore: ArcCompactor incompatible with FST".into(),
            ));
        }

        Ok(Self {
            states_region,
            compacts_region,
            nstates,
            ncompacts,
            narcs,
            start,
            arc_compactor,
            _marker: PhantomData,
        })
    }

    #[inline(always)]
    pub fn num_arcs(&self) -> usize {
        self.narcs
    }

    #[inline(always)]
    fn states_slice(&self) -> &[U] {
        if let Some(region) = &self.states_region {
            unsafe {
                std::slice::from_raw_parts(region.as_ref().as_ptr() as *const U, self.nstates + 1)
            }
        } else {
            &[]
        }
    }

    #[inline(always)]
    fn compacts_slice(&self) -> &[C::Element] {
        unsafe {
            std::slice::from_raw_parts(
                self.compacts_region.as_ref().as_ptr() as *const C::Element,
                self.ncompacts,
            )
        }
    }

    #[inline(always)]
    fn compacts_range(&self, state: A::StateId) -> (usize, usize) {
        let s = state.as_usize();
        if self.arc_compactor.size() == -1 {
            let states = self.states_slice();
            let start = states[s].as_usize();
            let len = states[s + 1].as_usize() - start;
            (start, len)
        } else {
            let size = self.arc_compactor.size() as usize;
            let start = s * size;
            (start, size)
        }
    }
}

pub struct CompactFst<'a, A: Arc, C: ArcCompactor<A>, U: Unsigned> {
    store: CompactArcStore<'a, A, C, U>,
    cache: CacheImpl<A>,
    properties: PropertyCache,
    input_symbols: Option<StdArc<SymbolTable>>,
    output_symbols: Option<StdArc<SymbolTable>>,
}

impl<'a, A: Arc, C: ArcCompactor<A>, U: Unsigned> CompactFst<'a, A, C, U> {
    /// Compacts `fst`.
    ///
    /// SICADA-DIVERGE: upstream reports an FST the compactor cannot represent
    /// by setting `kError` on the result and carrying on, so the caller gets a
    /// `CompactFst` that says something different from what it was given
    /// without reporting anything: a transducer compacted as an acceptor comes
    /// back with its output labels replaced by its input labels. Here it is an
    /// error.
    pub fn new<F: Fst<A>>(fst: &F, compactor: C, opts: CacheOptions) -> Result<Self, OpenFstError> {
        if !compactor.is_compatible(fst) {
            return Err(OpenFstError::InvalidOperation(format!(
                "CompactFst: the FST is not {}",
                missing_properties(
                    compactor.properties(),
                    fst.properties(compactor.properties(), true)
                )
            )));
        }
        let store = CompactArcStore::new(fst, compactor.clone())?;

        let mut properties = fst.properties(K_COPY_PROPERTIES, false);
        properties &= !K_WEIGHTED_CYCLES & !K_UNWEIGHTED_CYCLES;
        properties |= compactor.properties() | K_EXPANDED;

        Ok(Self {
            store,
            cache: CacheImpl::new(opts),
            properties: PropertyCache::new(properties),
            input_symbols: fst.input_symbols(),
            output_symbols: fst.output_symbols(),
        })
    }
}

/// Names the properties `needed` asks for that `have` does not carry.
fn missing_properties(needed: u64, have: u64) -> String {
    let missing = needed & !have;
    let names: Vec<&str> = (0..u64::BITS as usize)
        .filter(|bit| missing & (1u64 << bit) != 0)
        .map(property_name)
        .filter(|name| !name.is_empty())
        .collect();
    if names.is_empty() {
        format!("compatible with this compactor (missing {missing:#x})")
    } else {
        names.join(", ")
    }
}

/// Version written for a file whose regions are aligned.
const ALIGNED_FILE_VERSION: i32 = 1;
/// Version written for a file whose regions are not aligned.
const FILE_VERSION: i32 = 2;
/// The oldest version that can still be read.
const MIN_FILE_VERSION: i32 = 1;
/// Refuses a header claiming more states than could be stored.
const MAX_STATES: i64 = 0x0010_0000_0000_0000;
/// As [`MAX_STATES`], for arcs.
const MAX_ARCS: i64 = 0x0010_0000_0000_0000;
/// The properties every `CompactFst` has, whatever it holds.
const K_STATIC_PROPERTIES: u64 = K_EXPANDED;

impl<A: Arc, C: ArcCompactor<A> + Default, U: Unsigned> CompactFst<'static, A, C, U> {
    /// Reads a `CompactFst` from a stream.
    pub fn read<R: Read + Seek>(
        reader: &mut R,
        opts: &FstReadOptions,
    ) -> Result<Self, OpenFstError> {
        Self::read_regions(reader, opts, |reader, aligned, size| {
            if aligned {
                align_input(reader, ARCH_ALIGNMENT as u64)?;
            }
            let mut region = MappedFile::allocate(size, ARCH_ALIGNMENT)?;
            reader.read_exact(
                region
                    .as_mut_slice()
                    .expect("a freshly allocated region is writable"),
            )?;
            Ok(region)
        })
    }

    /// Reads a `CompactFst` from a file, mapping its regions where it can.
    pub fn read_from_file(
        path: impl AsRef<Path>,
        opts: &FstReadOptions,
    ) -> Result<Self, OpenFstError> {
        let mut file = File::open(path.as_ref())?;
        let opts = FstReadOptions {
            source: path.as_ref().display().to_string(),
            ..opts.clone()
        };
        let memorymap = opts.mode == FileReadMode::Map;
        Self::read_regions(&mut file, &opts, |file, aligned, size| {
            if aligned {
                align_input(file, ARCH_ALIGNMENT as u64)?;
            }
            Ok(MappedFile::map_or_read(file, memorymap, size)?)
        })
    }

    fn read_regions<R, F>(
        reader: &mut R,
        opts: &FstReadOptions,
        mut region: F,
    ) -> Result<Self, OpenFstError>
    where
        R: Read + Seek,
        F: FnMut(&mut R, bool, usize) -> Result<MappedFile<'static>, OpenFstError>,
    {
        let read = read_fst_header::<A, _>(
            reader,
            opts,
            U::compact_fst_type::<A, C>().as_str(),
            MIN_FILE_VERSION,
        )?;
        let header = read.header;
        let bad =
            |message: String| OpenFstError::InvalidFstHeader(format!("{}: {message}", opts.source));

        if header.num_states < 0 || header.num_states > MAX_STATES {
            return Err(bad(format!(
                "invalid number of states: {}",
                header.num_states
            )));
        }
        if header.num_arcs < 0 || header.num_arcs > MAX_ARCS {
            return Err(bad(format!("invalid number of arcs: {}", header.num_arcs)));
        }
        let nstates = header.num_states as usize;
        let narcs = header.num_arcs as usize;

        let start = if header.start == NO_STATE_ID as i64 {
            None
        } else {
            if header.start < 0 || header.start >= header.num_states {
                return Err(bad(format!(
                    "invalid start state {} for an FST with {nstates} states",
                    header.start
                )));
            }
            Some(A::StateId::from_usize(header.start as usize))
        };

        // The compactor's own data comes before the regions, since reading them
        // needs to know how big an element is.
        let arc_compactor = C::read(reader)?;
        let aligned =
            header.version == ALIGNED_FILE_VERSION || header.flags & flags::IS_ALIGNED != 0;
        let is_variable = arc_compactor.size() == -1;

        let states_region = if is_variable {
            Some(region(
                reader,
                aligned,
                (nstates + 1) * std::mem::size_of::<U>(),
            )?)
        } else {
            None
        };

        let ncompacts = match &states_region {
            // The last entry of the states array is the total, which is how a
            // variable-size compactor says how many elements follow.
            Some(states) => {
                let slice = unsafe {
                    std::slice::from_raw_parts(states.as_ref().as_ptr() as *const U, nstates + 1)
                };
                slice[nstates].as_usize()
            }
            None => nstates * arc_compactor.size() as usize,
        };
        if ncompacts > MAX_ARCS as usize {
            return Err(bad(format!("invalid number of elements: {ncompacts}")));
        }

        let compacts_region = region(
            reader,
            aligned,
            ncompacts * std::mem::size_of::<C::Element>(),
        )?;

        let store = CompactArcStore {
            states_region,
            compacts_region,
            nstates,
            ncompacts,
            narcs,
            start,
            arc_compactor,
            _marker: PhantomData,
        };

        // Each state names a range of the element array, and a range past its
        // end would be read out of bounds by every later access.
        if is_variable {
            let states: &[U] = store.states_slice();
            for s in 0..nstates {
                if states[s].as_usize() > ncompacts
                    || states[s].as_usize() > states[s + 1].as_usize()
                {
                    return Err(bad(format!("state {s} element range out of bounds")));
                }
            }
        }

        Ok(Self {
            store,
            cache: CacheImpl::new(CacheOptions::default()),
            properties: PropertyCache::new(header.properties | K_STATIC_PROPERTIES),
            input_symbols: read.isymbols,
            output_symbols: read.osymbols,
        })
    }
}

impl<A: Arc, C: ArcCompactor<A>, U: Unsigned> CompactFst<'_, A, C, U> {
    /// Writes this FST out.
    ///
    /// The element and state arrays go out as blocks of memory, exactly as
    /// [`ConstFst`](crate::fsts::const_fst::ConstFst)'s do, so the layout of
    /// the compactor's element type is part of the file format.
    pub fn write<W: Write>(
        &self,
        writer: &mut W,
        opts: &FstWriteOptions,
    ) -> Result<(), OpenFstError> {
        let header = FstHeader {
            fst_type: U::compact_fst_type::<A, C>().as_str().to_string(),
            arc_type: A::type_name().as_str().to_string(),
            version: if opts.align {
                ALIGNED_FILE_VERSION
            } else {
                FILE_VERSION
            },
            flags: 0,
            properties: self.properties.get() | K_STATIC_PROPERTIES,
            start: self
                .store
                .start
                .map_or(NO_STATE_ID as i64, |s| s.as_usize() as i64),
            num_states: self.store.nstates as i64,
            num_arcs: self.store.narcs as i64,
        };
        let mut writer = CountingWriter::new(writer, 0);
        write_fst_header(
            &mut writer,
            opts,
            &header,
            self.input_symbols.as_deref(),
            self.output_symbols.as_deref(),
        )?;

        self.store.arc_compactor.write(&mut writer)?;
        if let Some(states) = &self.store.states_region {
            if opts.align {
                writer.align(ARCH_ALIGNMENT as u64)?;
            }
            writer.write_all(&states.as_ref()[..(self.store.nstates + 1) * size_of::<U>()])?;
        }
        if opts.align {
            writer.align(ARCH_ALIGNMENT as u64)?;
        }
        writer.write_all(
            &self.store.compacts_region.as_ref()[..self.store.ncompacts * size_of::<C::Element>()],
        )?;
        Ok(())
    }
}

impl<'a, A: Arc, C: ArcCompactor<A>, U: Unsigned> CacheExpander<A> for CompactFst<'a, A, C, U> {
    fn expand_final(&self, state: A::StateId) -> Option<A::Weight> {
        let (start, len) = self.store.compacts_range(state);
        if len > 0 {
            let compacts = self.store.compacts_slice();
            let element = &compacts[start];
            let arc = self.store.arc_compactor.expand(state, element);
            if arc.ilabel() == A::Label::no_label() {
                return Some(arc.weight().clone());
            }
        }
        None
    }

    fn expand_arcs(&self, state: A::StateId) -> Vec<A> {
        let (start, len) = self.store.compacts_range(state);
        if len == 0 {
            return Vec::new();
        }

        let compacts = self.store.compacts_slice();
        let mut arcs = Vec::with_capacity(len);

        let mut offset = start;
        // Check if first element is a final weight
        let first_arc = self.store.arc_compactor.expand(state, &compacts[offset]);
        if first_arc.ilabel() == A::Label::no_label() {
            offset += 1;
        }

        for element in &compacts[offset..(start + len)] {
            arcs.push(self.store.arc_compactor.expand(state, element));
        }

        arcs
    }
}

impl<'a, A: Arc, C: ArcCompactor<A>, U: Unsigned> Fst<A> for CompactFst<'a, A, C, U> {
    type StateIter<'b>
        = std::iter::Map<std::ops::Range<usize>, fn(usize) -> A::StateId>
    where
        Self: 'b;
    type ArcIter<'b>
        = CacheArcIter<A>
    where
        Self: 'b;

    fn start(&self) -> Option<A::StateId> {
        self.store.start
    }

    fn final_weight(&self, state: A::StateId) -> A::Weight {
        self.cache.final_weight(state, self)
    }

    fn num_arcs(&self, state: A::StateId) -> usize {
        self.cache.num_arcs(state, self)
    }

    fn num_input_epsilons(&self, state: A::StateId) -> usize {
        self.cache.num_input_epsilons(state, self)
    }

    fn num_output_epsilons(&self, state: A::StateId) -> usize {
        self.cache.num_output_epsilons(state, self)
    }

    fn num_states_if_known(&self) -> Option<usize> {
        Some(self.store.nstates)
    }

    fn properties(&self, mask: u64, test: bool) -> u64 {
        cached_properties(self, &self.properties, mask, test)
    }

    #[inline(always)]
    fn fst_type(&self) -> &str {
        U::compact_fst_type::<A, C>().as_str()
    }

    fn input_symbols(&self) -> Option<StdArc<SymbolTable>> {
        self.input_symbols.clone()
    }

    fn output_symbols(&self) -> Option<StdArc<SymbolTable>> {
        self.output_symbols.clone()
    }

    fn states<'b>(&'b self) -> Self::StateIter<'b> {
        (0..self.store.nstates).map(A::StateId::from_usize)
    }

    fn arcs<'b>(&'b self, state: A::StateId) -> Self::ArcIter<'b> {
        self.cache.arcs_iter(state, self)
    }
}

impl<'a, A: Arc, C: ArcCompactor<A>, U: Unsigned> ExpandedFst<A> for CompactFst<'a, A, C, U> {
    fn num_states(&self) -> usize {
        self.store.nstates
    }
}

#[derive(Clone)]
pub struct StringCompactor<A: Arc> {
    _marker: PhantomData<A>,
}

impl<A: Arc> Default for StringCompactor<A> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<A: Arc + 'static> ArcCompactor<A> for StringCompactor<A> {
    type Element = A::Label;

    const COMPACT_TYPE_32: FstType = FstType::COMPACT_STRING_32;
    const COMPACT_TYPE_64: FstType = FstType::COMPACT_STRING_64;

    fn compact(&self, _state: A::StateId, arc: &A) -> Self::Element {
        arc.ilabel()
    }

    fn expand(&self, s: A::StateId, element: &Self::Element) -> A {
        let p = *element;
        let next = if p != A::Label::no_label() {
            A::StateId::from_usize(s.as_usize() + 1)
        } else {
            A::StateId::no_state()
        };
        A::new(p, p, A::Weight::one(), next)
    }

    fn size(&self) -> isize {
        1
    }

    fn properties(&self) -> u64 {
        K_COMPILED_STRING_PROPERTIES
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct WeightedStringElement<L, W> {
    label: L,
    weight: W,
}

#[derive(Clone)]
pub struct WeightedStringCompactor<A: Arc> {
    _marker: PhantomData<A>,
}

impl<A: Arc> Default for WeightedStringCompactor<A> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<A: Arc + 'static> ArcCompactor<A> for WeightedStringCompactor<A>
where
    A::Weight: Copy,
{
    type Element = WeightedStringElement<A::Label, A::Weight>;

    const COMPACT_TYPE_32: FstType = FstType::COMPACT_WEIGHTED_STRING_32;
    const COMPACT_TYPE_64: FstType = FstType::COMPACT_WEIGHTED_STRING_64;

    fn compact(&self, _state: A::StateId, arc: &A) -> Self::Element {
        WeightedStringElement {
            label: arc.ilabel(),
            weight: *arc.weight(),
        }
    }

    fn expand(&self, s: A::StateId, element: &Self::Element) -> A {
        let next = if element.label != A::Label::no_label() {
            A::StateId::from_usize(s.as_usize() + 1)
        } else {
            A::StateId::no_state()
        };
        A::new(element.label, element.label, element.weight, next)
    }

    fn size(&self) -> isize {
        1
    }

    fn properties(&self) -> u64 {
        K_STRING | K_ACCEPTOR
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct AcceptorElement<L, W, S> {
    label: L,
    weight: W,
    nextstate: S,
}

#[derive(Clone)]
pub struct AcceptorCompactor<A: Arc> {
    _marker: PhantomData<A>,
}

impl<A: Arc> Default for AcceptorCompactor<A> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<A: Arc + 'static> ArcCompactor<A> for AcceptorCompactor<A>
where
    A::Weight: Copy,
{
    type Element = AcceptorElement<A::Label, A::Weight, A::StateId>;

    const COMPACT_TYPE_32: FstType = FstType::COMPACT_ACCEPTOR_32;
    const COMPACT_TYPE_64: FstType = FstType::COMPACT_ACCEPTOR_64;

    fn compact(&self, _state: A::StateId, arc: &A) -> Self::Element {
        AcceptorElement {
            label: arc.ilabel(),
            weight: *arc.weight(),
            nextstate: arc.nextstate(),
        }
    }

    fn expand(&self, _s: A::StateId, element: &Self::Element) -> A {
        A::new(
            element.label,
            element.label,
            element.weight,
            element.nextstate,
        )
    }

    fn size(&self) -> isize {
        -1
    }

    fn properties(&self) -> u64 {
        K_ACCEPTOR
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct UnweightedAcceptorElement<L, S> {
    label: L,
    nextstate: S,
}

#[derive(Clone)]
pub struct UnweightedAcceptorCompactor<A: Arc> {
    _marker: PhantomData<A>,
}

impl<A: Arc> Default for UnweightedAcceptorCompactor<A> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<A: Arc + 'static> ArcCompactor<A> for UnweightedAcceptorCompactor<A> {
    type Element = UnweightedAcceptorElement<A::Label, A::StateId>;

    const COMPACT_TYPE_32: FstType = FstType::COMPACT_UNWEIGHTED_ACCEPTOR_32;
    const COMPACT_TYPE_64: FstType = FstType::COMPACT_UNWEIGHTED_ACCEPTOR_64;

    fn compact(&self, _state: A::StateId, arc: &A) -> Self::Element {
        UnweightedAcceptorElement {
            label: arc.ilabel(),
            nextstate: arc.nextstate(),
        }
    }

    fn expand(&self, _s: A::StateId, element: &Self::Element) -> A {
        A::new(
            element.label,
            element.label,
            A::Weight::one(),
            element.nextstate,
        )
    }

    fn size(&self) -> isize {
        -1
    }

    fn properties(&self) -> u64 {
        K_ACCEPTOR | K_UNWEIGHTED
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct UnweightedElement<L, S> {
    ilabel: L,
    olabel: L,
    nextstate: S,
}

#[derive(Clone)]
pub struct UnweightedCompactor<A: Arc> {
    _marker: PhantomData<A>,
}

impl<A: Arc> Default for UnweightedCompactor<A> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<A: Arc + 'static> ArcCompactor<A> for UnweightedCompactor<A> {
    type Element = UnweightedElement<A::Label, A::StateId>;

    const COMPACT_TYPE_32: FstType = FstType::COMPACT_UNWEIGHTED_32;
    const COMPACT_TYPE_64: FstType = FstType::COMPACT_UNWEIGHTED_64;

    fn compact(&self, _state: A::StateId, arc: &A) -> Self::Element {
        UnweightedElement {
            ilabel: arc.ilabel(),
            olabel: arc.olabel(),
            nextstate: arc.nextstate(),
        }
    }

    fn expand(&self, _s: A::StateId, element: &Self::Element) -> A {
        A::new(
            element.ilabel,
            element.olabel,
            A::Weight::one(),
            element.nextstate,
        )
    }

    fn size(&self) -> isize {
        -1
    }

    fn properties(&self) -> u64 {
        K_UNWEIGHTED
    }
}

// -----------------------------------------------------------------------------
// Type Aliases
// -----------------------------------------------------------------------------

pub type CompactStringFst<'a, A, U = u32> = CompactFst<'a, A, StringCompactor<A>, U>;
pub type CompactWeightedStringFst<'a, A, U = u32> =
    CompactFst<'a, A, WeightedStringCompactor<A>, U>;
pub type CompactAcceptorFst<'a, A, U = u32> = CompactFst<'a, A, AcceptorCompactor<A>, U>;
pub type CompactUnweightedFst<'a, A, U = u32> = CompactFst<'a, A, UnweightedCompactor<A>, U>;
pub type CompactUnweightedAcceptorFst<'a, A, U = u32> =
    CompactFst<'a, A, UnweightedAcceptorCompactor<A>, U>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::fst::MutableFst;
    use crate::fst::{FstReadOptions, FstWriteOptions};
    use crate::fst_header::FstHeader;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::weights::float_weight::TropicalWeight;
    use std::io::Cursor;

    /// A chain 0 → 1 → 2, which every compactor here can represent.
    fn build_string_fst() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(5, 5, TropicalWeight::one(), 1));
        fst.add_arc(1, StdArc::new(7, 7, TropicalWeight::one(), 2));
        fst.set_final(2, TropicalWeight::one());
        fst
    }

    /// Everything a caller can observe, compared against the FST it was built
    /// from.
    fn assert_same<C: ArcCompactor<StdArc>>(
        got: &CompactFst<'_, StdArc, C, u32>,
        want: &StdVectorFst,
    ) {
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
    fn test_compact_string() {
        let mut fst = StdVectorFst::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, StdArc::new(5, 5, TropicalWeight::one(), s1));

        let compactor = StringCompactor::default();
        let compact_fst: CompactStringFst<StdArc> =
            CompactFst::new(&fst, compactor, CacheOptions::default()).unwrap();

        assert_eq!(compact_fst.fst_type(), "compact_string");
        assert_eq!(compact_fst.num_states(), 2);
        assert_eq!(compact_fst.num_arcs(s0), 1);

        let mut arcs = compact_fst.arcs(s0);
        let arc = arcs.next().unwrap();
        assert_eq!(arc.ilabel(), 5);
        assert_eq!(arc.nextstate(), s1);
    }

    #[test]
    fn test_compact_string_64() {
        let mut fst = StdVectorFst::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, StdArc::new(5, 5, TropicalWeight::one(), s1));

        let compactor = StringCompactor::default();
        let compact_fst: CompactStringFst<StdArc, u64> =
            CompactFst::new(&fst, compactor, CacheOptions::default()).unwrap();

        assert_eq!(compact_fst.fst_type(), "compact64_string");
    }

    #[test]
    fn test_compact_acceptor() {
        let mut fst = StdVectorFst::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight(2.0));
        fst.add_arc(s0, StdArc::new(5, 5, TropicalWeight(1.5), s1));

        let compactor = AcceptorCompactor::default();
        let compact_fst: CompactAcceptorFst<StdArc> =
            CompactFst::new(&fst, compactor, CacheOptions::default()).unwrap();

        assert_eq!(compact_fst.fst_type(), "compact_acceptor");
        assert_eq!(compact_fst.final_weight(s1), TropicalWeight(2.0));

        let mut arcs = compact_fst.arcs(s0);
        let arc = arcs.next().unwrap();
        assert_eq!(arc.ilabel(), 5);
        assert_eq!(arc.weight(), &TropicalWeight(1.5));
    }

    /// A round trip through the format, for each compactor.
    ///
    /// The element and state arrays go to disk as blocks of memory, so this is
    /// also what checks that the packed layouts survive.
    #[test]
    fn a_compact_fst_round_trips() {
        let vfst = build_string_fst();

        // The string compactor: one element per state, no states array.
        let compact = CompactStringFst::<StdArc>::new(
            &vfst,
            StringCompactor::default(),
            CacheOptions::default(),
        )
        .unwrap();
        let mut bytes = Vec::new();
        compact
            .write(&mut bytes, &FstWriteOptions::default())
            .unwrap();
        let read =
            CompactStringFst::<StdArc>::read(&mut Cursor::new(bytes), &FstReadOptions::default())
                .unwrap();
        assert_same(&read, &vfst);

        // The unweighted compactor: variable size, so a states array is written.
        let compact = CompactUnweightedFst::<StdArc>::new(
            &vfst,
            UnweightedCompactor::default(),
            CacheOptions::default(),
        )
        .unwrap();
        let mut bytes = Vec::new();
        compact
            .write(&mut bytes, &FstWriteOptions::default())
            .unwrap();
        let read = CompactUnweightedFst::<StdArc>::read(
            &mut Cursor::new(bytes),
            &FstReadOptions::default(),
        )
        .unwrap();
        assert_same(&read, &vfst);
    }

    /// Aligning pads the regions so a reader can map them.
    #[test]
    fn an_aligned_compact_fst_round_trips() {
        let vfst = build_string_fst();
        let compact = CompactUnweightedFst::<StdArc>::new(
            &vfst,
            UnweightedCompactor::default(),
            CacheOptions::default(),
        )
        .unwrap();

        let opts = FstWriteOptions {
            align: true,
            ..Default::default()
        };
        let mut aligned = Vec::new();
        compact.write(&mut aligned, &opts).unwrap();
        let mut plain = Vec::new();
        compact
            .write(&mut plain, &FstWriteOptions::default())
            .unwrap();
        assert!(aligned.len() > plain.len(), "no padding was written");

        let header = FstHeader::read(&mut Cursor::new(aligned.clone())).unwrap();
        assert_eq!(header.version, ALIGNED_FILE_VERSION);
        let read = CompactUnweightedFst::<StdArc>::read(
            &mut Cursor::new(aligned),
            &FstReadOptions::default(),
        )
        .unwrap();
        assert_same(&read, &vfst);
    }

    #[test]
    fn symbol_tables_survive_the_round_trip() {
        let mut table = SymbolTable::new("input".to_string());
        table.add_symbol("<eps>", 0);
        table.add_symbol("a", 1);

        let mut vfst = build_string_fst();
        vfst.set_input_symbols(Some(crate::AtomicRc::new(table)));

        let compact = CompactUnweightedFst::<StdArc>::new(
            &vfst,
            UnweightedCompactor::default(),
            CacheOptions::default(),
        )
        .unwrap();
        let mut bytes = Vec::new();
        compact
            .write(&mut bytes, &FstWriteOptions::default())
            .unwrap();
        let read = CompactUnweightedFst::<StdArc>::read(
            &mut Cursor::new(bytes),
            &FstReadOptions::default(),
        )
        .unwrap();

        assert_eq!(read.input_symbols().unwrap().find_symbol(1), Some("a"));
        assert!(read.output_symbols().is_none());
        assert_same(&read, &vfst);
    }

    #[test]
    fn a_file_that_lies_about_its_shape_is_refused() {
        let vfst = build_string_fst();
        let compact = CompactUnweightedFst::<StdArc>::new(
            &vfst,
            UnweightedCompactor::default(),
            CacheOptions::default(),
        )
        .unwrap();
        let mut bytes = Vec::new();
        compact
            .write(&mut bytes, &FstWriteOptions::default())
            .unwrap();

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
            CompactUnweightedFst::<StdArc>::read(&mut Cursor::new(out), &FstReadOptions::default())
        };

        assert!(rewrite(&|_| {}).is_ok(), "the unedited file reads");
        assert!(rewrite(&|h| h.num_states = -1).is_err(), "negative states");
        assert!(rewrite(&|h| h.start = 99).is_err(), "start past the end");
        assert!(rewrite(&|h| h.start = -9).is_err(), "negative start");
        assert!(
            rewrite(&|h| h.fst_type = "vector".to_string()).is_err(),
            "wrong type"
        );
        assert!(
            rewrite(&|h| h.version = MIN_FILE_VERSION - 1).is_err(),
            "old version"
        );
    }

    /// A compactor keeps only what its properties promise, so an FST without
    /// those properties would come back saying something else. Upstream sets
    /// `kError` and hands the wrong FST over anyway.
    #[test]
    fn an_fst_the_compactor_cannot_represent_is_refused() {
        use crate::arc::StdArc;
        use crate::fst::MutableFst;
        use crate::fsts::vector_fst::VectorFst;
        use crate::weight::Weight;
        use crate::weights::float_weight::TropicalWeight;

        let mut transducer: VectorFst<StdArc> = VectorFst::new();
        for _ in 0..2 {
            transducer.add_state();
        }
        transducer.set_start(0);
        transducer.add_arc(0, StdArc::new(1, 2, TropicalWeight(0.5), 1));
        transducer.set_final(1, TropicalWeight::one());

        // An acceptor compactor stores one label per arc; the output side of
        // this transducer has nowhere to go.
        let Err(err) = CompactAcceptorFst::<StdArc, u32>::new(
            &transducer,
            AcceptorCompactor::default(),
            CacheOptions::default(),
        ) else {
            panic!("a transducer must not compact as an acceptor")
        };
        assert!(format!("{err}").contains("acceptor"), "{err}");

        // A weight compactor is refused for the same reason.
        let Err(err) = CompactUnweightedFst::<StdArc, u32>::new(
            &transducer,
            UnweightedCompactor::default(),
            CacheOptions::default(),
        ) else {
            panic!("a weighted FST must not compact as an unweighted one")
        };
        assert!(format!("{err}").contains("unweighted"), "{err}");

        // Make it an unweighted acceptor and both are accepted.
        let mut acceptor: VectorFst<StdArc> = VectorFst::new();
        for _ in 0..2 {
            acceptor.add_state();
        }
        acceptor.set_start(0);
        acceptor.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        acceptor.set_final(1, TropicalWeight::one());
        assert!(
            CompactAcceptorFst::<StdArc, u32>::new(
                &acceptor,
                AcceptorCompactor::default(),
                CacheOptions::default()
            )
            .is_ok()
        );
        assert!(
            CompactUnweightedFst::<StdArc, u32>::new(
                &acceptor,
                UnweightedCompactor::default(),
                CacheOptions::default()
            )
            .is_ok()
        );
    }
}
