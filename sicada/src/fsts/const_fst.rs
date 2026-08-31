//! An immutable FST held in two contiguous blocks.
//!
//! Port of OpenFst's `const-fst.h`. The states and the arcs each live in one
//! block, which lets a file be mapped rather than parsed: the bytes on disk are
//! the bytes in memory. The layout of what goes in those blocks is
//! therefore part of the file format; see [`ConstState`] and
//! [`ArcTpl`](crate::arc::ArcTpl).

use std::fmt::Debug;
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::marker::PhantomData;
use std::path::Path;

use crate::AtomicRc;
use crate::algorithms::test_properties::{cached_properties, check_properties};
use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::error::OpenFstError;
use crate::fst::{
    ContiguousArcsFst, ExpandedFst, FileReadMode, Fst, FstReadOptions, FstWriteOptions,
    NO_STATE_ID, PropertyCache, read_fst_header, write_fst_header,
};
use crate::fst_header::{FstHeader, flags};
use crate::fst_type::FstType;
use crate::fsts::compact_fst::Unsigned;
use crate::properties::{
    K_COPY_PROPERTIES, K_EXPANDED, K_MUTABLE, K_UNWEIGHTED_CYCLES, K_WEIGHTED_CYCLES,
};
use crate::symbol_table::SymbolTable;
use crate::utils::io::{CountingWriter, align_input};
use crate::utils::mapped_file::{ARCH_ALIGNMENT, MappedFile};
use crate::weight::Weight as _;

/// The properties every `ConstFst` has, whatever it holds.
const K_STATIC_PROPERTIES: u64 = K_EXPANDED;

/// Internal representation of a state in a ConstFst.
/// Note: Must be `#[repr(C)]` because it's directly mapped from byte arrays.
#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
pub struct ConstState<W, U> {
    pub final_weight: W,
    pub pos: U,
    pub narcs: U,
    pub niepsilons: U,
    pub noepsilons: U,
}

impl<W: crate::weight::Weight, U: Unsigned> Default for ConstState<W, U> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            final_weight: W::zero(),
            pos: U::default(),
            narcs: U::default(),
            niepsilons: U::default(),
            noepsilons: U::default(),
        }
    }
}

/// Simple concrete immutable FST whose states and arcs are each stored in single contiguous memory regions.
/// Backed by `MappedFile` to allow true zero-copy loading from disk.
pub struct ConstFst<'a, A: Arc, U: Unsigned> {
    states_region: MappedFile<'a>,
    arcs_region: MappedFile<'a>,
    nstates: usize,
    start: Option<A::StateId>,
    properties: PropertyCache,
    input_symbols: Option<AtomicRc<SymbolTable>>,
    output_symbols: Option<AtomicRc<SymbolTable>>,
    _marker: PhantomData<(A, U)>,
}

impl<'a, A: Arc, U: Unsigned> ConstFst<'a, A, U> {
    /// Creates a ConstFst from any generic Fst by fully expanding and copying its states and arcs.
    pub fn from_fst<F: Fst<A>>(fst: &F) -> Result<Self, OpenFstError> {
        let mut nstates = 0;
        let mut narcs = 0;

        for s in fst.states() {
            nstates += 1;
            narcs += fst.num_arcs(s);
        }

        let mut states_region = MappedFile::allocate_type::<ConstState<A::Weight, U>>(nstates)?;
        let mut arcs_region = MappedFile::allocate_type::<A>(narcs)?;

        if nstates > 0 {
            let states_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    states_region.as_mut_slice().unwrap().as_mut_ptr()
                        as *mut ConstState<A::Weight, U>,
                    nstates,
                )
            };

            let arcs_slice = if narcs > 0 {
                unsafe {
                    std::slice::from_raw_parts_mut(
                        arcs_region.as_mut_slice().unwrap().as_mut_ptr() as *mut A,
                        narcs,
                    )
                }
            } else {
                &mut []
            };

            let mut pos = 0;
            for (s_idx, s) in fst.states().enumerate() {
                let final_weight = fst.final_weight(s);
                let mut narcs_state = 0;
                let mut niepsilons = 0;
                let mut noepsilons = 0;

                for arc in fst.arcs(s) {
                    if arc.ilabel() == A::Label::epsilon() {
                        niepsilons += 1;
                    }
                    if arc.olabel() == A::Label::epsilon() {
                        noepsilons += 1;
                    }
                    arcs_slice[pos] = arc;
                    pos += 1;
                    narcs_state += 1;
                }

                states_slice[s_idx] = ConstState {
                    final_weight,
                    pos: U::from_usize(pos - narcs_state),
                    narcs: U::from_usize(narcs_state),
                    niepsilons: U::from_usize(niepsilons),
                    noepsilons: U::from_usize(noepsilons),
                };
            }
        }

        // A mutable FST keeps its bits up to date, so asking it settles them.
        // An immutable one may have come from a file written before the cycle
        // properties existed, so those are left out of what must already be
        // known, but stay in what a scan, if one happens, will settle.
        let properties = if (fst.properties(K_MUTABLE, false) & K_MUTABLE) != 0 {
            fst.properties(K_COPY_PROPERTIES, true)
        } else {
            check_properties(
                fst,
                K_COPY_PROPERTIES & !K_WEIGHTED_CYCLES & !K_UNWEIGHTED_CYCLES,
                K_COPY_PROPERTIES,
                false,
            )
        };

        Ok(Self {
            states_region,
            arcs_region,
            nstates,
            start: fst.start(),
            properties: PropertyCache::new(properties | K_STATIC_PROPERTIES),
            input_symbols: fst.input_symbols(),
            output_symbols: fst.output_symbols(),
            _marker: PhantomData,
        })
    }

    #[inline(always)]
    fn states_slice(&self) -> &[ConstState<A::Weight, U>] {
        if self.nstates == 0 {
            return &[];
        }
        unsafe {
            std::slice::from_raw_parts(
                self.states_region.as_ref().as_ptr() as *const ConstState<A::Weight, U>,
                self.nstates,
            )
        }
    }

    #[inline(always)]
    fn arcs_slice_internal(&self) -> &[A] {
        let size_in_arcs = self.arcs_region.as_ref().len() / std::mem::size_of::<A>();
        if size_in_arcs == 0 {
            return &[];
        }
        unsafe {
            std::slice::from_raw_parts(self.arcs_region.as_ref().as_ptr() as *const A, size_in_arcs)
        }
    }
}

/// Version written for a file whose regions are aligned.
///
/// Upstream numbers these the other way round to how they read: version 1 is
/// the *aligned* layout and version 2 the unaligned one, because alignment came
/// first and was later made optional.
const ALIGNED_FILE_VERSION: i32 = 1;
/// Version written for a file whose regions are not aligned.
const FILE_VERSION: i32 = 2;
/// The oldest version that can still be read.
const MIN_FILE_VERSION: i32 = 1;
/// Refuses a header claiming more states than could be stored, before that
/// count is used to size anything.
const MAX_STATES: i64 = 0x0010_0000_0000_0000;
/// As [`MAX_STATES`], for arcs.
const MAX_ARCS: i64 = 0x0010_0000_0000_0000;

impl<A: Arc, U: Unsigned> ConstFst<'static, A, U> {
    /// Reads a `ConstFst` from a stream.
    ///
    /// The regions are read into memory. Use
    /// [`read_from_file`](Self::read_from_file) to map them instead.
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

    /// Reads a `ConstFst` from a file, mapping its regions where it can.
    ///
    /// Mapping is advisory: a region that does not begin on an aligned offset
    /// cannot be viewed as the packed structures stored here, so it is read
    /// instead. That is why a file can be written aligned at all.
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

    /// The shared body of the two readers: everything but how a region is
    /// obtained.
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
            Self::fst_type_name().as_str(),
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

        // Version 1 predates the flag, and was always aligned.
        let aligned =
            header.version == ALIGNED_FILE_VERSION || header.flags & flags::IS_ALIGNED != 0;

        let states_region = region(
            reader,
            aligned,
            nstates * std::mem::size_of::<ConstState<A::Weight, U>>(),
        )?;
        let arcs_region = region(reader, aligned, narcs * std::mem::size_of::<A>())?;

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

        let fst = Self {
            states_region,
            arcs_region,
            nstates,
            start,
            properties: PropertyCache::new(header.properties | K_EXPANDED),
            input_symbols: read.isymbols,
            output_symbols: read.osymbols,
            _marker: PhantomData,
        };

        // Each state names a range of the arc array; a range reaching past it
        // would be read out of bounds by every later access.
        for (s, state) in fst.states_slice().iter().enumerate() {
            let pos = state.pos.as_usize();
            let narcs_s = state.narcs.as_usize();
            if pos > narcs || narcs_s > narcs - pos {
                return Err(bad(format!("state {s} arc range out of bounds")));
            }
        }

        if opts.verify {
            for arc in fst.arcs_slice_internal() {
                let nextstate = arc.nextstate();
                if nextstate == A::StateId::no_state() {
                    return Err(bad("disallowed next state".to_string()));
                }
                if nextstate < A::StateId::from_usize(0) {
                    return Err(bad("next state is negative".to_string()));
                }
                if nextstate.as_usize() >= nstates {
                    return Err(bad(format!(
                        "next state {} is larger than the number of states {nstates}",
                        nextstate.as_usize()
                    )));
                }
            }
        }

        Ok(fst)
    }
}

impl<A: Arc, U: Unsigned> ConstFst<'_, A, U> {
    /// The name this FST goes by in a file, which depends on the width of the
    /// arc indices.
    fn fst_type_name() -> FstType {
        if std::mem::size_of::<U>() == std::mem::size_of::<u32>() {
            FstType::CONST_32
        } else {
            FstType::CONST_64
        }
    }

    /// Writes any FST out in this format.
    ///
    /// SICADA-DIVERGE: upstream rewrites the header afterwards with the state
    /// and arc counts it actually saw, seeking back to do it, and falls back to
    /// counting them up front when the stream cannot seek. Counting first is
    /// what this always does: it costs one more pass over an FST that already
    /// knows its size, and in exchange the output is correct on a pipe and the
    /// writer needs no `Seek`.
    pub fn write_fst<F: Fst<A>, W: Write>(
        fst: &F,
        writer: &mut W,
        opts: &FstWriteOptions,
    ) -> Result<(), OpenFstError> {
        let mut nstates = 0i64;
        let mut narcs = 0i64;
        for s in fst.states() {
            nstates += 1;
            narcs += fst.num_arcs(s) as i64;
        }

        let header = FstHeader {
            fst_type: Self::fst_type_name().as_str().to_string(),
            arc_type: A::type_name().as_str().to_string(),
            version: if opts.align {
                ALIGNED_FILE_VERSION
            } else {
                FILE_VERSION
            },
            // Overwritten by `write_fst_header` from the tables it writes.
            flags: 0,
            properties: fst.properties(K_COPY_PROPERTIES, true) | K_STATIC_PROPERTIES,
            start: fst
                .start()
                .map_or(NO_STATE_ID as i64, |s| s.as_usize() as i64),
            num_states: nstates,
            num_arcs: narcs,
        };
        let isymbols = fst.input_symbols();
        let osymbols = fst.output_symbols();
        // Counted rather than seeked: alignment needs the file offset, and this
        // way it works on a stream that cannot report one.
        let mut writer = CountingWriter::new(writer, 0);
        write_fst_header(
            &mut writer,
            opts,
            &header,
            isymbols.as_deref(),
            osymbols.as_deref(),
        )?;

        // The states and arcs go out as blocks of memory, which is why their
        // layout is pinned; see `ConstState` and `ArcTpl`.
        let mut states = Vec::with_capacity(nstates as usize);
        let mut pos = U::from_usize(0);
        for s in fst.states() {
            let narcs_s = fst.num_arcs(s);
            states.push(ConstState {
                final_weight: fst.final_weight(s),
                pos,
                narcs: U::from_usize(narcs_s),
                niepsilons: U::from_usize(fst.num_input_epsilons(s)),
                noepsilons: U::from_usize(fst.num_output_epsilons(s)),
            });
            pos = U::from_usize(pos.as_usize() + narcs_s);
        }
        if opts.align {
            writer.align(ARCH_ALIGNMENT as u64)?;
        }
        writer.write_all(as_bytes(&states))?;

        if opts.align {
            writer.align(ARCH_ALIGNMENT as u64)?;
        }
        let mut arcs = Vec::with_capacity(narcs as usize);
        for s in fst.states() {
            arcs.extend(fst.arcs(s));
        }
        writer.write_all(as_bytes(&arcs))?;
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

/// A slice seen as the bytes it occupies.
///
/// # Safety
///
/// `T` must be `#[repr(C)]` with no uninitialized padding that would be written
/// out. Both types this is used for, `ConstState` and the arc type, are laid out
/// so that every byte belongs to a field.
fn as_bytes<T>(values: &[T]) -> &[u8] {
    // SAFETY: the slice is valid for `len * size_of::<T>()` bytes, and `u8` has
    // no alignment requirement or invalid values, so any byte pattern reads
    // back.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

pub struct ConstStateIter<S: ArcStateId> {
    nstates: usize,
    s: usize,
    _phantom: PhantomData<S>,
}

impl<S: ArcStateId> Iterator for ConstStateIter<S> {
    type Item = S;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.s < self.nstates {
            let s = S::from_usize(self.s);
            self.s += 1;
            Some(s)
        } else {
            None
        }
    }
}

pub struct ConstArcIter<'a, A: Arc> {
    arcs: &'a [A],
    pos: usize,
}

impl<'a, A: Arc> Clone for ConstArcIter<'a, A> {
    fn clone(&self) -> Self {
        Self {
            arcs: self.arcs,
            pos: self.pos,
        }
    }
}

impl<'a, A: Arc> Iterator for ConstArcIter<'a, A> {
    type Item = A;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.arcs.len() {
            let item = self.arcs[self.pos].clone();
            self.pos += 1;
            Some(item)
        } else {
            None
        }
    }
}

impl<'a, A: Arc, U: Unsigned> Fst<A> for ConstFst<'a, A, U> {
    type StateIter<'b>
        = ConstStateIter<A::StateId>
    where
        Self: 'b;
    type ArcIter<'b>
        = ConstArcIter<'b, A>
    where
        Self: 'b;

    #[inline]
    fn start(&self) -> Option<A::StateId> {
        self.start
    }

    #[inline]
    fn final_weight(&self, state: A::StateId) -> A::Weight {
        let s = state.as_usize();
        if s < self.nstates {
            self.states_slice()[s].final_weight.clone()
        } else {
            A::Weight::zero()
        }
    }

    #[inline]
    fn num_arcs(&self, state: A::StateId) -> usize {
        let s = state.as_usize();
        if s < self.nstates {
            self.states_slice()[s].narcs.as_usize()
        } else {
            0
        }
    }

    #[inline]
    fn num_input_epsilons(&self, state: A::StateId) -> usize {
        let s = state.as_usize();
        if s < self.nstates {
            self.states_slice()[s].niepsilons.as_usize()
        } else {
            0
        }
    }

    #[inline]
    fn num_output_epsilons(&self, state: A::StateId) -> usize {
        let s = state.as_usize();
        if s < self.nstates {
            self.states_slice()[s].noepsilons.as_usize()
        } else {
            0
        }
    }

    #[inline]
    fn num_states_if_known(&self) -> Option<usize> {
        Some(self.nstates)
    }

    #[inline]
    fn properties(&self, mask: u64, test: bool) -> u64 {
        cached_properties(self, &self.properties, mask, test)
    }

    #[inline]
    fn fst_type(&self) -> &str {
        if std::mem::size_of::<U>() != 4 {
            FstType::CONST_64.as_str()
        } else {
            FstType::CONST_32.as_str()
        }
    }

    #[inline]
    fn input_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
        self.input_symbols.clone()
    }

    #[inline]
    fn output_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
        self.output_symbols.clone()
    }

    #[inline]
    fn states<'b>(&'b self) -> Self::StateIter<'b> {
        ConstStateIter {
            nstates: self.nstates,
            s: 0,
            _phantom: PhantomData,
        }
    }

    #[inline]
    fn arcs<'b>(&'b self, state: A::StateId) -> Self::ArcIter<'b> {
        let s = state.as_usize();
        let arcs = if s < self.nstates {
            let st = &self.states_slice()[s];
            let start = st.pos.as_usize();
            let len = st.narcs.as_usize();
            &self.arcs_slice_internal()[start..start + len]
        } else {
            &[]
        };
        ConstArcIter { arcs, pos: 0 }
    }
}

impl<'a, A: Arc, U: Unsigned> ExpandedFst<A> for ConstFst<'a, A, U> {
    #[inline]
    fn num_states(&self) -> usize {
        self.nstates
    }
}

impl<'a, A: Arc, U: Unsigned> ContiguousArcsFst<A> for ConstFst<'a, A, U> {
    /// Returns a contiguous slice of arcs leaving the given state in O(1) time.
    #[inline]
    fn arcs_slice(&self, state: A::StateId) -> &[A] {
        let s = state.as_usize();
        if s < self.nstates {
            let st = &self.states_slice()[s];
            let start = st.pos.as_usize();
            let len = st.narcs.as_usize();
            &self.arcs_slice_internal()[start..start + len]
        } else {
            &[]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::float_weight::TropicalWeight;
    use crate::fst::MutableFst;
    use crate::vector_fst::StdVectorFst;
    use std::io::Cursor;

    #[test]
    fn test_const_fst_basic() {
        let mut vfst = StdVectorFst::new();
        let s0 = vfst.add_state();
        let s1 = vfst.add_state();

        vfst.set_start(s0);
        vfst.set_final(s1, TropicalWeight(1.0));

        vfst.add_arc(s0, StdArc::new(1, 2, TropicalWeight(0.5), s1));
        vfst.add_arc(s0, StdArc::new(0, 0, TropicalWeight(0.2), s0)); // Epsilon loop

        // Use u32 as the unsigned index type
        let cfst = ConstFst::<StdArc, u32>::from_fst(&vfst).unwrap();

        assert_eq!(cfst.fst_type(), "const");
        assert_eq!(cfst.start(), Some(0));
        assert_eq!(cfst.num_states(), 2);
        assert_eq!(cfst.final_weight(1).value(), 1.0);
        assert_eq!(cfst.final_weight(0), TropicalWeight::zero());

        assert_eq!(cfst.num_arcs(0), 2);
        assert_eq!(cfst.num_arcs(1), 0);

        // Epsilon counts test
        assert_eq!(cfst.num_input_epsilons(0), 1);
        assert_eq!(cfst.num_output_epsilons(0), 1);

        // Contiguous slice access test
        let arcs_s0 = cfst.arcs_slice(0);
        assert_eq!(arcs_s0.len(), 2);
        assert_eq!(arcs_s0[0].ilabel(), 1);
        assert_eq!(arcs_s0[0].olabel(), 2);
        assert_eq!(arcs_s0[0].weight().value(), 0.5);
        assert_eq!(arcs_s0[0].nextstate(), 1);
    }

    #[test]
    fn test_const_fst_64() {
        let mut vfst = StdVectorFst::new();
        vfst.add_state();
        vfst.set_start(0);

        // Use u64 as the unsigned index type
        let cfst = ConstFst::<StdArc, u64>::from_fst(&vfst).unwrap();
        assert_eq!(cfst.fst_type(), "const64");
    }

    #[test]
    fn test_const_fst_empty() {
        let vfst = StdVectorFst::new();
        let cfst = ConstFst::<StdArc, u32>::from_fst(&vfst).unwrap();

        assert_eq!(cfst.num_states(), 0);
        assert_eq!(cfst.start(), None);
    }

    /// The layout of what goes on disk, checked against what OpenFst produces.
    ///
    /// From `tests/oracles/const-fst-golden.cc`. A `ConstFst` file is the
    /// arrays of these two structures written out as blocks of memory, so a
    /// changed size, alignment or field offset is a changed file format.
    #[test]
    fn the_on_disk_structures_are_laid_out_as_openfst_lays_them_out() {
        use std::mem::{align_of, offset_of, size_of};

        type Arc32 = StdArc;
        assert_eq!(size_of::<Arc32>(), 16);
        assert_eq!(align_of::<Arc32>(), 4);
        assert_eq!(offset_of!(Arc32, ilabel), 0);
        assert_eq!(offset_of!(Arc32, olabel), 4);
        assert_eq!(offset_of!(Arc32, weight), 8);
        assert_eq!(offset_of!(Arc32, nextstate), 12);

        type State32 = ConstState<TropicalWeight, u32>;
        assert_eq!(size_of::<State32>(), 20);
        assert_eq!(align_of::<State32>(), 4);
        assert_eq!(offset_of!(State32, final_weight), 0);
        assert_eq!(offset_of!(State32, pos), 4);
        assert_eq!(offset_of!(State32, narcs), 8);
        assert_eq!(offset_of!(State32, niepsilons), 12);
        assert_eq!(offset_of!(State32, noepsilons), 16);

        type State64 = ConstState<TropicalWeight, u64>;
        assert_eq!(size_of::<State64>(), 40);
        assert_eq!(align_of::<State64>(), 8);
        assert_eq!(offset_of!(State64, final_weight), 0);
        assert_eq!(offset_of!(State64, pos), 8);
        assert_eq!(offset_of!(State64, narcs), 16);
        assert_eq!(offset_of!(State64, niepsilons), 24);
        assert_eq!(offset_of!(State64, noepsilons), 32);
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// 0 -> 1 -> 2, with 2 final.
    fn chain() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight::one(), 2));
        fst.set_final(2, TropicalWeight::one());
        fst
    }

    /// The exact bytes of the states and arcs blocks, against the ones OpenFst
    /// writes for the same FST.
    #[test]
    fn the_state_and_arc_blocks_are_the_bytes_openfst_writes() {
        const GOLDEN_STATES: &str = concat!(
            "0000807f00000000010000000000000000000000",
            "0000807f01000000010000000000000000000000",
            "0000000002000000000000000000000000000000",
        );
        const GOLDEN_ARCS: &str =
            "0100000001000000000000000100000002000000020000000000000002000000";

        let mut bytes = Vec::new();
        ConstFst::<StdArc, u32>::write_fst(&chain(), &mut bytes, &FstWriteOptions::default())
            .unwrap();

        // Everything after the header is the two blocks, back to back.
        let mut header = Vec::new();
        let read = FstHeader::read(&mut Cursor::new(bytes.clone())).unwrap();
        read.write(&mut header).unwrap();
        let body = &bytes[header.len()..];

        let states_len = 3 * std::mem::size_of::<ConstState<TropicalWeight, u32>>();
        assert_eq!(hex(&body[..states_len]), GOLDEN_STATES);
        assert_eq!(hex(&body[states_len..]), GOLDEN_ARCS);
    }

    #[test]
    fn a_const_fst_round_trips_through_a_stream() {
        let source = chain();
        let mut bytes = Vec::new();
        ConstFst::<StdArc, u32>::write_fst(&source, &mut bytes, &FstWriteOptions::default())
            .unwrap();

        let read =
            ConstFst::<StdArc, u32>::read(&mut Cursor::new(bytes), &FstReadOptions::default())
                .unwrap();

        assert_eq!(read.start(), Some(0));
        assert_eq!(read.num_states(), 3);
        assert_eq!(read.final_weight(2), TropicalWeight::one());
        assert_eq!(read.final_weight(0), TropicalWeight::zero());
        for state in 0..3 {
            let want: Vec<_> = source.arcs(state).collect();
            let got: Vec<_> = read.arcs(state).collect();
            assert_eq!(got, want, "state {state}");
        }
    }

    /// Writing aligned pads the blocks so a reader can map them; the contents
    /// have to survive the padding.
    #[test]
    fn an_aligned_file_round_trips_and_is_padded() {
        let source = chain();
        let opts = FstWriteOptions {
            align: true,
            ..Default::default()
        };
        let mut aligned = Vec::new();
        ConstFst::<StdArc, u32>::write_fst(&source, &mut aligned, &opts).unwrap();

        let mut plain = Vec::new();
        ConstFst::<StdArc, u32>::write_fst(&source, &mut plain, &FstWriteOptions::default())
            .unwrap();
        assert!(aligned.len() > plain.len(), "no padding was written");

        let header = FstHeader::read(&mut Cursor::new(aligned.clone())).unwrap();
        assert_eq!(header.version, ALIGNED_FILE_VERSION);
        assert_ne!(header.flags & flags::IS_ALIGNED, 0);

        let read =
            ConstFst::<StdArc, u32>::read(&mut Cursor::new(aligned), &FstReadOptions::default())
                .unwrap();
        assert_eq!(read.num_states(), 3);
        let arcs: Vec<_> = read.arcs(0).collect();
        assert_eq!(arcs, source.arcs(0).collect::<Vec<_>>());
    }

    /// Version 1 predates the alignment flag and was always aligned, so it has
    /// to be read as aligned whatever the flags say.
    #[test]
    fn version_one_is_read_as_aligned_even_without_the_flag() {
        let opts = FstWriteOptions {
            align: true,
            ..Default::default()
        };
        let mut bytes = Vec::new();
        ConstFst::<StdArc, u32>::write_fst(&chain(), &mut bytes, &opts).unwrap();

        // Clear the flag, leaving only the version to say so.
        let mut header = FstHeader::read(&mut Cursor::new(bytes.clone())).unwrap();
        let header_len = {
            let mut probe = Vec::new();
            header.write(&mut probe).unwrap();
            probe.len()
        };
        assert_eq!(header.version, ALIGNED_FILE_VERSION);
        header.flags &= !flags::IS_ALIGNED;
        let mut rewritten = Vec::new();
        header.write(&mut rewritten).unwrap();
        rewritten.extend_from_slice(&bytes[header_len..]);

        let read =
            ConstFst::<StdArc, u32>::read(&mut Cursor::new(rewritten), &FstReadOptions::default())
                .unwrap();
        assert_eq!(read.num_states(), 3);
        assert_eq!(read.arcs(0).next().unwrap().ilabel(), 1);
    }

    #[test]
    fn a_file_that_lies_about_its_shape_is_refused() {
        let mut bytes = Vec::new();
        ConstFst::<StdArc, u32>::write_fst(&chain(), &mut bytes, &FstWriteOptions::default())
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
            ConstFst::<StdArc, u32>::read(&mut Cursor::new(out), &FstReadOptions::default())
        };

        assert!(rewrite(&|h| h.num_states = -1).is_err(), "negative states");
        assert!(
            rewrite(&|h| h.num_states = i64::MAX).is_err(),
            "absurd states"
        );
        assert!(rewrite(&|h| h.num_arcs = -1).is_err(), "negative arcs");
        assert!(rewrite(&|h| h.start = 9).is_err(), "start past the end");
        assert!(rewrite(&|h| h.start = -7).is_err(), "negative start");
        assert!(
            rewrite(&|h| h.fst_type = "vector".to_string()).is_err(),
            "wrong type"
        );
        assert!(
            rewrite(&|h| h.arc_type = "log".to_string()).is_err(),
            "wrong arc type"
        );
        assert!(
            rewrite(&|h| h.version = MIN_FILE_VERSION - 1).is_err(),
            "old version"
        );
        // Unchanged, as a control.
        assert!(rewrite(&|_| {}).is_ok());
    }

    /// With `verify`, an arc pointing at a state that does not exist is caught
    /// at read time rather than at the first traversal.
    #[test]
    fn an_arc_leading_outside_the_fst_is_caught_when_verifying() {
        let mut bytes = Vec::new();
        ConstFst::<StdArc, u32>::write_fst(&chain(), &mut bytes, &FstWriteOptions::default())
            .unwrap();

        // The last four bytes are the final arc's nextstate.
        let len = bytes.len();
        bytes[len - 4..].copy_from_slice(&9i32.to_le_bytes());

        assert!(
            ConstFst::<StdArc, u32>::read(
                &mut Cursor::new(bytes.clone()),
                &FstReadOptions::default()
            )
            .is_err()
        );
        // Without verifying it is let through, as upstream also does.
        let opts = FstReadOptions {
            verify: false,
            ..Default::default()
        };
        assert!(ConstFst::<StdArc, u32>::read(&mut Cursor::new(bytes), &opts).is_ok());
    }

    /// The symbol tables sit between the header and the blocks; a reader that
    /// skipped them would decode the states out of the middle of a table.
    #[test]
    fn symbol_tables_survive_the_round_trip() {
        let mut table = SymbolTable::new("input".to_string());
        table.add_symbol("<eps>", 0);
        table.add_symbol("a", 1);
        table.add_symbol("b", 2);

        let mut source = chain();
        source.set_input_symbols(Some(AtomicRc::new(table)));

        let mut bytes = Vec::new();
        ConstFst::<StdArc, u32>::write_fst(&source, &mut bytes, &FstWriteOptions::default())
            .unwrap();
        let read =
            ConstFst::<StdArc, u32>::read(&mut Cursor::new(bytes), &FstReadOptions::default())
                .unwrap();

        assert_eq!(read.num_states(), 3);
        let symbols = read.input_symbols().expect("the table came back");
        assert_eq!(symbols.find_symbol(1), Some("a"));
        assert_eq!(symbols.find_symbol(2), Some("b"));
        assert!(read.output_symbols().is_none());
        assert_eq!(read.arcs(0).next().unwrap().ilabel(), 1);
    }

    /// Reading through a file is the path that can map rather than copy.
    #[test]
    fn a_const_fst_round_trips_through_a_file() {
        use std::io::Write as _;

        let source = chain();
        let mut bytes = Vec::new();
        ConstFst::<StdArc, u32>::write_fst(
            &source,
            &mut bytes,
            &FstWriteOptions {
                align: true,
                ..Default::default()
            },
        )
        .unwrap();

        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        file.flush().unwrap();

        for mode in [FileReadMode::Read, FileReadMode::Map] {
            let read = ConstFst::<StdArc, u32>::read_from_file(
                file.path(),
                &FstReadOptions::default().mode(mode),
            )
            .unwrap();
            assert_eq!(read.num_states(), 3, "{mode:?}");
            assert_eq!(read.start(), Some(0));
            for state in 0..3 {
                assert_eq!(
                    read.arcs(state).collect::<Vec<_>>(),
                    source.arcs(state).collect::<Vec<_>>(),
                    "{mode:?} state {state}"
                );
            }
        }
    }
}
