//! Reading an FST whose type is only known from its header.
//!
//! Port of OpenFst's `fst-types.cc`, which is the list of FST types the library
//! registers by default.
//!
//! SICADA-DIVERGE: upstream's registration is a global dynamic registry keyed
//! by the type-name string, falling back to `dlopen`ing `<type>-fst.so` for a
//! name it does not know. sicada does not take that design, for two reasons:
//! [`Fst`] is dyn-incompatible because its iterators are GATs, so the
//! `Fst<Arc>*` the registry hands back has no Rust equivalent that does not box
//! the iterators; and loading plugins out of shared objects is not a library's
//! job. [`AnyFst`] is the closed enum
//! that replaces it. Every branch is resolved statically, and the set of types
//! is fixed at compile time, as it already was in practice, since nothing in
//! the OpenFst distribution ships an FST plugin.

use std::fs::File;
use std::io::{Read, Seek, Write};
use std::path::Path;

use crate::AtomicRc;
use crate::arc::Arc;
use crate::error::OpenFstError;
use crate::fst::{ExpandedFst, Fst, FstReadOptions, FstWriteOptions};
use crate::fst_header::FstHeader;
use crate::fst_type::FstType;
use crate::fsts::compact_fst::{
    CompactAcceptorFst, CompactStringFst, CompactUnweightedAcceptorFst, CompactUnweightedFst,
    CompactWeightedStringFst,
};
use crate::fsts::const_fst::ConstFst;
use crate::fsts::edit_fst::EditFst;
use crate::fsts::vector_fst::VectorFst;
use crate::symbol_table::SymbolTable;
use crate::utils::io::FstScalar;
use crate::weight::WeightIo;

/// Generates the enum over concrete FST types, its two iterators, and the
/// delegation of every [`Fst`] method to whichever variant is present.
macro_rules! any_iter_ty {
    (plain, $t:ty) => { $t };
    (boxed, $t:ty) => { Box<$t> };
}

macro_rules! any_iter_wrap {
    (plain, $e:expr) => {
        $e
    };
    (boxed, $e:expr) => {
        Box::new($e)
    };
}

macro_rules! define_any_fst {
    ($( $(#[$meta:meta])* $variant:ident => $ty:ty, $fst_type:expr, $kind:ident; )*) => {
        /// An FST of one of the types sicada knows how to read.
        ///
        /// The variants are boxed so that the enum stays one word wide however
        /// large the largest implementation is, and so that a type wrapping
        /// another FST can hold one of these.
        pub enum AnyFst<'f, A: Arc + 'static>
        where A::Weight: Copy
        {
            $( $(#[$meta])* $variant(Box<$ty>), )*
        }

        /// The state iterator of whichever FST [`AnyFst`] holds.
        pub enum AnyStateIter<'f, 'a, A: Arc + 'static>
        where 'f: 'a, A::Weight: Copy
        {
            $( #[allow(missing_docs)] $variant(<$ty as Fst<A>>::StateIter<'a>), )*
        }

        /// The arc iterator of whichever FST [`AnyFst`] holds.
        pub enum AnyArcIter<'f, 'a, A: Arc + 'static>
        where 'f: 'a, A::Weight: Copy
        {
            $( #[allow(missing_docs)]
               $variant(any_iter_ty!($kind, <$ty as Fst<A>>::ArcIter<'a>)), )*
        }

        impl<'f, 'a, A: Arc + 'static> Iterator for AnyStateIter<'f, 'a, A>
        where 'f: 'a, A::Weight: Copy
        {
            type Item = A::StateId;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                match self { $( Self::$variant(it) => it.next(), )* }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                match self { $( Self::$variant(it) => it.size_hint(), )* }
            }
        }

        impl<'f, 'a, A: Arc + 'static> Iterator for AnyArcIter<'f, 'a, A>
        where 'f: 'a, A::Weight: Copy
        {
            type Item = A;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                match self { $( Self::$variant(it) => it.next(), )* }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                match self { $( Self::$variant(it) => it.size_hint(), )* }
            }
        }

        impl<'f, 'a, A: Arc + 'static> Clone for AnyArcIter<'f, 'a, A>
        where 'f: 'a, A::Weight: Copy
        {
            fn clone(&self) -> Self {
                match self { $( Self::$variant(it) => Self::$variant(it.clone()), )* }
            }
        }

        impl<'f, A: Arc + 'static> Fst<A> for AnyFst<'f, A>
        where A::Weight: Copy
        {
            type StateIter<'a> = AnyStateIter<'f, 'a, A> where Self: 'a;
            type ArcIter<'a> = AnyArcIter<'f, 'a, A> where Self: 'a;

            fn start(&self) -> Option<A::StateId> {
                match self { $( Self::$variant(f) => f.start(), )* }
            }

            fn final_weight(&self, state: A::StateId) -> A::Weight {
                match self { $( Self::$variant(f) => f.final_weight(state), )* }
            }

            fn num_arcs(&self, state: A::StateId) -> usize {
                match self { $( Self::$variant(f) => f.num_arcs(state), )* }
            }

            fn num_input_epsilons(&self, state: A::StateId) -> usize {
                match self { $( Self::$variant(f) => f.num_input_epsilons(state), )* }
            }

            fn num_output_epsilons(&self, state: A::StateId) -> usize {
                match self { $( Self::$variant(f) => f.num_output_epsilons(state), )* }
            }

            fn num_states_if_known(&self) -> Option<usize> {
                match self { $( Self::$variant(f) => f.num_states_if_known(), )* }
            }

            fn properties(&self, mask: u64, test: bool) -> u64 {
                match self { $( Self::$variant(f) => f.properties(mask, test), )* }
            }

            fn fst_type(&self) -> &str {
                match self { $( Self::$variant(f) => f.fst_type(), )* }
            }

            fn input_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
                match self { $( Self::$variant(f) => f.input_symbols(), )* }
            }

            fn output_symbols(&self) -> Option<AtomicRc<SymbolTable>> {
                match self { $( Self::$variant(f) => f.output_symbols(), )* }
            }

            fn states<'a>(&'a self) -> Self::StateIter<'a> {
                match self {
                    $( Self::$variant(f) => AnyStateIter::$variant(f.states()), )*
                }
            }

            fn arcs<'a>(&'a self, state: A::StateId) -> Self::ArcIter<'a> {
                match self {
                    $( Self::$variant(f) =>
                        AnyArcIter::$variant(any_iter_wrap!($kind, f.arcs(state))), )*
                }
            }
        }

        impl<'f, A: Arc + 'static> ExpandedFst<A> for AnyFst<'f, A>
        where A::Weight: Copy
        {
            fn num_states(&self) -> usize {
                match self { $( Self::$variant(f) => f.num_states(), )* }
            }
        }

        impl<'f, A: Arc + 'static> AnyFst<'f, A>
        where A::Weight: Copy
        {
            /// The type name this FST would be written under.
            pub fn fst_type_name(&self) -> FstType {
                match self { $( Self::$variant(_) => $fst_type, )* }
            }

        }

        impl<'f, A: Arc + 'static> AnyFst<'f, A>
        where
            A::Label: FstScalar,
            A::StateId: FstScalar,
            A::Weight: Copy + WeightIo,
        {
            /// Writes the FST in its own format.
            pub fn write<W: Write>(
                &self,
                writer: &mut W,
                opts: &FstWriteOptions,
            ) -> Result<(), OpenFstError> {
                match self { $( Self::$variant(f) => f.write(writer, opts), )* }
            }
        }
    };
}

define_any_fst! {
    /// A mutable FST, states and arcs written field by field.
    Vector => VectorFst<A>, FstType::VECTOR, plain;
    /// An immutable FST with 32-bit offsets.
    Const32 => ConstFst<'f, A, u32>, FstType::CONST_32, plain;
    /// An immutable FST with 64-bit offsets.
    Const64 => ConstFst<'f, A, u64>, FstType::CONST_64, plain;
    /// An unweighted string, 32-bit offsets.
    CompactString32 => CompactStringFst<'f, A, u32>, FstType::COMPACT_STRING_32, plain;
    /// An unweighted string, 64-bit offsets.
    CompactString64 => CompactStringFst<'f, A, u64>, FstType::COMPACT_STRING_64, plain;
    /// A weighted string, 32-bit offsets.
    CompactWeightedString32 =>
        CompactWeightedStringFst<'f, A, u32>, FstType::COMPACT_WEIGHTED_STRING_32, plain;
    /// A weighted string, 64-bit offsets.
    CompactWeightedString64 =>
        CompactWeightedStringFst<'f, A, u64>, FstType::COMPACT_WEIGHTED_STRING_64, plain;
    /// An acceptor, 32-bit offsets.
    CompactAcceptor32 => CompactAcceptorFst<'f, A, u32>, FstType::COMPACT_ACCEPTOR_32, plain;
    /// An acceptor, 64-bit offsets.
    CompactAcceptor64 => CompactAcceptorFst<'f, A, u64>, FstType::COMPACT_ACCEPTOR_64, plain;
    /// An unweighted FST, 32-bit offsets.
    CompactUnweighted32 => CompactUnweightedFst<'f, A, u32>, FstType::COMPACT_UNWEIGHTED_32, plain;
    /// An unweighted FST, 64-bit offsets.
    CompactUnweighted64 => CompactUnweightedFst<'f, A, u64>, FstType::COMPACT_UNWEIGHTED_64, plain;
    /// An unweighted acceptor, 32-bit offsets.
    CompactUnweightedAcceptor32 =>
        CompactUnweightedAcceptorFst<'f, A, u32>, FstType::COMPACT_UNWEIGHTED_ACCEPTOR_32, plain;
    /// An unweighted acceptor, 64-bit offsets.
    CompactUnweightedAcceptor64 =>
        CompactUnweightedAcceptorFst<'f, A, u64>, FstType::COMPACT_UNWEIGHTED_ACCEPTOR_64, plain;
    /// An FST with edits recorded beside it.
    ///
    /// The arc iterator is boxed because this is the one variant that can hold
    /// an [`AnyFst`] of its own, so the iterator type would otherwise be
    /// infinite. That is one allocation per state visited, and only for this
    /// variant; upstream pays a virtual call per arc on every type, because its
    /// type-erased `ArcIterator<Fst<Arc>>` is virtual throughout.
    Edit => EditFst<A, AnyFst<'f, A>>, FstType::EDIT, boxed;
}

impl<A: Arc + 'static> AnyFst<'static, A>
where
    A::Label: FstScalar,
    A::StateId: FstScalar,
    A::Weight: Copy + WeightIo,
{
    /// Reads whichever FST a stream holds, from the type name in its header.
    ///
    /// The header is read once here and handed to the implementation through
    /// [`FstReadOptions::header`], so the stream is never rewound and this
    /// works on a pipe.
    pub fn read<R: Read + Seek>(
        reader: &mut R,
        opts: &FstReadOptions,
    ) -> Result<Self, OpenFstError> {
        let header = FstHeader::read(&mut *reader)?;
        let fst_type = Self::identify(&header, &opts.source)?;
        let opts = FstReadOptions {
            header: Some(header),
            ..opts.clone()
        };
        Self::read_as(fst_type, reader, &opts)
    }

    /// Reads whichever FST a file holds, mapping its regions where the format
    /// allows it.
    pub fn read_from_file(
        path: impl AsRef<Path>,
        opts: &FstReadOptions,
    ) -> Result<Self, OpenFstError> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let opts = FstReadOptions {
            source: path.display().to_string(),
            ..opts.clone()
        };
        let header = FstHeader::read(&mut file)?;
        let fst_type = Self::identify(&header, &opts.source)?;
        // Only the memory-mapped formats have a separate file path; the rest
        // read straight from the handle, which is already past the header.
        match fst_type {
            // These two hold their contents in the stream rather than in a
            // block that could be mapped, so they read on from the handle --
            // which is past the header, so the header read above is handed on
            // rather than read again.
            FstType::VECTOR | FstType::EDIT => Self::read_as(
                fst_type,
                &mut file,
                &FstReadOptions {
                    header: Some(header),
                    ..opts
                },
            ),
            // The mapped formats open the path for themselves, so their handle
            // starts at byte zero and they have to read their own header. The
            // one read above must *not* be handed on: the reader would then
            // believe it was past a header it had not consumed, and every
            // offset after it would be short by the header's length.
            _ => Self::map_as(
                fst_type,
                path,
                &FstReadOptions {
                    header: None,
                    ..opts
                },
            ),
        }
    }

    /// The type a header names, refusing anything sicada cannot read.
    fn identify(header: &FstHeader, source: &str) -> Result<FstType, OpenFstError> {
        let fst_type = FstType::from_name(&header.fst_type).ok_or_else(|| {
            OpenFstError::InvalidFstHeader(format!(
                "{source}: unknown FST type '{}'",
                header.fst_type
            ))
        })?;
        let arc_type = A::type_name();
        if header.arc_type != arc_type.as_str() {
            return Err(OpenFstError::InvalidFstHeader(format!(
                "{source}: arc not of type '{}', found '{}'",
                arc_type.as_str(),
                header.arc_type
            )));
        }
        Ok(fst_type)
    }

    fn unreadable(fst_type: FstType, source: &str) -> OpenFstError {
        OpenFstError::InvalidFstHeader(format!(
            "{source}: FST type '{fst_type}' cannot be read from a stream"
        ))
    }

    fn read_as<R: Read + Seek>(
        fst_type: FstType,
        reader: &mut R,
        opts: &FstReadOptions,
    ) -> Result<Self, OpenFstError> {
        Ok(match fst_type {
            FstType::VECTOR => Self::Vector(Box::new(VectorFst::read(reader, opts)?)),
            FstType::CONST_32 => Self::Const32(Box::new(ConstFst::read(reader, opts)?)),
            FstType::CONST_64 => Self::Const64(Box::new(ConstFst::read(reader, opts)?)),
            FstType::COMPACT_STRING_32 => {
                Self::CompactString32(Box::new(CompactStringFst::read(reader, opts)?))
            }
            FstType::COMPACT_STRING_64 => {
                Self::CompactString64(Box::new(CompactStringFst::read(reader, opts)?))
            }
            FstType::COMPACT_WEIGHTED_STRING_32 => Self::CompactWeightedString32(Box::new(
                CompactWeightedStringFst::read(reader, opts)?,
            )),
            FstType::COMPACT_WEIGHTED_STRING_64 => Self::CompactWeightedString64(Box::new(
                CompactWeightedStringFst::read(reader, opts)?,
            )),
            FstType::COMPACT_ACCEPTOR_32 => {
                Self::CompactAcceptor32(Box::new(CompactAcceptorFst::read(reader, opts)?))
            }
            FstType::COMPACT_ACCEPTOR_64 => {
                Self::CompactAcceptor64(Box::new(CompactAcceptorFst::read(reader, opts)?))
            }
            FstType::COMPACT_UNWEIGHTED_32 => {
                Self::CompactUnweighted32(Box::new(CompactUnweightedFst::read(reader, opts)?))
            }
            FstType::COMPACT_UNWEIGHTED_64 => {
                Self::CompactUnweighted64(Box::new(CompactUnweightedFst::read(reader, opts)?))
            }
            FstType::COMPACT_UNWEIGHTED_ACCEPTOR_32 => Self::CompactUnweightedAcceptor32(Box::new(
                CompactUnweightedAcceptorFst::read(reader, opts)?,
            )),
            FstType::COMPACT_UNWEIGHTED_ACCEPTOR_64 => Self::CompactUnweightedAcceptor64(Box::new(
                CompactUnweightedAcceptorFst::read(reader, opts)?,
            )),
            FstType::EDIT => Self::Edit(Box::new(EditFst::read(reader, opts)?)),
            other => return Err(Self::unreadable(other, &opts.source)),
        })
    }

    fn map_as(fst_type: FstType, path: &Path, opts: &FstReadOptions) -> Result<Self, OpenFstError> {
        Ok(match fst_type {
            FstType::CONST_32 => Self::Const32(Box::new(ConstFst::read_from_file(path, opts)?)),
            FstType::CONST_64 => Self::Const64(Box::new(ConstFst::read_from_file(path, opts)?)),
            FstType::COMPACT_STRING_32 => {
                Self::CompactString32(Box::new(CompactStringFst::read_from_file(path, opts)?))
            }
            FstType::COMPACT_STRING_64 => {
                Self::CompactString64(Box::new(CompactStringFst::read_from_file(path, opts)?))
            }
            FstType::COMPACT_WEIGHTED_STRING_32 => Self::CompactWeightedString32(Box::new(
                CompactWeightedStringFst::read_from_file(path, opts)?,
            )),
            FstType::COMPACT_WEIGHTED_STRING_64 => Self::CompactWeightedString64(Box::new(
                CompactWeightedStringFst::read_from_file(path, opts)?,
            )),
            FstType::COMPACT_ACCEPTOR_32 => {
                Self::CompactAcceptor32(Box::new(CompactAcceptorFst::read_from_file(path, opts)?))
            }
            FstType::COMPACT_ACCEPTOR_64 => {
                Self::CompactAcceptor64(Box::new(CompactAcceptorFst::read_from_file(path, opts)?))
            }
            FstType::COMPACT_UNWEIGHTED_32 => Self::CompactUnweighted32(Box::new(
                CompactUnweightedFst::read_from_file(path, opts)?,
            )),
            FstType::COMPACT_UNWEIGHTED_64 => Self::CompactUnweighted64(Box::new(
                CompactUnweightedFst::read_from_file(path, opts)?,
            )),
            FstType::COMPACT_UNWEIGHTED_ACCEPTOR_32 => Self::CompactUnweightedAcceptor32(Box::new(
                CompactUnweightedAcceptorFst::read_from_file(path, opts)?,
            )),
            FstType::COMPACT_UNWEIGHTED_ACCEPTOR_64 => Self::CompactUnweightedAcceptor64(Box::new(
                CompactUnweightedAcceptorFst::read_from_file(path, opts)?,
            )),
            other => return Err(Self::unreadable(other, &opts.source)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::cache::CacheOptions;
    use crate::fst::MutableFst;
    use crate::properties::K_FST_PROPERTIES;
    use crate::weight::Weight;
    use crate::weights::float_weight::TropicalWeight;
    use std::io::Cursor;

    type Any = AnyFst<'static, StdArc>;

    /// 0 -1:2/0.5-> 1 -3:4/1.5-> 2, with 2 final.
    fn sample() -> VectorFst<StdArc> {
        let mut fst = VectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 2, TropicalWeight(0.5), 1));
        fst.add_arc(1, StdArc::new(3, 4, TropicalWeight(1.5), 2));
        fst.set_final(2, TropicalWeight(2.5));
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    /// An acceptor, which the compact formats that fold the two sides need.
    fn acceptor() -> VectorFst<StdArc> {
        let mut fst = VectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(0.5), 1));
        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight(1.5), 2));
        fst.set_final(2, TropicalWeight(2.5));
        fst.properties(K_FST_PROPERTIES, true);
        fst
    }

    /// A file whose header names a mapped format is read through the path, not
    /// the handle the header came off.
    ///
    /// The two readers differ in where the stream starts. `read_as` is handed a
    /// handle already past the header and so is given that header to reuse;
    /// `map_as` opens the path itself, at byte zero, and has to read its own.
    /// Passing the header to both made every mapped format come back short by
    /// the header's length: `ConstFst` and all five compact families, read
    /// from a file, failed with "arc range out of bounds". The stream reader
    /// was fine, which is why nothing here caught it: only
    /// `tests/test_const_fst.rs` goes through a file.
    #[test]
    fn a_mapped_format_reads_from_a_file() {
        use crate::fst::FstWriteOptions;
        use std::io::Write as _;

        let source = sample();
        let mut bytes = Vec::new();
        ConstFst::<StdArc, u32>::write_fst(&source, &mut bytes, &FstWriteOptions::default())
            .expect("written");
        let mut file = tempfile::NamedTempFile::new().expect("a temporary file");
        file.write_all(&bytes).expect("written to the file");
        file.flush().expect("flushed");

        let read = Any::read_from_file(file.path(), &FstReadOptions::default())
            .expect("a const FST read through its header");
        assert_eq!(read.fst_type_name(), FstType::CONST_32);
        assert_eq!(shape(&read), shape(&source));
    }

    /// What an FST looks like from outside, for comparing two of them.
    fn shape<F: Fst<StdArc>>(fst: &F) -> (Option<i32>, Vec<(TropicalWeight, Vec<StdArc>)>) {
        let states: Vec<(TropicalWeight, Vec<StdArc>)> = fst
            .states()
            .map(|s| (fst.final_weight(s), fst.arcs(s).collect()))
            .collect();
        (fst.start(), states)
    }

    fn round_trip(bytes: Vec<u8>) -> Any {
        AnyFst::read(&mut Cursor::new(bytes), &FstReadOptions::default()).unwrap()
    }

    /// Every format reads back as the type its header names, with the same
    /// contents.
    #[test]
    fn every_format_is_recognized_from_its_header() {
        let opts = FstWriteOptions::default();
        let source = sample();
        let acc = acceptor();
        let want = shape(&source);
        let want_acc = shape(&acc);

        let mut bytes = Vec::new();
        VectorFst::write_fst(&source, &mut bytes, &opts).unwrap();
        let fst = round_trip(bytes);
        assert_eq!(fst.fst_type_name(), FstType::VECTOR);
        assert_eq!(shape(&fst), want);

        for (name, bytes) in [
            (FstType::CONST_32, {
                let mut b = Vec::new();
                ConstFst::<StdArc, u32>::write_fst(&source, &mut b, &opts).unwrap();
                b
            }),
            (FstType::CONST_64, {
                let mut b = Vec::new();
                ConstFst::<StdArc, u64>::write_fst(&source, &mut b, &opts).unwrap();
                b
            }),
        ] {
            let fst = round_trip(bytes);
            assert_eq!(fst.fst_type_name(), name, "{name}");
            assert_eq!(shape(&fst), want, "{name}");
        }

        // The compact formats keep one label per arc, so they take an acceptor.
        let mut b = Vec::new();
        CompactWeightedStringFst::<StdArc, u32>::new(
            &acc,
            Default::default(),
            CacheOptions::default(),
        )
        .unwrap()
        .write(&mut b, &opts)
        .unwrap();
        let fst = round_trip(b);
        assert_eq!(fst.fst_type_name(), FstType::COMPACT_WEIGHTED_STRING_32);
        assert_eq!(shape(&fst), want_acc);

        let mut b = Vec::new();
        CompactAcceptorFst::<StdArc, u64>::new(&acc, Default::default(), CacheOptions::default())
            .unwrap()
            .write(&mut b, &opts)
            .unwrap();
        let fst = round_trip(b);
        assert_eq!(fst.fst_type_name(), FstType::COMPACT_ACCEPTOR_64);
        assert_eq!(shape(&fst), want_acc);
    }

    /// An FST read as `AnyFst` writes back to the same bytes.
    #[test]
    fn writing_what_was_read_gives_the_same_bytes() {
        let opts = FstWriteOptions::default();
        let mut bytes = Vec::new();
        ConstFst::<StdArc, u32>::write_fst(&sample(), &mut bytes, &opts).unwrap();

        let fst = round_trip(bytes.clone());
        let mut again = Vec::new();
        fst.write(&mut again, &opts).unwrap();
        assert_eq!(again, bytes);
    }

    #[test]
    fn a_header_naming_a_type_sicada_cannot_read_is_refused() {
        let opts = FstWriteOptions::default();
        let mut bytes = Vec::new();
        VectorFst::write_fst(&sample(), &mut bytes, &opts).unwrap();

        // "vector" is 6 bytes; "vektor" is not a type at all.
        let at = bytes.windows(6).position(|w| w == b"vector").unwrap();
        bytes[at..at + 6].copy_from_slice(b"vektor");
        let Err(err) = AnyFst::<StdArc>::read(&mut Cursor::new(bytes), &FstReadOptions::default())
        else {
            panic!("a header naming no known type must not read")
        };
        assert!(format!("{err}").contains("unknown FST type"), "{err}");
    }

    #[test]
    fn an_fst_of_the_wrong_arc_type_is_refused() {
        use crate::arc::LogArc;
        use crate::weights::float_weight::LogWeight;

        let mut fst: VectorFst<LogArc> = VectorFst::new();
        fst.add_state();
        fst.set_start(0);
        fst.set_final(0, LogWeight::one());
        let mut bytes = Vec::new();
        VectorFst::write_fst(&fst, &mut bytes, &FstWriteOptions::default()).unwrap();

        let Err(err) = AnyFst::<StdArc>::read(&mut Cursor::new(bytes), &FstReadOptions::default())
        else {
            panic!("an FST of another arc type must not read")
        };
        assert!(format!("{err}").contains("arc not of type"), "{err}");
    }

    /// An `EditFst` wrapping a `ConstFst` comes back with both parts intact,
    /// which is the case upstream reads through an unchecked downcast.
    #[test]
    fn an_edit_fst_round_trips_with_whatever_it_wraps() {
        let opts = FstWriteOptions::default();
        let mut bytes = Vec::new();
        ConstFst::<StdArc, u32>::write_fst(&sample(), &mut bytes, &opts).unwrap();
        let wrapped = round_trip(bytes);

        let mut edited = EditFst::new(wrapped);
        edited.set_final(1, TropicalWeight(9.0));
        edited.add_arc(0, StdArc::new(7, 8, TropicalWeight(0.25), 2));
        let added = edited.add_state();
        edited.set_final(added, TropicalWeight(4.0));
        let want = shape(&edited);

        let mut bytes = Vec::new();
        edited.write(&mut bytes, &opts).unwrap();
        let fst = round_trip(bytes);

        assert_eq!(fst.fst_type_name(), FstType::EDIT);
        assert_eq!(shape(&fst), want);
        let AnyFst::Edit(inner) = &fst else {
            panic!("expected an edit FST")
        };
        assert_eq!(
            inner.wrapped().fst_type_name(),
            FstType::CONST_32,
            "the wrapped FST kept its own type"
        );
    }

    /// The edit maps are written in a fixed order, so the same FST always
    /// writes to the same bytes.
    #[test]
    fn an_edit_fst_writes_the_same_bytes_every_time() {
        let opts = FstWriteOptions::default();
        let mut bytes = Vec::new();
        VectorFst::write_fst(&sample(), &mut bytes, &opts).unwrap();

        let build = || {
            let mut edited = EditFst::new(round_trip(bytes.clone()));
            for state in [2, 0, 1] {
                edited.set_final(state, TropicalWeight(state as f32));
            }
            edited.add_arc(1, StdArc::new(5, 6, TropicalWeight(1.0), 0));
            let mut out = Vec::new();
            edited.write(&mut out, &opts).unwrap();
            out
        };
        assert_eq!(build(), build());
    }

    /// Symbol tables survive a round trip through the edit format, which
    /// upstream's loses.
    #[test]
    fn an_edit_fst_keeps_the_symbol_tables_of_what_it_wraps() {
        let mut syms = SymbolTable::new("input");
        syms.add_symbol("a", 1);
        let mut source = sample();
        source.set_input_symbols(Some(AtomicRc::new(syms)));

        let opts = FstWriteOptions::default();
        let mut bytes = Vec::new();
        VectorFst::write_fst(&source, &mut bytes, &opts).unwrap();
        let edited = EditFst::new(round_trip(bytes));

        let mut bytes = Vec::new();
        edited.write(&mut bytes, &opts).unwrap();
        let fst = round_trip(bytes);
        assert_eq!(fst.input_symbols().unwrap().name(), "input");
    }

    /// An edit FST wrapping an edit FST: the recursion the boxed arc iterator
    /// exists for.
    #[test]
    fn an_edit_fst_can_wrap_another_one() {
        let opts = FstWriteOptions::default();
        let mut bytes = Vec::new();
        VectorFst::write_fst(&sample(), &mut bytes, &opts).unwrap();

        let mut inner = EditFst::new(round_trip(bytes));
        inner.set_final(0, TropicalWeight(1.0));
        let mut bytes = Vec::new();
        inner.write(&mut bytes, &opts).unwrap();

        let mut outer = EditFst::new(round_trip(bytes));
        outer.add_arc(2, StdArc::new(9, 9, TropicalWeight(3.0), 0));
        let want = shape(&outer);

        let mut bytes = Vec::new();
        outer.write(&mut bytes, &opts).unwrap();
        let fst = round_trip(bytes);
        assert_eq!(shape(&fst), want);
        assert_eq!(fst.num_states(), 3);
    }

    /// Every type name sicada can produce is one it can also read back.
    #[test]
    fn the_names_a_header_can_carry_and_the_types_that_can_be_read_agree() {
        let opts = FstWriteOptions::default();
        let mut bytes = Vec::new();
        VectorFst::write_fst(&sample(), &mut bytes, &opts).unwrap();
        let vector = round_trip(bytes);

        // Every FstType in the enum's own list resolves back to itself.
        assert_eq!(FstType::from_name(vector.fst_type()), Some(FstType::VECTOR));
        for name in [
            FstType::EDIT,
            FstType::CONST_32,
            FstType::CONST_64,
            FstType::COMPACT_STRING_32,
            FstType::COMPACT_UNWEIGHTED_ACCEPTOR_64,
        ] {
            assert_eq!(
                FstType::from_name(name.as_str()),
                Some(name.clone()),
                "{name}"
            );
        }
    }
}
