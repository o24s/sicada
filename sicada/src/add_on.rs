//! Attaching an arbitrary serializable object to an FST.
//!
//! Port of OpenFst's `add-on.h`. Some FSTs carry precomputed data alongside the
//! states and arcs (`matcher-fst.h` attaches the tables its lookahead matcher
//! needs), and that data has to survive a round trip through a file. An add-on
//! is that data, plus how to read and write it.

use std::io::{Read, Write};
use std::sync::Arc as StdArc;

use crate::error::OpenFstError;
use crate::fst::{FstReadOptions, FstWriteOptions};
use crate::utils::io::{read_scalar, write_scalar};

#[cfg(feature = "fst-types")]
pub use impl_::AddOnImpl;

/// Identifies stream data as belonging to an add-on FST.
pub const ADD_ON_MAGIC_NUMBER: i32 = 446_681_434;

/// Data that can travel with an FST through a file.
///
/// Upstream states this contract in a comment, as `T* Read(std::istream &)` and
/// `bool Write(std::ostream &)`, and relies on the template instantiating. A
/// trait says the same thing where the compiler can check it.
pub trait AddOn: Sized {
    /// Reads the object, having already consumed whatever precedes it.
    fn read<R: Read>(reader: &mut R, opts: &FstReadOptions) -> Result<Self, OpenFstError>;

    /// Writes the object.
    fn write<W: Write>(&self, writer: &mut W, opts: &FstWriteOptions) -> Result<(), OpenFstError>;
}

/// An add-on with nothing to save.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NullAddOn;

impl AddOn for NullAddOn {
    fn read<R: Read>(_reader: &mut R, _opts: &FstReadOptions) -> Result<Self, OpenFstError> {
        Ok(Self)
    }

    fn write<W: Write>(
        &self,
        _writer: &mut W,
        _opts: &FstWriteOptions,
    ) -> Result<(), OpenFstError> {
        Ok(())
    }
}

/// Two add-ons travelling together, either of which may be absent.
///
/// Each half is preceded on the wire by a `bool` saying whether it follows, so
/// an absent half costs one byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddOnPair<A1, A2> {
    first: Option<StdArc<A1>>,
    second: Option<StdArc<A2>>,
}

impl<A1, A2> AddOnPair<A1, A2> {
    /// Pairs two add-ons, either of which may be absent.
    pub fn new(first: Option<StdArc<A1>>, second: Option<StdArc<A2>>) -> Self {
        Self { first, second }
    }

    /// The first half, if present.
    pub fn first(&self) -> Option<&A1> {
        self.first.as_deref()
    }

    /// The second half, if present.
    pub fn second(&self) -> Option<&A2> {
        self.second.as_deref()
    }

    /// The first half, sharing ownership.
    pub fn shared_first(&self) -> Option<StdArc<A1>> {
        self.first.clone()
    }

    /// The second half, sharing ownership.
    pub fn shared_second(&self) -> Option<StdArc<A2>> {
        self.second.clone()
    }
}

impl<A1: AddOn, A2: AddOn> AddOn for AddOnPair<A1, A2> {
    fn read<R: Read>(reader: &mut R, opts: &FstReadOptions) -> Result<Self, OpenFstError> {
        let has_first: bool = read_scalar(reader)?;
        let first = if has_first {
            Some(StdArc::new(A1::read(reader, opts)?))
        } else {
            None
        };
        let has_second: bool = read_scalar(reader)?;
        let second = if has_second {
            Some(StdArc::new(A2::read(reader, opts)?))
        } else {
            None
        };
        Ok(Self::new(first, second))
    }

    fn write<W: Write>(&self, writer: &mut W, opts: &FstWriteOptions) -> Result<(), OpenFstError> {
        write_scalar(writer, self.first.is_some())?;
        if let Some(first) = &self.first {
            first.write(writer, opts)?;
        }
        write_scalar(writer, self.second.is_some())?;
        if let Some(second) = &self.second {
            second.write(writer, opts)?;
        }
        Ok(())
    }
}

#[cfg(feature = "fst-types")]
mod impl_ {
    use std::io::{Read, Seek, Write};
    use std::sync::Arc as StdArc;

    use super::{ADD_ON_MAGIC_NUMBER, AddOn};
    use crate::arc::ArcStateId;
    use crate::error::OpenFstError;
    use crate::fst::{
        ExpandedFst, Fst, FstReadOptions, FstWriteOptions, read_fst_header, write_fst_header,
    };
    use crate::fst_header::FstHeader;
    use crate::fst_type::FstType;
    use crate::fsts::any_fst::AnyFst;
    use crate::properties::K_FST_PROPERTIES;
    use crate::utils::io::{FstScalar, read_scalar, write_scalar};
    use crate::weight::WeightIo;

    /// An FST with an add-on attached, and the file format that keeps the two
    /// together.
    ///
    /// Port of upstream's `internal::AddOnImpl`. The bytes are: this FST's own
    /// header, the add-on magic number, the contained FST *with its own header*,
    /// and then the add-on itself behind a `bool` saying whether it is there.
    ///
    /// SICADA-DIVERGE: upstream is templated on the contained FST type, so a
    /// `MatcherFst<ConstFst<StdArc>, …>` can only ever hold a `ConstFst`. The
    /// contained FST is an [`AnyFst`] here, resolved to whatever its header
    /// says it is. That is the same closed-enum dispatch
    /// [`EditFst`](crate::fsts::edit_fst::EditFst) uses, and the reason reading
    /// works without a dynamic registry.
    pub struct AddOnImpl<'f, A: crate::arc::Arc + 'static, T>
    where
        A::Weight: Copy,
    {
        fst: AnyFst<'f, A>,
        /// The name this FST goes by in a header, e.g. `ilabel_lookahead`.
        fst_type: FstType,
        add_on: Option<StdArc<T>>,
    }

    /// The version this writes, matching upstream's `kFileVersion`.
    const FILE_VERSION: i32 = 1;
    /// The oldest version it reads, matching upstream's `kMinFileVersion`.
    const MIN_FILE_VERSION: i32 = 1;

    impl<'f, A: crate::arc::Arc + 'static, T> AddOnImpl<'f, A, T>
    where
        A::Weight: Copy,
    {
        /// Attaches `add_on` to `fst`, which will go by `fst_type` in a header.
        pub fn new(fst: AnyFst<'f, A>, fst_type: FstType, add_on: Option<StdArc<T>>) -> Self {
            Self {
                fst,
                fst_type,
                add_on,
            }
        }

        /// The FST underneath.
        pub fn fst(&self) -> &AnyFst<'f, A> {
            &self.fst
        }

        /// The name this FST goes by in a header.
        pub fn fst_type(&self) -> FstType {
            self.fst_type.clone()
        }

        /// What is attached, if anything.
        pub fn add_on(&self) -> Option<&T> {
            self.add_on.as_deref()
        }

        /// What is attached, sharing ownership.
        pub fn shared_add_on(&self) -> Option<StdArc<T>> {
            self.add_on.clone()
        }

        /// Attaches something else.
        pub fn set_add_on(&mut self, add_on: Option<StdArc<T>>) {
            self.add_on = add_on;
        }
    }

    impl<'f, A, T> AddOnImpl<'f, A, T>
    where
        A: crate::arc::Arc + 'static,
        A::Label: FstScalar,
        A::StateId: FstScalar,
        A::Weight: Copy + WeightIo,
        T: AddOn,
    {
        /// Writes the header, the contained FST and the add-on.
        pub fn write<W: Write>(
            &self,
            writer: &mut W,
            opts: &FstWriteOptions,
        ) -> Result<(), OpenFstError> {
            let header = FstHeader {
                fst_type: self.fst_type.as_str().to_string(),
                arc_type: A::type_name().as_str().to_string(),
                version: FILE_VERSION,
                flags: 0,
                properties: self.fst.properties(K_FST_PROPERTIES, false),
                start: self.fst.start().map_or(-1, |s| s.as_usize() as i64),
                num_states: self.fst.num_states() as i64,
                num_arcs: -1,
            };
            // The contained FST writes its own header and its own symbol tables;
            // a second copy out here could only disagree with them.
            let header_opts = FstWriteOptions {
                write_isymbols: false,
                write_osymbols: false,
                ..opts.clone()
            };
            write_fst_header(writer, &header_opts, &header, None, None)?;
            write_scalar(writer, ADD_ON_MAGIC_NUMBER)?;
            let contained = FstWriteOptions {
                write_header: true,
                ..opts.clone()
            };
            self.fst.write(writer, &contained)?;
            write_scalar(writer, self.add_on.is_some())?;
            if let Some(add_on) = &self.add_on {
                add_on.write(writer, opts)?;
            }
            Ok(())
        }
    }

    impl<A, T> AddOnImpl<'static, A, T>
    where
        A: crate::arc::Arc + 'static,
        A::Label: FstScalar,
        A::StateId: FstScalar,
        A::Weight: Copy + WeightIo,
        T: AddOn,
    {
        /// Reads what [`write`](Self::write) wrote, refusing anything whose header
        /// does not name `fst_type`.
        pub fn read<R: Read + Seek>(
            reader: &mut R,
            opts: &FstReadOptions,
            fst_type: FstType,
        ) -> Result<Self, OpenFstError> {
            read_fst_header::<A, _>(reader, opts, fst_type.as_str(), MIN_FILE_VERSION)?;
            let magic: i32 = read_scalar(reader)?;
            if magic != ADD_ON_MAGIC_NUMBER {
                return Err(OpenFstError::InvalidFstHeader(format!(
                    "{}: bad add-on header",
                    opts.source
                )));
            }
            // The contained FST carries its own header, so the outer one must not
            // be offered to it.
            let contained = FstReadOptions {
                header: None,
                ..opts.clone()
            };
            let fst = AnyFst::read(reader, &contained)?;
            let has_add_on: bool = read_scalar(reader)?;
            let add_on = if has_add_on {
                Some(StdArc::new(T::read(reader, &contained)?))
            } else {
                None
            };
            Ok(Self::new(fst, fst_type, add_on))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// A stand-in for real add-on data, so the framing can be tested without a
    /// matcher.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Counter(i32);

    impl AddOn for Counter {
        fn read<R: Read>(reader: &mut R, _opts: &FstReadOptions) -> Result<Self, OpenFstError> {
            Ok(Self(read_scalar(reader)?))
        }

        fn write<W: Write>(
            &self,
            writer: &mut W,
            _opts: &FstWriteOptions,
        ) -> Result<(), OpenFstError> {
            write_scalar(writer, self.0).map_err(Into::into)
        }
    }

    fn round_trip<T: AddOn>(value: &T) -> T {
        let mut bytes = Vec::new();
        value
            .write(&mut bytes, &FstWriteOptions::default())
            .unwrap();
        T::read(&mut Cursor::new(bytes), &FstReadOptions::default()).unwrap()
    }

    #[test]
    fn the_magic_number_matches_openfst() {
        assert_eq!(ADD_ON_MAGIC_NUMBER, 446_681_434);
    }

    #[test]
    fn the_null_add_on_writes_nothing() {
        let mut bytes = Vec::new();
        NullAddOn
            .write(&mut bytes, &FstWriteOptions::default())
            .unwrap();
        assert!(bytes.is_empty());
        assert_eq!(round_trip(&NullAddOn), NullAddOn);
    }

    #[test]
    fn a_pair_round_trips_with_both_halves() {
        let pair = AddOnPair::new(
            Some(StdArc::new(Counter(7))),
            Some(StdArc::new(Counter(-3))),
        );
        let read = round_trip(&pair);
        assert_eq!(read.first(), Some(&Counter(7)));
        assert_eq!(read.second(), Some(&Counter(-3)));
    }

    /// An absent half costs one byte and comes back absent, so an FST can carry
    /// only the tables it actually has.
    #[test]
    fn an_absent_half_costs_one_byte() {
        let pair: AddOnPair<Counter, Counter> = AddOnPair::new(None, Some(StdArc::new(Counter(1))));
        let mut bytes = Vec::new();
        pair.write(&mut bytes, &FstWriteOptions::default()).unwrap();
        // false, then true, then the four bytes of the counter.
        assert_eq!(bytes.len(), 1 + 1 + 4);
        assert_eq!(bytes[0], 0);
        assert_eq!(bytes[1], 1);

        let read = round_trip(&pair);
        assert!(read.first().is_none());
        assert_eq!(read.second(), Some(&Counter(1)));
    }

    #[test]
    fn a_pair_with_neither_half_is_two_bytes() {
        let pair: AddOnPair<Counter, Counter> = AddOnPair::new(None, None);
        let mut bytes = Vec::new();
        pair.write(&mut bytes, &FstWriteOptions::default()).unwrap();
        assert_eq!(bytes, vec![0, 0]);
        let read = round_trip(&pair);
        assert!(read.first().is_none() && read.second().is_none());
    }

    #[test]
    fn pairs_nest() {
        type Nested = AddOnPair<AddOnPair<Counter, Counter>, Counter>;
        let inner = AddOnPair::new(Some(StdArc::new(Counter(1))), None);
        let outer: Nested = AddOnPair::new(Some(StdArc::new(inner)), Some(StdArc::new(Counter(2))));
        let read = round_trip(&outer);
        assert_eq!(read.first().unwrap().first(), Some(&Counter(1)));
        assert!(read.first().unwrap().second().is_none());
        assert_eq!(read.second(), Some(&Counter(2)));
    }
}
