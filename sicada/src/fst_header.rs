//! The header every FST file begins with.
//!
//! Port of `FstHeader` in OpenFst's `fst.h` and `fst.cc`. The layout is a magic
//! number, the FST and arc type names, a version, flags, the property bits, and
//! the start state with the state and arc counts. Everything here is part of the
//! binary format; `tests/oracles/fst-header-golden.cc` produced the bytes the
//! tests pin.

use std::fmt;
use std::io::{Read, Seek, SeekFrom, Write};

use crate::error::OpenFstError;
use crate::utils::io::{read_scalar, read_string, write_scalar, write_string};

pub const FST_MAGIC_NUMBER: i32 = 2125659606;

/// Flags recorded in the header.
///
/// The alignment flag matters for reading: a region that is not aligned cannot
/// be mapped and has to be read instead.
pub mod flags {
    /// The file carries an input symbol table.
    pub const HAS_ISYMBOLS: u32 = 0x1;
    /// The file carries an output symbol table.
    pub const HAS_OSYMBOLS: u32 = 0x2;
    /// The FST's regions are aligned for memory mapping.
    pub const IS_ALIGNED: u32 = 0x4;
}

/// FST Binary Header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FstHeader {
    pub fst_type: String,
    pub arc_type: String,
    pub version: i32,
    /// See [`flags`]. Upstream stores this as `uint32_t`.
    pub flags: u32,
    pub properties: u64,
    pub start: i64,
    pub num_states: i64,
    pub num_arcs: i64,
}

impl FstHeader {
    /// Reads a header, consuming it.
    ///
    /// SICADA-DIVERGE: upstream takes a `rewind` flag and demands a seekable
    /// stream whether or not it is set. Splitting the two apart means reading a
    /// header off a pipe does not need one, which is the case upstream's own
    /// `stream_write` option exists to support on the writing side.
    /// [`peek`](Self::peek) is the `rewind = true` case.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, OpenFstError> {
        let magic: i32 = read_scalar(&mut reader)?;
        if magic != FST_MAGIC_NUMBER {
            return Err(OpenFstError::InvalidMagicNumber {
                expected: FST_MAGIC_NUMBER,
                found: magic,
            });
        }

        let fst_type = read_string(&mut reader)?;
        let arc_type = read_string(&mut reader)?;

        let version: i32 = read_scalar(&mut reader)?;
        let flags: u32 = read_scalar(&mut reader)?;
        let properties: u64 = read_scalar(&mut reader)?;
        let start: i64 = read_scalar(&mut reader)?;
        let num_states: i64 = read_scalar(&mut reader)?;
        let num_arcs: i64 = read_scalar(&mut reader)?;

        Ok(Self {
            fst_type,
            arc_type,
            version,
            flags,
            properties,
            start,
            num_states,
            num_arcs,
        })
    }

    /// Reads a header without consuming it, so a caller can probe a stream.
    ///
    /// The stream is left where it started whether or not a header was there.
    pub fn peek<R: Read + Seek>(mut reader: R) -> Result<Self, OpenFstError> {
        let pos = reader.stream_position()?;
        let header = Self::read(&mut reader);
        reader.seek(SeekFrom::Start(pos))?;
        header
    }

    /// Writes the header.
    pub fn write<W: Write>(&self, mut writer: W) -> Result<(), std::io::Error> {
        write_scalar(&mut writer, FST_MAGIC_NUMBER)?;
        write_string(&mut writer, &self.fst_type)?;
        write_string(&mut writer, &self.arc_type)?;
        write_scalar(&mut writer, self.version)?;
        write_scalar(&mut writer, self.flags)?;
        write_scalar(&mut writer, self.properties)?;
        write_scalar(&mut writer, self.start)?;
        write_scalar(&mut writer, self.num_states)?;
        write_scalar(&mut writer, self.num_arcs)?;
        Ok(())
    }
}

/// The text upstream's `FstHeader::DebugString` produces, so a header logged by
/// sicada and one logged by OpenFst read the same.
impl fmt::Display for FstHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "fsttype: \"{}\" arctype: \"{}\" version: \"{}\" flags: \"{}\" \
             properties: \"{}\" start: \"{}\" numstates: \"{}\" numarcs: \"{}\"",
            self.fst_type,
            self.arc_type,
            self.version,
            self.flags,
            self.properties,
            self.start,
            self.num_states,
            self.num_arcs
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample() -> FstHeader {
        FstHeader {
            fst_type: "vector".to_string(),
            arc_type: "standard".to_string(),
            version: 2,
            flags: flags::HAS_ISYMBOLS | flags::HAS_OSYMBOLS,
            properties: 0x0000_0000_0001_0007,
            start: 0,
            num_states: 5,
            num_arcs: 7,
        }
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The exact bytes OpenFst writes for a header.
    ///
    /// Taken from tests/oracles/fst-header-golden.cc, which follows
    /// FstHeader::Write with the util.h overloads. Every FST file starts with
    /// this, so a change here is a change to the format.
    #[test]
    fn a_header_serializes_to_the_bytes_openfst_writes() {
        const GOLDEN: &str = "d6fdb27e06000000766563746f72080000007374616e6461726402000000030000000700010000000000000000000000000005000000000000000700000000000000";

        let mut bytes = Vec::new();
        sample().write(&mut bytes).unwrap();
        assert_eq!(hex(&bytes), GOLDEN);
        assert_eq!(bytes.len(), 66);
    }

    #[test]
    fn a_header_round_trips() {
        let header = sample();
        let mut bytes = Vec::new();
        header.write(&mut bytes).unwrap();
        let read = FstHeader::read(Cursor::new(bytes)).unwrap();
        assert_eq!(read, header);
    }

    #[test]
    fn the_magic_number_is_what_openfst_uses() {
        assert_eq!(FST_MAGIC_NUMBER, 2125659606);
        let mut bytes = Vec::new();
        sample().write(&mut bytes).unwrap();
        assert_eq!(&bytes[..4], &FST_MAGIC_NUMBER.to_le_bytes());
    }

    #[test]
    fn a_wrong_magic_number_is_rejected() {
        let mut bytes = Vec::new();
        sample().write(&mut bytes).unwrap();
        bytes[0] ^= 0xFF;
        assert!(matches!(
            FstHeader::read(Cursor::new(bytes)),
            Err(OpenFstError::InvalidMagicNumber { .. })
        ));
    }

    /// Probing for a header must leave the stream where it found it, so a caller
    /// can try several readers against the same input.
    #[test]
    fn peeking_restores_the_position_on_a_mismatch() {
        let mut cursor = Cursor::new(vec![0u8; 32]);
        cursor.set_position(4);
        assert!(FstHeader::peek(&mut cursor).is_err());
        assert_eq!(cursor.position(), 4);

        // A plain read consumes the magic number.
        cursor.set_position(4);
        assert!(FstHeader::read(&mut cursor).is_err());
        assert_eq!(cursor.position(), 8);
    }

    #[test]
    fn a_truncated_header_is_an_error() {
        let mut bytes = Vec::new();
        sample().write(&mut bytes).unwrap();
        for cut in [4, 10, 30, 65] {
            assert!(
                FstHeader::read(Cursor::new(bytes[..cut].to_vec())).is_err(),
                "a header cut to {cut} bytes should not parse"
            );
        }
    }

    #[test]
    fn the_flag_values_match_openfst() {
        assert_eq!(flags::HAS_ISYMBOLS, 0x1);
        assert_eq!(flags::HAS_OSYMBOLS, 0x2);
        assert_eq!(flags::IS_ALIGNED, 0x4);
    }
}
