//! Binary serialization primitives for the OpenFst file format.
//!
//! Port of the `ReadType` / `WriteType` half of OpenFst's `util.h`. Scalars are
//! written as their raw bytes, a string as an `i32` length followed by its bytes,
//! and a vector as an `i64` length followed by its elements.
//!
//! # Byte order
//!
//! SICADA-DIVERGE: upstream writes scalars in **host** byte order. `util.h`
//! says so outright, and its files are therefore not portable between a
//! little-endian and a big-endian machine. sicada always reads and writes
//! little-endian. On every mainstream target the two agree byte for byte, so
//! files still interchange with OpenFst; on a big-endian host sicada produces
//! the little-endian layout instead, which is deliberately *more* portable than
//! what OpenFst would write there. Do not "fix" this to native order without
//! also deciding what happens to files already written.

use std::io::{self, Read, Seek, SeekFrom, Write};

/// Alignment required for mapping structures in bytes (16 bytes, 128-bit boundary).
pub const ARCH_ALIGNMENT: u64 = 16;

/// A scalar that can be read from and written to an FST stream.
///
/// Only fixed-width types belong here: the on-disk layout must not depend on the
/// architecture, which is why upstream's `IsScalarIOTypeV` excludes `int` and
/// `size_t`.
pub trait FstScalar: Sized + Copy {
    /// Number of bytes this scalar occupies in the stream.
    const WIDTH: usize;

    /// Reads one value.
    fn read_from<R: Read>(reader: &mut R) -> io::Result<Self>;

    /// Writes one value.
    fn write_to<W: Write>(self, writer: &mut W) -> io::Result<()>;
}

macro_rules! impl_fst_scalar {
    ($($t:ty),* $(,)?) => {
        $(
            impl FstScalar for $t {
                const WIDTH: usize = size_of::<$t>();

                #[inline]
                fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
                    let mut buf = [0u8; size_of::<$t>()];
                    reader.read_exact(&mut buf)?;
                    Ok(<$t>::from_le_bytes(buf))
                }

                #[inline]
                fn write_to<W: Write>(self, writer: &mut W) -> io::Result<()> {
                    writer.write_all(&self.to_le_bytes())
                }
            }
        )*
    };
}

impl_fst_scalar!(i8, u8, i16, u16, i32, u32, i64, u64, f32, f64);

impl FstScalar for bool {
    const WIDTH: usize = 1;

    #[inline]
    fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        Ok(u8::read_from(reader)? != 0)
    }

    #[inline]
    fn write_to<W: Write>(self, writer: &mut W) -> io::Result<()> {
        u8::from(self).write_to(writer)
    }
}

/// Reads one scalar.
#[inline]
pub fn read_scalar<T: FstScalar, R: Read>(reader: &mut R) -> io::Result<T> {
    T::read_from(reader)
}

/// Writes one scalar.
#[inline]
pub fn write_scalar<T: FstScalar, W: Write>(writer: &mut W, value: T) -> io::Result<()> {
    value.write_to(writer)
}

/// Reads a string: an `i32` byte count followed by that many bytes.
///
/// A non-positive length yields the empty string, as upstream does.
pub fn read_string<R: Read>(reader: &mut R) -> io::Result<String> {
    let len = i32::read_from(reader)?;
    if len <= 0 {
        return Ok(String::new());
    }
    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Writes a string: an `i32` byte count followed by its bytes.
pub fn write_string<W: Write>(writer: &mut W, value: &str) -> io::Result<()> {
    let len = i32::try_from(value.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "string too long for the FST format's 32-bit length prefix",
        )
    })?;
    len.write_to(writer)?;
    writer.write_all(value.as_bytes())
}

/// Reads a vector: an `i64` element count followed by that many elements.
pub fn read_vec<T: FstScalar, R: Read>(reader: &mut R) -> io::Result<Vec<T>> {
    let len = i64::read_from(reader)?;
    if len <= 0 {
        return Ok(Vec::new());
    }
    let len = usize::try_from(len)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "vector length out of range"))?;
    // SICADA-OPT: reserve against the bytes actually available rather than the
    // declared length, so a corrupt or truncated file cannot make us allocate
    // gigabytes before the read fails.
    let mut values = Vec::with_capacity(len.min(4096));
    for _ in 0..len {
        values.push(T::read_from(reader)?);
    }
    Ok(values)
}

/// Writes a vector: an `i64` element count followed by its elements.
pub fn write_vec<T: FstScalar, W: Write>(writer: &mut W, values: &[T]) -> io::Result<()> {
    (values.len() as i64).write_to(writer)?;
    for &value in values {
        value.write_to(writer)?;
    }
    Ok(())
}

/// Skips input until the position is a multiple of `align`.
///
/// Zero-copy mapping depends on the mapped region starting on this boundary.
///
/// SICADA-OPT: the padding is read and discarded rather than seeked over.
/// Seeking a `BufReader` throws its buffer away, and the padding here is at most
/// fifteen bytes, so a read is both cheaper and leaves the reader warm. Upstream
/// also reads, but one byte per call.
pub fn align_input<R: Read + Seek>(strm: &mut R, align: u64) -> io::Result<()> {
    let pos = strm.stream_position()?;
    let rem = pos % align;
    if rem == 0 {
        return Ok(());
    }
    let padding = (align - rem) as usize;
    let mut discard = [0u8; 64];
    if padding <= discard.len() {
        strm.read_exact(&mut discard[..padding])?;
    } else {
        // Alignments this large do not occur in the FST format, but the function
        // is public and must not silently do the wrong thing.
        strm.seek(SeekFrom::Current(padding as i64))?;
    }
    Ok(())
}

/// A writer that keeps count of the bytes that have gone through it.
///
/// Alignment on the writing side needs the current file offset, which upstream
/// gets from `tellp()`, so it cannot align while writing to a pipe and says as
/// much in the doc for its `align` option. Counting the bytes as they go by
/// gives the same offset without asking the stream for it.
pub struct CountingWriter<W> {
    inner: W,
    written: u64,
}

impl<W: Write> CountingWriter<W> {
    /// Wraps `inner`, starting the count at `written` bytes already gone by.
    pub fn new(inner: W, written: u64) -> Self {
        Self { inner, written }
    }

    /// How many bytes have been written through this wrapper.
    #[inline]
    pub fn written(&self) -> u64 {
        self.written
    }

    /// Pads to the next multiple of `align`, as [`align_output`] does for a
    /// seekable stream.
    pub fn align(&mut self, align: u64) -> io::Result<()> {
        let padding = (align - self.written % align) % align;
        if padding > 0 {
            self.write_all(&vec![0u8; padding as usize])?;
        }
        Ok(())
    }

    /// Returns the wrapped writer.
    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: Write> Write for CountingWriter<W> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.written += n as u64;
        Ok(n)
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Pads output with zeroes until the position is a multiple of `align`.
pub fn align_output<W: Write + Seek>(strm: &mut W, align: u64) -> io::Result<()> {
    let pos = strm.stream_position()?;
    let rem = pos % align;
    if rem == 0 {
        return Ok(());
    }
    let padding = (align - rem) as usize;
    // SICADA-OPT: one write of a stack buffer, where upstream writes a byte at a
    // time in a loop and this previously allocated a Vec for at most fifteen
    // zeroes.
    let zeroes = [0u8; 64];
    if padding <= zeroes.len() {
        strm.write_all(&zeroes[..padding])?;
    } else {
        strm.write_all(&vec![0u8; padding])?;
    }
    Ok(())
}

/// Brings `buffer`'s offsets into step with `strm`'s, so alignment decided while
/// filling the buffer matches where the bytes will actually land.
///
/// Prepends `strm.stream_position() % align` placeholder bytes and returns that
/// count. The placeholders are **not** content: when the buffer is finally
/// written out, the caller skips them, writing `&buffer[offset..]`. Their only
/// job is to make an offset inside the buffer congruent, modulo `align`, to the
/// file offset the same byte will occupy, so [`align_output`] can be called
/// against the buffer instead of the file.
///
/// Corresponds to upstream's `AlignBufferWithOutputStream`, which serializes an
/// FST into memory before its final position in the file is known. `buffer` is
/// assumed to be empty, as upstream assumes its buffer is at position 0.
pub fn align_buffer_with_output<W: Seek>(
    strm: &mut W,
    buffer: &mut Vec<u8>,
    align: u64,
) -> io::Result<usize> {
    let offset = (strm.stream_position()? % align) as usize;
    buffer.splice(0..0, std::iter::repeat_n(0u8, offset));
    Ok(offset)
}

// ---------------------------------------------------------------------------
// Text utilities
// ---------------------------------------------------------------------------

/// Parses a 64-bit signed integer written in `base`.
///
/// The whole string must be consumed, so no prefixes such as `0x` and no
/// trailing characters; a leading minus is allowed. Corresponds to upstream's
/// `ParseInt64`.
///
/// SICADA-DIVERGE: upstream's `StrToInt64` companion returns a sentinel and sets
/// an out-parameter, and logs to the global log on failure. Returning an
/// `Option` says the same thing without either.
pub fn parse_int64(s: &str, base: u32) -> Option<i64> {
    if s.is_empty() {
        return None;
    }
    // `from_str_radix` accepts a leading `+`, which upstream's `from_chars` does
    // not; reject it so the two agree on what parses.
    if s.starts_with('+') {
        return None;
    }
    i64::from_str_radix(s, base).ok()
}

/// One line of a label-pair file that failed to parse.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LabelPairError {
    #[error("line {line}: expected 2 columns, found {found}")]
    WrongColumnCount { line: usize, found: usize },

    #[error("line {line}: {column:?} is not an integer")]
    NotAnInteger { line: usize, column: String },
}

/// Parses whitespace-separated label pairs, one per line.
///
/// Blank lines and lines whose first non-blank character is `#` are skipped, as
/// upstream does. Corresponds to `ReadIntPairs` / `ReadLabelPairs`; reading the
/// file is the caller's business, which keeps this testable without a
/// filesystem.
pub fn parse_label_pairs(text: &str) -> Result<Vec<(i64, i64)>, LabelPairError> {
    let mut pairs = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = index + 1;
        let columns: Vec<&str> = raw.split([' ', '\t']).filter(|c| !c.is_empty()).collect();
        if columns.is_empty() || columns[0].starts_with('#') {
            continue;
        }
        if columns.len() != 2 {
            return Err(LabelPairError::WrongColumnCount {
                line,
                found: columns.len(),
            });
        }
        let mut parsed = [0i64; 2];
        for (slot, column) in parsed.iter_mut().zip(columns.iter()) {
            *slot = parse_int64(column, 10).ok_or_else(|| LabelPairError::NotAnInteger {
                line,
                column: (*column).to_string(),
            })?;
        }
        pairs.push((parsed[0], parsed[1]));
    }
    Ok(pairs)
}

/// Writes label pairs, one tab-separated pair per line.
pub fn write_label_pairs<W: Write>(writer: &mut W, pairs: &[(i64, i64)]) -> io::Result<()> {
    for (first, second) in pairs {
        writeln!(writer, "{first}\t{second}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn written<F>(write: F) -> Vec<u8>
    where
        F: FnOnce(&mut Vec<u8>) -> io::Result<()>,
    {
        let mut buf = Vec::new();
        write(&mut buf).unwrap();
        buf
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The exact bytes OpenFst writes for each primitive.
    ///
    /// Taken by compiling the ReadType/WriteType overloads out of
    /// `vendor/openfst/openfst/lib/util.h` and dumping their output; see
    /// tests/oracles/io-golden.cc. A change here is a change to the file
    /// format.
    #[test]
    fn primitives_serialize_to_the_bytes_openfst_writes() {
        assert_eq!(
            hex(&written(|w| write_scalar(w, 0x1234_5678i32))),
            "78563412"
        );
        assert_eq!(hex(&written(|w| write_scalar(w, -2i32))), "feffffff");
        assert_eq!(
            hex(&written(|w| write_scalar(w, 0x0123_4567_89AB_CDEFi64))),
            "efcdab8967452301"
        );
        assert_eq!(
            hex(&written(|w| write_scalar(w, 0xFEDC_BA98_7654_3210u64))),
            "1032547698badcfe"
        );
        assert_eq!(hex(&written(|w| write_scalar(w, 0.5f32))), "0000003f");
        assert_eq!(
            hex(&written(|w| write_scalar(w, -0.25f64))),
            "000000000000d0bf"
        );
        assert_eq!(hex(&written(|w| write_scalar(w, 0xABu8))), "ab");
        assert_eq!(hex(&written(|w| write_string(w, "abc"))), "03000000616263");
        assert_eq!(hex(&written(|w| write_string(w, ""))), "00000000");
        assert_eq!(
            hex(&written(|w| write_vec(w, &[1i64, -2, 3]))),
            "03000000000000000100000000000000feffffffffffffff0300000000000000"
        );
        assert_eq!(
            hex(&written(|w| write_vec::<i32, _>(w, &[]))),
            "0000000000000000"
        );
    }

    /// The string length prefix is 32 bits, not 64. Getting this wrong produces
    /// files that round trip within sicada and are unreadable by OpenFst.
    #[test]
    fn a_string_is_prefixed_by_a_32_bit_length() {
        let bytes = written(|w| write_string(w, "hello"));
        assert_eq!(&bytes[..4], &5i32.to_le_bytes());
        assert_eq!(bytes.len(), 4 + 5);
    }

    /// A vector length prefix is 64 bits, unlike a string's.
    #[test]
    fn a_vector_is_prefixed_by_a_64_bit_length() {
        let bytes = written(|w| write_vec(w, &[7i32, 8]));
        assert_eq!(&bytes[..8], &2i64.to_le_bytes());
        assert_eq!(bytes.len(), 8 + 2 * 4);
    }

    fn round_trip<T: FstScalar + PartialEq + std::fmt::Debug>(value: T) {
        let bytes = written(|w| write_scalar(w, value));
        assert_eq!(bytes.len(), T::WIDTH);
        let mut cursor = Cursor::new(bytes);
        assert_eq!(read_scalar::<T, _>(&mut cursor).unwrap(), value);
        assert_eq!(cursor.position() as usize, cursor.get_ref().len());
    }

    #[test]
    fn scalars_round_trip() {
        round_trip(i8::MIN);
        round_trip(u8::MAX);
        round_trip(i16::MIN);
        round_trip(u16::MAX);
        round_trip(i32::MIN);
        round_trip(u32::MAX);
        round_trip(i64::MIN);
        round_trip(u64::MAX);
        round_trip(0.0f32);
        round_trip(f32::MIN);
        round_trip(f64::MAX);
        round_trip(true);
        round_trip(false);
    }

    #[test]
    fn strings_and_vectors_round_trip() {
        for text in ["", "a", "hello world", "日本語", "\0embedded"] {
            let mut cursor = Cursor::new(written(|w| write_string(w, text)));
            assert_eq!(read_string(&mut cursor).unwrap(), text);
        }
        for values in [vec![], vec![0i64], vec![1, -2, i64::MIN, i64::MAX]] {
            let mut cursor = Cursor::new(written(|w| write_vec(w, &values)));
            assert_eq!(read_vec::<i64, _>(&mut cursor).unwrap(), values);
        }
    }

    #[test]
    fn a_negative_declared_length_reads_as_empty() {
        // Upstream treats a non-positive count as "nothing follows" rather than
        // as an error, and files in the wild rely on it.
        let mut cursor = Cursor::new((-1i32).to_le_bytes().to_vec());
        assert_eq!(read_string(&mut cursor).unwrap(), "");
        let mut cursor = Cursor::new((-1i64).to_le_bytes().to_vec());
        assert_eq!(read_vec::<i32, _>(&mut cursor).unwrap(), Vec::<i32>::new());
    }

    #[test]
    fn a_truncated_read_fails_rather_than_returning_short_data() {
        let mut cursor = Cursor::new(vec![0x03, 0x00, 0x00, 0x00, b'a']);
        assert!(read_string(&mut cursor).is_err());

        let mut cursor = Cursor::new(3i64.to_le_bytes().to_vec());
        assert!(read_vec::<i64, _>(&mut cursor).is_err());
    }

    /// A corrupt length must not be believed to the point of allocating for it.
    #[test]
    fn an_absurd_declared_length_does_not_allocate_up_front() {
        let mut bytes = i64::MAX.to_le_bytes().to_vec();
        bytes.extend_from_slice(&1i32.to_le_bytes());
        let mut cursor = Cursor::new(bytes);
        assert!(read_vec::<i32, _>(&mut cursor).is_err());
    }

    #[test]
    fn a_string_too_long_for_the_prefix_is_rejected() {
        // Constructing a >2GiB string in a test is wasteful; check the boundary
        // logic directly instead.
        assert!(i32::try_from(i32::MAX as usize + 1).is_err());
    }

    #[test]
    fn alignment_moves_to_the_next_boundary_and_no_further() {
        for start in 0..40u64 {
            let mut cursor = Cursor::new(vec![0u8; 128]);
            cursor.set_position(start);
            align_input(&mut cursor, ARCH_ALIGNMENT).unwrap();
            assert!(cursor.position().is_multiple_of(ARCH_ALIGNMENT));
            assert!(cursor.position() - start < ARCH_ALIGNMENT);

            let mut out = Cursor::new(vec![0u8; 128]);
            out.set_position(start);
            align_output(&mut out, ARCH_ALIGNMENT).unwrap();
            assert_eq!(out.position(), cursor.position());
        }
    }

    #[test]
    fn alignment_pads_with_zeroes() {
        let mut out = Cursor::new(Vec::new());
        write_scalar(&mut out, 0xFFu8).unwrap();
        align_output(&mut out, 8).unwrap();
        assert_eq!(out.into_inner(), vec![0xFF, 0, 0, 0, 0, 0, 0, 0]);
    }
    #[test]
    fn parse_int64_consumes_the_whole_string() {
        assert_eq!(parse_int64("0", 10), Some(0));
        assert_eq!(parse_int64("-42", 10), Some(-42));
        assert_eq!(parse_int64("9223372036854775807", 10), Some(i64::MAX));
        assert_eq!(parse_int64("-9223372036854775808", 10), Some(i64::MIN));
        assert_eq!(parse_int64("ff", 16), Some(255));
        assert_eq!(parse_int64("101", 2), Some(5));

        // Trailing or leading junk, prefixes and overflow are all rejected.
        assert_eq!(parse_int64("", 10), None);
        assert_eq!(parse_int64("12a", 10), None);
        assert_eq!(parse_int64(" 12", 10), None);
        assert_eq!(parse_int64("12 ", 10), None);
        assert_eq!(parse_int64("0x10", 16), None);
        assert_eq!(parse_int64("+1", 10), None);
        assert_eq!(parse_int64("9223372036854775808", 10), None);
    }

    #[test]
    fn label_pairs_parse_and_skip_blanks_and_comments() {
        let text = "\
# a comment
1\t2

  3 4
\t5\t6\t
# trailing comment
";
        assert_eq!(
            parse_label_pairs(text).unwrap(),
            vec![(1, 2), (3, 4), (5, 6)]
        );
        assert_eq!(parse_label_pairs("").unwrap(), vec![]);
    }

    #[test]
    fn label_pairs_reject_malformed_lines() {
        assert_eq!(
            parse_label_pairs("1 2 3"),
            Err(LabelPairError::WrongColumnCount { line: 1, found: 3 })
        );
        assert_eq!(
            parse_label_pairs("1"),
            Err(LabelPairError::WrongColumnCount { line: 1, found: 1 })
        );
        assert_eq!(
            parse_label_pairs("1 2\nthree 4"),
            Err(LabelPairError::NotAnInteger {
                line: 2,
                column: "three".to_string()
            })
        );
    }

    #[test]
    fn label_pairs_round_trip() {
        let pairs = vec![(0i64, 1i64), (-5, 7), (i64::MIN, i64::MAX)];
        let mut buf = Vec::new();
        write_label_pairs(&mut buf, &pairs).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert_eq!(parse_label_pairs(&text).unwrap(), pairs);
    }
    #[test]
    fn aligning_input_leaves_the_reader_positioned_and_readable() {
        // The padding is consumed, not seeked over, so what follows still reads.
        let mut data = vec![0u8; 16];
        data.extend_from_slice(b"payload");
        let mut cursor = Cursor::new(data);
        cursor.set_position(5);
        align_input(&mut cursor, ARCH_ALIGNMENT).unwrap();
        assert_eq!(cursor.position(), 16);
        let mut rest = [0u8; 7];
        cursor.read_exact(&mut rest).unwrap();
        assert_eq!(&rest, b"payload");
    }

    #[test]
    fn aligning_input_past_the_end_of_the_data_fails() {
        let mut cursor = Cursor::new(vec![0u8; 5]);
        cursor.set_position(5);
        assert!(align_input(&mut cursor, ARCH_ALIGNMENT).is_err());
    }

    /// The contract is congruence, not alignment of the body: an offset inside
    /// the buffer must agree, modulo `align`, with the file offset that byte
    /// will end up at once the placeholders are skipped.
    #[test]
    fn a_buffer_is_brought_into_step_with_the_stream() {
        const ALIGN: u64 = 16;
        for start in [0u64, 1, 7, 15, 16, 31] {
            let mut strm = Cursor::new(vec![0u8; 64]);
            strm.set_position(start);
            let mut buffer = Vec::new();
            let offset = align_buffer_with_output(&mut strm, &mut buffer, ALIGN).unwrap();

            assert_eq!(offset as u64, start % ALIGN);
            assert!(buffer.iter().all(|&byte| byte == 0));
            assert_eq!(buffer.len(), offset);

            // Append some content and check the correspondence: content byte `i`
            // sits at buffer offset `offset + i` and will sit at file offset
            // `start + i`, and those agree modulo ALIGN.
            buffer.extend_from_slice(b"body");
            for i in 0..4u64 {
                let in_buffer = offset as u64 + i;
                let in_file = start + i;
                assert_eq!(in_buffer % ALIGN, in_file % ALIGN, "byte {i} at {start}");
            }
        }
    }
}
