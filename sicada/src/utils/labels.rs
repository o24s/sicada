//! Conversions between byte strings, UTF-8 byte streams and label sequences.
//!
//! Port of OpenFst's `icu.h`, which implements an **unrestricted Thompson/Pike
//! UTF-8** parser and serializer. Standard UTF-8 is a restricted subset of that
//! encoding: Thompson/Pike allows sequences of up to six bytes carrying 31 bits,
//! and places no limit at `U+10FFFF`, forbids nothing around the surrogate range,
//! and rejects no overlong form. See <http://en.wikipedia.org/wiki/UTF-8>.
//!
//! That is why these functions exist at all instead of deferring to `str`:
//! Rust's `char` and `str` only model Unicode scalar values, so they cannot round
//! trip the label sequences OpenFst writes into string FSTs. Serialization
//! therefore produces a `Vec<u8>` rather than a `String`: the output is only
//! valid UTF-8 when every label is a Unicode scalar value.

use crate::error::LabelStringError;

/// Copies a byte string into labels, one label per byte.
///
/// Usable with as little as 8 bits of label precision.
pub fn byte_string_to_labels<L: From<u8>>(bytes: &[u8], labels: &mut Vec<L>) {
    labels.reserve(bytes.len());
    labels.extend(bytes.iter().copied().map(L::from));
}

/// Parses an unrestricted Thompson/Pike UTF-8 byte stream into labels.
///
/// Usable with 16 bits of label precision for the Basic Multilingual Plane, 21
/// bits for all of Unicode, and 31 bits for the full unrestricted encoding.
///
/// SICADA-DIVERGE: upstream silently truncates a code point that does not fit
/// the label type ("truncating if necessary"). Silent truncation turns one label
/// into a different, valid-looking label, so this returns
/// [`LabelStringError::LabelOutOfRange`] instead. Callers that want the old
/// behaviour can mask the values themselves.
pub fn utf8_string_to_labels<L>(bytes: &[u8], labels: &mut Vec<L>) -> Result<(), LabelStringError>
where
    L: TryFrom<u32>,
{
    let mut i = 0;
    while i < bytes.len() {
        let lead = bytes[i];
        let offset = i;
        i += 1;

        if lead & 0x80 == 0 {
            push_label(labels, u32::from(lead), offset)?;
            continue;
        }
        if lead & 0xc0 == 0x80 {
            return Err(LabelStringError::ContinuationAsLeadByte { byte: lead, offset });
        }

        // Number of continuation bytes, 1..=5, from how far into the lead-byte
        // range the byte sits.
        let count = u32::from(lead >= 0xc0)
            + u32::from(lead >= 0xe0)
            + u32::from(lead >= 0xf0)
            + u32::from(lead >= 0xf8)
            + u32::from(lead >= 0xfc);
        let mut label = u32::from(lead) & ((1u32 << (6 - count)) - 1);

        for _ in 0..count {
            if i == bytes.len() {
                return Err(LabelStringError::TruncatedSequence { offset });
            }
            let byte = bytes[i];
            i += 1;
            if byte & 0xc0 != 0x80 {
                return Err(LabelStringError::MissingContinuationByte {
                    byte,
                    offset: i - 1,
                });
            }
            label = (label << 6) | u32::from(byte & 0x3f);
        }
        push_label(labels, label, offset)?;
    }
    Ok(())
}

/// Writes labels out as a byte string, one byte per label, skipping epsilons.
///
/// SICADA-DIVERGE: upstream narrows each label to `char` and only then compares
/// it against zero, so a label such as 256 both truncates *and* disappears, since
/// its low byte is zero. Here epsilon is decided on the label itself and anything
/// that does not fit a byte is an error rather than a silent rewrite.
pub fn labels_to_byte_string<L>(labels: &[L], bytes: &mut Vec<u8>) -> Result<(), LabelStringError>
where
    L: Copy + Into<i64>,
{
    bytes.reserve(labels.len());
    for (index, &label) in labels.iter().enumerate() {
        let label = label.into();
        match label {
            0 => continue,
            v if v < 0 => return Err(LabelStringError::NegativeLabel { label: v, index }),
            v if v > u8::MAX as i64 => {
                return Err(LabelStringError::LabelOutOfRange { label: v, index });
            }
            v => bytes.push(v as u8),
        }
    }
    Ok(())
}

/// Serializes labels as an unrestricted Thompson/Pike UTF-8 byte stream,
/// skipping epsilons.
///
/// The result is valid UTF-8 only if every label is a Unicode scalar value; see
/// the module docs.
pub fn labels_to_utf8_string<L>(labels: &[L], bytes: &mut Vec<u8>) -> Result<(), LabelStringError>
where
    L: Copy + Into<i64>,
{
    bytes.reserve(labels.len());
    for (index, &label) in labels.iter().enumerate() {
        let label = label.into();
        if label == 0 {
            continue;
        }
        if label < 0 {
            return Err(LabelStringError::NegativeLabel { label, index });
        }
        if label > i64::from(MAX_LABEL) {
            return Err(LabelStringError::LabelOutOfRange { label, index });
        }
        encode_one(label as u32, bytes);
    }
    Ok(())
}

/// Largest label the six-byte Thompson/Pike encoding can represent.
pub const MAX_LABEL: u32 = 0x7fff_ffff;

fn encode_one(label: u32, bytes: &mut Vec<u8>) {
    // Lead byte carries `8 - n - 1` payload bits for an n-byte sequence, with the
    // remaining bits split into six-bit continuation bytes.
    const CONT: u32 = 0x80;
    match label {
        0..=0x7f => bytes.push(label as u8),
        0x80..=0x7ff => {
            bytes.push(((label >> 6) | 0xc0) as u8);
            bytes.push(((label & 0x3f) | CONT) as u8);
        }
        0x800..=0xffff => {
            bytes.push(((label >> 12) | 0xe0) as u8);
            bytes.push((((label >> 6) & 0x3f) | CONT) as u8);
            bytes.push(((label & 0x3f) | CONT) as u8);
        }
        0x1_0000..=0x1f_ffff => {
            bytes.push(((label >> 18) | 0xf0) as u8);
            bytes.push((((label >> 12) & 0x3f) | CONT) as u8);
            bytes.push((((label >> 6) & 0x3f) | CONT) as u8);
            bytes.push(((label & 0x3f) | CONT) as u8);
        }
        0x20_0000..=0x3ff_ffff => {
            bytes.push(((label >> 24) | 0xf8) as u8);
            bytes.push((((label >> 18) & 0x3f) | CONT) as u8);
            bytes.push((((label >> 12) & 0x3f) | CONT) as u8);
            bytes.push((((label >> 6) & 0x3f) | CONT) as u8);
            bytes.push(((label & 0x3f) | CONT) as u8);
        }
        _ => {
            bytes.push(((label >> 30) | 0xfc) as u8);
            bytes.push((((label >> 24) & 0x3f) | CONT) as u8);
            bytes.push((((label >> 18) & 0x3f) | CONT) as u8);
            bytes.push((((label >> 12) & 0x3f) | CONT) as u8);
            bytes.push((((label >> 6) & 0x3f) | CONT) as u8);
            bytes.push(((label & 0x3f) | CONT) as u8);
        }
    }
}

#[inline]
fn push_label<L: TryFrom<u32>>(
    labels: &mut Vec<L>,
    label: u32,
    index: usize,
) -> Result<(), LabelStringError> {
    match L::try_from(label) {
        Ok(label) => {
            labels.push(label);
            Ok(())
        }
        Err(_) => Err(LabelStringError::LabelOutOfRange {
            label: i64::from(label),
            index,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(bytes: &[u8]) -> Result<Vec<i32>, LabelStringError> {
        let mut labels = Vec::new();
        utf8_string_to_labels(bytes, &mut labels)?;
        Ok(labels)
    }

    fn serialize(labels: &[i32]) -> Result<Vec<u8>, LabelStringError> {
        let mut bytes = Vec::new();
        labels_to_utf8_string(labels, &mut bytes)?;
        Ok(bytes)
    }

    #[test]
    fn byte_strings_map_one_label_per_byte() {
        let mut labels: Vec<i32> = Vec::new();
        byte_string_to_labels(b"a\xffb", &mut labels);
        assert_eq!(labels, vec![0x61, 0xff, 0x62]);
    }

    #[test]
    fn parses_standard_utf8() {
        assert_eq!(parse(b"hello").unwrap(), b"hello".map(i32::from).to_vec());
        // U+00E9, U+65E5, U+1F600: two-, three- and four-byte sequences.
        assert_eq!(parse("é".as_bytes()).unwrap(), vec![0xe9]);
        assert_eq!(parse("日".as_bytes()).unwrap(), vec![0x65e5]);
        assert_eq!(parse("😀".as_bytes()).unwrap(), vec![0x1f600]);
        assert_eq!(
            parse("aé日😀".as_bytes()).unwrap(),
            vec![0x61, 0xe9, 0x65e5, 0x1f600]
        );
    }

    #[test]
    fn parses_sequences_standard_utf8_rejects() {
        // Five- and six-byte forms, above U+10FFFF: valid Thompson/Pike, not
        // valid UTF-8, and unrepresentable as a Rust `char`.
        assert_eq!(
            parse(&[0xf8, 0x88, 0x80, 0x80, 0x80]).unwrap(),
            vec![0x20_0000]
        );
        assert_eq!(
            parse(&[0xfc, 0x84, 0x80, 0x80, 0x80, 0x80]).unwrap(),
            vec![0x400_0000]
        );
        assert_eq!(
            parse(&[0xfd, 0xbf, 0xbf, 0xbf, 0xbf, 0xbf]).unwrap(),
            vec![MAX_LABEL as i32]
        );
        // A surrogate code point, and an overlong encoding of 'A'.
        assert_eq!(parse(&[0xed, 0xa0, 0x80]).unwrap(), vec![0xd800]);
        assert_eq!(parse(&[0xc1, 0x81]).unwrap(), vec![0x41]);
        // Built at run time so the compiler does not fold this into a lint about
        // an always-invalid literal.
        let surrogate: Vec<u8> = vec![0xed, 0xa0, 0x80];
        assert!(std::str::from_utf8(&surrogate).is_err());
    }

    #[test]
    fn rejects_malformed_sequences() {
        assert!(matches!(
            parse(&[0x80]),
            Err(LabelStringError::ContinuationAsLeadByte {
                byte: 0x80,
                offset: 0
            })
        ));
        assert!(matches!(
            parse(&[0xe6, 0x97]),
            Err(LabelStringError::TruncatedSequence { offset: 0 })
        ));
        assert!(matches!(
            parse(&[0xe6, 0x97, 0x41]),
            Err(LabelStringError::MissingContinuationByte {
                byte: 0x41,
                offset: 2
            })
        ));
    }

    #[test]
    fn rejects_labels_that_do_not_fit() {
        let mut labels: Vec<u8> = Vec::new();
        assert!(matches!(
            utf8_string_to_labels("日".as_bytes(), &mut labels),
            Err(LabelStringError::LabelOutOfRange { label: 0x65e5, .. })
        ));
    }

    #[test]
    fn serializes_back_to_the_same_bytes() {
        for text in ["", "hello", "é", "日本語", "a😀b"] {
            let labels = parse(text.as_bytes()).unwrap();
            assert_eq!(serialize(&labels).unwrap(), text.as_bytes(), "{text}");
        }
    }

    #[test]
    fn epsilon_labels_are_skipped() {
        assert_eq!(serialize(&[0x61, 0, 0x62]).unwrap(), b"ab");
        let mut bytes = Vec::new();
        labels_to_byte_string(&[0x61, 0, 0x62], &mut bytes).unwrap();
        assert_eq!(bytes, b"ab");
    }

    #[test]
    fn byte_string_output_rejects_what_upstream_would_silently_mangle() {
        // Upstream narrows to `char` first, so 256 becomes 0 and is then dropped
        // as an epsilon; 300 would become 44 (','). Both are errors here.
        let mut bytes = Vec::new();
        assert!(matches!(
            labels_to_byte_string(&[0x61, 256], &mut bytes),
            Err(LabelStringError::LabelOutOfRange {
                label: 256,
                index: 1
            })
        ));
        bytes.clear();
        assert!(matches!(
            labels_to_byte_string(&[300], &mut bytes),
            Err(LabelStringError::LabelOutOfRange { label: 300, .. })
        ));
        bytes.clear();
        assert!(matches!(
            labels_to_byte_string(&[-1], &mut bytes),
            Err(LabelStringError::NegativeLabel { label: -1, .. })
        ));
    }

    #[test]
    fn utf8_output_rejects_negative_and_oversized_labels() {
        assert!(matches!(
            serialize(&[-5]),
            Err(LabelStringError::NegativeLabel { label: -5, .. })
        ));
        let mut bytes = Vec::new();
        assert!(matches!(
            labels_to_utf8_string(&[i64::from(MAX_LABEL) + 1], &mut bytes),
            Err(LabelStringError::LabelOutOfRange { .. })
        ));
    }

    /// Every label the encoding can represent must survive a round trip, across
    /// all six sequence lengths and their boundaries.
    #[test]
    fn round_trips_every_sequence_length() {
        let boundaries = [
            1, 0x7f, 0x80, 0x7ff, 0x800, 0xffff, 0x1_0000, 0x1f_ffff, 0x20_0000, 0x3ff_ffff,
            0x400_0000, MAX_LABEL,
        ];
        let expected_len = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6];
        for (&label, &len) in boundaries.iter().zip(expected_len.iter()) {
            let mut bytes = Vec::new();
            labels_to_utf8_string(&[label as i64], &mut bytes).unwrap();
            assert_eq!(bytes.len(), len, "label {label:#x} encoded as {bytes:02x?}");
            assert_eq!(
                parse(&bytes).unwrap(),
                vec![label as i32],
                "label {label:#x}"
            );
        }
    }

    /// Round trip over a pseudo-random spread of labels, run as one sequence so
    /// sequence boundaries are exercised too.
    #[test]
    fn round_trips_random_label_sequences() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for _ in 0..200 {
            let labels: Vec<i32> = (0..64)
                .map(|_| (next() % u64::from(MAX_LABEL)) as i32 + 1)
                .collect();
            let bytes = serialize(&labels).unwrap();
            assert_eq!(parse(&bytes).unwrap(), labels);
        }
    }
}
