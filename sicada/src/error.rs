use std::io;
use thiserror::Error;

use crate::fst::MatchType;

/// Errors that can occur when splitting composite text formats.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SplitCompositeError {
    #[error("Unmatched close parenthesis at byte offset {offset}")]
    UnmatchedCloseParenthesis { offset: usize },

    #[error("Unmatched open parenthesis at the end of the string")]
    UnmatchedOpenParenthesis,
}

/// Errors that can occur when parsing weights, symbols, or FST formats from text.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ParseError {
    #[error(transparent)]
    SplitComposite(#[from] SplitCompositeError),

    #[error("Invalid number of elements: expected {expected}, found {found}")]
    InvalidElementCount { expected: usize, found: usize },

    #[error("Float parsing error: {0}")]
    ParseFloat(#[from] std::num::ParseFloatError),

    #[error("Integer parsing error: {0}")]
    ParseInt(#[from] std::num::ParseIntError),

    #[error("Invalid format: {0}")]
    InvalidFormat(String),
}

/// Errors from converting between byte / UTF-8 strings and label sequences.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LabelStringError {
    #[error("UTF-8 continuation byte {byte:#04x} used as a lead byte at offset {offset}")]
    ContinuationAsLeadByte { byte: u8, offset: usize },

    #[error("truncated UTF-8 byte sequence starting at offset {offset}")]
    TruncatedSequence { offset: usize },

    #[error("expected a UTF-8 continuation byte at offset {offset}, found {byte:#04x}")]
    MissingContinuationByte { byte: u8, offset: usize },

    #[error("label {label} at index {index} does not fit the target label type")]
    LabelOutOfRange { label: i64, index: usize },

    #[error("label {label} at index {index} is negative")]
    NegativeLabel { label: i64, index: usize },
}

/// An error that occurred during an OpenFst operation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OpenFstError {
    #[error("OpenFst error: {0}")]
    Message(String),

    #[error(
        "FST property verification failed: required mask {mask:016x}, expected {expected:016x}, but found {actual:016x}"
    )]
    PropertyVerificationFailed {
        mask: u64,
        expected: u64,
        actual: u64,
    },

    #[error("Unsupported or invalid operation: {0}")]
    InvalidOperation(String),

    /// An FST failed [`verify`](crate::algorithms::verify::verify).
    #[error("Verify: {0}")]
    VerificationFailed(String),

    #[error("Symbol table error: {0}")]
    SymbolTable(String),

    #[error("I/O error: {0}")]
    Io(String),

    #[error(transparent)]
    Parse(#[from] ParseError),

    #[error("Bad FST header: Magic number not matched. Expected {expected}, got {found}")]
    InvalidMagicNumber { expected: i32, found: i32 },

    #[error("Invalid FST header: {0}")]
    InvalidFstHeader(String),

    #[error("Invalid MatchType {match_type:?} provided for matcher '{matcher_name}'")]
    MatcherInvalidMatchType {
        matcher_name: &'static str,
        match_type: MatchType,
    },

    #[error("Invalid configuration for matcher '{matcher_name}': {reason}")]
    MatcherInvalidConfiguration {
        matcher_name: &'static str,
        reason: &'static str,
    },

    #[error("FST is not linear: start state is missing")]
    NoStartState,

    #[error("FST is not linear: path does not reach a final state")]
    DoesNotReachFinalState,

    #[error("FST is not linear: state {state} has multiple outgoing arcs")]
    MultipleOutgoingArcs { state: usize },

    #[error("FST is not linear: final state {state} has outgoing arc(s)")]
    FinalStateHasOutgoingArcs { state: usize },

    #[error("Failed to decode FST to string: label value exceeds u8 capacity")]
    LabelToByteConversion,

    #[error("Failed to parse FST bytes as UTF-8 string: {0}")]
    Utf8Parse(#[from] std::string::FromUtf8Error),

    #[error(transparent)]
    LabelString(#[from] LabelStringError),
}

impl OpenFstError {
    /// Creates a new `OpenFstError` with the given message.
    pub fn new(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

impl From<io::Error> for OpenFstError {
    fn from(err: io::Error) -> Self {
        Self::Io(err.to_string())
    }
}
