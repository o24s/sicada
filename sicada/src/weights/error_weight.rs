//! A weight that has no values.
//!
//! Port of OpenFst's `error-weight.h`. It exists so that a container with no
//! FSTs in it, such as an empty FST archive, still has an arc type to name. It
//! is not a semiring and never carries data.

use std::fmt;
use std::str::FromStr;

use crate::error::ParseError;
use crate::fst_type::WeightType;
use crate::weight::Weight;

/// A weight with no inhabitants.
///
/// SICADA-DIVERGE: upstream's `ErrorWeight` is a struct whose constructor logs
/// `FSTERROR() << "ErrorWeight::ErrorWeight called"` and carries on, so a value
/// does exist and every operation on it has to be defined defensively:
/// `operator==` returns false even against itself, `Member()` is false, and
/// `Plus`/`Times`/`Divide` all hand back another error value. Here the type is
/// uninhabited, so "cannot be instantiated" is enforced by the compiler and
/// every method taking `&self` is statically unreachable. Only the associated
/// functions that must produce a value out of nothing (`zero`, `one` and
/// `no_weight`) can be called at all, and those panic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorWeight {}

/// Message shared by every operation that has to conjure a value.
const NO_VALUES: &str = "ErrorWeight has no values; it exists only to name the \
                         arc type of an empty archive";

impl fmt::Display for ErrorWeight {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        // No value of this type exists, so this cannot be reached.
        match *self {}
    }
}

impl FromStr for ErrorWeight {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Err(ParseError::InvalidFormat(format!(
            "cannot parse {s:?} as an error weight: {NO_VALUES}"
        )))
    }
}

impl Weight for ErrorWeight {
    type ReverseWeight = ErrorWeight;

    fn zero() -> Self {
        panic!("{NO_VALUES}")
    }

    fn one() -> Self {
        panic!("{NO_VALUES}")
    }

    fn no_weight() -> Self {
        panic!("{NO_VALUES}")
    }

    fn type_name() -> WeightType {
        WeightType::new("error")
    }

    fn properties() -> u64 {
        0
    }

    fn plus(&self, _rhs: &Self) -> Self {
        match *self {}
    }

    fn times(&self, _rhs: &Self) -> Self {
        match *self {}
    }

    fn reverse(&self) -> Self::ReverseWeight {
        match *self {}
    }

    fn is_member(&self) -> bool {
        match *self {}
    }

    fn approx_equal(&self, _other: &Self, _delta: f32) -> bool {
        match *self {}
    }

    fn quantize(&self, _delta: f32) -> Self {
        match *self {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The point of the type: no value of it can exist.
    #[test]
    fn the_type_is_uninhabited() {
        assert_eq!(size_of::<ErrorWeight>(), 0);
        assert_eq!(size_of::<Option<ErrorWeight>>(), 0, "Option is too");
        assert!(Vec::<ErrorWeight>::new().is_empty());
    }

    #[test]
    fn it_names_itself_error_and_claims_no_properties() {
        assert_eq!(ErrorWeight::type_name().as_str(), "error");
        assert_eq!(ErrorWeight::properties(), 0);
    }

    #[test]
    fn parsing_always_fails() {
        assert!("".parse::<ErrorWeight>().is_err());
        assert!("0".parse::<ErrorWeight>().is_err());
    }

    // The compiler knows these calls cannot return, since their result type is
    // uninhabited, and warns that everything after them is unreachable. That is
    // precisely the property under test.
    #[test]
    #[should_panic(expected = "ErrorWeight has no values")]
    #[allow(unreachable_code)]
    fn zero_panics() {
        let _ = ErrorWeight::zero();
    }

    #[test]
    #[should_panic(expected = "ErrorWeight has no values")]
    #[allow(unreachable_code)]
    fn one_panics() {
        let _ = ErrorWeight::one();
    }

    #[test]
    #[should_panic(expected = "ErrorWeight has no values")]
    #[allow(unreachable_code)]
    fn no_weight_panics() {
        let _ = ErrorWeight::no_weight();
    }
}
