//! Strongly-typed names for the FST, arc and weight types a file header carries.
//!
//! The header stores these as strings, and a reader has to decide which concrete
//! implementation to build from them. Upstream does that through
//! `register.h`'s global registry; sicada looks the name up here and dispatches
//! over a closed set instead.

use std::borrow::Cow;
use std::fmt;

/// A strongly-typed representation of a Weight's type name.
///
/// Holds a `Cow` because a composite weight's name is built from its components
/// (`expectation_tropical_tropical`, `power_log_3`) and so is only known at run
/// time. The alternative, returning `&'static str`, forces every such name to be
/// leaked; sicada did that in six places before this changed, once per call.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WeightType(Cow<'static, str>);

impl WeightType {
    pub const TROPICAL: Self = Self(Cow::Borrowed("tropical"));
    pub const LOG: Self = Self(Cow::Borrowed("log"));
    pub const REAL: Self = Self(Cow::Borrowed("real"));
    pub const MINMAX: Self = Self(Cow::Borrowed("minmax"));

    /// Names a weight whose type name is a compile-time constant.
    #[inline]
    pub const fn new(name: &'static str) -> Self {
        Self(Cow::Borrowed(name))
    }

    /// Names a composite weight, whose name is assembled from its components.
    #[inline]
    pub fn new_dynamic(name: String) -> Self {
        Self(Cow::Owned(name))
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for WeightType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A strongly-typed representation of an Arc's type name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArcType(Cow<'static, str>);

impl ArcType {
    pub const STANDARD: Self = Self(Cow::Borrowed("standard"));
    pub const LOG: Self = Self(Cow::Borrowed("log"));

    #[inline]
    pub fn new_static(name: &'static str) -> Self {
        Self(Cow::Borrowed(name))
    }

    #[inline]
    pub fn new_dynamic(name: String) -> Self {
        Self(Cow::Owned(name))
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl ArcType {
    /// Every arc type sicada names statically.
    ///
    /// An arc type is `<weight type>` for a standard-shaped arc, so the list is
    /// open-ended: [`ArcType::new_dynamic`] covers a weight whose name is built
    /// at run time. `from_name` therefore always answers, and the caller checks
    /// the name against the arc type it expects.
    pub const ALL: &'static [Self] = &[Self::STANDARD, Self::LOG];

    /// Recovers the typed name from the string a header carries.
    pub fn from_name(name: &str) -> Self {
        Self::ALL
            .iter()
            .find(|candidate| candidate.as_str() == name)
            .cloned()
            .unwrap_or_else(|| Self::new_dynamic(name.to_string()))
    }
}

impl fmt::Display for ArcType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A strongly-typed representation of an FST's structural type name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FstType(&'static str);

macro_rules! define_fst_types {
    // FST types that never change based on 32/64 bit size (e.g. "vector", "compose")
    ( single { $($name:ident => $str:literal),* $(,)? } ) => {
        $(
            pub const $name: Self = Self($str);
        )*
    };

    // FST types that append "64" to the end (e.g. "const", "const64")
    ( sized { $($name:ident => $str:literal),* $(,)? } ) => {
        pastey::paste! {
            $(
                pub const [<$name _32>]: Self = Self($str);
                pub const [<$name _64>]: Self = Self(concat!($str, "64"));
            )*
        }
    };

    // Compact FST types that insert "64" in the middle (e.g. "compact_string", "compact64_string")
    ( compact { $($name:ident => $str:literal),* $(,)? } ) => {
        pastey::paste! {
            $(
                pub const [<COMPACT_ $name _32>]: Self = Self(concat!("compact_", $str));
                pub const [<COMPACT_ $name _64>]: Self = Self(concat!("compact64_", $str));
            )*
        }
    };
}

impl FstType {
    define_fst_types! {
        single {
            VECTOR => "vector",
            ARC_MAP => "arc_map",
            COMPLEMENT => "complement",
            COMPOSE => "compose",
            EDIT => "edit",
            MERGE => "merge",
            EXPANDER => "expander",
            ARC_LOOKAHEAD => "arc_lookahead",
            ILABEL_LOOKAHEAD => "ilabel_lookahead",
            OLABEL_LOOKAHEAD => "olabel_lookahead",
        }
    }

    define_fst_types! {
        sized {
            CONST => "const",
        }
    }

    define_fst_types! {
        compact {
            STRING => "string",
            WEIGHTED_STRING => "weighted_string",
            ACCEPTOR => "acceptor",
            UNWEIGHTED => "unweighted",
            UNWEIGHTED_ACCEPTOR => "unweighted_acceptor",
        }
    }

    #[inline]
    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    #[inline]
    pub const fn as_str(&self) -> &'static str {
        self.0
    }
}

impl FstType {
    /// Every FST type sicada knows how to name.
    ///
    /// Reading a file whose header names something outside this list is an
    /// error: upstream would try to `dlopen` a matching shared object, which
    /// sicada does not do.
    pub const ALL: &'static [Self] = &[
        Self::VECTOR,
        Self::EDIT,
        Self::ARC_MAP,
        Self::COMPLEMENT,
        Self::COMPOSE,
        Self::EXPANDER,
        Self::ARC_LOOKAHEAD,
        Self::ILABEL_LOOKAHEAD,
        Self::OLABEL_LOOKAHEAD,
        Self::CONST_32,
        Self::CONST_64,
        Self::COMPACT_STRING_32,
        Self::COMPACT_STRING_64,
        Self::COMPACT_WEIGHTED_STRING_32,
        Self::COMPACT_WEIGHTED_STRING_64,
        Self::COMPACT_ACCEPTOR_32,
        Self::COMPACT_ACCEPTOR_64,
        Self::COMPACT_UNWEIGHTED_32,
        Self::COMPACT_UNWEIGHTED_64,
        Self::COMPACT_UNWEIGHTED_ACCEPTOR_32,
        Self::COMPACT_UNWEIGHTED_ACCEPTOR_64,
    ];

    /// Recovers the typed name from the string a header carries.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .find(|candidate| candidate.as_str() == name)
            .cloned()
    }
}

impl fmt::Display for FstType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every named FST type must be recoverable from its own string, since that
    /// round trip is how a file header becomes a concrete implementation.
    #[test]
    fn fst_type_names_round_trip() {
        for expected in FstType::ALL {
            assert_eq!(
                FstType::from_name(expected.as_str()).as_ref(),
                Some(expected),
                "{expected} did not round trip"
            );
        }
    }

    #[test]
    fn fst_type_names_are_distinct() {
        for (i, left) in FstType::ALL.iter().enumerate() {
            for right in &FstType::ALL[i + 1..] {
                assert_ne!(left.as_str(), right.as_str(), "duplicate name {left}");
            }
        }
    }

    #[test]
    fn an_unknown_fst_type_is_rejected() {
        assert_eq!(FstType::from_name("not-an-fst"), None);
        assert_eq!(FstType::from_name(""), None);
        // Upstream would try to dlopen "vector-fst.so" for a near miss; we do not.
        assert_eq!(FstType::from_name("Vector"), None);
    }

    /// The size suffix has to land in the right place: `const` grows a trailing
    /// `64`, while a compact type takes it in the middle.
    #[test]
    fn size_suffixes_follow_the_upstream_spelling() {
        assert_eq!(FstType::CONST_32.as_str(), "const");
        assert_eq!(FstType::CONST_64.as_str(), "const64");
        assert_eq!(FstType::COMPACT_STRING_32.as_str(), "compact_string");
        assert_eq!(FstType::COMPACT_STRING_64.as_str(), "compact64_string");
        assert_eq!(
            FstType::COMPACT_UNWEIGHTED_ACCEPTOR_64.as_str(),
            "compact64_unweighted_acceptor"
        );
    }

    #[test]
    fn arc_type_names_round_trip_and_unknown_ones_are_kept() {
        for expected in ArcType::ALL {
            assert_eq!(&ArcType::from_name(expected.as_str()), expected);
        }
        // An arc type is named after its weight, so unfamiliar names are carried
        // through rather than rejected.
        let dynamic = ArcType::from_name("tropical64");
        assert_eq!(dynamic.as_str(), "tropical64");
    }
}
