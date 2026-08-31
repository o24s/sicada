use crate::fst_type::WeightType;
use crate::weight::{
    COMMUTATIVE, CommutativeWeight, Divide, DivideType, IDEMPOTENT, IdempotentWeight,
    LEFT_SEMIRING, LeftSemiring, Minus, PATH, PathWeight, RIGHT_SEMIRING, RightSemiring, Weight,
    impl_weight_convert,
};
use std::fmt;
use std::str::FromStr;

/// Trait to generalize over f32 and f64 limits required by weights.
pub trait FloatExt: Copy + PartialEq + PartialOrd + fmt::Display + FromStr {
    fn pos_infinity() -> Self;
    fn neg_infinity() -> Self;
    fn nan() -> Self;
    fn is_nan(self) -> bool;
    fn is_greater_than_neg_inf(self) -> bool;
}

impl FloatExt for f32 {
    #[inline(always)]
    fn pos_infinity() -> Self {
        f32::INFINITY
    }
    #[inline(always)]
    fn neg_infinity() -> Self {
        f32::NEG_INFINITY
    }
    #[inline(always)]
    fn nan() -> Self {
        f32::NAN
    }
    #[inline(always)]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    #[inline(always)]
    fn is_greater_than_neg_inf(self) -> bool {
        self > f32::NEG_INFINITY
    }
}

impl FloatExt for f64 {
    #[inline(always)]
    fn pos_infinity() -> Self {
        f64::INFINITY
    }
    #[inline(always)]
    fn neg_infinity() -> Self {
        f64::NEG_INFINITY
    }
    #[inline(always)]
    fn nan() -> Self {
        f64::NAN
    }
    #[inline(always)]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    #[inline(always)]
    fn is_greater_than_neg_inf(self) -> bool {
        self > f64::NEG_INFINITY
    }
}

macro_rules! impl_display_fromstr {
    ($name:ident, $t:ident) => {
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.0 == $t::pos_infinity() {
                    write!(f, "Infinity")
                } else if self.0 == $t::neg_infinity() {
                    write!(f, "-Infinity")
                } else if self.0.is_nan() {
                    write!(f, "BadNumber")
                } else {
                    write!(f, "{}", self.0)
                }
            }
        }

        impl FromStr for $name {
            type Err = <$t as FromStr>::Err;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    "Infinity" => Ok(Self($t::pos_infinity())),
                    "-Infinity" => Ok(Self($t::neg_infinity())),
                    "BadNumber" => Ok(Self($t::nan())),
                    _ => s.parse::<$t>().map(Self),
                }
            }
        }
    };
}

// Tropical Semiring: (min, +, inf, 0)
macro_rules! define_tropical {
    ($name:ident, $t:ident, $type_name:expr) => {
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy)]
        pub struct $name(pub $t);

        /// The bytes upstream's `FloatWeight::Write` produces: the value and
        /// nothing else.
        impl crate::weight::WeightIo for $name {
            #[inline]
            fn read<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
                Ok(Self(crate::utils::io::read_scalar::<$t, _>(reader)?))
            }

            #[inline]
            fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
                crate::utils::io::write_scalar(writer, self.0)
            }
        }

        impl $name {
            #[inline(always)]
            pub const fn new(value: $t) -> Self {
                Self(value)
            }

            #[inline(always)]
            pub const fn value(&self) -> $t {
                self.0
            }

            #[inline(always)]
            pub fn minus(&self, _rhs: &Self) -> Self {
                // Not mathematically well-defined for all values in Tropical.
                // OpenFst actually omits Minus for Tropical, we return NoWeight.
                Self::no_weight()
            }
        }

        impl PartialEq for $name {
            #[inline(always)]
            fn eq(&self, other: &Self) -> bool {
                if self.0.is_nan() {
                    other.0.is_nan()
                } else if other.0.is_nan() {
                    false
                } else {
                    self.0 == other.0
                }
            }
        }

        impl Eq for $name {}

        impl PartialOrd for $name {
            #[inline(always)]
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                if self.0.is_nan() {
                    if other.0.is_nan() {
                        Some(std::cmp::Ordering::Equal)
                    } else {
                        None
                    }
                } else if other.0.is_nan() {
                    None
                } else {
                    self.0.partial_cmp(&other.0)
                }
            }
        }

        impl LeftSemiring for $name {}
        impl RightSemiring for $name {}
        impl CommutativeWeight for $name {}
        impl PathWeight for $name {}
        impl IdempotentWeight for $name {}

        impl_display_fromstr!($name, $t);

        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                if self.0.is_nan() {
                    $t::nan().to_bits().hash(state);
                } else {
                    self.0.to_bits().hash(state);
                }
            }
        }

        impl Weight for $name {
            type ReverseWeight = Self;

            #[inline(always)]
            fn zero() -> Self {
                Self($t::pos_infinity())
            }
            #[inline(always)]
            fn one() -> Self {
                Self(0.0)
            }
            #[inline(always)]
            fn no_weight() -> Self {
                Self($t::nan())
            }
            #[inline(always)]
            fn type_name() -> WeightType {
                WeightType::new($type_name)
            }

            #[inline(always)]
            fn properties() -> u64 {
                LEFT_SEMIRING | RIGHT_SEMIRING | COMMUTATIVE | PATH | IDEMPOTENT
            }

            #[inline(always)]
            fn is_member(&self) -> bool {
                self.0.is_greater_than_neg_inf()
            }

            #[inline(always)]
            fn approx_equal(&self, other: &Self, delta: f32) -> bool {
                if !self.is_member() || !other.is_member() {
                    !self.is_member() && !other.is_member()
                } else {
                    self.0 <= other.0 + (delta as $t) && other.0 <= self.0 + (delta as $t)
                }
            }

            #[inline(always)]
            fn quantize(&self, delta: f32) -> Self {
                if !self.is_member() || self.0 == $t::pos_infinity() {
                    *self
                } else {
                    Self((self.0 / (delta as $t) + 0.5).floor() * (delta as $t))
                }
            }

            #[inline(always)]
            fn reverse(&self) -> Self::ReverseWeight {
                *self
            }

            #[inline(always)]
            fn plus(&self, rhs: &Self) -> Self {
                if !self.is_member() || !rhs.is_member() {
                    Self::no_weight()
                } else if self.0 < rhs.0 {
                    *self
                } else {
                    *rhs
                }
            }

            #[inline(always)]
            fn times(&self, rhs: &Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }

        impl Divide for $name {
            #[inline(always)]
            fn divide(&self, rhs: &Self, _typ: DivideType) -> Self {
                if rhs.is_member() {
                    Self(self.0 - rhs.0)
                } else {
                    Self::no_weight()
                }
            }
        }
    };
}
define_tropical!(TropicalWeight, f32, "tropical");
define_tropical!(TropicalWeight64, f64, "tropical64");

// Log Semiring: (-log(e^-x + e^-y), +, inf, 0)
macro_rules! define_log {
    ($name:ident, $t:ident, $type_name:expr) => {
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy)]
        pub struct $name(pub $t);

        /// The bytes upstream's `FloatWeight::Write` produces: the value and
        /// nothing else.
        impl crate::weight::WeightIo for $name {
            #[inline]
            fn read<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
                Ok(Self(crate::utils::io::read_scalar::<$t, _>(reader)?))
            }

            #[inline]
            fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
                crate::utils::io::write_scalar(writer, self.0)
            }
        }

        impl $name {
            #[inline(always)]
            pub const fn new(value: $t) -> Self {
                Self(value)
            }

            #[inline(always)]
            pub const fn value(&self) -> $t {
                self.0
            }

            #[inline(always)]
            pub fn minus(&self, rhs: &Self) -> Self {
                let f1 = self.0;
                let f2 = rhs.0;
                if f1.is_nan() || f2.is_nan() || f1 > f2 {
                    return Self::no_weight();
                }
                if f2 == $t::pos_infinity() {
                    return *self;
                }
                let d = f2 - f1;
                if d == $t::pos_infinity() {
                    return *self;
                }
                Self((f1 as f64 - super::log_neg_exp(d as f64)) as $t)
            }
        }

        impl PartialEq for $name {
            #[inline(always)]
            fn eq(&self, other: &Self) -> bool {
                if self.0.is_nan() {
                    other.0.is_nan()
                } else if other.0.is_nan() {
                    false
                } else {
                    self.0 == other.0
                }
            }
        }

        impl Eq for $name {}

        impl PartialOrd for $name {
            #[inline(always)]
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                if self.0.is_nan() {
                    if other.0.is_nan() {
                        Some(std::cmp::Ordering::Equal)
                    } else {
                        None
                    }
                } else if other.0.is_nan() {
                    None
                } else {
                    self.0.partial_cmp(&other.0)
                }
            }
        }

        impl LeftSemiring for $name {}
        impl RightSemiring for $name {}
        impl CommutativeWeight for $name {}

        impl_display_fromstr!($name, $t);

        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                if self.0.is_nan() {
                    $t::nan().to_bits().hash(state);
                } else {
                    self.0.to_bits().hash(state);
                }
            }
        }

        impl Weight for $name {
            type ReverseWeight = Self;

            #[inline(always)]
            fn zero() -> Self {
                Self($t::pos_infinity())
            }
            #[inline(always)]
            fn one() -> Self {
                Self(0.0)
            }
            #[inline(always)]
            fn no_weight() -> Self {
                Self($t::nan())
            }
            #[inline(always)]
            fn type_name() -> WeightType {
                WeightType::new($type_name)
            }

            #[inline(always)]
            fn properties() -> u64 {
                LEFT_SEMIRING | RIGHT_SEMIRING | COMMUTATIVE
            }

            #[inline(always)]
            fn is_member(&self) -> bool {
                self.0.is_greater_than_neg_inf()
            }

            #[inline(always)]
            fn approx_equal(&self, other: &Self, delta: f32) -> bool {
                if !self.is_member() || !other.is_member() {
                    !self.is_member() && !other.is_member()
                } else {
                    self.0 <= other.0 + (delta as $t) && other.0 <= self.0 + (delta as $t)
                }
            }

            #[inline(always)]
            fn quantize(&self, delta: f32) -> Self {
                if !self.is_member() || self.0 == $t::pos_infinity() {
                    *self
                } else {
                    Self((self.0 / (delta as $t) + 0.5).floor() * (delta as $t))
                }
            }

            #[inline(always)]
            fn reverse(&self) -> Self::ReverseWeight {
                *self
            }

            #[inline(always)]
            fn plus(&self, rhs: &Self) -> Self {
                let f1 = self.0;
                let f2 = rhs.0;
                if f1.is_nan() || f2.is_nan() {
                    Self::no_weight()
                } else if f1 == $t::pos_infinity() {
                    *rhs
                } else if f2 == $t::pos_infinity() {
                    *self
                } else if f1 > f2 {
                    Self((f2 as f64 - super::log_pos_exp((f1 - f2) as f64)) as $t)
                } else {
                    Self((f1 as f64 - super::log_pos_exp((f2 - f1) as f64)) as $t)
                }
            }

            #[inline(always)]
            fn times(&self, rhs: &Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }

        impl Divide for $name {
            #[inline(always)]
            fn divide(&self, rhs: &Self, _typ: DivideType) -> Self {
                if rhs.is_member() {
                    Self(self.0 - rhs.0)
                } else {
                    Self::no_weight()
                }
            }
        }

        impl Minus for $name {
            #[inline(always)]
            fn minus(&self, rhs: &Self) -> Self {
                self.minus(rhs)
            }
        }

        pastey::paste! {
            #[derive(Debug, Clone)]
            pub struct [<KahanAdder $name>] {
                sum: f64,
                c: f64,
            }

            impl [<KahanAdder $name>] {
                pub fn new(w: $name) -> Self {
                    Self { sum: w.0 as f64, c: 0.0 }
                }

                #[inline]
                pub fn add(&mut self, w: &$name) {
                    let f = w.0 as f64;
                    if f.is_nan() || self.sum.is_nan() {
                        self.sum = f64::NAN;
                        return;
                    }
                    if f == f64::INFINITY {
                        return;
                    } else if self.sum == f64::INFINITY {
                        self.sum = f;
                        self.c = 0.0;
                    } else if f > self.sum {
                        let y = -super::log_pos_exp(f - self.sum) - self.c;
                        let t = self.sum + y;
                        self.c = (t - self.sum) - y;
                        self.sum = t;
                    } else {
                        let y = -super::log_pos_exp(self.sum - f) - self.c;
                        let t = f + y;
                        self.c = (t - f) - y;
                        self.sum = t;
                    }
                }

                #[inline]
                pub fn sum(&self) -> $name {
                    $name(self.sum as $t)
                }

                #[inline]
                pub fn reset(&mut self, w: $name) {
                    self.sum = w.0 as f64;
                    self.c = 0.0;
                }
            }

            impl Default for [<KahanAdder $name>] {
                fn default() -> Self {
                    Self::new(<$name as Weight>::zero())
                }
            }
        }
    };
}
define_log!(LogWeight, f32, "log");
define_log!(Log64Weight, f64, "log64");

// Real Semiring: (+, *, 0, 1)
macro_rules! define_real {
    ($name:ident, $t:ident, $type_name:expr) => {
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy)]
        pub struct $name(pub $t);

        /// The bytes upstream's `FloatWeight::Write` produces: the value and
        /// nothing else.
        impl crate::weight::WeightIo for $name {
            #[inline]
            fn read<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
                Ok(Self(crate::utils::io::read_scalar::<$t, _>(reader)?))
            }

            #[inline]
            fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
                crate::utils::io::write_scalar(writer, self.0)
            }
        }

        impl $name {
            #[inline(always)]
            pub const fn new(value: $t) -> Self {
                Self(value)
            }

            #[inline(always)]
            pub const fn value(&self) -> $t {
                self.0
            }

            #[inline(always)]
            pub fn minus(&self, rhs: &Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }

        impl PartialEq for $name {
            #[inline(always)]
            fn eq(&self, other: &Self) -> bool {
                if self.0.is_nan() {
                    other.0.is_nan()
                } else if other.0.is_nan() {
                    false
                } else {
                    self.0 == other.0
                }
            }
        }

        impl Eq for $name {}

        impl PartialOrd for $name {
            #[inline(always)]
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                if self.0.is_nan() {
                    if other.0.is_nan() {
                        Some(std::cmp::Ordering::Equal)
                    } else {
                        None
                    }
                } else if other.0.is_nan() {
                    None
                } else {
                    self.0.partial_cmp(&other.0)
                }
            }
        }

        impl LeftSemiring for $name {}
        impl RightSemiring for $name {}
        impl CommutativeWeight for $name {}

        impl_display_fromstr!($name, $t);

        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                if self.0.is_nan() {
                    $t::nan().to_bits().hash(state);
                } else {
                    self.0.to_bits().hash(state);
                }
            }
        }

        impl Weight for $name {
            type ReverseWeight = Self;

            #[inline(always)]
            fn zero() -> Self {
                Self(0.0)
            }
            #[inline(always)]
            fn one() -> Self {
                Self(1.0)
            }
            #[inline(always)]
            fn no_weight() -> Self {
                Self($t::nan())
            }
            #[inline(always)]
            fn type_name() -> WeightType {
                WeightType::new($type_name)
            }

            #[inline(always)]
            fn properties() -> u64 {
                LEFT_SEMIRING | RIGHT_SEMIRING | COMMUTATIVE
            }

            #[inline(always)]
            fn is_member(&self) -> bool {
                self.0.is_greater_than_neg_inf()
            }

            #[inline(always)]
            fn approx_equal(&self, other: &Self, delta: f32) -> bool {
                if !self.is_member() || !other.is_member() {
                    !self.is_member() && !other.is_member()
                } else {
                    self.0 <= other.0 + (delta as $t) && other.0 <= self.0 + (delta as $t)
                }
            }

            #[inline(always)]
            fn quantize(&self, delta: f32) -> Self {
                if !self.is_member() || self.0 == $t::pos_infinity() {
                    *self
                } else {
                    Self((self.0 / (delta as $t) + 0.5).floor() * (delta as $t))
                }
            }

            #[inline(always)]
            fn reverse(&self) -> Self::ReverseWeight {
                *self
            }

            #[inline(always)]
            fn plus(&self, rhs: &Self) -> Self {
                Self(self.0 + rhs.0)
            }

            #[inline(always)]
            fn times(&self, rhs: &Self) -> Self {
                Self(self.0 * rhs.0)
            }
        }

        impl Divide for $name {
            #[inline(always)]
            fn divide(&self, rhs: &Self, _typ: DivideType) -> Self {
                if rhs.is_member() {
                    Self(self.0 / rhs.0)
                } else {
                    Self::no_weight()
                }
            }
        }

        impl Minus for $name {
            #[inline(always)]
            fn minus(&self, rhs: &Self) -> Self {
                self.minus(rhs)
            }
        }

        pastey::paste! {
            #[derive(Debug, Clone)]
            pub struct [<KahanAdder $name>] {
                sum: f64,
                c: f64,
            }

            impl [<KahanAdder $name>] {
                pub fn new(w: $name) -> Self {
                    Self { sum: w.0 as f64, c: 0.0 }
                }

                #[inline]
                pub fn add(&mut self, w: &$name) {
                    let f = w.0 as f64;
                    if f.is_nan() || self.sum.is_nan() {
                        self.sum = f64::NAN;
                        return;
                    }
                    if f == f64::INFINITY {
                        self.sum = f;
                    } else if self.sum == f64::INFINITY {
                        // Already infinity, do nothing
                    } else {
                        let y = f - self.c;
                        let t = self.sum + y;
                        self.c = (t - self.sum) - y;
                        self.sum = t;
                    }
                }

                #[inline]
                pub fn sum(&self) -> $name {
                    $name(self.sum as $t)
                }

                #[inline]
                pub fn reset(&mut self, w: $name) {
                    self.sum = w.0 as f64;
                    self.c = 0.0;
                }
            }

            impl Default for [<KahanAdder $name>] {
                fn default() -> Self {
                    Self::new(<$name as Weight>::zero())
                }
            }
        }
    };
}
define_real!(RealWeight, f32, "real");
define_real!(Real64Weight, f64, "real64");

// MinMax Semiring: (min, max, inf, -inf)
macro_rules! define_minmax {
    ($name:ident, $t:ident, $type_name:expr) => {
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy)]
        pub struct $name(pub $t);

        /// The bytes upstream's `FloatWeight::Write` produces: the value and
        /// nothing else.
        impl crate::weight::WeightIo for $name {
            #[inline]
            fn read<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
                Ok(Self(crate::utils::io::read_scalar::<$t, _>(reader)?))
            }

            #[inline]
            fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
                crate::utils::io::write_scalar(writer, self.0)
            }
        }

        impl $name {
            #[inline(always)]
            pub const fn new(value: $t) -> Self {
                Self(value)
            }

            #[inline(always)]
            pub const fn value(&self) -> $t {
                self.0
            }
        }

        impl PartialEq for $name {
            #[inline(always)]
            fn eq(&self, other: &Self) -> bool {
                if self.0.is_nan() {
                    other.0.is_nan()
                } else if other.0.is_nan() {
                    false
                } else {
                    self.0 == other.0
                }
            }
        }

        impl Eq for $name {}

        impl PartialOrd for $name {
            #[inline(always)]
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                if self.0.is_nan() {
                    if other.0.is_nan() {
                        Some(std::cmp::Ordering::Equal)
                    } else {
                        None
                    }
                } else if other.0.is_nan() {
                    None
                } else {
                    self.0.partial_cmp(&other.0)
                }
            }
        }

        impl LeftSemiring for $name {}
        impl RightSemiring for $name {}
        impl CommutativeWeight for $name {}
        impl PathWeight for $name {}
        impl IdempotentWeight for $name {}

        impl_display_fromstr!($name, $t);

        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                if self.0.is_nan() {
                    $t::nan().to_bits().hash(state);
                } else {
                    self.0.to_bits().hash(state);
                }
            }
        }

        impl Weight for $name {
            type ReverseWeight = Self;

            #[inline(always)]
            fn zero() -> Self {
                Self($t::pos_infinity())
            }
            #[inline(always)]
            fn one() -> Self {
                Self($t::neg_infinity())
            }
            #[inline(always)]
            fn no_weight() -> Self {
                Self($t::nan())
            }
            #[inline(always)]
            fn type_name() -> WeightType {
                WeightType::new($type_name)
            }

            #[inline(always)]
            fn properties() -> u64 {
                LEFT_SEMIRING | RIGHT_SEMIRING | COMMUTATIVE | IDEMPOTENT | PATH
            }

            #[inline(always)]
            fn is_member(&self) -> bool {
                !self.0.is_nan()
            }

            #[inline(always)]
            fn approx_equal(&self, other: &Self, delta: f32) -> bool {
                if !self.is_member() || !other.is_member() {
                    !self.is_member() && !other.is_member()
                } else {
                    self.0 <= other.0 + (delta as $t) && other.0 <= self.0 + (delta as $t)
                }
            }

            #[inline(always)]
            fn quantize(&self, delta: f32) -> Self {
                if !self.is_member() || self.0 == $t::neg_infinity() || self.0 == $t::pos_infinity()
                {
                    *self
                } else {
                    Self((self.0 / (delta as $t) + 0.5).floor() * (delta as $t))
                }
            }

            #[inline(always)]
            fn reverse(&self) -> Self::ReverseWeight {
                *self
            }

            #[inline(always)]
            fn plus(&self, rhs: &Self) -> Self {
                if !self.is_member() || !rhs.is_member() {
                    Self::no_weight()
                } else if self.0 < rhs.0 {
                    *self
                } else {
                    *rhs
                }
            }

            #[inline(always)]
            fn times(&self, rhs: &Self) -> Self {
                if !self.is_member() || !rhs.is_member() {
                    Self::no_weight()
                } else if self.0 >= rhs.0 {
                    *self
                } else {
                    *rhs
                }
            }
        }

        impl Divide for $name {
            #[inline(always)]
            fn divide(&self, rhs: &Self, _typ: DivideType) -> Self {
                if self.0 >= rhs.0 {
                    *self
                } else {
                    Self::no_weight()
                }
            }
        }
    };
}
define_minmax!(MinMaxWeight, f32, "minmax");
define_minmax!(MinMaxWeight64, f64, "minmax64");

// Log <-> Tropical
impl_weight_convert!(LogWeight, TropicalWeight, |w: LogWeight| TropicalWeight(
    w.value()
));
impl_weight_convert!(Log64Weight, TropicalWeight64, |w: Log64Weight| {
    TropicalWeight64(w.value())
});
impl_weight_convert!(TropicalWeight, LogWeight, |w: TropicalWeight| LogWeight(
    w.value()
));
impl_weight_convert!(TropicalWeight64, Log64Weight, |w: TropicalWeight64| {
    Log64Weight(w.value())
});

// Real -> Log (-ln(x))
impl_weight_convert!(RealWeight, LogWeight, |w: RealWeight| LogWeight(
    -w.value().ln()
));
impl_weight_convert!(Real64Weight, Log64Weight, |w: Real64Weight| Log64Weight(
    -w.value().ln()
));

// Log -> Real (exp(-x))
impl_weight_convert!(LogWeight, RealWeight, |w: LogWeight| RealWeight(
    (-w.value()).exp()
));
impl_weight_convert!(Log64Weight, Real64Weight, |w: Log64Weight| Real64Weight(
    (-w.value()).exp()
));

// The conversions above pair each width with its own. Upstream generalized these
// in 694dc53 ("Greatly reduce number of `WeightConvert` specializations via
// partial specialization") so that any width converts to any other, since
// nothing about the conversion depends on the width. The mixed-width cases
// follow.

// Log <-> Tropical, across widths.
impl_weight_convert!(
    LogWeight,
    TropicalWeight64,
    |w: LogWeight| TropicalWeight64(w.value() as f64)
);
impl_weight_convert!(
    Log64Weight,
    TropicalWeight,
    |w: Log64Weight| TropicalWeight(w.value() as f32)
);
impl_weight_convert!(
    TropicalWeight,
    Log64Weight,
    |w: TropicalWeight| Log64Weight(w.value() as f64)
);
impl_weight_convert!(
    TropicalWeight64,
    LogWeight,
    |w: TropicalWeight64| LogWeight(w.value() as f32)
);

// Real <-> Log, across widths. The exponential runs at the wider precision so a
// narrow input does not lose the exponent before it is taken.
impl_weight_convert!(RealWeight, Log64Weight, |w: RealWeight| Log64Weight(
    -(w.value() as f64).ln()
));
impl_weight_convert!(Real64Weight, LogWeight, |w: Real64Weight| LogWeight(
    -w.value().ln() as f32
));
impl_weight_convert!(LogWeight, Real64Weight, |w: LogWeight| Real64Weight(
    (-(w.value() as f64)).exp()
));
impl_weight_convert!(Log64Weight, RealWeight, |w: Log64Weight| RealWeight(
    (-w.value()).exp() as f32
));

// Between the two widths of one semiring.
impl_weight_convert!(LogWeight, Log64Weight, |w: LogWeight| Log64Weight(
    w.value() as f64
));
impl_weight_convert!(Log64Weight, LogWeight, |w: Log64Weight| LogWeight(
    w.value() as f32
));
impl_weight_convert!(RealWeight, Real64Weight, |w: RealWeight| Real64Weight(
    w.value() as f64
));
impl_weight_convert!(Real64Weight, RealWeight, |w: Real64Weight| RealWeight(
    w.value() as f32
));
// SICADA-DIVERGE: upstream has no conversion between the two widths of tropical
// or of minmax, though it has them for log and real. Nothing about those two
// differs, so they are provided here as well.
impl_weight_convert!(TropicalWeight, TropicalWeight64, |w: TropicalWeight| {
    TropicalWeight64(w.value() as f64)
});
impl_weight_convert!(TropicalWeight64, TropicalWeight, |w: TropicalWeight64| {
    TropicalWeight(w.value() as f32)
});
impl_weight_convert!(MinMaxWeight, MinMaxWeight64, |w: MinMaxWeight| {
    MinMaxWeight64(w.value() as f64)
});
impl_weight_convert!(MinMaxWeight64, MinMaxWeight, |w: MinMaxWeight64| {
    MinMaxWeight(w.value() as f32)
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::axioms;
    use crate::weight::{COMMUTATIVE, IDEMPOTENT, PATH, SEMIRING};

    // -----------------------------------------------------------------------
    // Semiring axioms
    // -----------------------------------------------------------------------

    /// Each float semiring must satisfy exactly what its `properties()` claims.
    /// The claims differ: tropical and minmax are idempotent path semirings,
    /// log and real are not.
    #[test]
    fn every_float_semiring_satisfies_its_claims() {
        axioms::check(&[
            TropicalWeight(0.5),
            TropicalWeight(2.0),
            TropicalWeight(-1.5),
        ]);
        axioms::check(&[TropicalWeight64(0.5), TropicalWeight64(-3.25)]);
        axioms::check(&[LogWeight(0.5), LogWeight(2.0), LogWeight(-1.5)]);
        axioms::check(&[Log64Weight(0.5), Log64Weight(-3.25)]);
        // Real weights are probabilities, so the samples stay in [0, 1].
        axioms::check(&[RealWeight(0.25), RealWeight(0.5), RealWeight(0.75)]);
        axioms::check(&[Real64Weight(0.25), Real64Weight(0.75)]);
        axioms::check(&[MinMaxWeight(0.5), MinMaxWeight(2.0), MinMaxWeight(-1.5)]);
        axioms::check(&[MinMaxWeight64(0.5), MinMaxWeight64(-3.25)]);
    }

    #[test]
    fn the_claimed_properties_match_openfst() {
        assert_eq!(
            TropicalWeight::properties(),
            SEMIRING | COMMUTATIVE | IDEMPOTENT | PATH
        );
        assert_eq!(LogWeight::properties(), SEMIRING | COMMUTATIVE);
        assert_eq!(RealWeight::properties(), SEMIRING | COMMUTATIVE);
        assert_eq!(
            MinMaxWeight::properties(),
            SEMIRING | COMMUTATIVE | IDEMPOTENT | PATH
        );
    }

    #[test]
    fn division_recovers_the_other_factor() {
        axioms::check_divide(&[TropicalWeight(0.5), TropicalWeight(2.0)]);
        axioms::check_divide(&[TropicalWeight64(0.5), TropicalWeight64(2.0)]);
        axioms::check_divide(&[LogWeight(0.5), LogWeight(2.0)]);
        axioms::check_divide(&[Log64Weight(0.5), Log64Weight(2.0)]);
        axioms::check_divide(&[RealWeight(0.25), RealWeight(0.5)]);
        axioms::check_divide(&[MinMaxWeight(0.5), MinMaxWeight(2.0)]);
    }

    // -----------------------------------------------------------------------
    // The operations themselves
    // -----------------------------------------------------------------------

    #[test]
    fn tropical_plus_is_min_and_times_is_addition() {
        assert_eq!(
            TropicalWeight(1.0).plus(&TropicalWeight(2.0)),
            TropicalWeight(1.0)
        );
        assert_eq!(
            TropicalWeight(1.0).times(&TropicalWeight(2.0)),
            TropicalWeight(3.0)
        );
        assert_eq!(TropicalWeight::zero(), TropicalWeight(f32::INFINITY));
        assert_eq!(TropicalWeight::one(), TropicalWeight(0.0));
    }

    #[test]
    fn minmax_plus_is_min_and_times_is_max() {
        assert_eq!(
            MinMaxWeight(1.0).plus(&MinMaxWeight(2.0)),
            MinMaxWeight(1.0)
        );
        assert_eq!(
            MinMaxWeight(1.0).times(&MinMaxWeight(2.0)),
            MinMaxWeight(2.0)
        );
    }

    #[test]
    fn real_plus_is_addition_and_times_is_multiplication() {
        assert_eq!(RealWeight(0.25).plus(&RealWeight(0.5)), RealWeight(0.75));
        assert_eq!(RealWeight(0.25).times(&RealWeight(0.5)), RealWeight(0.125));
        assert_eq!(RealWeight::zero(), RealWeight(0.0));
        assert_eq!(RealWeight::one(), RealWeight(1.0));
    }

    /// The log semiring's plus is `-log(e^-x + e^-y)`, computed so that the
    /// larger operand dominates and no intermediate overflows.
    #[test]
    fn log_plus_combines_probabilities() {
        // Two events of probability e^-1 each sum to 2e^-1.
        let sum = LogWeight(1.0).plus(&LogWeight(1.0));
        let expected = -(2.0f32 * (-1.0f32).exp()).ln();
        assert!((sum.value() - expected).abs() < 1e-6, "{sum} vs {expected}");

        // Zero is +infinity and contributes nothing.
        assert_eq!(LogWeight(1.0).plus(&LogWeight::zero()), LogWeight(1.0));
        assert_eq!(LogWeight::zero().plus(&LogWeight(1.0)), LogWeight(1.0));
    }

    // -----------------------------------------------------------------------
    // NaN and the no_weight sentinel
    // -----------------------------------------------------------------------

    /// `no_weight` is NaN, and the comparison contract around it is load-bearing
    /// for every hash map keyed by a weight.
    #[test]
    fn the_no_weight_sentinel_compares_reflexively() {
        macro_rules! check {
            ($($ty:ident),*) => {$({
                let none = $ty::no_weight();
                assert!(!none.is_member(), concat!(stringify!($ty), " no_weight is a member"));
                assert_eq!(none, none, concat!(stringify!($ty), " breaks reflexivity"));
                assert_ne!(none, $ty::one());
                assert_ne!($ty::one(), none);
                assert!(none.approx_equal(&none, 0.1));
                assert!(!none.approx_equal(&$ty::one(), f32::INFINITY));
                // Two NaNs must hash alike, since they compare equal.
                use std::hash::{Hash, Hasher};
                use std::collections::hash_map::DefaultHasher;
                let hash_of = |w: &$ty| {
                    let mut hasher = DefaultHasher::new();
                    w.hash(&mut hasher);
                    hasher.finish()
                };
                assert_eq!(hash_of(&none), hash_of(&$ty::no_weight()));
            })*};
        }
        check!(
            TropicalWeight,
            TropicalWeight64,
            LogWeight,
            Log64Weight,
            RealWeight,
            Real64Weight,
            MinMaxWeight,
            MinMaxWeight64
        );
    }

    #[test]
    fn the_sentinel_is_unordered_against_real_weights() {
        let none = TropicalWeight::no_weight();
        assert_eq!(none.partial_cmp(&TropicalWeight(1.0)), None);
        assert_eq!(TropicalWeight(1.0).partial_cmp(&none), None);
        assert_eq!(none.partial_cmp(&none), Some(std::cmp::Ordering::Equal));
    }

    #[test]
    fn quantize_keeps_full_precision_at_the_wider_width() {
        // Quantizing must not route a 64-bit weight through f32.
        let weight = TropicalWeight64(1.0 + 1e-9);
        let quantized = weight.quantize(1e-12);
        assert!(
            (quantized.value() - weight.value()).abs() < 1e-10,
            "{quantized} lost precision from {weight}"
        );
        // Quantizing to a coarse delta snaps to a multiple of it.
        assert_eq!(TropicalWeight(1.24).quantize(0.5), TropicalWeight(1.0));
        assert_eq!(TropicalWeight(1.26).quantize(0.5), TropicalWeight(1.5));
        // Zero and non-members pass through untouched.
        assert_eq!(TropicalWeight::zero().quantize(0.5), TropicalWeight::zero());
        assert!(!TropicalWeight::no_weight().quantize(0.5).is_member());
    }

    // -----------------------------------------------------------------------
    // Text representation
    // -----------------------------------------------------------------------

    #[test]
    fn weights_round_trip_through_their_text_form() {
        for value in [0.0f32, 1.5, -2.25, 1e10] {
            let weight = TropicalWeight(value);
            let parsed: TropicalWeight = weight.to_string().parse().unwrap();
            assert_eq!(parsed, weight, "{weight} did not round trip");
        }
        assert_eq!(
            TropicalWeight::zero()
                .to_string()
                .parse::<TropicalWeight>()
                .unwrap(),
            TropicalWeight::zero()
        );
    }

    #[test]
    fn an_unparsable_weight_is_an_error() {
        assert!("not a number".parse::<TropicalWeight>().is_err());
        assert!("".parse::<LogWeight>().is_err());
    }

    #[test]
    fn the_type_names_match_openfst() {
        assert_eq!(TropicalWeight::type_name().as_str(), "tropical");
        assert_eq!(TropicalWeight64::type_name().as_str(), "tropical64");
        assert_eq!(LogWeight::type_name().as_str(), "log");
        assert_eq!(Log64Weight::type_name().as_str(), "log64");
        assert_eq!(RealWeight::type_name().as_str(), "real");
        assert_eq!(Real64Weight::type_name().as_str(), "real64");
        assert_eq!(MinMaxWeight::type_name().as_str(), "minmax");
        assert_eq!(MinMaxWeight64::type_name().as_str(), "minmax64");
    }

    // -----------------------------------------------------------------------
    // Conversions
    // -----------------------------------------------------------------------

    #[test]
    fn log_and_tropical_share_their_stored_value() {
        // Both hold a negative log; only `plus` differs.
        assert_eq!(TropicalWeight::from(LogWeight(2.5)), TropicalWeight(2.5));
        assert_eq!(LogWeight::from(TropicalWeight(2.5)), LogWeight(2.5));
        assert_eq!(Log64Weight::from(TropicalWeight(2.5)), Log64Weight(2.5));
        assert_eq!(TropicalWeight::from(Log64Weight(2.5)), TropicalWeight(2.5));
    }

    #[test]
    fn crossing_into_real_exponentiates() {
        // A probability of e^-1 is a log weight of 1.
        let real = RealWeight::from(LogWeight(1.0));
        assert!((real.value() - std::f32::consts::E.recip()).abs() < 1e-6);

        let back = LogWeight::from(real);
        assert!((back.value() - 1.0).abs() < 1e-6);
    }

    /// Zero and One have to land on the target semiring's Zero and One, or a
    /// converted FST stops meaning what it meant.
    #[test]
    fn conversions_carry_the_semiring_identities_across() {
        assert_eq!(
            TropicalWeight::from(LogWeight::zero()),
            TropicalWeight::zero()
        );
        assert_eq!(
            TropicalWeight::from(LogWeight::one()),
            TropicalWeight::one()
        );
        assert_eq!(LogWeight::from(TropicalWeight::zero()), LogWeight::zero());
        assert_eq!(LogWeight::from(TropicalWeight::one()), LogWeight::one());

        assert_eq!(RealWeight::from(LogWeight::zero()), RealWeight::zero());
        assert_eq!(RealWeight::from(LogWeight::one()), RealWeight::one());
        assert_eq!(LogWeight::from(RealWeight::one()), LogWeight::one());
    }

    #[test]
    fn widening_and_narrowing_round_trips_for_representable_values() {
        for value in [0.0f32, 1.5, -2.25, 1e10] {
            assert_eq!(
                TropicalWeight::from(TropicalWeight64::from(TropicalWeight(value))),
                TropicalWeight(value)
            );
            assert_eq!(
                LogWeight::from(Log64Weight::from(LogWeight(value))),
                LogWeight(value)
            );
            assert_eq!(
                RealWeight::from(Real64Weight::from(RealWeight(value))),
                RealWeight(value)
            );
            assert_eq!(
                MinMaxWeight::from(MinMaxWeight64::from(MinMaxWeight(value))),
                MinMaxWeight(value)
            );
        }
    }
}
