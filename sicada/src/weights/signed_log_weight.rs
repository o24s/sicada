use std::fmt;
use std::str::FromStr;

use crate::error::ParseError;
use crate::fst_type::WeightType;
use crate::weight::{
    COMMUTATIVE, Divide, DivideType, LEFT_SEMIRING, LeftSemiring, Minus, RIGHT_SEMIRING,
    RightSemiring, Weight, impl_weight_convert,
};
use crate::weights::float_weight::{
    Log64Weight, LogWeight, Real64Weight, RealWeight, TropicalWeight, TropicalWeight64,
};

// Signed Log Semiring
macro_rules! define_signed_log {
    ($name:ident, $adder_name:ident, $t:ident, $type_name:expr) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $name {
            pub sign: f32,
            pub neg_log_prob: $t,
        }

        impl $name {
            #[inline(always)]
            pub const fn new(sign: $t, neg_log_prob: $t) -> Self {
                Self {
                    sign: sign as f32,
                    neg_log_prob,
                }
            }

            #[inline(always)]
            pub fn is_positive(&self) -> bool {
                self.sign > 0.0
            }

            #[inline(always)]
            pub fn minus(&self, rhs: &Self) -> Self {
                let minus_w2 = Self {
                    sign: -rhs.sign,
                    neg_log_prob: rhs.neg_log_prob,
                };
                self.plus(&minus_w2)
            }
        }

        impl Eq for $name {}

        impl LeftSemiring for $name {}
        impl RightSemiring for $name {}

        #[allow(clippy::derived_hash_with_manual_eq)]
        /// SICADA-BUGFIX: the derived `PartialEq` compared both fields, so the
        /// two spellings of `Zero`, sign `+1` and sign `-1` with an infinite
        /// magnitude in both cases, did not compare equal. Multiplying `Zero` by a
        /// negative weight produces the negative spelling, so distributivity
        /// failed and `Hash`, which already normalises the sign at infinity,
        /// disagreed with equality. Upstream compares exactly as below.
        impl PartialEq for $name {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                if self.is_positive() == other.is_positive() {
                    // As in the float weights, the `no_weight` sentinel is NaN
                    // and has to compare equal to itself.
                    if self.neg_log_prob.is_nan() {
                        other.neg_log_prob.is_nan()
                    } else if other.neg_log_prob.is_nan() {
                        false
                    } else {
                        self.neg_log_prob == other.neg_log_prob
                    }
                } else {
                    // Opposite signs agree only on Zero, whose magnitude is
                    // infinite in the negative-log representation.
                    self.neg_log_prob == $t::INFINITY && other.neg_log_prob == $t::INFINITY
                }
            }
        }

        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                let h1 = if self.neg_log_prob == $t::INFINITY || self.is_positive() {
                    1.0f32.to_bits() as u64
                } else {
                    (-1.0f32).to_bits() as u64
                };
                let h2 = self.neg_log_prob.to_bits() as u64;
                let combined = h1.wrapping_shl(5) ^ h1.wrapping_shr(64 - 5) ^ h2;
                combined.hash(state);
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let w1_str = if self.sign == f32::INFINITY {
                    "Infinity".to_string()
                } else if self.sign == f32::NEG_INFINITY {
                    "-Infinity".to_string()
                } else if self.sign.is_nan() {
                    "BadNumber".to_string()
                } else {
                    self.sign.to_string()
                };

                let w2_str = if self.neg_log_prob == $t::INFINITY {
                    "Infinity".to_string()
                } else if self.neg_log_prob == $t::NEG_INFINITY {
                    "-Infinity".to_string()
                } else if self.neg_log_prob.is_nan() {
                    "BadNumber".to_string()
                } else {
                    self.neg_log_prob.to_string()
                };
                write!(f, "{},{}", w1_str, w2_str)
            }
        }

        impl FromStr for $name {
            type Err = ParseError;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let parts: Vec<&str> = s.split(',').collect();
                if parts.len() != 2 {
                    return Err(ParseError::InvalidElementCount {
                        expected: 2,
                        found: parts.len(),
                    });
                }

                let parse_f32 = |s: &str| -> Result<f32, ParseError> {
                    match s {
                        "Infinity" => Ok(f32::INFINITY),
                        "-Infinity" => Ok(f32::NEG_INFINITY),
                        "BadNumber" => Ok(f32::NAN),
                        _ => s.parse::<f32>().map_err(|_| {
                            ParseError::InvalidFormat(format!("Failed to parse sign: {}", s))
                        }),
                    }
                };

                let parse_t = |s: &str| -> Result<$t, ParseError> {
                    match s {
                        "Infinity" => Ok($t::INFINITY),
                        "-Infinity" => Ok($t::NEG_INFINITY),
                        "BadNumber" => Ok($t::NAN),
                        _ => s.parse::<$t>().map_err(|_| {
                            ParseError::InvalidFormat(format!("Failed to parse value: {}", s))
                        }),
                    }
                };

                let sign = parse_f32(parts[0])?;
                let neg_log_prob = parse_t(parts[1])?;
                Ok(Self { sign, neg_log_prob })
            }
        }

        impl Weight for $name {
            type ReverseWeight = Self;

            #[inline(always)]
            fn zero() -> Self {
                Self {
                    sign: 1.0,
                    neg_log_prob: $t::INFINITY,
                }
            }

            #[inline(always)]
            fn one() -> Self {
                Self {
                    sign: 1.0,
                    neg_log_prob: 0.0,
                }
            }

            #[inline(always)]
            fn no_weight() -> Self {
                Self {
                    sign: 1.0,
                    neg_log_prob: $t::NAN,
                }
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
                !self.sign.is_nan()
                    && !self.neg_log_prob.is_nan()
                    && self.neg_log_prob != $t::NEG_INFINITY
            }

            #[inline(always)]
            fn approx_equal(&self, other: &Self, delta: f32) -> bool {
                if self.is_positive() == other.is_positive() {
                    self.neg_log_prob <= other.neg_log_prob + (delta as $t)
                        && other.neg_log_prob <= self.neg_log_prob + (delta as $t)
                } else {
                    // SICADA-BUGFIX: upstream compares both magnitudes against
                    // the component semiring's `Zero()`, which is `+inf` in the
                    // negative-log representation. Comparing against `0.0`
                    // instead, as this did, is comparing against `One`, so a
                    // weight and its own negation were reported approximately
                    // equal.
                    self.neg_log_prob == $t::INFINITY && other.neg_log_prob == $t::INFINITY
                }
            }

            #[inline(always)]
            fn quantize(&self, delta: f32) -> Self {
                if !self.is_member() || self.neg_log_prob == $t::INFINITY {
                    *self
                } else {
                    let qw2 =
                        (((self.neg_log_prob as f32) / delta) + 0.5).floor() as $t * (delta as $t);
                    Self {
                        sign: self.sign,
                        neg_log_prob: qw2,
                    }
                }
            }

            #[inline(always)]
            fn reverse(&self) -> Self::ReverseWeight {
                *self
            }

            #[inline(always)]
            fn plus(&self, rhs: &Self) -> Self {
                if !self.is_member() || !rhs.is_member() {
                    return Self::no_weight();
                }
                let equal = self.is_positive() == rhs.is_positive();
                let f1 = self.neg_log_prob;
                let f2 = rhs.neg_log_prob;

                if f1 == $t::INFINITY {
                    *rhs
                } else if f2 == $t::INFINITY {
                    *self
                } else if f1 == f2 {
                    if equal {
                        let ln2 = std::f64::consts::LN_2 as $t;
                        Self {
                            sign: self.sign,
                            neg_log_prob: f2 - ln2,
                        }
                    } else {
                        Self::zero()
                    }
                } else if f1 > f2 {
                    // f1 > f2
                    if equal {
                        Self {
                            sign: self.sign,
                            neg_log_prob: (f2 as f64 - super::log_pos_exp((f1 - f2) as f64)) as $t,
                        }
                    } else {
                        Self {
                            sign: rhs.sign,
                            neg_log_prob: (f2 as f64 - super::log_neg_exp((f1 - f2) as f64)) as $t,
                        }
                    }
                } else {
                    // f1 < f2
                    if equal {
                        Self {
                            sign: rhs.sign,
                            neg_log_prob: (f1 as f64 - super::log_pos_exp((f2 - f1) as f64)) as $t,
                        }
                    } else {
                        Self {
                            sign: self.sign,
                            neg_log_prob: (f1 as f64 - super::log_neg_exp((f2 - f1) as f64)) as $t,
                        }
                    }
                }
            }

            #[inline(always)]
            fn times(&self, rhs: &Self) -> Self {
                if !self.is_member() || !rhs.is_member() {
                    return Self::no_weight();
                }
                let sign = if self.is_positive() == rhs.is_positive() {
                    1.0
                } else {
                    -1.0
                };
                Self {
                    sign,
                    neg_log_prob: self.neg_log_prob + rhs.neg_log_prob,
                }
            }
        }

        impl Divide for $name {
            #[inline(always)]
            fn divide(&self, rhs: &Self, _typ: DivideType) -> Self {
                if !self.is_member() || !rhs.is_member() {
                    return Self::no_weight();
                }
                let f1 = self.neg_log_prob;
                let f2 = rhs.neg_log_prob;

                if f2 == $t::INFINITY {
                    Self {
                        sign: 1.0,
                        neg_log_prob: $t::NAN,
                    }
                } else if f1 == $t::INFINITY {
                    Self {
                        sign: 1.0,
                        neg_log_prob: $t::INFINITY,
                    }
                } else {
                    let sign = if self.is_positive() == rhs.is_positive() {
                        1.0
                    } else {
                        -1.0
                    };
                    Self {
                        sign,
                        neg_log_prob: f1 - f2,
                    }
                }
            }
        }

        impl Minus for $name {
            #[inline(always)]
            fn minus(&self, rhs: &Self) -> Self {
                self.minus(rhs)
            }
        }

        /// Kahan compensated adder specifically optimized for signed log weights.
        #[derive(Debug, Clone)]
        pub struct $adder_name {
            ssum: bool,
            sum: f64,
            c: f64,
        }

        impl $adder_name {
            #[inline]
            pub fn new(w: $name) -> Self {
                Self {
                    ssum: w.is_positive(),
                    sum: w.neg_log_prob as f64,
                    c: 0.0,
                }
            }

            #[inline]
            pub fn add(&mut self, w: &$name) -> $name {
                let sw = w.is_positive();
                let f = w.neg_log_prob as f64;
                let equal = self.ssum == sw;

                if !self.sum().is_member() || f == f64::INFINITY {
                    return self.sum();
                } else if !w.is_member() || self.sum == f64::INFINITY {
                    self.sum = f;
                    self.ssum = sw;
                    self.c = 0.0;
                } else if f == self.sum {
                    if equal {
                        self.sum = super::kahan_log_sum(self.sum, f, &mut self.c);
                    } else {
                        self.sum = f64::INFINITY;
                        self.ssum = true;
                        self.c = 0.0;
                    }
                } else if f > self.sum {
                    if equal {
                        self.sum = super::kahan_log_sum(self.sum, f, &mut self.c);
                    } else {
                        self.sum = super::kahan_log_diff(self.sum, f, &mut self.c);
                    }
                } else {
                    if equal {
                        self.sum = super::kahan_log_sum(f, self.sum, &mut self.c);
                    } else {
                        self.sum = super::kahan_log_diff(f, self.sum, &mut self.c);
                        self.ssum = sw;
                    }
                }
                self.sum()
            }

            #[inline]
            pub fn sum(&self) -> $name {
                $name {
                    sign: if self.ssum { 1.0 } else { -1.0 },
                    neg_log_prob: self.sum as $t,
                }
            }

            #[inline]
            pub fn reset(&mut self, w: $name) {
                self.ssum = w.is_positive();
                self.sum = w.neg_log_prob as f64;
                self.c = 0.0;
            }
        }

        impl Default for $adder_name {
            fn default() -> Self {
                Self::new(<$name as Weight>::zero())
            }
        }
    };
}

define_signed_log!(
    SignedLogWeight,
    SignedLogAdder,
    f32,
    "signed_log_tropical_log"
);
define_signed_log!(
    SignedLog64Weight,
    SignedLog64Adder,
    f64,
    "signed_log_tropical_log64"
);

// --- To Tropical ---
impl_weight_convert!(SignedLogWeight, TropicalWeight, |w: SignedLogWeight| {
    if !w.is_member() || !w.is_positive() {
        TropicalWeight::no_weight()
    } else {
        TropicalWeight(w.neg_log_prob)
    }
});
impl_weight_convert!(SignedLog64Weight, TropicalWeight, |w: SignedLog64Weight| {
    if !w.is_member() || !w.is_positive() {
        TropicalWeight::no_weight()
    } else {
        TropicalWeight(w.neg_log_prob as f32)
    }
});

// --- To Log ---
impl_weight_convert!(SignedLogWeight, LogWeight, |w: SignedLogWeight| {
    if !w.is_member() || !w.is_positive() {
        LogWeight::no_weight()
    } else {
        LogWeight(w.neg_log_prob)
    }
});
impl_weight_convert!(SignedLog64Weight, LogWeight, |w: SignedLog64Weight| {
    if !w.is_member() || !w.is_positive() {
        LogWeight::no_weight()
    } else {
        LogWeight(w.neg_log_prob as f32)
    }
});

// --- To Log64 ---
impl_weight_convert!(SignedLogWeight, Log64Weight, |w: SignedLogWeight| {
    if !w.is_member() || !w.is_positive() {
        Log64Weight::no_weight()
    } else {
        Log64Weight(w.neg_log_prob as f64)
    }
});
impl_weight_convert!(SignedLog64Weight, Log64Weight, |w: SignedLog64Weight| {
    if !w.is_member() || !w.is_positive() {
        Log64Weight::no_weight()
    } else {
        Log64Weight(w.neg_log_prob)
    }
});

// --- To Real ---
impl_weight_convert!(SignedLogWeight, RealWeight, |w: SignedLogWeight| {
    RealWeight(w.sign * (-w.neg_log_prob).exp())
});
impl_weight_convert!(SignedLog64Weight, RealWeight, |w: SignedLog64Weight| {
    RealWeight(w.sign * (-(w.neg_log_prob as f32)).exp())
});

// --- To Real64 ---
impl_weight_convert!(SignedLogWeight, Real64Weight, |w: SignedLogWeight| {
    Real64Weight(w.sign as f64 * (-(w.neg_log_prob as f64)).exp())
});
impl_weight_convert!(SignedLog64Weight, Real64Weight, |w: SignedLog64Weight| {
    Real64Weight(w.sign as f64 * (-w.neg_log_prob).exp())
});

// --- From Tropical ---
impl_weight_convert!(TropicalWeight, SignedLogWeight, |w: TropicalWeight| {
    SignedLogWeight::new(1.0, w.value())
});
impl_weight_convert!(TropicalWeight, SignedLog64Weight, |w: TropicalWeight| {
    SignedLog64Weight::new(1.0, w.value() as f64)
});
impl_weight_convert!(
    TropicalWeight64,
    SignedLog64Weight,
    |w: TropicalWeight64| { SignedLog64Weight::new(1.0, w.value()) }
);

// --- From Log ---
impl_weight_convert!(LogWeight, SignedLogWeight, |w: LogWeight| {
    SignedLogWeight::new(1.0, w.value())
});
impl_weight_convert!(LogWeight, SignedLog64Weight, |w: LogWeight| {
    SignedLog64Weight::new(1.0, w.value() as f64)
});
impl_weight_convert!(Log64Weight, SignedLog64Weight, |w: Log64Weight| {
    SignedLog64Weight::new(1.0, w.value())
});
impl_weight_convert!(Log64Weight, SignedLogWeight, |w: Log64Weight| {
    SignedLogWeight::new(1.0, w.value() as f32)
});

// --- From Real ---
impl_weight_convert!(RealWeight, SignedLogWeight, |w: RealWeight| {
    let sign = if w.value() >= 0.0 { 1.0 } else { -1.0 };
    SignedLogWeight::new(sign, -w.value().abs().ln())
});
impl_weight_convert!(RealWeight, SignedLog64Weight, |w: RealWeight| {
    let sign = if w.value() >= 0.0 { 1.0 } else { -1.0 };
    SignedLog64Weight::new(sign, -(w.value().abs() as f64).ln())
});

// --- From Real64 ---
impl_weight_convert!(Real64Weight, SignedLogWeight, |w: Real64Weight| {
    let sign = if w.value() >= 0.0 { 1.0 } else { -1.0 };
    SignedLogWeight::new(sign, -(w.value().abs() as f32).ln())
});
impl_weight_convert!(Real64Weight, SignedLog64Weight, |w: Real64Weight| {
    let sign = if w.value() >= 0.0 { 1.0 } else { -1.0 };
    SignedLog64Weight::new(sign, -w.value().abs().ln())
});

// --- Between SignedLog and SignedLog64 ---
impl_weight_convert!(
    SignedLog64Weight,
    SignedLogWeight,
    |w: SignedLog64Weight| {
        SignedLogWeight {
            sign: w.sign,
            neg_log_prob: w.neg_log_prob as f32,
        }
    }
);
impl_weight_convert!(SignedLogWeight, SignedLog64Weight, |w: SignedLogWeight| {
    SignedLog64Weight {
        sign: w.sign,
        neg_log_prob: w.neg_log_prob as f64,
    }
});

// Upstream generalized these to any pair of widths in 694dc53 ("Greatly reduce
// number of `WeightConvert` specializations via partial specialization"). The
// combinations sicada was missing follow.
impl_weight_convert!(SignedLogWeight, TropicalWeight64, |w: SignedLogWeight| {
    if !w.is_member() || !w.is_positive() {
        TropicalWeight64::no_weight()
    } else {
        TropicalWeight64(w.neg_log_prob as f64)
    }
});
impl_weight_convert!(
    SignedLog64Weight,
    TropicalWeight64,
    |w: SignedLog64Weight| {
        if !w.is_member() || !w.is_positive() {
            TropicalWeight64::no_weight()
        } else {
            TropicalWeight64(w.neg_log_prob)
        }
    }
);
impl_weight_convert!(TropicalWeight64, SignedLogWeight, |w: TropicalWeight64| {
    SignedLogWeight::new(1.0, w.value() as f32)
});

#[cfg(test)]
mod tests {
    use super::*;

    /// Zero has two spellings, either sign with an infinite magnitude, and they
    /// have to compare equal, hash alike, and be approximately equal. Multiplying
    /// Zero by a negative weight produces the negative spelling, so this is not a
    /// corner case: distributivity depends on it.
    #[test]
    fn the_two_spellings_of_zero_are_the_same_weight() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let positive_zero = SignedLogWeight::new(1.0, f32::INFINITY);
        let negative_zero = SignedLogWeight::new(-1.0, f32::INFINITY);
        assert_eq!(positive_zero, SignedLogWeight::zero());
        assert_eq!(positive_zero, negative_zero);
        assert!(positive_zero.approx_equal(&negative_zero, 1e-6));

        let hash_of = |w: &SignedLogWeight| {
            let mut hasher = DefaultHasher::new();
            w.hash(&mut hasher);
            hasher.finish()
        };
        assert_eq!(hash_of(&positive_zero), hash_of(&negative_zero));

        // And Zero times a negative weight is still Zero, however it is spelled.
        let negative = SignedLogWeight::new(-1.0, 2.0);
        assert_eq!(
            SignedLogWeight::zero().times(&negative),
            SignedLogWeight::zero()
        );
    }

    /// A weight and its negation are opposites, not near-equals. `approx_equal`
    /// used to compare the magnitudes against `0.0`, which is One rather than
    /// Zero, and so reported `+1` and `-1` as approximately equal.
    #[test]
    fn a_weight_and_its_negation_are_not_approximately_equal() {
        let positive = SignedLogWeight::new(1.0, 0.0);
        let negative = SignedLogWeight::new(-1.0, 0.0);
        assert_ne!(positive, negative);
        assert!(!positive.approx_equal(&negative, 1e-6));
        assert!(
            !positive.approx_equal(&negative, 10.0),
            "not at any tolerance"
        );
    }

    /// The signed log semiring adds a sign to the log semiring, so it loses
    /// idempotence and the path property but keeps commutativity.
    #[test]
    fn it_satisfies_the_axioms_it_claims() {
        use crate::weight::axioms;

        axioms::check(&[
            SignedLogWeight::new(1.0, 1.0),
            SignedLogWeight::new(-1.0, 2.0),
            SignedLogWeight::new(1.0, 3.0),
        ]);
        axioms::check(&[
            SignedLog64Weight::new(1.0, 1.0),
            SignedLog64Weight::new(-1.0, 2.0),
        ]);
    }

    /// A negative weight has no counterpart in the unsigned semirings, so the
    /// conversion has to refuse rather than silently drop the sign.
    #[test]
    fn converting_a_negative_weight_out_yields_no_weight() {
        let negative = SignedLogWeight::new(-1.0, 2.0);
        assert!(!TropicalWeight::from(negative).is_member());
        assert!(!LogWeight::from(negative).is_member());
        assert!(!TropicalWeight64::from(negative).is_member());

        let positive = SignedLogWeight::new(1.0, 2.0);
        assert_eq!(TropicalWeight::from(positive), TropicalWeight(2.0));
        assert_eq!(LogWeight::from(positive), LogWeight(2.0));
        assert_eq!(TropicalWeight64::from(positive), TropicalWeight64(2.0));
    }

    #[test]
    fn converting_in_from_an_unsigned_weight_gives_a_positive_one() {
        assert!(SignedLogWeight::from(TropicalWeight(2.0)).is_positive());
        assert!(SignedLogWeight::from(LogWeight(2.0)).is_positive());
        assert!(SignedLogWeight::from(TropicalWeight64(2.0)).is_positive());
    }

    #[test]
    fn the_two_widths_convert_both_ways() {
        let narrow = SignedLogWeight::new(-1.0, 2.5);
        let wide = SignedLog64Weight::from(narrow);
        assert_eq!(wide.sign, narrow.sign);
        assert_eq!(wide.neg_log_prob, 2.5);
        assert_eq!(SignedLogWeight::from(wide), narrow);
    }

    /// Adding a weight to its own negation must cancel exactly.
    #[test]
    fn a_weight_and_its_negation_cancel() {
        let positive = SignedLogWeight::new(1.0, 2.0);
        let negative = SignedLogWeight::new(-1.0, 2.0);
        assert_eq!(positive.plus(&negative), SignedLogWeight::zero());
        assert_eq!(negative.plus(&positive), SignedLogWeight::zero());
    }

    /// The compensated adder must not drift over a long run of equal addends,
    /// which is the reason upstream specializes it.
    #[test]
    fn the_compensated_adder_stays_accurate_over_a_long_sum() {
        let term = SignedLogWeight::new(1.0, 1.0);
        let mut adder = SignedLogAdder::new(SignedLogWeight::zero());
        for _ in 0..10_000 {
            adder.add(&term);
        }
        // Summing n copies of e^-1 gives n * e^-1, so the weight is
        // -ln(n) + 1 in the negative-log representation.
        let expected = 1.0 - (10_000f32).ln();
        let sum = adder.sum();
        assert!(sum.is_positive());
        assert!(
            (sum.neg_log_prob - expected).abs() < 1e-2,
            "{} vs {expected}",
            sum.neg_log_prob
        );
    }

    #[test]
    fn test_signed_log_weight_parse_display() {
        let text = "-1,2.5";
        let w: SignedLogWeight = text.parse().unwrap();
        assert_eq!(w.to_string(), text);
        assert!(!w.is_positive());
        assert_eq!(w.neg_log_prob, 2.5);

        let zero = SignedLogWeight::zero();
        assert_eq!(zero.to_string(), "1,Infinity");
    }

    #[test]
    fn test_signed_log_weight_plus() {
        // e^-2 + e^-3, which is -ln(e^-2 + e^-3).
        let w1 = SignedLogWeight::new(1.0, 2.0);
        let w2 = SignedLogWeight::new(1.0, 3.0);
        let p = w1.plus(&w2);

        let expected = -((-2.0f32).exp() + (-3.0f32).exp()).ln();
        assert!((p.neg_log_prob - expected).abs() < 1e-5);
    }

    #[test]
    fn test_signed_log_weight_minus() {
        // Adding opposite signs, so the magnitudes subtract.
        let w1 = SignedLogWeight::new(1.0, 2.0);
        let w2 = SignedLogWeight::new(-1.0, 3.0);
        let p = w1.plus(&w2);

        // 1.0*e^-2 - 1.0*e^-3: positive, and -ln(e^-2 - e^-3).
        let expected = -((-2.0f32).exp() - (-3.0f32).exp()).ln();
        assert!(p.is_positive());
        assert!((p.neg_log_prob - expected).abs() < 1e-5);
    }

    #[test]
    fn test_signed_log_adder() {
        let mut adder = SignedLogAdder::new(SignedLogWeight::zero());
        adder.add(&SignedLogWeight::new(1.0, 2.0));
        adder.add(&SignedLogWeight::new(-1.0, 3.0));

        let p = adder.sum();
        let expected = -((-2.0f32).exp() - (-3.0f32).exp()).ln();
        assert!(p.is_positive());
        assert!((p.neg_log_prob - expected).abs() < 1e-5);
    }
}
