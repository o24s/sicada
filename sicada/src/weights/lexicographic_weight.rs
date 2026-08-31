use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use crate::error::ParseError;
use crate::fst_type::WeightType;
use crate::utils::split_composite_weight;
use crate::weight::{
    COMMUTATIVE, Divide, DivideType, IDEMPOTENT, LEFT_SEMIRING, PATH, RIGHT_SEMIRING, Weight,
};

/// Lexicographic weight set and associated semiring operation definitions.
///
/// A lexicographic weight is a sequence of weights, each of which must have the
/// path property and Times() must be (strongly) cancellative.
/// The + operation on two weights a and b is the lexicographically prior of a and b.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LexicographicWeight<W1: Weight, W2: Weight> {
    pub value1: W1,
    pub value2: W2,
}

impl<W1: Weight, W2: Weight> LexicographicWeight<W1, W2> {
    #[inline(always)]
    pub fn new(value1: W1, value2: W2) -> Self {
        Self { value1, value2 }
    }
}

impl<W1: Weight, W2: Weight> Weight for LexicographicWeight<W1, W2> {
    type ReverseWeight = LexicographicWeight<W1::ReverseWeight, W2::ReverseWeight>;

    #[inline(always)]
    fn zero() -> Self {
        Self {
            value1: W1::zero(),
            value2: W2::zero(),
        }
    }

    #[inline(always)]
    fn one() -> Self {
        Self {
            value1: W1::one(),
            value2: W2::one(),
        }
    }

    #[inline(always)]
    fn no_weight() -> Self {
        Self {
            value1: W1::no_weight(),
            value2: W2::no_weight(),
        }
    }

    fn type_name() -> WeightType {
        let s = format!("{}_LT_{}", W1::type_name(), W2::type_name());
        WeightType::new_dynamic(s)
    }

    #[inline(always)]
    fn properties() -> u64 {
        W1::properties()
            & W2::properties()
            & (LEFT_SEMIRING | RIGHT_SEMIRING | PATH | IDEMPOTENT | COMMUTATIVE)
    }

    #[inline(always)]
    fn is_member(&self) -> bool {
        if !self.value1.is_member() || !self.value2.is_member() {
            return false;
        }

        let is_z1 = self.value1 == W1::zero();
        let is_z2 = self.value2 == W2::zero();

        // Lexicographic weights cannot mix zeroes and non-zeroes.
        is_z1 == is_z2
    }

    #[inline]
    fn approx_equal(&self, other: &Self, delta: f32) -> bool {
        W1::approx_equal(&self.value1, &other.value1, delta)
            && W2::approx_equal(&self.value2, &other.value2, delta)
    }

    #[inline]
    fn quantize(&self, delta: f32) -> Self {
        Self {
            value1: W1::quantize(&self.value1, delta),
            value2: W2::quantize(&self.value2, delta),
        }
    }

    #[inline]
    fn reverse(&self) -> Self::ReverseWeight {
        LexicographicWeight {
            value1: W1::reverse(&self.value1),
            value2: W2::reverse(&self.value2),
        }
    }

    #[inline]
    fn plus(&self, rhs: &Self) -> Self {
        if !self.is_member() || !rhs.is_member() {
            return Self::no_weight();
        }

        // SICADA-OPT: upstream's `NaturalLess(a, b)` is `Plus(a, b) == a && a != b`,
        // which computes the sum whether or not the two are equal. Comparing first
        // means the sum is only computed when they differ.

        // On the first component.
        if self.value1 != rhs.value1 {
            let p1 = W1::plus(&self.value1, &rhs.value1);
            if p1 == self.value1 {
                return self.clone(); // self comes first
            }
            if p1 == rhs.value1 {
                return rhs.clone(); // rhs comes first
            }
        }

        // The first components are equal, so the second decides.
        if self.value2 != rhs.value2 {
            let p2 = W2::plus(&self.value2, &rhs.value2);
            if p2 == self.value2 {
                return self.clone(); // self comes first
            }
            if p2 == rhs.value2 {
                return rhs.clone(); // rhs comes first
            }
        }

        // Equal throughout, or not ordered against each other at all.
        self.clone()
    }

    #[inline]
    fn times(&self, rhs: &Self) -> Self {
        Self {
            value1: W1::times(&self.value1, &rhs.value1),
            value2: W2::times(&self.value2, &rhs.value2),
        }
    }
}

impl<W1: Weight + Divide, W2: Weight + Divide> Divide for LexicographicWeight<W1, W2> {
    #[inline]
    fn divide(&self, rhs: &Self, typ: DivideType) -> Self {
        Self {
            value1: W1::divide(&self.value1, &rhs.value1, typ),
            value2: W2::divide(&self.value2, &rhs.value2, typ),
        }
    }
}

impl<W1: Weight, W2: Weight> fmt::Display for LexicographicWeight<W1, W2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{},{}", self.value1, self.value2)
    }
}

impl<W1: Weight + FromStr, W2: Weight + FromStr> FromStr for LexicographicWeight<W1, W2> {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts = split_composite_weight(s, ',', '(', ')')?;
        if parts.len() != 2 {
            return Err(ParseError::InvalidElementCount {
                expected: 2,
                found: parts.len(),
            });
        }

        let value1 = parts[0].parse::<W1>().map_err(|_| {
            ParseError::InvalidFormat(format!(
                "Failed to parse W1 of LexicographicWeight: {}",
                parts[0]
            ))
        })?;
        let value2 = parts[1].parse::<W2>().map_err(|_| {
            ParseError::InvalidFormat(format!(
                "Failed to parse W2 of LexicographicWeight: {}",
                parts[1]
            ))
        })?;

        Ok(Self { value1, value2 })
    }
}

#[cfg(test)]
mod tests {
    /// The lexicographic semiring orders by the first component and breaks ties
    /// with the second, so it is a semiring only when both components are.
    #[test]
    fn it_satisfies_the_axioms_it_claims() {
        use crate::weight::axioms;
        use crate::weights::float_weight::TropicalWeight;

        type Lex = LexicographicWeight<TropicalWeight, TropicalWeight>;
        axioms::check(&[
            Lex::new(TropicalWeight(1.0), TropicalWeight(2.0)),
            Lex::new(TropicalWeight(1.0), TropicalWeight(3.0)),
            Lex::new(TropicalWeight(2.0), TropicalWeight(1.0)),
        ]);
        axioms::check_divide(&[Lex::new(TropicalWeight(1.0), TropicalWeight(2.0))]);
    }

    /// Ties in the first component must be broken by the second, which is the
    /// entire purpose of the weight.
    #[test]
    fn plus_breaks_ties_with_the_second_component() {
        use crate::weights::float_weight::TropicalWeight;

        let cheaper_tail = LexicographicWeight::<TropicalWeight, TropicalWeight>::new(
            TropicalWeight(1.0),
            TropicalWeight(2.0),
        );
        let costlier_tail = LexicographicWeight::<TropicalWeight, TropicalWeight>::new(
            TropicalWeight(1.0),
            TropicalWeight(5.0),
        );
        assert_eq!(cheaper_tail.plus(&costlier_tail), cheaper_tail);

        // But the first component still dominates.
        let cheaper_head = LexicographicWeight::<TropicalWeight, TropicalWeight>::new(
            TropicalWeight(0.5),
            TropicalWeight(9.0),
        );
        assert_eq!(cheaper_head.plus(&cheaper_tail), cheaper_head);
    }

    use super::*;
    use crate::weights::float_weight::TropicalWeight;

    type LexTropical = LexicographicWeight<TropicalWeight, TropicalWeight>;

    #[test]
    fn test_lexicographic_weight_parse_display() {
        let text = "2.5,4.5";
        let w: LexTropical = text.parse().unwrap();

        assert_eq!(w.to_string(), text);
        assert_eq!(w.value1.value(), 2.5);
        assert_eq!(w.value2.value(), 4.5);
    }

    #[test]
    fn test_lexicographic_weight_plus() {
        // ⊕ over the tropical semiring is `min`, so the cheaper of the two comes
        // first in the lexicographic order.

        // The first components differ.
        let w1: LexTropical = "2,8".parse().unwrap();
        let w2: LexTropical = "5,1".parse().unwrap();
        let w3 = w1.plus(&w2);
        // 2 < 5 on the first component, so w1 wins.
        assert_eq!(w3.to_string(), "2,8");

        // The first components are equal.
        let w4: LexTropical = "2,8".parse().unwrap();
        let w5: LexTropical = "2,5".parse().unwrap();
        let w6 = w4.plus(&w5);
        // So the second decides: 5 < 8.
        assert_eq!(w6.to_string(), "2,5");
    }

    #[test]
    fn test_lexicographic_weight_times() {
        let w1: LexTropical = "2,4".parse().unwrap();
        let w2: LexTropical = "3,5".parse().unwrap();

        let w3 = w1.times(&w2);
        // ⊗ over the tropical semiring adds: 2+3=5, 4+5=9.
        assert_eq!(w3.to_string(), "5,9");
    }

    #[test]
    fn test_lexicographic_weight_divide() {
        let w1: LexTropical = "8,9".parse().unwrap();
        let w2: LexTropical = "2,5".parse().unwrap();

        let w3 = w1.divide(&w2, DivideType::Any);
        // Division subtracts: 8-2=6, 9-5=4.
        assert_eq!(w3.to_string(), "6,4");
    }

    #[test]
    fn test_lexicographic_weight_member() {
        let zero = LexTropical::zero();
        let one = LexTropical::one();

        assert!(zero.is_member());
        assert!(one.is_member());

        // A pair may not mix zero with a weight that is not zero.
        let invalid = LexTropical::new(TropicalWeight::zero(), TropicalWeight::one());
        assert!(!invalid.is_member());
    }
}
