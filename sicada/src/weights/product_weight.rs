use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use crate::error::ParseError;
use crate::fst_type::WeightType;
use crate::utils::split_composite_weight;
use crate::weight::{
    Adder, COMMUTATIVE, CommutativeWeight, Divide, DivideType, IDEMPOTENT, IdempotentWeight,
    LEFT_SEMIRING, LeftSemiring, RIGHT_SEMIRING, RightSemiring, Weight,
};

/// Product weight set and associated semiring operation definitions.
///
/// Product semiring: W1 * W2.
/// The operations (Plus, Times, Divide) are computed independently on each component.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProductWeight<W1: Weight, W2: Weight> {
    pub value1: W1,
    pub value2: W2,
}

impl<W1: Weight, W2: Weight> ProductWeight<W1, W2> {
    #[inline(always)]
    pub fn new(value1: W1, value2: W2) -> Self {
        Self { value1, value2 }
    }
}

impl<W1: Weight, W2: Weight> Weight for ProductWeight<W1, W2> {
    type ReverseWeight = ProductWeight<W1::ReverseWeight, W2::ReverseWeight>;

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
        let s = format!("{}_X_{}", W1::type_name(), W2::type_name());
        WeightType::new_dynamic(s)
    }

    #[inline(always)]
    fn properties() -> u64 {
        W1::properties()
            & W2::properties()
            & (LEFT_SEMIRING | RIGHT_SEMIRING | COMMUTATIVE | IDEMPOTENT)
    }

    #[inline(always)]
    fn is_member(&self) -> bool {
        self.value1.is_member() && self.value2.is_member()
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
        ProductWeight {
            value1: W1::reverse(&self.value1),
            value2: W2::reverse(&self.value2),
        }
    }

    #[inline]
    fn plus(&self, rhs: &Self) -> Self {
        Self {
            value1: W1::plus(&self.value1, &rhs.value1),
            value2: W2::plus(&self.value2, &rhs.value2),
        }
    }

    #[inline]
    fn times(&self, rhs: &Self) -> Self {
        Self {
            value1: W1::times(&self.value1, &rhs.value1),
            value2: W2::times(&self.value2, &rhs.value2),
        }
    }
}

impl<W1: Weight + Divide, W2: Weight + Divide> Divide for ProductWeight<W1, W2> {
    #[inline]
    fn divide(&self, rhs: &Self, typ: DivideType) -> Self {
        Self {
            value1: W1::divide(&self.value1, &rhs.value1, typ),
            value2: W2::divide(&self.value2, &rhs.value2, typ),
        }
    }
}

impl<W1: Weight, W2: Weight> fmt::Display for ProductWeight<W1, W2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{},{}", self.value1, self.value2)
    }
}

impl<W1: Weight + FromStr, W2: Weight + FromStr> FromStr for ProductWeight<W1, W2> {
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
            ParseError::InvalidFormat(format!("Failed to parse W1 of ProductWeight: {}", parts[0]))
        })?;
        let value2 = parts[1].parse::<W2>().map_err(|_| {
            ParseError::InvalidFormat(format!("Failed to parse W2 of ProductWeight: {}", parts[1]))
        })?;

        Ok(Self { value1, value2 })
    }
}

/// Accumulator specifically tailored for ProductWeight.
/// It wraps independent accumulators for each component.
#[derive(Debug, Clone)]
pub struct ProductAdder<W1: Weight, W2: Weight> {
    adder1: Adder<W1>,
    adder2: Adder<W2>,
}

impl<W1: Weight, W2: Weight> ProductAdder<W1, W2> {
    #[inline]
    pub fn new(w: ProductWeight<W1, W2>) -> Self {
        let mut adder1 = Adder::new();
        adder1.reset(w.value1);

        let mut adder2 = Adder::new();
        adder2.reset(w.value2);

        Self { adder1, adder2 }
    }

    #[inline]
    pub fn add(&mut self, w: &ProductWeight<W1, W2>) -> ProductWeight<W1, W2> {
        self.adder1.add(&w.value1);
        self.adder2.add(&w.value2);
        self.sum()
    }

    #[inline]
    pub fn sum(&self) -> ProductWeight<W1, W2> {
        ProductWeight::new(self.adder1.sum(), self.adder2.sum())
    }

    #[inline]
    pub fn reset(&mut self, w: ProductWeight<W1, W2>) {
        self.adder1.reset(w.value1);
        self.adder2.reset(w.value2);
    }
}

impl<W1: Weight, W2: Weight> Default for ProductAdder<W1, W2> {
    fn default() -> Self {
        Self::new(ProductWeight::zero())
    }
}

// A product has whatever both halves have, which is exactly what upstream's
// `Properties()` computes at run time.
//
// SICADA-DIVERGE: upstream reports these only as run-time bits, so an algorithm
// that needs left distributivity can be instantiated with a product that does
// not have it. The bits are unchanged; these let a bound say the same thing.
impl<W1: Weight + LeftSemiring, W2: Weight + LeftSemiring> LeftSemiring for ProductWeight<W1, W2> {}
impl<W1: Weight + RightSemiring, W2: Weight + RightSemiring> RightSemiring
    for ProductWeight<W1, W2>
{
}
impl<W1: Weight + CommutativeWeight, W2: Weight + CommutativeWeight> CommutativeWeight
    for ProductWeight<W1, W2>
{
}
impl<W1: Weight + IdempotentWeight, W2: Weight + IdempotentWeight> IdempotentWeight
    for ProductWeight<W1, W2>
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::float_weight::TropicalWeight;

    type ProdTropical = ProductWeight<TropicalWeight, TropicalWeight>;

    #[test]
    fn it_satisfies_the_axioms_it_claims() {
        use crate::weight::axioms;
        use crate::weights::float_weight::LogWeight;

        // Both components tropical: idempotent, path-like componentwise.
        axioms::check(&[
            ProdTropical::new(TropicalWeight(1.0), TropicalWeight(2.0)),
            ProdTropical::new(TropicalWeight(2.0), TropicalWeight(1.0)),
        ]);
        axioms::check_divide(&[ProdTropical::new(TropicalWeight(1.0), TropicalWeight(2.0))]);

        // Mixed components: the claim is the intersection of the two.
        type Mixed = ProductWeight<TropicalWeight, LogWeight>;
        assert_eq!(
            Mixed::properties(),
            TropicalWeight::properties() & LogWeight::properties() & Mixed::properties()
        );
        axioms::check(&[
            Mixed::new(TropicalWeight(1.0), LogWeight(2.0)),
            Mixed::new(TropicalWeight(2.0), LogWeight(1.0)),
        ]);
    }

    #[test]
    fn test_product_weight_parse_display() {
        let text = "2.5,4.5";
        let w: ProdTropical = text.parse().unwrap();

        assert_eq!(w.to_string(), text);
        assert_eq!(w.value1.value(), 2.5);
        assert_eq!(w.value2.value(), 4.5);
    }

    #[test]
    fn test_product_weight_plus() {
        let w1: ProdTropical = "2,8".parse().unwrap();
        let w2: ProdTropical = "5,1".parse().unwrap();

        let w3 = w1.plus(&w2);
        // TropicalWeight Plus operates independently as min(a, b) on each side
        // min(2, 5) = 2, min(8, 1) = 1
        assert_eq!(w3.to_string(), "2,1");
    }

    #[test]
    fn test_product_weight_times() {
        let w1: ProdTropical = "2,4".parse().unwrap();
        let w2: ProdTropical = "3,5".parse().unwrap();

        let w3 = w1.times(&w2);
        // TropicalWeight Times computes the sum independently
        // 2+3 = 5, 4+5 = 9
        assert_eq!(w3.to_string(), "5,9");
    }

    #[test]
    fn test_product_weight_divide() {
        let w1: ProdTropical = "8,9".parse().unwrap();
        let w2: ProdTropical = "2,5".parse().unwrap();

        let w3 = w1.divide(&w2, DivideType::Any);
        // TropicalWeight Divide computes the difference independently
        // 8-2 = 6, 9-5 = 4
        assert_eq!(w3.to_string(), "6,4");
    }

    #[test]
    fn test_product_weight_member() {
        let zero = ProdTropical::zero();
        let one = ProdTropical::one();

        assert!(zero.is_member());
        assert!(one.is_member());

        // Unlike LexicographicWeight, ProductWeight allows mixed Zero/NonZero status.
        let mixed = ProdTropical::new(TropicalWeight::zero(), TropicalWeight::one());
        assert!(mixed.is_member());
    }

    /// The marker traits a product carries have to be the ones its property
    /// bits report, which for a product is the intersection of its halves'.
    #[test]
    fn the_marker_traits_agree_with_the_property_bits() {
        use crate::weights::float_weight::TropicalWeight;
        use crate::weights::string_weight::{StringLeft, StringWeight};

        fn left<W: Weight + LeftSemiring>() {
            assert_ne!(W::properties() & LEFT_SEMIRING, 0, "{}", W::type_name());
        }
        fn right<W: Weight + RightSemiring>() {
            assert_ne!(W::properties() & RIGHT_SEMIRING, 0, "{}", W::type_name());
        }
        fn commutative<W: Weight + CommutativeWeight>() {
            assert_ne!(W::properties() & COMMUTATIVE, 0, "{}", W::type_name());
        }
        fn idempotent<W: Weight + IdempotentWeight>() {
            assert_ne!(W::properties() & IDEMPOTENT, 0, "{}", W::type_name());
        }

        type Both = ProductWeight<TropicalWeight, TropicalWeight>;
        left::<Both>();
        right::<Both>();
        commutative::<Both>();
        idempotent::<Both>();

        // Pairing with a left-only string weight loses right distributivity,
        // and the bit agrees.
        type LeftOnly = ProductWeight<StringWeight<i32, StringLeft>, TropicalWeight>;
        left::<LeftOnly>();
        assert_eq!(LeftOnly::properties() & RIGHT_SEMIRING, 0);
    }
}
