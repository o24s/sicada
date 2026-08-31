use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use crate::error::ParseError;
use crate::fst_type::WeightType;
use crate::utils::split_composite_weight;
use crate::weight::{
    COMMUTATIVE, Divide, DivideType, IDEMPOTENT, LEFT_SEMIRING, RIGHT_SEMIRING, Weight,
};

/// Cartesian power semiring: W ^ n
///
/// Forms:
///  - a left semimodule when W is a left semiring,
///  - a right semimodule when W is a right semiring,
///  - a bisemimodule when W is a semiring,
///    the free semimodule of rank n over W
///
/// The Times operation is overloaded to provide the left and right scalar products.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PowerWeight<W: Weight, const N: usize> {
    pub elements: [W; N],
}

impl<W: Weight, const N: usize> PowerWeight<W, N> {
    #[inline]
    pub fn new(elements: [W; N]) -> Self {
        Self { elements }
    }

    /// Builds a vector holding `weight` at `index` and `default_weight` elsewhere.
    ///
    /// Corresponds to upstream's `PowerWeight(size_t index, const W &weight,
    /// const W &default_weight)`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of range.
    #[inline]
    pub fn from_component(index: usize, weight: W, default_weight: W) -> Self {
        assert!(
            index < N,
            "component index {index} out of range for rank {N}"
        );
        let mut elements = std::array::from_fn(|_| default_weight.clone());
        elements[index] = weight;
        Self { elements }
    }

    #[inline]
    pub fn value(&self, index: usize) -> &W {
        &self.elements[index]
    }

    #[inline]
    pub fn set_value(&mut self, index: usize, weight: W) {
        self.elements[index] = weight;
    }

    /// Semimodule left scalar product.
    #[inline]
    pub fn times_scalar_left(scalar: &W, weight: &Self) -> Self {
        Self {
            elements: std::array::from_fn(|i| W::times(scalar, &weight.elements[i])),
        }
    }

    /// Semimodule right scalar product.
    #[inline]
    pub fn times_scalar_right(weight: &Self, scalar: &W) -> Self {
        Self {
            elements: std::array::from_fn(|i| W::times(&weight.elements[i], scalar)),
        }
    }

    /// Semimodule dot product.
    #[inline]
    pub fn dot_product(&self, other: &Self) -> W {
        let mut result = W::zero();
        for i in 0..N {
            result = W::plus(&result, &W::times(&self.elements[i], &other.elements[i]));
        }
        result
    }
}

// `FromStr` is left off the bound, since requiring it here sends the
// trait solver round a cycle.
impl<W: Weight, const N: usize> Weight for PowerWeight<W, N> {
    type ReverseWeight = PowerWeight<W::ReverseWeight, N>;

    #[inline(always)]
    fn zero() -> Self {
        Self {
            elements: std::array::from_fn(|_| W::zero()),
        }
    }

    #[inline(always)]
    fn one() -> Self {
        Self {
            elements: std::array::from_fn(|_| W::one()),
        }
    }

    #[inline(always)]
    fn no_weight() -> Self {
        Self {
            elements: std::array::from_fn(|_| W::no_weight()),
        }
    }

    fn type_name() -> WeightType {
        let s = format!("{}_^{}", W::type_name(), N);
        WeightType::new_dynamic(s)
    }

    #[inline(always)]
    fn properties() -> u64 {
        W::properties() & (LEFT_SEMIRING | RIGHT_SEMIRING | COMMUTATIVE | IDEMPOTENT)
    }

    #[inline(always)]
    fn is_member(&self) -> bool {
        self.elements.iter().all(|w| w.is_member())
    }

    #[inline]
    fn approx_equal(&self, other: &Self, delta: f32) -> bool {
        self.elements
            .iter()
            .zip(other.elements.iter())
            .all(|(a, b)| W::approx_equal(a, b, delta))
    }

    #[inline]
    fn quantize(&self, delta: f32) -> Self {
        Self {
            elements: std::array::from_fn(|i| W::quantize(&self.elements[i], delta)),
        }
    }

    #[inline]
    fn reverse(&self) -> Self::ReverseWeight {
        PowerWeight {
            elements: std::array::from_fn(|i| W::reverse(&self.elements[i])),
        }
    }

    #[inline]
    fn plus(&self, rhs: &Self) -> Self {
        Self {
            elements: std::array::from_fn(|i| W::plus(&self.elements[i], &rhs.elements[i])),
        }
    }

    #[inline]
    fn times(&self, rhs: &Self) -> Self {
        Self {
            elements: std::array::from_fn(|i| W::times(&self.elements[i], &rhs.elements[i])),
        }
    }
}

impl<W: Weight + Divide, const N: usize> Divide for PowerWeight<W, N> {
    #[inline]
    fn divide(&self, rhs: &Self, typ: DivideType) -> Self {
        Self {
            elements: std::array::from_fn(|i| W::divide(&self.elements[i], &rhs.elements[i], typ)),
        }
    }
}

impl<W: Weight, const N: usize> fmt::Display for PowerWeight<W, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, w) in self.elements.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", w)?;
        }
        Ok(())
    }
}

impl<W: Weight, const N: usize> FromStr for PowerWeight<W, N> {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts = split_composite_weight(s, ',', '(', ')')?;
        if parts.len() != N {
            return Err(ParseError::InvalidElementCount {
                expected: N,
                found: parts.len(),
            });
        }

        // One element at a time.
        // Collected into a `Vec` first, since there is no way to build the fixed
        // size array element by element without assuming a `Default`.
        let mut parsed = Vec::with_capacity(N);
        for p in parts {
            let w = p.parse::<W>().map_err(|_| {
                ParseError::InvalidFormat(format!("Failed to parse element: {}", p))
            })?;
            parsed.push(w);
        }

        let elements: [W; N] = parsed.try_into().map_err(|_| {
            ParseError::InvalidFormat("Conversion from Vec to array failed".to_string())
        })?;

        Ok(Self { elements })
    }
}

impl<W: Weight, const N: usize> Hash for PowerWeight<W, N>
where
    W: Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.elements.hash(state);
    }
}

#[cfg(test)]
mod tests {
    /// The free semimodule of rank n: componentwise plus and times, so it
    /// inherits whatever the component semiring satisfies.
    #[test]
    fn it_satisfies_the_axioms_it_claims() {
        use crate::weight::axioms;
        use crate::weights::float_weight::{LogWeight, TropicalWeight};

        type Power3 = PowerWeight<TropicalWeight, 3>;
        axioms::check(&[
            Power3::new([
                TropicalWeight(1.0),
                TropicalWeight(2.0),
                TropicalWeight(3.0),
            ]),
            Power3::new([
                TropicalWeight(3.0),
                TropicalWeight(1.0),
                TropicalWeight(2.0),
            ]),
        ]);
        axioms::check_divide(&[Power3::new([
            TropicalWeight(1.0),
            TropicalWeight(2.0),
            TropicalWeight(3.0),
        ])]);

        type PowerLog2 = PowerWeight<LogWeight, 2>;
        axioms::check(&[
            PowerLog2::new([LogWeight(1.0), LogWeight(2.0)]),
            PowerLog2::new([LogWeight(2.0), LogWeight(1.0)]),
        ]);
    }

    /// A rank-zero power weight is the trivial semiring with one element.
    #[test]
    fn rank_zero_is_the_trivial_semiring() {
        use crate::weight::axioms;
        use crate::weights::float_weight::TropicalWeight;

        type Power0 = PowerWeight<TropicalWeight, 0>;
        assert_eq!(Power0::zero(), Power0::one());
        axioms::check::<Power0>(&[]);
    }

    use super::*;
    use crate::weights::float_weight::TropicalWeight;

    type PowerTropical = PowerWeight<TropicalWeight, 3>;

    #[test]
    fn test_power_weight_parse_display() {
        let text = "1.5,2.5,3.5";
        let w: PowerTropical = text.parse().unwrap();

        assert_eq!(w.to_string(), text);
        assert_eq!(w.value(0).value(), 1.5);
        assert_eq!(w.value(1).value(), 2.5);
        assert_eq!(w.value(2).value(), 3.5);
    }

    #[test]
    fn test_power_weight_plus() {
        let w1: PowerTropical = "2,4,6".parse().unwrap();
        let w2: PowerTropical = "5,3,8".parse().unwrap();

        let w3 = w1.plus(&w2);
        // ⊕ over the tropical semiring is `min`.
        assert_eq!(w3.to_string(), "2,3,6");
    }

    #[test]
    fn test_power_weight_times() {
        let w1: PowerTropical = "2,4,6".parse().unwrap();
        let w2: PowerTropical = "5,3,8".parse().unwrap();

        let w3 = w1.times(&w2);
        // ⊗ adds.
        assert_eq!(w3.to_string(), "7,7,14");
    }

    #[test]
    fn test_power_weight_divide() {
        let w1: PowerTropical = "5,8,12".parse().unwrap();
        let w2: PowerTropical = "2,3,4".parse().unwrap();

        let w3 = w1.divide(&w2, DivideType::Any);
        // Division subtracts.
        assert_eq!(w3.to_string(), "3,5,8");
    }

    #[test]
    fn test_power_weight_dot_product() {
        let w1: PowerTropical = "2,4,6".parse().unwrap();
        let w2: PowerTropical = "5,3,8".parse().unwrap();

        let dot = w1.dot_product(&w2);
        // Times: 2+5=7, 4+3=7, 6+8=14
        // ⊕ over the products: min(7, 7, 14) = 7.
        assert_eq!(dot.value(), 7.0);
    }

    #[test]
    fn test_power_weight_scalar_products() {
        let w1: PowerTropical = "2,4,6".parse().unwrap();
        let scalar = TropicalWeight(10.0);

        let left = PowerTropical::times_scalar_left(&scalar, &w1);
        let right = PowerTropical::times_scalar_right(&w1, &scalar);

        assert_eq!(left.to_string(), "12,14,16");
        assert_eq!(right.to_string(), "12,14,16");
    }

    #[test]
    fn test_power_weight_approx_equal() {
        let w1: PowerTropical = "2.001,4,6".parse().unwrap();
        let w2: PowerTropical = "2,4,6".parse().unwrap();

        // The difference is within a tolerance of 1e-2.
        assert!(w1.approx_equal(&w2, 1e-2));
        // And larger than 1e-4, so not equal at that tolerance.
        assert!(!w1.approx_equal(&w2, 1e-4));
    }
}
