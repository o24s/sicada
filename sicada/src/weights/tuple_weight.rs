use std::array;
use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use crate::error::ParseError;
use crate::utils::split_composite_weight;
use crate::weight::Weight;

/// An n-tuple weight container, an element of the n-th Cartesian power of W.
///
/// Note: In OpenFst, `TupleWeight` is just a base container class and does NOT
/// form a semiring on its own (it lacks `Plus`, `Times`, and `Divide`).
///
/// Because it does not implement the `Weight` trait, the Rust compiler will
/// strictly prevent you from using it directly in `Arc` or FST algorithms.
/// Use `PowerWeight` if you need Cartesian power semiring operations.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TupleWeight<W, const N: usize>(pub [W; N]);

impl<W, const N: usize> TupleWeight<W, N> {
    #[inline(always)]
    pub fn value(&self, index: usize) -> &W {
        &self.0[index]
    }

    #[inline(always)]
    pub fn set_value(&mut self, index: usize, w: W) {
        self.0[index] = w;
    }
}

impl<W: Clone, const N: usize> TupleWeight<W, N> {
    /// Fills the tuple with the specified weight.
    #[inline(always)]
    pub fn new(weight: W) -> Self {
        Self(array::from_fn(|_| weight.clone()))
    }

    /// Initializes component `index` to `weight` and all other components to `default_weight`.
    #[inline(always)]
    pub fn new_with_default(index: usize, weight: W, default_weight: W) -> Self {
        let mut arr = array::from_fn(|_| default_weight.clone());
        arr[index] = weight;
        Self(arr)
    }
}

impl<W: Weight, const N: usize> TupleWeight<W, N> {
    #[inline(always)]
    pub fn zero() -> Self {
        Self(array::from_fn(|_| W::zero()))
    }

    #[inline(always)]
    pub fn one() -> Self {
        Self(array::from_fn(|_| W::one()))
    }

    #[inline(always)]
    pub fn no_weight() -> Self {
        Self(array::from_fn(|_| W::no_weight()))
    }

    #[inline(always)]
    pub fn is_member(&self) -> bool {
        self.0.iter().all(|w| w.is_member())
    }

    #[inline(always)]
    pub fn approx_equal(w1: &Self, w2: &Self, delta: f32) -> bool {
        w1.0.iter()
            .zip(w2.0.iter())
            .all(|(a, b)| W::approx_equal(a, b, delta))
    }

    pub fn quantize(&self, delta: f32) -> Self {
        Self(array::from_fn(|i| W::quantize(&self.0[i], delta)))
    }

    #[inline(always)]
    pub fn reverse(&self) -> TupleWeight<W::ReverseWeight, N> {
        TupleWeight(array::from_fn(|i| W::reverse(&self.0[i])))
    }
}

impl<W: Hash, const N: usize> Hash for TupleWeight<W, N> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hashing the entire array is exactly equivalent to hashing each element.
        self.0.hash(state);
    }
}

impl<W: fmt::Display, const N: usize> fmt::Display for TupleWeight<W, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Output format matching OpenFst default behavior (no outer parens, separated by commas).
        for (i, w) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", w)?;
        }
        Ok(())
    }
}

impl<W, const N: usize> FromStr for TupleWeight<W, N>
where
    W: FromStr,
    <W as FromStr>::Err: Into<ParseError>,
{
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Uses the robust composite parser to properly handle nested groupings.
        let parts = split_composite_weight(s, ',', '(', ')')?;

        if parts.len() != N {
            return Err(ParseError::InvalidElementCount {
                expected: N,
                found: parts.len(),
            });
        }

        // We use a Vec to safely accumulate parsed items, then convert it to an array
        // without requiring W to implement Default or Copy.
        let mut vec = Vec::with_capacity(N);
        for p in parts {
            let w = p.parse::<W>().map_err(Into::into)?;
            vec.push(w);
        }

        // The conversion from Vec to [W; N] cannot fail because we verified parts.len() == N.
        let arr = vec
            .try_into()
            .map_err(|_| ParseError::InvalidElementCount {
                expected: N,
                found: 0,
            })?;

        Ok(Self(arr))
    }
}

#[cfg(test)]
mod tests {
    type Triple = TupleWeight<TropicalWeight, 3>;

    /// Like `PairWeight`, this is storage rather than a semiring: upstream gives
    /// it no Plus or Times, leaving those to `PowerWeight`. Everything it does
    /// provide applies componentwise.
    #[test]
    fn the_identities_fill_every_component() {
        assert_eq!(Triple::zero(), Triple::new(TropicalWeight::zero()));
        assert_eq!(Triple::one(), Triple::new(TropicalWeight::one()));
        for index in 0..3 {
            assert_eq!(Triple::zero().value(index), &TropicalWeight::zero());
        }
    }

    #[test]
    fn membership_needs_every_component() {
        assert!(Triple::one().is_member());
        for index in 0..3 {
            let mut tuple = Triple::one();
            tuple.set_value(index, TropicalWeight::no_weight());
            assert!(
                !tuple.is_member(),
                "component {index} should spoil the tuple"
            );
        }
    }

    #[test]
    fn a_single_component_can_be_set_against_a_default() {
        let tuple = Triple::new_with_default(1, TropicalWeight(2.5), TropicalWeight::one());
        assert_eq!(tuple.value(0), &TropicalWeight::one());
        assert_eq!(tuple.value(1), &TropicalWeight(2.5));
        assert_eq!(tuple.value(2), &TropicalWeight::one());
    }

    #[test]
    fn quantize_and_approx_equal_are_componentwise() {
        let mut tuple = Triple::new(TropicalWeight(1.24));
        tuple.set_value(2, TropicalWeight(1.26));
        let quantized = tuple.quantize(0.5);
        assert_eq!(quantized.value(0), &TropicalWeight(1.0));
        assert_eq!(quantized.value(2), &TropicalWeight(1.5));

        let nudged = Triple::new(TropicalWeight(1.29));
        assert!(TupleWeight::approx_equal(&tuple, &nudged, 0.1));
        assert!(!TupleWeight::approx_equal(&tuple, &nudged, 0.01));
    }

    #[test]
    fn reverse_reverses_every_component() {
        let tuple = Triple::new(TropicalWeight(1.5));
        let reversed = tuple.reverse();
        for index in 0..3 {
            assert_eq!(reversed.value(index), &TropicalWeight(1.5).reverse());
        }
    }

    #[test]
    fn a_tuple_of_the_wrong_arity_is_rejected() {
        assert!("1.0,2.0".parse::<Triple>().is_err());
        assert!("1.0,2.0,3.0,4.0".parse::<Triple>().is_err());
        assert!("1.0,nonsense,3.0".parse::<Triple>().is_err());
    }

    #[test]
    fn equal_tuples_hash_alike() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hash_of = |tuple: &Triple| {
            let mut hasher = DefaultHasher::new();
            tuple.hash(&mut hasher);
            hasher.finish()
        };
        assert_eq!(
            hash_of(&Triple::new(TropicalWeight(1.5))),
            hash_of(&Triple::new(TropicalWeight(1.5)))
        );
    }

    use super::*;
    use crate::float_weight::TropicalWeight;

    type Tuple3 = TupleWeight<TropicalWeight, 3>;
    type NestedTuple = TupleWeight<TupleWeight<TropicalWeight, 2>, 2>;

    #[test]
    fn test_tuple_weight_parse() {
        let text = "1,2,3.5";
        let tw = text.parse::<Tuple3>().unwrap();
        assert_eq!(tw.value(0).value(), 1.0);
        assert_eq!(tw.value(1).value(), 2.0);
        assert_eq!(tw.value(2).value(), 3.5);

        assert_eq!(tw.to_string(), text);
    }

    #[test]
    fn test_nested_tuple_weight_parse() {
        // Tests that a TupleWeight containing TupleWeights parses correctly
        // without the inner commas destroying the outer boundaries.
        let text = "(1.0,2.0),(3.0,4.0)";
        let tw = text.parse::<NestedTuple>().unwrap();

        assert_eq!(tw.value(0).value(0).value(), 1.0);
        assert_eq!(tw.value(0).value(1).value(), 2.0);
        assert_eq!(tw.value(1).value(0).value(), 3.0);
        assert_eq!(tw.value(1).value(1).value(), 4.0);

        // Without outer parentheses provided to inner tuples in display by default,
        // string formatting is flat in OpenFst unless overridden.
        assert_eq!(tw.to_string(), "1,2,3,4");
    }

    #[test]
    fn test_tuple_weight_initializers() {
        let def = Tuple3::new_with_default(1, TropicalWeight(9.0), TropicalWeight(0.0));
        assert_eq!(def.value(0).value(), 0.0);
        assert_eq!(def.value(1).value(), 9.0);
        assert_eq!(def.value(2).value(), 0.0);
    }
}
