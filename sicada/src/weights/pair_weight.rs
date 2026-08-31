use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use crate::error::ParseError;
use crate::utils::split_composite_weight;
use crate::weight::Weight;

/// Pair weight container for weight classes that contain two weights.
///
/// Note: Like `TupleWeight`, `PairWeight` is a base container and does NOT
/// form a semiring on its own (it lacks `Plus`, `Times`, and `Divide`).
/// It is used as a building block for weights like `ProductWeight` or `LexicographicWeight`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairWeight<W1, W2> {
    pub value1: W1,
    pub value2: W2,
}

impl<W1, W2> PairWeight<W1, W2> {
    #[inline(always)]
    pub fn new(value1: W1, value2: W2) -> Self {
        Self { value1, value2 }
    }

    #[inline(always)]
    pub fn value1(&self) -> &W1 {
        &self.value1
    }

    #[inline(always)]
    pub fn value2(&self) -> &W2 {
        &self.value2
    }

    #[inline(always)]
    pub fn set_value1(&mut self, weight: W1) {
        self.value1 = weight;
    }

    #[inline(always)]
    pub fn set_value2(&mut self, weight: W2) {
        self.value2 = weight;
    }
}

impl<W1: Weight, W2: Weight> PairWeight<W1, W2> {
    #[inline(always)]
    pub fn zero() -> Self {
        Self::new(W1::zero(), W2::zero())
    }

    #[inline(always)]
    pub fn one() -> Self {
        Self::new(W1::one(), W2::one())
    }

    #[inline(always)]
    pub fn no_weight() -> Self {
        Self::new(W1::no_weight(), W2::no_weight())
    }

    #[inline(always)]
    pub fn is_member(&self) -> bool {
        self.value1.is_member() && self.value2.is_member()
    }

    #[inline(always)]
    pub fn approx_equal(w1: &Self, w2: &Self, delta: f32) -> bool {
        W1::approx_equal(&w1.value1, &w2.value1, delta)
            && W2::approx_equal(&w1.value2, &w2.value2, delta)
    }

    pub fn quantize(&self, delta: f32) -> Self {
        Self::new(
            W1::quantize(&self.value1, delta),
            W2::quantize(&self.value2, delta),
        )
    }

    #[inline(always)]
    pub fn reverse(&self) -> PairWeight<W1::ReverseWeight, W2::ReverseWeight> {
        PairWeight::new(W1::reverse(&self.value1), W2::reverse(&self.value2))
    }
}

impl<W1: Hash, W2: Hash> Hash for PairWeight<W1, W2> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.value1.hash(state);
        self.value2.hash(state);
    }
}

impl<W1: fmt::Display, W2: fmt::Display> fmt::Display for PairWeight<W1, W2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Omitting outer parentheses to match OpenFst default behavior
        // unless they are explicitly requested by higher-level configuration.
        write!(f, "{},{}", self.value1, self.value2)
    }
}

impl<W1, W2> FromStr for PairWeight<W1, W2>
where
    W1: FromStr,
    W2: FromStr,
    <W1 as FromStr>::Err: Into<ParseError>,
    <W2 as FromStr>::Err: Into<ParseError>,
{
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Use the robust split_composite_weight to safely ignore commas inside nested parentheses.
        let parts = split_composite_weight(s, ',', '(', ')')?;

        if parts.len() != 2 {
            return Err(ParseError::InvalidElementCount {
                expected: 2,
                found: parts.len(),
            });
        }

        let w1 = parts[0].parse::<W1>().map_err(Into::into)?;
        let w2 = parts[1].parse::<W2>().map_err(Into::into)?;

        Ok(Self::new(w1, w2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float_weight::{LogWeight, TropicalWeight};

    use crate::weight::Weight;

    type Pair = PairWeight<TropicalWeight, LogWeight>;

    /// `PairWeight` is storage, not a semiring: upstream gives it no Plus or
    /// Times, leaving those to the types built on it (product, lexicographic,
    /// expectation). What it does provide is elementwise.
    #[test]
    fn the_identities_and_predicates_are_elementwise() {
        assert_eq!(
            Pair::zero(),
            Pair::new(TropicalWeight::zero(), LogWeight::zero())
        );
        assert_eq!(
            Pair::one(),
            Pair::new(TropicalWeight::one(), LogWeight::one())
        );

        assert!(Pair::zero().is_member());
        assert!(!Pair::no_weight().is_member());
        // One bad component is enough to spoil the pair.
        assert!(!Pair::new(TropicalWeight::no_weight(), LogWeight::one()).is_member());
        assert!(!Pair::new(TropicalWeight::one(), LogWeight::no_weight()).is_member());
    }

    #[test]
    fn quantize_and_approx_equal_are_elementwise() {
        let pair = Pair::new(TropicalWeight(1.24), LogWeight(2.76));
        assert_eq!(
            pair.quantize(0.5),
            Pair::new(TropicalWeight(1.0), LogWeight(3.0))
        );

        let close = Pair::new(TropicalWeight(1.0), LogWeight(2.0));
        let nudged = Pair::new(TropicalWeight(1.05), LogWeight(2.05));
        assert!(PairWeight::approx_equal(&close, &nudged, 0.1));
        assert!(!PairWeight::approx_equal(&close, &nudged, 0.01));
    }

    #[test]
    fn reverse_reverses_each_component() {
        let pair = Pair::new(TropicalWeight(1.5), LogWeight(2.5));
        let reversed = pair.reverse();
        assert_eq!(reversed.value1(), &TropicalWeight(1.5).reverse());
        assert_eq!(reversed.value2(), &LogWeight(2.5).reverse());
    }

    #[test]
    fn the_text_form_round_trips_including_nesting() {
        let pair = Pair::new(TropicalWeight(1.5), LogWeight(2.5));
        assert_eq!(pair.to_string(), "1.5,2.5");
        assert_eq!(pair.to_string().parse::<Pair>().unwrap(), pair);

        // A nested pair keeps its own separator inside parentheses, so the outer
        // split must not be fooled by it.
        type Nested = PairWeight<Pair, TropicalWeight>;
        let nested = Nested::new(pair, TropicalWeight(3.0));
        assert_eq!(
            "(1.5,2.5),3".parse::<Nested>().unwrap().value1(),
            nested.value1()
        );
    }

    #[test]
    fn a_pair_with_the_wrong_number_of_elements_is_rejected() {
        assert!("1.0".parse::<Pair>().is_err());
        assert!("1.0,2.0,3.0".parse::<Pair>().is_err());
        assert!("nonsense,2.0".parse::<Pair>().is_err());
    }

    /// Equal pairs must hash alike, since pairs end up as hash-map keys in the
    /// state tables composition builds.
    #[test]
    fn equal_pairs_hash_alike() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hash_of = |pair: &Pair| {
            let mut hasher = DefaultHasher::new();
            pair.hash(&mut hasher);
            hasher.finish()
        };
        let left = Pair::new(TropicalWeight(1.5), LogWeight(2.5));
        let right = Pair::new(TropicalWeight(1.5), LogWeight(2.5));
        assert_eq!(left, right);
        assert_eq!(hash_of(&left), hash_of(&right));
    }

    #[test]
    fn test_pair_weight_parse() {
        let text = "1.5,2.5";
        let pw = text
            .parse::<PairWeight<TropicalWeight, TropicalWeight>>()
            .unwrap();
        assert_eq!(pw.value1.value(), 1.5);
        assert_eq!(pw.value2.value(), 2.5);

        let nested_text = "(1.0,2.0),3.0";
        let nested_pw = nested_text
            .parse::<PairWeight<PairWeight<TropicalWeight, TropicalWeight>, TropicalWeight>>()
            .unwrap();

        assert_eq!(nested_pw.value1.value1.value(), 1.0);
        assert_eq!(nested_pw.value1.value2.value(), 2.0);
        assert_eq!(nested_pw.value2.value(), 3.0);
    }
}
