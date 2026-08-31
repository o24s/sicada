use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use crate::error::ParseError;
use crate::fst_type::WeightType;
use crate::utils::split_composite_weight;
use crate::weight::{
    COMMUTATIVE, Divide, DivideType, IDEMPOTENT, LEFT_SEMIRING, RIGHT_SEMIRING, Weight,
};
use crate::weights::sparse_tuple_weight::SparseTupleWeight;

/// Sparse cartesian power semiring: W ^ n
///
/// Forms:
///  - a left semimodule when W is a left semiring,
///  - a right semimodule when W is a right semiring,
///  - a bisemimodule when W is a semiring,
///    the free semimodule of rank n over W
///
/// The Times operation is overloaded to provide the left and right scalar products.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparsePowerWeight<W: Weight, K = i64> {
    pub inner: SparseTupleWeight<W, K>,
}

impl<W: Weight, K: Copy + Ord + Hash + FromStr + fmt::Display + fmt::Debug>
    SparsePowerWeight<W, K>
{
    #[inline]
    pub fn new(inner: SparseTupleWeight<W, K>) -> Self {
        Self { inner }
    }

    #[inline]
    pub fn new_with_default(default_value: W) -> Self {
        Self {
            inner: SparseTupleWeight::new(default_value),
        }
    }

    /// Builds a vector holding `weight` at `key` and `default_weight` elsewhere.
    ///
    /// Corresponds to upstream's `SparsePowerWeight(const K &key, const W
    /// &weight, const W &default_weight)`.
    #[inline]
    pub fn from_component(key: K, weight: W, default_weight: W) -> Self {
        let mut inner = SparseTupleWeight::new(default_weight);
        inner.set_value(key, weight);
        Self { inner }
    }

    #[inline]
    pub fn dot_product(&self, other: &Self) -> W {
        let product_inner = self.inner.map(&other.inner, |_, v1, v2| W::times(v1, v2));
        let mut result = W::zero();
        for (_, w) in product_inner.iter() {
            result = W::plus(&result, w);
        }
        result
    }
}

impl<W: Weight, K> Weight for SparsePowerWeight<W, K>
where
    K: Copy + Ord + Hash + FromStr + fmt::Display + fmt::Debug + 'static,
{
    type ReverseWeight = SparsePowerWeight<W::ReverseWeight, K>;

    #[inline(always)]
    fn zero() -> Self {
        Self {
            inner: SparseTupleWeight::zero(),
        }
    }

    #[inline(always)]
    fn one() -> Self {
        Self {
            inner: SparseTupleWeight::one(),
        }
    }

    #[inline(always)]
    fn no_weight() -> Self {
        Self {
            inner: SparseTupleWeight::no_weight(),
        }
    }

    fn type_name() -> WeightType {
        let mut s = format!("{}_^n", W::type_name());
        // The width is appended unless `K` is 4 bytes, which is what upstream's
        // `uint32_t` gives and therefore the name it writes unadorned.
        if std::mem::size_of::<K>() != 4 {
            s.push_str(&format!("_{}", std::mem::size_of::<K>() * 8));
        }
        WeightType::new_dynamic(s)
    }

    #[inline(always)]
    fn properties() -> u64 {
        W::properties() & (LEFT_SEMIRING | RIGHT_SEMIRING | COMMUTATIVE | IDEMPOTENT)
    }

    #[inline(always)]
    fn is_member(&self) -> bool {
        self.inner.is_member()
    }

    #[inline]
    fn approx_equal(&self, other: &Self, delta: f32) -> bool {
        let mapped = self.inner.map(&other.inner, |_, v1, v2| {
            if W::approx_equal(v1, v2, delta) {
                W::one()
            } else {
                W::zero()
            }
        });
        mapped == SparseTupleWeight::one()
    }

    #[inline]
    fn quantize(&self, delta: f32) -> Self {
        Self {
            inner: self.inner.quantize(delta),
        }
    }

    #[inline]
    fn reverse(&self) -> Self::ReverseWeight {
        SparsePowerWeight {
            inner: self.inner.reverse(),
        }
    }

    #[inline]
    fn plus(&self, rhs: &Self) -> Self {
        Self {
            inner: self.inner.map(&rhs.inner, |_, v1, v2| W::plus(v1, v2)),
        }
    }

    #[inline]
    fn times(&self, rhs: &Self) -> Self {
        Self {
            inner: self.inner.map(&rhs.inner, |_, v1, v2| W::times(v1, v2)),
        }
    }
}

impl<W: Weight + Divide, K> Divide for SparsePowerWeight<W, K>
where
    K: Copy + Ord + Hash + FromStr + fmt::Display + fmt::Debug + 'static,
{
    #[inline]
    fn divide(&self, rhs: &Self, typ: DivideType) -> Self {
        Self {
            inner: self
                .inner
                .map(&rhs.inner, |_, v1, v2| W::divide(v1, v2, typ)),
        }
    }
}

impl<W: Weight, K> fmt::Display for SparsePowerWeight<W, K>
where
    SparseTupleWeight<W, K>: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
    }
}

impl<W: Weight, K: Copy + Ord + Hash + FromStr + fmt::Display> FromStr for SparsePowerWeight<W, K> {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts = split_composite_weight(s, ',', '(', ')')?;
        if parts.is_empty() {
            return Err(ParseError::InvalidElementCount {
                expected: 1,
                found: 0,
            });
        }

        let def = parts[0].parse::<W>().map_err(|_| {
            ParseError::InvalidFormat(format!("Failed to parse default weight: {}", parts[0]))
        })?;
        let mut weight = SparseTupleWeight::new(def);

        let mut i = 1;
        while i + 1 < parts.len() {
            let key = parts[i].parse::<K>().map_err(|_| {
                ParseError::InvalidFormat(format!("Failed to parse key: {}", parts[i]))
            })?;
            let val = parts[i + 1].parse::<W>().map_err(|_| {
                ParseError::InvalidFormat(format!("Failed to parse weight: {}", parts[i + 1]))
            })?;
            weight.push_back(key, val);
            i += 2;
        }

        if i < parts.len() {
            return Err(ParseError::InvalidElementCount {
                expected: parts.len() + 1,
                found: parts.len(),
            });
        }

        Ok(Self { inner: weight })
    }
}

impl<W: Weight, K> Hash for SparsePowerWeight<W, K>
where
    SparseTupleWeight<W, K>: Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::axioms;
    use crate::weights::float_weight::TropicalWeight;

    type Sparse = SparsePowerWeight<TropicalWeight, i64>;

    fn sparse_of(entries: &[(i64, f32)], default: f32) -> Sparse {
        let mut inner = SparseTupleWeight::new(TropicalWeight(default));
        for &(key, value) in entries {
            inner.set_value(key, TropicalWeight(value));
        }
        Sparse::new(inner)
    }

    /// The sparse free semimodule: componentwise, including over the infinitely
    /// many keys represented by the default.
    #[test]
    fn it_satisfies_the_axioms_it_claims() {
        axioms::check(&[
            sparse_of(&[(1, 1.0), (3, 2.0)], 0.0),
            sparse_of(&[(2, 3.0)], 0.0),
        ]);
        axioms::check_divide(&[sparse_of(&[(1, 1.0), (3, 2.0)], 0.0)]);
    }

    #[test]
    fn it_claims_only_what_the_component_supports() {
        assert_eq!(
            Sparse::properties(),
            TropicalWeight::properties() & Sparse::properties()
        );
    }

    /// Zero and One fill every key, so they are represented entirely by the
    /// default with no explicit entries.
    #[test]
    fn the_identities_are_pure_defaults() {
        assert_eq!(Sparse::zero().inner.size(), 0);
        assert_eq!(Sparse::one().inner.size(), 0);
        assert_eq!(
            Sparse::zero().inner.default_value(),
            &TropicalWeight::zero()
        );
        assert_eq!(Sparse::one().inner.default_value(), &TropicalWeight::one());
    }

    #[test]
    fn operations_reach_keys_that_only_one_side_stores() {
        let left = sparse_of(&[(1, 1.0)], 0.0);
        let right = sparse_of(&[(2, 2.0)], 0.0);
        let product = left.times(&right);
        // Tropical times is addition, and the default is 0 (One), so each key
        // keeps its own value.
        assert_eq!(product.inner.value(1), &TropicalWeight(1.0));
        assert_eq!(product.inner.value(2), &TropicalWeight(2.0));
        assert_eq!(product.inner.value(3), &TropicalWeight(0.0));
    }

    #[test]
    fn a_non_member_anywhere_spoils_the_weight() {
        assert!(sparse_of(&[(1, 1.0)], 0.0).is_member());
        let mut inner = SparseTupleWeight::new(TropicalWeight::zero());
        inner.set_value(1, TropicalWeight::no_weight());
        assert!(!Sparse::new(inner).is_member());
        assert!(!Sparse::new(SparseTupleWeight::new(TropicalWeight::no_weight())).is_member());
    }
}
