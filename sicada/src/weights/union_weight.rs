use std::cmp::Ordering;
use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use crate::error::ParseError;
use crate::utils::split_composite_weight;
use crate::weight::{
    COMMUTATIVE, Divide, DivideType, IDEMPOTENT, LEFT_SEMIRING, RIGHT_SEMIRING, Weight,
};

/// Options for configuring the behavior of a `UnionWeight`.
pub trait UnionWeightOptions<W: Weight>: Clone + PartialEq + Eq + fmt::Debug + 'static {
    type ReverseOptions: UnionWeightOptions<W::ReverseWeight>;

    /// Comparison function. Returns `true` if `w1` is strictly "better" (sorted earlier) than `w2`.
    fn compare(w1: &W, w2: &W) -> bool;

    /// Merges two weights that are considered equivalent.
    fn merge(w1: W, w2: W) -> W;

    /// Normalizer functor. Invoked after `Plus` or `Times` operations.
    fn normalize(weights: &mut Vec<W>);
}

#[derive(Clone, PartialEq, Eq)]
pub struct UnionWeight<W, O> {
    elements: Option<Vec<W>>,
    _phantom: std::marker::PhantomData<O>,
}

impl<W: Weight, O: UnionWeightOptions<W>> UnionWeight<W, O> {
    #[inline(always)]
    fn new_bad() -> Self {
        Self {
            elements: None,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub fn empty_set() -> Self {
        Self {
            elements: Some(Vec::new()),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn new(weight: W) -> Self {
        if !weight.is_member() {
            Self::new_bad()
        } else {
            Self {
                elements: Some(vec![weight]),
                _phantom: std::marker::PhantomData,
            }
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.elements = Some(Vec::new());
    }

    #[inline(always)]
    pub fn size(&self) -> usize {
        self.elements.as_ref().map_or(0, |e| e.len())
    }

    #[inline(always)]
    pub fn is_bad_set(&self) -> bool {
        self.elements.is_none()
    }

    #[inline]
    pub fn push_back(&mut self, weight: W, sorted: bool) {
        let elems = match self.elements.as_mut() {
            Some(e) => e,
            None => return,
        };

        if !weight.is_member() {
            self.elements = None;
            return;
        }

        if elems.is_empty() {
            elems.push(weight);
            return;
        }

        if sorted {
            let back = elems.last_mut().unwrap();
            if O::compare(back, &weight) {
                elems.push(weight);
            } else if !O::compare(&weight, back) {
                let old_back = std::mem::replace(back, W::no_weight());
                *back = O::merge(old_back, weight);
            } else {
                self.insert(weight);
            }
        } else {
            self.insert(weight);
        }
    }

    fn insert(&mut self, weight: W) {
        let elems = self.elements.as_mut().unwrap();
        match elems.binary_search_by(|w| {
            if O::compare(w, &weight) {
                Ordering::Less
            } else if O::compare(&weight, w) {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }) {
            Ok(idx) => {
                let old_w = std::mem::replace(&mut elems[idx], W::no_weight());
                elems[idx] = O::merge(old_w, weight);
            }
            Err(idx) => elems.insert(idx, weight),
        }
    }

    pub fn sort(&mut self) {
        if let Some(elems) = self.elements.as_mut() {
            elems.sort_unstable_by(|a, b| {
                if O::compare(a, b) {
                    Ordering::Less
                } else if O::compare(b, a) {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            });
            elems.dedup_by(|a, b| {
                if !O::compare(a, b) && !O::compare(b, a) {
                    let old_b = std::mem::replace(b, W::no_weight());
                    let old_a = std::mem::replace(a, W::no_weight());
                    *b = O::merge(old_b, old_a);
                    true
                } else {
                    false
                }
            });
        }
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, W> {
        self.elements.as_ref().map_or([].iter(), |e| e.iter())
    }
}

impl<W: Weight, O: UnionWeightOptions<W>> Weight for UnionWeight<W, O> {
    type ReverseWeight = UnionWeight<W::ReverseWeight, O::ReverseOptions>;

    #[inline(always)]
    fn zero() -> Self {
        Self::empty_set()
    }

    #[inline(always)]
    fn one() -> Self {
        Self::new(W::one())
    }

    #[inline(always)]
    fn no_weight() -> Self {
        Self::new_bad()
    }

    #[inline(always)]
    fn type_name() -> crate::fst_type::WeightType {
        let s = format!("{}_union", W::type_name());
        crate::fst_type::WeightType::new_dynamic(s)
    }

    #[inline(always)]
    fn properties() -> u64 {
        W::properties() & (LEFT_SEMIRING | RIGHT_SEMIRING | COMMUTATIVE | IDEMPOTENT)
    }

    #[inline]
    fn is_member(&self) -> bool {
        if let Some(elems) = &self.elements {
            if elems.is_empty() {
                return true;
            }
            elems.iter().all(|w| w.is_member())
        } else {
            false
        }
    }

    #[inline]
    fn approx_equal(&self, other: &Self, delta: f32) -> bool {
        if self.size() != other.size() {
            return false;
        }
        if self.is_bad_set() && other.is_bad_set() {
            return true;
        }
        let e1 = self.elements.as_ref().unwrap();
        let e2 = other.elements.as_ref().unwrap();

        e1.iter()
            .zip(e2.iter())
            .all(|(a, b)| W::approx_equal(a, b, delta))
    }

    fn quantize(&self, delta: f32) -> Self {
        if let Some(elems) = &self.elements {
            let mut q = Self::empty_set();
            for w in elems {
                q.push_back(W::quantize(w, delta), true);
            }
            q
        } else {
            Self::new_bad()
        }
    }

    fn reverse(&self) -> Self::ReverseWeight {
        if let Some(elems) = &self.elements {
            let mut rw = UnionWeight::<W::ReverseWeight, O::ReverseOptions>::empty_set();
            for w in elems {
                rw.push_back(W::reverse(w), false);
            }
            rw.sort();
            rw
        } else {
            UnionWeight::<W::ReverseWeight, O::ReverseOptions>::new_bad()
        }
    }

    fn plus(&self, rhs: &Self) -> Self {
        if !self.is_member() || !rhs.is_member() {
            return Self::no_weight();
        }
        if *self == Self::zero() {
            return rhs.clone();
        }
        if *rhs == Self::zero() {
            return self.clone();
        }

        let e1 = self.elements.as_ref().unwrap();
        let e2 = rhs.elements.as_ref().unwrap();

        let mut sum = Self::empty_set();
        let sum_elems = sum.elements.as_mut().unwrap();
        sum_elems.reserve(e1.len() + e2.len());

        let mut i = 0;
        let mut j = 0;

        while i < e1.len() && j < e2.len() {
            let v1 = &e1[i];
            let v2 = &e2[j];

            if O::compare(v1, v2) {
                sum.push_back(v1.clone(), true);
                i += 1;
            } else if O::compare(v2, v1) {
                sum.push_back(v2.clone(), true);
                j += 1;
            } else {
                let merged = O::merge(v1.clone(), v2.clone());
                sum.push_back(merged, true);
                i += 1;
                j += 1;
            }
        }
        for v in &e1[i..] {
            sum.push_back(v.clone(), true);
        }
        for v in &e2[j..] {
            sum.push_back(v.clone(), true);
        }

        O::normalize(sum.elements.as_mut().unwrap());

        sum
    }

    fn times(&self, rhs: &Self) -> Self {
        if !self.is_member() || !rhs.is_member() {
            return Self::no_weight();
        }
        if *self == Self::zero() || *rhs == Self::zero() {
            return Self::zero();
        }

        let e1 = self.elements.as_ref().unwrap();
        let e2 = rhs.elements.as_ref().unwrap();

        let mut prod1 = Self::zero();

        for v1 in e1 {
            let mut prod2 = Self::zero();
            for v2 in e2 {
                prod2.push_back(W::times(v1, v2), true);
            }
            prod1 = Self::plus(&prod1, &prod2);
        }

        prod1
    }
}

impl<W: Weight + Divide, O: UnionWeightOptions<W>> Divide for UnionWeight<W, O> {
    fn divide(&self, rhs: &Self, typ: DivideType) -> Self {
        if !self.is_member() || !rhs.is_member() {
            return Self::no_weight();
        }
        if *self == Self::zero() || *rhs == Self::zero() {
            return Self::zero();
        }

        let e1 = self.elements.as_ref().unwrap();
        let e2 = rhs.elements.as_ref().unwrap();

        let mut quot = Self::empty_set();

        if e1.len() == 1 {
            for v2 in e2.iter().rev() {
                quot.push_back(W::divide(&e1[0], v2, typ), true);
            }
        } else if e2.len() == 1 {
            for v1 in e1 {
                quot.push_back(W::divide(v1, &e2[0], typ), true);
            }
        } else {
            return Self::no_weight();
        }

        quot
    }
}

impl<W: Weight, O: UnionWeightOptions<W>> fmt::Debug for UnionWeight<W, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_bad_set() {
            write!(f, "BadSet")
        } else if self.size() == 0 {
            write!(f, "EmptySet")
        } else {
            f.debug_list().entries(self.iter()).finish()
        }
    }
}

impl<W: Weight, O: UnionWeightOptions<W>> fmt::Display for UnionWeight<W, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_bad_set() {
            write!(f, "BadSet")
        } else if self.size() == 0 {
            write!(f, "EmptySet")
        } else {
            for (i, v) in self.iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?;
                }
                write!(f, "{}", v)?;
            }
            Ok(())
        }
    }
}

impl<W: Weight, O: UnionWeightOptions<W>> FromStr for UnionWeight<W, O> {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s == "EmptySet" {
            return Ok(Self::empty_set());
        }
        if s == "BadSet" {
            return Ok(Self::new_bad());
        }

        let parts = split_composite_weight(s, ',', '(', ')')?;
        let mut weight = Self::empty_set();

        for p in parts {
            let v = p.parse::<W>().map_err(|_| {
                ParseError::InvalidFormat(format!(
                    "Failed to parse inner weight of UnionWeight: {}",
                    p
                ))
            })?;
            weight.push_back(v, true);
        }

        Ok(weight)
    }
}

impl<W: Weight, O: UnionWeightOptions<W>> Hash for UnionWeight<W, O>
where
    W: Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        if let Some(elems) = &self.elements {
            for w in elems {
                w.hash(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::axioms;
    use crate::weight::{COMMUTATIVE, IDEMPOTENT, LEFT_SEMIRING, RIGHT_SEMIRING};
    use crate::weights::float_weight::TropicalWeight;

    /// `UnionWeight` has no instantiation in sicada yet (upstream's is the
    /// gallic union weight in `string-weight.h`), so the tests supply their own
    /// options. Ordering by cost and merging equivalents with `plus` is the same
    /// shape the gallic options use.
    #[derive(Clone, PartialEq, Eq, Debug)]
    struct ByCost;

    impl UnionWeightOptions<TropicalWeight> for ByCost {
        type ReverseOptions = ByCost;

        fn compare(w1: &TropicalWeight, w2: &TropicalWeight) -> bool {
            w1.value() < w2.value()
        }

        fn merge(w1: TropicalWeight, w2: TropicalWeight) -> TropicalWeight {
            w1.plus(&w2)
        }

        fn normalize(_weights: &mut Vec<TropicalWeight>) {}
    }

    type Union = UnionWeight<TropicalWeight, ByCost>;

    fn union_of(values: &[f32]) -> Union {
        let mut weight = Union::empty_set();
        for &value in values {
            weight.push_back(TropicalWeight(value), false);
        }
        weight.sort();
        weight
    }

    #[test]
    fn it_claims_only_what_the_underlying_weight_supports() {
        // Upstream masks the component's properties down to the four that
        // survive taking unions.
        assert_eq!(
            Union::properties(),
            TropicalWeight::properties()
                & (LEFT_SEMIRING | RIGHT_SEMIRING | COMMUTATIVE | IDEMPOTENT)
        );
    }

    #[test]
    fn it_satisfies_the_axioms_it_claims() {
        axioms::check(&[union_of(&[1.0]), union_of(&[2.0]), union_of(&[1.0, 2.0])]);
    }

    #[test]
    fn zero_is_the_empty_union_and_one_is_the_singleton_identity() {
        assert_eq!(Union::zero().size(), 0);
        assert_eq!(Union::one(), Union::new(TropicalWeight::one()));
    }

    /// Plus merges the two sorted sets, which makes the union a union rather
    /// than a multiset.
    #[test]
    fn plus_merges_the_sets_in_order() {
        let left = union_of(&[1.0, 3.0]);
        let right = union_of(&[2.0, 4.0]);
        let sum = left.plus(&right);
        assert_eq!(sum.size(), 4);
        let values: Vec<f32> = sum.iter().map(|w| w.value()).collect();
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(values, sorted, "the merge must stay ordered");
    }

    /// Times distributes over the union, so it is the pairwise product.
    #[test]
    fn times_takes_the_pairwise_product() {
        let left = union_of(&[1.0, 2.0]);
        let right = union_of(&[10.0, 20.0]);
        let product = left.times(&right);
        // Tropical times is addition: {11, 21, 12, 22}.
        let mut values: Vec<f32> = product.iter().map(|w| w.value()).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(values, vec![11.0, 12.0, 21.0, 22.0]);
    }

    #[test]
    fn a_bad_set_is_not_a_member_and_infects_the_operations() {
        let bad = Union::no_weight();
        assert!(!bad.is_member());
        assert!(bad.is_bad_set());

        let good = union_of(&[1.0]);
        assert!(!bad.plus(&good).is_member());
        assert!(!good.plus(&bad).is_member());
        assert!(!bad.times(&good).is_member());
        assert!(!good.times(&bad).is_member());
    }

    #[test]
    fn a_non_member_component_makes_the_union_bad() {
        assert!(!Union::new(TropicalWeight::no_weight()).is_member());
    }

    #[test]
    fn quantize_and_reverse_reach_every_element() {
        let weight = union_of(&[1.24, 1.76]);
        let quantized = weight.quantize(0.5);
        let values: Vec<f32> = quantized.iter().map(|w| w.value()).collect();
        assert_eq!(values, vec![1.0, 2.0]);

        let reversed = weight.reverse();
        assert_eq!(reversed.size(), weight.size());
    }
}
