use std::fmt;
use std::str::FromStr;

use std::fmt::Debug;

use crate::error::ParseError;
use crate::fst_type::WeightType;
use crate::weight::{
    COMMUTATIVE, Divide, DivideType, IDEMPOTENT, LEFT_SEMIRING, RIGHT_SEMIRING, Weight,
};

pub const K_SET_EMPTY: i64 = 0;
pub const K_SET_UNIV: i64 = -1;
pub const K_SET_BAD: i64 = -2;

/// A marker trait for the mathematical behavior of `SetWeight`.
pub trait SetSemiringType: Clone + Copy + PartialEq + Eq + Debug + 'static {
    /// Which pair of operations this semiring uses, known at compile time.
    const KIND: SetKind;

    fn type_name() -> &'static str;
}

/// Which pair of set operations a [`SetWeight`] uses for `plus` and `times`.
///
/// SICADA-OPT: this replaces dispatching on `S::type_name()`, which compared
/// string contents on every `plus` and `times` for a choice that is fixed at
/// compile time by the type parameter. Matching on an associated constant folds
/// the dispatch away entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetKind {
    /// `plus` intersects, `times` unions.
    IntersectUnion,
    /// `plus` unions, `times` intersects.
    UnionIntersect,
    /// As `IntersectUnion`, but `plus` requires its arguments to be equal.
    IntersectUnionRestrict,
    /// Every non-`Zero` element is equivalent.
    Boolean,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntersectUnion;
impl SetSemiringType for IntersectUnion {
    const KIND: SetKind = SetKind::IntersectUnion;

    #[inline(always)]
    fn type_name() -> &'static str {
        "intersect_union_set"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnionIntersect;
impl SetSemiringType for UnionIntersect {
    const KIND: SetKind = SetKind::UnionIntersect;

    #[inline(always)]
    fn type_name() -> &'static str {
        "union_intersect_set"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntersectUnionRestrict;
impl SetSemiringType for IntersectUnionRestrict {
    const KIND: SetKind = SetKind::IntersectUnionRestrict;

    #[inline(always)]
    fn type_name() -> &'static str {
        "restricted_set_intersect_union"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BooleanSet;
impl SetSemiringType for BooleanSet {
    const KIND: SetKind = SetKind::Boolean;

    #[inline(always)]
    fn type_name() -> &'static str {
        "boolean_set"
    }
}

/// Set semiring of integral labels.
///
/// `L` is the label type (usually `i64`).
/// `S` defines whether `Plus` acts as `Intersect` or `Union` (and `Times` conversely).
#[derive(Debug, Clone)]
pub struct SetWeight<L, S> {
    /// A sorted, unique vector of labels.
    /// `None` represents a special set state (Empty, Univ, or Bad),
    /// encoded by placing `K_SET_EMPTY` or `K_SET_UNIV` inside `[0]`.
    labels: Vec<L>,
    _phantom: std::marker::PhantomData<S>,
}

impl<L: Copy + Ord + Into<i64> + TryFrom<i64>, S: SetSemiringType> SetWeight<L, S> {
    #[inline]
    fn new_special(special_val: i64) -> Self {
        let label = L::try_from(special_val)
            .unwrap_or_else(|_| panic!("Failed to convert special value into Label type"));
        Self {
            labels: vec![label],
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub fn empty_set() -> Self {
        Self::new_special(K_SET_EMPTY)
    }

    #[inline(always)]
    pub fn univ_set() -> Self {
        Self::new_special(K_SET_UNIV)
    }

    #[inline(always)]
    pub fn bad_set() -> Self {
        Self::new_special(K_SET_BAD)
    }

    /// Creates a SetWeight from a sorted, unique vector of positive labels.
    pub fn from_sorted_vec(labels: Vec<L>) -> Self {
        if labels.is_empty() {
            Self::empty_set()
        } else {
            Self {
                labels,
                _phantom: std::marker::PhantomData,
            }
        }
    }

    #[inline(always)]
    fn first_val(&self) -> i64 {
        if self.labels.is_empty() {
            K_SET_EMPTY
        } else {
            self.labels[0].into()
        }
    }

    #[inline(always)]
    pub fn is_empty_set(&self) -> bool {
        self.labels.len() == 1 && self.first_val() == K_SET_EMPTY
    }

    #[inline(always)]
    pub fn is_univ_set(&self) -> bool {
        self.labels.len() == 1 && self.first_val() == K_SET_UNIV
    }

    #[inline(always)]
    pub fn is_bad_set(&self) -> bool {
        self.labels.len() == 1 && self.first_val() == K_SET_BAD
    }

    #[inline(always)]
    pub fn iter(&self) -> std::slice::Iter<'_, L> {
        self.labels.iter()
    }
}

// Implement union logic
fn union_sets<L, S: SetSemiringType>(w1: &SetWeight<L, S>, w2: &SetWeight<L, S>) -> SetWeight<L, S>
where
    L: Copy + Ord + Into<i64> + TryFrom<i64> + fmt::Display + fmt::Debug + std::str::FromStr,
    <L as std::str::FromStr>::Err: fmt::Display,
{
    if !w1.is_member() || !w2.is_member() {
        return SetWeight::bad_set();
    }
    if w1.is_empty_set() {
        return w2.clone();
    }
    if w2.is_empty_set() {
        return w1.clone();
    }
    if w1.is_univ_set() {
        return w1.clone();
    }
    if w2.is_univ_set() {
        return w2.clone();
    }

    let mut result = Vec::with_capacity(w1.labels.len() + w2.labels.len());
    let mut i = 0;
    let mut j = 0;

    while i < w1.labels.len() && j < w2.labels.len() {
        let v1 = w1.labels[i];
        let v2 = w2.labels[j];
        if v1 < v2 {
            result.push(v1);
            i += 1;
        } else if v1 > v2 {
            result.push(v2);
            j += 1;
        } else {
            result.push(v1);
            i += 1;
            j += 1;
        }
    }
    result.extend_from_slice(&w1.labels[i..]);
    result.extend_from_slice(&w2.labels[j..]);

    SetWeight::from_sorted_vec(result)
}

// Implement intersect logic
fn intersect_sets<L, S: SetSemiringType>(
    w1: &SetWeight<L, S>,
    w2: &SetWeight<L, S>,
) -> SetWeight<L, S>
where
    L: Copy + Ord + Into<i64> + TryFrom<i64> + fmt::Display + fmt::Debug + std::str::FromStr,
    <L as std::str::FromStr>::Err: fmt::Display,
{
    if !w1.is_member() || !w2.is_member() {
        return SetWeight::bad_set();
    }
    if w1.is_empty_set() {
        return w1.clone();
    }
    if w2.is_empty_set() {
        return w2.clone();
    }
    if w1.is_univ_set() {
        return w2.clone();
    }
    if w2.is_univ_set() {
        return w1.clone();
    }

    let mut result = Vec::with_capacity(std::cmp::min(w1.labels.len(), w2.labels.len()));
    let mut i = 0;
    let mut j = 0;

    while i < w1.labels.len() && j < w2.labels.len() {
        let v1 = w1.labels[i];
        let v2 = w2.labels[j];
        if v1 < v2 {
            i += 1;
        } else if v1 > v2 {
            j += 1;
        } else {
            result.push(v1);
            i += 1;
            j += 1;
        }
    }

    if result.is_empty() {
        SetWeight::empty_set()
    } else {
        SetWeight::from_sorted_vec(result)
    }
}

// Implement difference logic
fn difference_sets<L, S: SetSemiringType>(
    w1: &SetWeight<L, S>,
    w2: &SetWeight<L, S>,
) -> SetWeight<L, S>
where
    L: Copy + Ord + Into<i64> + TryFrom<i64> + fmt::Display + fmt::Debug + std::str::FromStr,
    <L as std::str::FromStr>::Err: fmt::Display,
{
    if !w1.is_member() || !w2.is_member() {
        return SetWeight::bad_set();
    }
    if w1.is_empty_set() {
        return w1.clone();
    }
    if w2.is_empty_set() {
        return w1.clone();
    }
    if w2.is_univ_set() {
        return SetWeight::empty_set();
    }

    let mut result = Vec::with_capacity(w1.labels.len());
    let mut i = 0;
    let mut j = 0;

    while i < w1.labels.len() && j < w2.labels.len() {
        let v1 = w1.labels[i];
        let v2 = w2.labels[j];
        if v1 < v2 {
            result.push(v1);
            i += 1;
        } else if v1 > v2 {
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    result.extend_from_slice(&w1.labels[i..]);

    if result.is_empty() {
        SetWeight::empty_set()
    } else {
        SetWeight::from_sorted_vec(result)
    }
}

impl<L, S> Weight for SetWeight<L, S>
where
    L: Copy + Ord + Into<i64> + TryFrom<i64> + fmt::Display + fmt::Debug + std::str::FromStr,
    <L as std::str::FromStr>::Err: fmt::Display,
    S: SetSemiringType,
{
    type ReverseWeight = Self;

    #[inline]
    fn zero() -> Self {
        match S::KIND {
            SetKind::UnionIntersect => Self::empty_set(),
            _ => Self::univ_set(),
        }
    }

    #[inline]
    fn one() -> Self {
        match S::KIND {
            SetKind::UnionIntersect => Self::univ_set(),
            _ => Self::empty_set(),
        }
    }

    #[inline]
    fn no_weight() -> Self {
        Self::bad_set()
    }

    #[inline]
    fn type_name() -> WeightType {
        WeightType::new(S::type_name())
    }

    #[inline(always)]
    fn properties() -> u64 {
        IDEMPOTENT | LEFT_SEMIRING | RIGHT_SEMIRING | COMMUTATIVE
    }

    #[inline(always)]
    fn is_member(&self) -> bool {
        !self.is_bad_set()
    }

    #[inline(always)]
    fn approx_equal(&self, other: &Self, _delta: f32) -> bool {
        self == other
    }

    #[inline(always)]
    fn quantize(&self, _delta: f32) -> Self {
        self.clone()
    }

    #[inline(always)]
    fn reverse(&self) -> Self::ReverseWeight {
        self.clone()
    }

    #[inline]
    fn plus(&self, rhs: &Self) -> Self {
        match S::KIND {
            SetKind::UnionIntersect => union_sets(self, rhs),
            SetKind::IntersectUnion => intersect_sets(self, rhs),
            SetKind::IntersectUnionRestrict => {
                // Plus is partial here: it is only defined when the arguments
                // agree, which is how determinization checks that its input has a
                // unique labelled path weight.
                if !self.is_member() || !rhs.is_member() {
                    return Self::no_weight();
                }
                if *self == Self::zero() {
                    return rhs.clone();
                }
                if *rhs == Self::zero() {
                    return self.clone();
                }
                if self != rhs {
                    Self::no_weight()
                } else {
                    self.clone()
                }
            }
            SetKind::Boolean => {
                // Or, over the quotient this set type compares on.
                if !self.is_member() || !rhs.is_member() {
                    return Self::no_weight();
                }
                if *self == Self::one() {
                    return self.clone();
                }
                if *rhs == Self::one() {
                    return rhs.clone();
                }
                Self::zero()
            }
        }
    }

    #[inline]
    fn times(&self, rhs: &Self) -> Self {
        match S::KIND {
            SetKind::UnionIntersect => intersect_sets(self, rhs),
            SetKind::Boolean => {
                // And, on the same reading.
                if !self.is_member() || !rhs.is_member() {
                    return Self::no_weight();
                }
                if *self == Self::one() {
                    return rhs.clone();
                }
                self.clone()
            }
            SetKind::IntersectUnion | SetKind::IntersectUnionRestrict => union_sets(self, rhs),
        }
    }
}

/// Equality, which for `BooleanSet` is not the equality of the sets.
///
/// `SET_BOOLEAN` reads every value other than `Zero` as the same truth value,
/// so its equality is that of the quotient rather than of the labels: `{1}` and
/// `One` are the same weight. Upstream says so with its own `operator==` for
/// that set type, and its `Plus` and `Times` are written on top of it, testing
/// `w == One()` where they mean "is true". Comparing labels here instead would
/// make those two operations disagree with themselves.
///
/// SICADA-DIVERGE: upstream's boolean `operator==` answers false when either
/// side is not a member, which leaves `NoWeight` unequal to itself. Keeping
/// that would cost `Eq`, and nothing in either library depends on it, so a bad
/// set equals a bad set here.
impl<L: Copy + Ord + Into<i64> + TryFrom<i64>, S: SetSemiringType> PartialEq for SetWeight<L, S> {
    fn eq(&self, other: &Self) -> bool {
        if S::KIND == SetKind::Boolean {
            return self.is_univ_set() == other.is_univ_set()
                && self.is_bad_set() == other.is_bad_set();
        }
        self.labels == other.labels
    }
}

impl<L: Copy + Ord + Into<i64> + TryFrom<i64>, S: SetSemiringType> Eq for SetWeight<L, S> {}

impl<L, S> Divide for SetWeight<L, S>
where
    L: Copy + Ord + Into<i64> + TryFrom<i64> + fmt::Display + fmt::Debug + std::str::FromStr,
    <L as std::str::FromStr>::Err: fmt::Display,
    S: SetSemiringType,
{
    #[inline]
    fn divide(&self, rhs: &Self, _typ: DivideType) -> Self {
        match S::KIND {
            SetKind::UnionIntersect => {
                if !self.is_member() || !rhs.is_member() {
                    return Self::no_weight();
                }
                if self == rhs {
                    Self::univ_set()
                } else {
                    self.clone()
                }
            }
            SetKind::Boolean => {
                // Recovers a `b` with `rhs * b == self`, on the same reading.
                if !self.is_member() || !rhs.is_member() {
                    return Self::no_weight();
                }
                if *self == Self::one() || *rhs == Self::zero() {
                    Self::one()
                } else {
                    Self::zero()
                }
            }
            SetKind::IntersectUnion | SetKind::IntersectUnionRestrict => difference_sets(self, rhs),
        }
    }
}

impl<L: Copy + Ord + Into<i64> + TryFrom<i64> + fmt::Display, S: SetSemiringType> fmt::Display
    for SetWeight<L, S>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty_set() {
            write!(f, "EmptySet")
        } else if self.is_univ_set() {
            write!(f, "UnivSet")
        } else if self.is_bad_set() {
            write!(f, "BadSet")
        } else {
            for (i, label) in self.labels.iter().enumerate() {
                if i > 0 {
                    write!(f, "_")?;
                }
                write!(f, "{}", label)?;
            }
            Ok(())
        }
    }
}

impl<L: Copy + Ord + Into<i64> + TryFrom<i64> + fmt::Display + FromStr, S: SetSemiringType> FromStr
    for SetWeight<L, S>
where
    <L as FromStr>::Err: std::fmt::Display,
{
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s == "EmptySet" {
            Ok(Self::empty_set())
        } else if s == "UnivSet" {
            Ok(Self::univ_set())
        } else if s == "BadSet" {
            Ok(Self::bad_set())
        } else {
            let parts: Result<Vec<L>, _> = s
                .split('_')
                .map(|p| {
                    p.parse::<L>().map_err(|e| {
                        ParseError::InvalidFormat(format!("Failed to parse label: {}", e))
                    })
                })
                .collect();

            let mut labels = parts?;
            labels.sort_unstable();
            labels.dedup();

            Ok(Self::from_sorted_vec(labels))
        }
    }
}

// Boilerplate type aliases
pub type IntersectUnionSetWeight<L> = SetWeight<L, IntersectUnion>;
pub type UnionIntersectSetWeight<L> = SetWeight<L, UnionIntersect>;
pub type BooleanSetWeight<L> = SetWeight<L, BooleanSet>;
pub type RestrictedIntersectUnionSetWeight<L> = SetWeight<L, IntersectUnionRestrict>;

#[cfg(test)]
mod tests {
    use crate::weight::axioms;

    /// All four set semirings must satisfy the axioms they claim. The
    /// combinations differ in which operation is Plus and which is Times, so a
    /// mistake in one is not a mistake in the others.
    #[test]
    fn every_set_semiring_satisfies_its_claims() {
        fn samples<S: SetSemiringType>() -> Vec<SetWeight<i64, S>> {
            vec![
                SetWeight::from_sorted_vec(vec![1]),
                SetWeight::from_sorted_vec(vec![1, 2]),
                SetWeight::from_sorted_vec(vec![2, 3]),
            ]
        }
        axioms::check(&samples::<IntersectUnion>());
        axioms::check(&samples::<UnionIntersect>());
        // The other two are absent on purpose; see the tests below.
    }

    /// `BooleanSet` is a semiring, including on the values that spell true
    /// some other way.
    ///
    /// Its documented reading is that "all non-Zero elements are equivalent",
    /// so the carrier is `{false, true}` and a weight like `{1}` is another
    /// spelling of true. Equality is the quotient rather than the sets, which
    /// is what makes the laws hold for those spellings too, and is what
    /// upstream's own `operator==` for this set type does.
    #[test]
    fn the_boolean_semiring_holds_on_every_spelling_of_true() {
        type Set = SetWeight<i64, BooleanSet>;

        // Zero is the universal set and One the empty set, as upstream has them.
        assert!(Set::zero().is_univ_set());
        assert!(Set::one().is_empty_set());

        // A non-canonical true, which is the same weight as One.
        let truthy = Set::from_sorted_vec(vec![1]);
        assert_eq!(truthy, Set::one());
        assert_ne!(truthy, Set::zero());

        // The axioms over both spellings, not just the canonical pair.
        axioms::check::<Set>(&[truthy.clone(), Set::zero(), Set::one()]);
        axioms::check_divide::<Set>(&[truthy.clone(), Set::zero(), Set::one()]);

        // Or and And, which upstream writes against One and which now mean the
        // same thing here.
        assert_eq!(truthy.plus(&Set::zero()), Set::one());
        assert_eq!(Set::zero().plus(&truthy), Set::one());
        assert_eq!(truthy.times(&Set::zero()), Set::zero());
        assert_eq!(Set::zero().times(&truthy), Set::zero());
        assert_eq!(truthy.times(&Set::one()), Set::one());
        assert_eq!(
            truthy.times(&Set::zero()),
            Set::zero().times(&truthy),
            "times must commute"
        );
    }

    /// `IntersectUnionRestrict` is not a semiring, despite claiming to be one.
    ///
    /// Its `plus` is partial: defined only when the arguments agree, and
    /// signalling an error otherwise. That is deliberate, since determinization
    /// uses it to check that its input has a unique labelled path weight, but it
    /// means distributivity fails wherever the restriction fires. Upstream still
    /// declares `kLeftSemiring | kRightSemiring` for it.
    #[test]
    fn the_restricted_semiring_is_partial_and_says_so_only_by_erroring() {
        type Set = SetWeight<i64, IntersectUnionRestrict>;
        let one = Set::from_sorted_vec(vec![1]);
        let two = Set::from_sorted_vec(vec![2]);

        // Equal arguments: plus is the identity.
        assert_eq!(one.plus(&one), one);
        // Zero is still absorbed on either side.
        assert_eq!(Set::zero().plus(&one), one);
        assert_eq!(one.plus(&Set::zero()), one);
        // Unequal arguments are rejected.
        assert!(!one.plus(&two).is_member());

        // And this is what breaks distributivity, which the axiom harness finds:
        // a * (b + c) is a bad set while a*b + a*c is not.
        let a = Set::zero();
        let b = Set::one();
        let c = one.clone();
        assert!(!a.times(&b.plus(&c)).is_member());
        assert!(a.times(&b).plus(&a.times(&c)).is_member());

        // There is no sample set the harness can run on: it always includes Zero
        // and One, and plus(One, anything else) already trips the restriction.
        // The laws hold only where every summed pair is equal, which is the
        // precondition determinization is checking for in the first place.
    }

    /// A set is stored sorted and duplicate-free, so two spellings of the same
    /// set have to compare equal.
    #[test]
    fn a_set_is_normalized_however_it_is_built() {
        type Set = SetWeight<i64, IntersectUnion>;
        let ascending = Set::from_sorted_vec(vec![1, 2, 3]);
        let parsed = "3_1_2_1_3".parse::<Set>().unwrap();
        assert_eq!(ascending, parsed, "parsing must sort and deduplicate");
        assert_eq!(parsed.to_string(), "1_2_3");
    }

    use super::*;

    #[test]
    fn test_set_weight_parse_and_display() {
        let w1 = "1_2_3".parse::<IntersectUnionSetWeight<i64>>().unwrap();
        assert_eq!(w1.labels, vec![1, 2, 3]);
        assert_eq!(w1.to_string(), "1_2_3");

        let empty = "EmptySet".parse::<IntersectUnionSetWeight<i64>>().unwrap();
        assert!(empty.is_empty_set());
        assert_eq!(empty.to_string(), "EmptySet");
    }

    #[test]
    fn test_intersect_union() {
        // For IntersectUnion, Plus = Intersect, Times = Union
        let w1 = "1_2".parse::<IntersectUnionSetWeight<i64>>().unwrap();
        let w2 = "2_3".parse::<IntersectUnionSetWeight<i64>>().unwrap();

        let plus_res = IntersectUnionSetWeight::plus(&w1, &w2); // Intersect -> 2
        assert_eq!(plus_res.to_string(), "2");

        let times_res = IntersectUnionSetWeight::times(&w1, &w2); // Union -> 1_2_3
        assert_eq!(times_res.to_string(), "1_2_3");
    }

    #[test]
    fn test_union_intersect() {
        // For UnionIntersect, Plus = Union, Times = Intersect
        let w1 = "1_2".parse::<UnionIntersectSetWeight<i64>>().unwrap();
        let w2 = "2_3".parse::<UnionIntersectSetWeight<i64>>().unwrap();

        let plus_res = UnionIntersectSetWeight::plus(&w1, &w2); // Union -> 1_2_3
        assert_eq!(plus_res.to_string(), "1_2_3");

        let times_res = UnionIntersectSetWeight::times(&w1, &w2); // Intersect -> 2
        assert_eq!(times_res.to_string(), "2");
    }
}
