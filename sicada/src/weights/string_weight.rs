use std::cmp::Ordering;
use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use crate::arc::ArcLabel;
use crate::error::ParseError;
use crate::fst_type::WeightType;
use crate::weight::{
    Divide, DivideType, IDEMPOTENT, IdempotentWeight, LEFT_SEMIRING, LeftSemiring, RIGHT_SEMIRING,
    RightSemiring, Weight,
};
use crate::weights::product_weight::ProductWeight;
use crate::weights::union_weight::{UnionWeight, UnionWeightOptions};

pub trait StringTypeMarker: Clone + PartialEq + Eq + Hash + fmt::Debug + 'static {
    const VALUE: u8;
    type Reverse: StringTypeMarker;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StringLeft;
impl StringTypeMarker for StringLeft {
    const VALUE: u8 = 0;
    type Reverse = StringRight;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StringRight;
impl StringTypeMarker for StringRight {
    const VALUE: u8 = 1;
    type Reverse = StringLeft;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StringRestrict;
impl StringTypeMarker for StringRestrict {
    const VALUE: u8 = 2;
    type Reverse = StringRestrict;
}

pub trait GallicTypeMarker: Clone + PartialEq + Eq + Hash + fmt::Debug + 'static {
    const VALUE: u8;

    /// Prefix this flavour contributes to a gallic arc's type name, which is
    /// what an FST file header records.
    const ARC_PREFIX: &'static str;
    type StringType: StringTypeMarker;
    type Reverse: GallicTypeMarker<StringType = <Self::StringType as StringTypeMarker>::Reverse>;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GallicLeft;
impl GallicTypeMarker for GallicLeft {
    const ARC_PREFIX: &'static str = "left_gallic_";
    const VALUE: u8 = 0;
    type StringType = StringLeft;
    type Reverse = GallicRight;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GallicRight;
impl GallicTypeMarker for GallicRight {
    const ARC_PREFIX: &'static str = "right_gallic_";
    const VALUE: u8 = 1;
    type StringType = StringRight;
    type Reverse = GallicLeft;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GallicRestrict;
impl GallicTypeMarker for GallicRestrict {
    const ARC_PREFIX: &'static str = "restricted_gallic_";
    const VALUE: u8 = 2;
    type StringType = StringRestrict;
    type Reverse = GallicRestrict;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GallicMin;
impl GallicTypeMarker for GallicMin {
    const ARC_PREFIX: &'static str = "min_gallic_";
    const VALUE: u8 = 3;
    type StringType = StringRestrict;
    type Reverse = GallicMin;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StringWeightValue<L> {
    Zero,
    NoWeight,
    Labels(Vec<L>),
}

/// String semiring: (longest_common_prefix/suffix, ., Infinity, Epsilon)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StringWeight<L: ArcLabel, S: StringTypeMarker> {
    pub value: StringWeightValue<L>,
    pub _marker: std::marker::PhantomData<S>,
}

impl<L: ArcLabel, S: StringTypeMarker> StringWeight<L, S> {
    #[inline(always)]
    pub fn new(labels: Vec<L>) -> Self {
        Self {
            value: StringWeightValue::Labels(labels),
            _marker: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub fn size(&self) -> usize {
        match &self.value {
            StringWeightValue::Labels(v) => v.len(),
            _ => 0,
        }
    }
}

impl<L: ArcLabel, S: StringTypeMarker> Weight for StringWeight<L, S> {
    type ReverseWeight = StringWeight<L, S::Reverse>;

    #[inline(always)]
    fn zero() -> Self {
        Self {
            value: StringWeightValue::Zero,
            _marker: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    fn one() -> Self {
        Self {
            value: StringWeightValue::Labels(Vec::new()),
            _marker: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    fn no_weight() -> Self {
        Self {
            value: StringWeightValue::NoWeight,
            _marker: std::marker::PhantomData,
        }
    }

    fn type_name() -> WeightType {
        let name = match S::VALUE {
            0 => "left_string",
            1 => "right_string",
            _ => "restricted_string",
        };
        WeightType::new(name)
    }

    #[inline(always)]
    fn properties() -> u64 {
        let mut p = IDEMPOTENT;
        if S::VALUE == 0 {
            p |= LEFT_SEMIRING;
        } else if S::VALUE == 1 {
            p |= RIGHT_SEMIRING;
        } else {
            p |= LEFT_SEMIRING | RIGHT_SEMIRING;
        }
        p
    }

    #[inline(always)]
    fn is_member(&self) -> bool {
        !matches!(self.value, StringWeightValue::NoWeight)
    }

    #[inline(always)]
    fn approx_equal(&self, other: &Self, _delta: f32) -> bool {
        self == other
    }

    #[inline(always)]
    fn quantize(&self, _delta: f32) -> Self {
        self.clone()
    }

    #[inline]
    fn reverse(&self) -> Self::ReverseWeight {
        match &self.value {
            StringWeightValue::Zero => StringWeight::zero(),
            StringWeightValue::NoWeight => StringWeight::no_weight(),
            StringWeightValue::Labels(v) => {
                let mut rev = v.clone();
                rev.reverse();
                StringWeight {
                    value: StringWeightValue::Labels(rev),
                    _marker: std::marker::PhantomData,
                }
            }
        }
    }

    #[inline]
    fn plus(&self, rhs: &Self) -> Self {
        if !self.is_member() || !rhs.is_member() {
            return Self::no_weight();
        }
        if matches!(self.value, StringWeightValue::Zero) {
            return rhs.clone();
        }
        if matches!(rhs.value, StringWeightValue::Zero) {
            return self.clone();
        }

        let v1 = match &self.value {
            StringWeightValue::Labels(v) => v,
            _ => return Self::no_weight(),
        };
        let v2 = match &rhs.value {
            StringWeightValue::Labels(v) => v,
            _ => return Self::no_weight(),
        };

        match S::VALUE {
            0 => {
                let mut sum = Vec::new();
                for (a, b) in v1.iter().zip(v2.iter()) {
                    if a == b {
                        sum.push(*a);
                    } else {
                        break;
                    }
                }
                Self::new(sum)
            }
            1 => {
                let mut sum = Vec::new();
                for (a, b) in v1.iter().rev().zip(v2.iter().rev()) {
                    if a == b {
                        sum.push(*a);
                    } else {
                        break;
                    }
                }
                sum.reverse();
                Self::new(sum)
            }
            _ => {
                if self == rhs {
                    self.clone()
                } else {
                    Self::no_weight()
                }
            }
        }
    }

    #[inline]
    fn times(&self, rhs: &Self) -> Self {
        if !self.is_member() || !rhs.is_member() {
            return Self::no_weight();
        }
        if matches!(self.value, StringWeightValue::Zero)
            || matches!(rhs.value, StringWeightValue::Zero)
        {
            return Self::zero();
        }

        if let (StringWeightValue::Labels(v1), StringWeightValue::Labels(v2)) =
            (&self.value, &rhs.value)
        {
            let mut prod = Vec::with_capacity(v1.len() + v2.len());
            prod.extend_from_slice(v1);
            prod.extend_from_slice(v2);
            Self::new(prod)
        } else {
            Self::no_weight()
        }
    }
}

impl<L: ArcLabel, S: StringTypeMarker> Divide for StringWeight<L, S> {
    #[inline]
    fn divide(&self, rhs: &Self, typ: DivideType) -> Self {
        if !self.is_member() || !rhs.is_member() {
            return Self::no_weight();
        }
        if matches!(rhs.value, StringWeightValue::Zero) {
            return Self::no_weight();
        }
        if matches!(self.value, StringWeightValue::Zero) {
            return Self::zero();
        }

        let v1 = match &self.value {
            StringWeightValue::Labels(v) => v,
            _ => return Self::no_weight(),
        };
        let v2 = match &rhs.value {
            StringWeightValue::Labels(v) => v,
            _ => return Self::no_weight(),
        };

        if typ == DivideType::Left {
            if S::VALUE == 1 {
                return Self::no_weight();
            }
            if v1.len() >= v2.len() && &v1[0..v2.len()] == v2.as_slice() {
                Self::new(v1[v2.len()..].to_vec())
            } else {
                Self::no_weight()
            }
        } else if typ == DivideType::Right {
            if S::VALUE == 0 {
                return Self::no_weight();
            }
            if v1.len() >= v2.len() && &v1[v1.len() - v2.len()..] == v2.as_slice() {
                Self::new(v1[..v1.len() - v2.len()].to_vec())
            } else {
                Self::no_weight()
            }
        } else {
            Self::no_weight()
        }
    }
}

impl<L: ArcLabel, S: StringTypeMarker> fmt::Display for StringWeight<L, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.value {
            StringWeightValue::Zero => write!(f, "Infinity"),
            StringWeightValue::NoWeight => write!(f, "BadString"),
            StringWeightValue::Labels(v) => {
                if v.is_empty() {
                    write!(f, "Epsilon")
                } else {
                    for (i, label) in v.iter().enumerate() {
                        if i > 0 {
                            write!(f, "_")?;
                        }
                        write!(f, "{}", label)?;
                    }
                    Ok(())
                }
            }
        }
    }
}

impl<L: ArcLabel, S: StringTypeMarker> FromStr for StringWeight<L, S> {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "Infinity" {
            Ok(Self::zero())
        } else if s == "Epsilon" {
            Ok(Self::new(Vec::new()))
        } else if s == "BadString" {
            Ok(Self::no_weight())
        } else {
            let parts: Result<Vec<L>, _> = s.split('_').map(|part| part.parse::<L>()).collect();
            match parts {
                Ok(v) => Ok(Self::new(v)),
                Err(_) => Err(ParseError::InvalidFormat(format!(
                    "Failed to parse labels in string: {}",
                    s
                ))),
            }
        }
    }
}

/// Product of string weight and an arbitrary weight.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GallicWeight<L: ArcLabel, W: Weight, G: GallicTypeMarker> {
    pub product: ProductWeight<StringWeight<L, G::StringType>, W>,
    pub _marker: std::marker::PhantomData<G>,
}

impl<L: ArcLabel, W: Weight, G: GallicTypeMarker> GallicWeight<L, W, G> {
    pub fn new(product: ProductWeight<StringWeight<L, G::StringType>, W>) -> Self {
        Self {
            product,
            _marker: std::marker::PhantomData,
        }
    }

    /// Builds a gallic weight from its two sides.
    ///
    /// Corresponds to upstream's `GallicWeight(SW w1, W w2)`, which sicada was
    /// missing; every caller had to assemble the `ProductWeight` itself.
    pub fn from_parts(labels: StringWeight<L, G::StringType>, weight: W) -> Self {
        Self::new(ProductWeight::new(labels, weight))
    }

    /// The label sequence.
    pub fn labels(&self) -> &StringWeight<L, G::StringType> {
        &self.product.value1
    }

    /// The weight carried alongside the labels.
    pub fn weight(&self) -> &W {
        &self.product.value2
    }
}

impl<L: ArcLabel, W: Weight, G: GallicTypeMarker> Weight for GallicWeight<L, W, G> {
    type ReverseWeight = GallicWeight<L, W::ReverseWeight, G::Reverse>;

    #[inline(always)]
    fn zero() -> Self {
        Self::new(ProductWeight::zero())
    }

    #[inline(always)]
    fn one() -> Self {
        Self::new(ProductWeight::one())
    }

    #[inline(always)]
    fn no_weight() -> Self {
        Self::new(ProductWeight::no_weight())
    }

    fn type_name() -> WeightType {
        let name = match G::VALUE {
            0 => "left_gallic",
            1 => "right_gallic",
            2 => "restricted_gallic",
            3 => "min_gallic",
            _ => "gallic",
        };
        WeightType::new(name)
    }

    #[inline(always)]
    fn properties() -> u64 {
        ProductWeight::<StringWeight<L, G::StringType>, W>::properties()
    }

    #[inline(always)]
    fn is_member(&self) -> bool {
        self.product.is_member()
    }

    #[inline(always)]
    fn approx_equal(&self, other: &Self, delta: f32) -> bool {
        self.product.approx_equal(&other.product, delta)
    }

    #[inline(always)]
    fn quantize(&self, delta: f32) -> Self {
        Self::new(self.product.quantize(delta))
    }

    #[inline]
    fn reverse(&self) -> Self::ReverseWeight {
        GallicWeight::new(self.product.reverse())
    }

    #[inline]
    fn plus(&self, rhs: &Self) -> Self {
        if G::VALUE == 3 {
            // GALLIC_MIN: relies on W's NaturalLess behavior inside W::plus
            let p = W::plus(&self.product.value2, &rhs.product.value2);
            if p == self.product.value2 && self.product.value2 != rhs.product.value2 {
                self.clone()
            } else {
                rhs.clone()
            }
        } else {
            Self::new(self.product.plus(&rhs.product))
        }
    }

    #[inline]
    fn times(&self, rhs: &Self) -> Self {
        Self::new(self.product.times(&rhs.product))
    }
}

impl<L: ArcLabel, W: Weight + Divide, G: GallicTypeMarker> Divide for GallicWeight<L, W, G> {
    #[inline]
    fn divide(&self, rhs: &Self, typ: DivideType) -> Self {
        Self::new(self.product.divide(&rhs.product, typ))
    }
}

impl<L: ArcLabel, W: Weight, G: GallicTypeMarker> fmt::Display for GallicWeight<L, W, G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.product)
    }
}

impl<L: ArcLabel, W: Weight, G: GallicTypeMarker> FromStr for GallicWeight<L, W, G> {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::new(s.parse()?))
    }
}

pub struct GallicUnionWeightOptions<L, W>(std::marker::PhantomData<(L, W)>);

impl<L, W> Clone for GallicUnionWeightOptions<L, W> {
    fn clone(&self) -> Self {
        Self(std::marker::PhantomData)
    }
}
impl<L, W> PartialEq for GallicUnionWeightOptions<L, W> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl<L, W> Eq for GallicUnionWeightOptions<L, W> {}
impl<L, W> fmt::Debug for GallicUnionWeightOptions<L, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GallicUnionWeightOptions")
    }
}

// To break the recursive cycle of `'static` and `ReverseWeight` bounds in Rust,
// we restrict GeneralGallicWeight to weights `W` that are their own reverse
// (like TropicalWeight, LogWeight, RealWeight etc.), which covers 99% of use-cases.
impl<L: ArcLabel, W: Weight<ReverseWeight = W> + 'static>
    UnionWeightOptions<GallicWeight<L, W, GallicRestrict>> for GallicUnionWeightOptions<L, W>
{
    type ReverseOptions = GallicUnionWeightOptions<L, W>;

    fn compare(
        w1: &GallicWeight<L, W, GallicRestrict>,
        w2: &GallicWeight<L, W, GallicRestrict>,
    ) -> bool {
        let s1 = &w1.product.value1;
        let s2 = &w2.product.value1;

        if s1.size() < s2.size() {
            return true;
        }
        if s1.size() > s2.size() {
            return false;
        }

        if let (StringWeightValue::Labels(v1), StringWeightValue::Labels(v2)) =
            (&s1.value, &s2.value)
        {
            for (l1, l2) in v1.iter().zip(v2.iter()) {
                match l1.cmp(l2) {
                    Ordering::Less => return true,
                    Ordering::Greater => return false,
                    Ordering::Equal => continue,
                }
            }
        }
        false
    }

    fn merge(
        w1: GallicWeight<L, W, GallicRestrict>,
        w2: GallicWeight<L, W, GallicRestrict>,
    ) -> GallicWeight<L, W, GallicRestrict> {
        let p_w = W::plus(&w1.product.value2, &w2.product.value2);
        let prod = ProductWeight::new(w1.product.value1, p_w);
        GallicWeight::new(prod)
    }

    fn normalize(_weights: &mut Vec<GallicWeight<L, W, GallicRestrict>>) {
        // No normalization required
    }
}

/// Represents the general `gallic` type, mapping to OpenFst's UnionWeight-based GallicWeight.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GeneralGallicWeight<L: ArcLabel, W: Weight<ReverseWeight = W> + 'static> {
    pub union: UnionWeight<GallicWeight<L, W, GallicRestrict>, GallicUnionWeightOptions<L, W>>,
}

impl<L: ArcLabel, W: Weight<ReverseWeight = W> + 'static> Weight for GeneralGallicWeight<L, W> {
    type ReverseWeight = Self;

    #[inline(always)]
    fn zero() -> Self {
        Self {
            union: UnionWeight::zero(),
        }
    }

    #[inline(always)]
    fn one() -> Self {
        Self {
            union: UnionWeight::one(),
        }
    }

    #[inline(always)]
    fn no_weight() -> Self {
        Self {
            union: UnionWeight::no_weight(),
        }
    }

    fn type_name() -> WeightType {
        WeightType::new("gallic")
    }

    #[inline(always)]
    fn properties() -> u64 {
        UnionWeight::<GallicWeight<L, W, GallicRestrict>, GallicUnionWeightOptions<L, W>>::properties()
    }

    #[inline(always)]
    fn is_member(&self) -> bool {
        self.union.is_member()
    }

    #[inline(always)]
    fn approx_equal(&self, other: &Self, delta: f32) -> bool {
        self.union.approx_equal(&other.union, delta)
    }

    #[inline(always)]
    fn quantize(&self, delta: f32) -> Self {
        Self {
            union: self.union.quantize(delta),
        }
    }

    #[inline]
    fn reverse(&self) -> Self::ReverseWeight {
        GeneralGallicWeight {
            union: self.union.reverse(),
        }
    }

    #[inline]
    fn plus(&self, rhs: &Self) -> Self {
        Self {
            union: self.union.plus(&rhs.union),
        }
    }

    #[inline]
    fn times(&self, rhs: &Self) -> Self {
        Self {
            union: self.union.times(&rhs.union),
        }
    }
}

impl<L: ArcLabel, W: Weight<ReverseWeight = W> + Divide + 'static> Divide
    for GeneralGallicWeight<L, W>
{
    #[inline]
    fn divide(&self, rhs: &Self, typ: DivideType) -> Self {
        Self {
            union: self.union.divide(&rhs.union, typ),
        }
    }
}

impl<L: ArcLabel, W: Weight<ReverseWeight = W> + 'static> fmt::Display
    for GeneralGallicWeight<L, W>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.union)
    }
}

impl<L: ArcLabel, W: Weight<ReverseWeight = W> + 'static> FromStr for GeneralGallicWeight<L, W> {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self { union: s.parse()? })
    }
}

// The semiring properties, as marker traits.
//
// SICADA-DIVERGE: upstream states these only as run-time bits in
// `Properties()`, so an algorithm needing left distributivity can be
// instantiated with a weight that does not have it and finds out at run time,
// if at all. The bits are still reported the same way; these say the same thing
// where a bound can use it.
//
// Which side a string semiring distributes over is which end its multiplication
// keeps: a left string weight takes the longest common *prefix*, so ⊗
// distributes over ⊕ on the left only.
impl<L: ArcLabel> LeftSemiring for StringWeight<L, StringLeft> {}
impl<L: ArcLabel> RightSemiring for StringWeight<L, StringRight> {}
// A restricted string weight is only ever combined with an equal one, so it
// distributes on both sides.
impl<L: ArcLabel> LeftSemiring for StringWeight<L, StringRestrict> {}
impl<L: ArcLabel> RightSemiring for StringWeight<L, StringRestrict> {}
impl<L: ArcLabel, S: StringTypeMarker> IdempotentWeight for StringWeight<L, S> {}

// A gallic weight is a product, so it has whatever both halves have.
impl<L, W, G> LeftSemiring for GallicWeight<L, W, G>
where
    L: ArcLabel,
    W: Weight + LeftSemiring,
    G: GallicTypeMarker,
    StringWeight<L, G::StringType>: LeftSemiring,
{
}

impl<L, W, G> RightSemiring for GallicWeight<L, W, G>
where
    L: ArcLabel,
    W: Weight + RightSemiring,
    G: GallicTypeMarker,
    StringWeight<L, G::StringType>: RightSemiring,
{
}

impl<L, W, G> IdempotentWeight for GallicWeight<L, W, G>
where
    L: ArcLabel,
    W: Weight + IdempotentWeight,
    G: GallicTypeMarker,
{
}

#[cfg(test)]
mod tests {
    use crate::weight::axioms;

    /// The string semiring: `times` concatenates, `plus` takes the longest
    /// common prefix (or suffix). It distributes on one side only, which is why
    /// the left and right flavours exist.
    #[test]
    fn the_left_and_right_string_semirings_satisfy_their_claims() {
        fn samples<S: StringTypeMarker>() -> Vec<StringWeight<i32, S>> {
            vec![
                StringWeight::new(vec![1]),
                StringWeight::new(vec![1, 2]),
                StringWeight::new(vec![2, 1]),
            ]
        }
        axioms::check(&samples::<StringLeft>());
        axioms::check(&samples::<StringRight>());
        // StringRestrict is absent for the same reason as the restricted set
        // semiring; see the test below.
    }

    #[test]
    fn the_left_semiring_takes_the_longest_common_prefix() {
        type Left = StringWeight<i32, StringLeft>;
        let abc = Left::new(vec![1, 2, 3]);
        let abd = Left::new(vec![1, 2, 4]);
        assert_eq!(abc.plus(&abd), Left::new(vec![1, 2]));
        // Times concatenates.
        assert_eq!(
            Left::new(vec![1]).times(&Left::new(vec![2])),
            Left::new(vec![1, 2])
        );
    }

    #[test]
    fn the_right_semiring_takes_the_longest_common_suffix() {
        type Right = StringWeight<i32, StringRight>;
        let acd = Right::new(vec![1, 3, 4]);
        let bcd = Right::new(vec![2, 3, 4]);
        assert_eq!(acd.plus(&bcd), Right::new(vec![3, 4]));
    }

    /// The gallic weight pairs a string with a weight, so a transducer's output
    /// label sequence rides along with its cost. It is the product of the two,
    /// and inherits the string side's one-sidedness.
    #[test]
    fn the_gallic_weights_satisfy_their_claims() {
        use crate::weights::float_weight::TropicalWeight;

        fn samples<G: GallicTypeMarker>() -> Vec<GallicWeight<i32, TropicalWeight, G>>
        where
            GallicWeight<i32, TropicalWeight, G>: Weight,
        {
            vec![
                GallicWeight::from_parts(StringWeight::new(vec![1]), TropicalWeight(1.0)),
                GallicWeight::from_parts(StringWeight::new(vec![1, 2]), TropicalWeight(2.0)),
                GallicWeight::from_parts(StringWeight::new(vec![2]), TropicalWeight(3.0)),
            ]
        }
        axioms::check(&samples::<GallicLeft>());
        axioms::check(&samples::<GallicRight>());
    }

    /// The gallic product must combine the two sides independently: strings
    /// concatenate while weights multiply.
    #[test]
    fn the_gallic_product_combines_both_sides() {
        use crate::weights::float_weight::TropicalWeight;

        type Gallic = GallicWeight<i32, TropicalWeight, GallicLeft>;
        let left = Gallic::from_parts(StringWeight::new(vec![1]), TropicalWeight(1.0));
        let right = Gallic::from_parts(StringWeight::new(vec![2]), TropicalWeight(2.0));
        let product = left.times(&right);
        assert_eq!(
            product,
            Gallic::from_parts(StringWeight::new(vec![1, 2]), TropicalWeight(3.0))
        );
    }

    /// `STRING_RESTRICT` makes `plus` partial in the same way the restricted set
    /// semiring does: it requires its arguments to be equal, so that a
    /// determinized transducer has one output string per path. Distributivity
    /// fails wherever the restriction fires, though upstream declares it.
    #[test]
    fn the_restricted_string_semiring_is_partial() {
        type Restrict = StringWeight<i32, StringRestrict>;
        let one = Restrict::new(vec![1]);
        let two = Restrict::new(vec![2]);

        assert_eq!(one.plus(&one), one);
        assert_eq!(Restrict::zero().plus(&one), one);
        assert_eq!(one.plus(&Restrict::zero()), one);
        assert!(
            !one.plus(&two).is_member(),
            "unequal arguments are rejected"
        );
    }

    use super::*;
    use crate::weights::float_weight::TropicalWeight;

    // For tests, assume i64 implements ArcLabel.
    type LeftString = StringWeight<i64, StringLeft>;
    type RightString = StringWeight<i64, StringRight>;
    type RestrictString = StringWeight<i64, StringRestrict>;

    #[test]
    fn test_string_weight_parse_display() {
        let text = "1_2_3";
        let w: LeftString = text.parse().unwrap();
        assert_eq!(w.to_string(), text);
        assert_eq!(w.size(), 3);

        let eps: LeftString = "Epsilon".parse().unwrap();
        assert_eq!(eps.size(), 0);
        assert_eq!(eps.to_string(), "Epsilon");

        let inf: LeftString = "Infinity".parse().unwrap();
        assert_eq!(inf, StringWeight::zero());
    }

    #[test]
    fn test_string_weight_plus() {
        let w1: LeftString = "1_2_3".parse().unwrap();
        let w2: LeftString = "1_2_4_5".parse().unwrap();

        // Left Plus -> Longest Common Prefix
        let sum_left = w1.plus(&w2);
        assert_eq!(sum_left.to_string(), "1_2");

        let w3: RightString = "1_2_3".parse().unwrap();
        let w4: RightString = "5_2_3".parse().unwrap();

        // Right Plus -> Longest Common Suffix
        let sum_right = w3.plus(&w4);
        assert_eq!(sum_right.to_string(), "2_3");

        let w5: RestrictString = "1_2".parse().unwrap();
        let w6: RestrictString = "1_2".parse().unwrap();
        let w7: RestrictString = "1_3".parse().unwrap();

        // Restrict Plus -> Exact match or BadString
        assert_eq!(w5.plus(&w6).to_string(), "1_2");
        assert_eq!(w5.plus(&w7), StringWeight::no_weight());
    }

    #[test]
    fn test_string_weight_times() {
        let w1: LeftString = "1_2".parse().unwrap();
        let w2: LeftString = "3_4".parse().unwrap();

        // Times -> Concatenation
        let prod = w1.times(&w2);
        assert_eq!(prod.to_string(), "1_2_3_4");
    }

    #[test]
    fn test_gallic_weight_min() {
        // Gallic MIN test
        type MinGallic = GallicWeight<i64, TropicalWeight, GallicMin>;

        let gw1: MinGallic = "1_2,10.0".parse().unwrap();
        let gw2: MinGallic = "3_4,5.0".parse().unwrap();

        // Plus for MIN uses the better (lighter) W weight. 5.0 is better than 10.0 in Tropical.
        let sum = gw1.plus(&gw2);
        assert_eq!(sum.to_string(), "3_4,5");
    }

    /// A marker trait is a claim about the semiring, and `properties()` is the
    /// same claim as a bit. They have to agree, or an algorithm bounded on the
    /// trait is relying on something the weight does not report.
    #[test]
    fn the_marker_traits_agree_with_the_property_bits() {
        fn left<W: Weight + LeftSemiring>() {
            assert_ne!(W::properties() & LEFT_SEMIRING, 0, "{}", W::type_name());
        }
        fn right<W: Weight + RightSemiring>() {
            assert_ne!(W::properties() & RIGHT_SEMIRING, 0, "{}", W::type_name());
        }
        fn idempotent<W: Weight + IdempotentWeight>() {
            assert_ne!(W::properties() & IDEMPOTENT, 0, "{}", W::type_name());
        }

        left::<StringWeight<i32, StringLeft>>();
        right::<StringWeight<i32, StringRight>>();
        left::<StringWeight<i32, StringRestrict>>();
        right::<StringWeight<i32, StringRestrict>>();
        idempotent::<StringWeight<i32, StringLeft>>();
        idempotent::<StringWeight<i32, StringRight>>();
        idempotent::<StringWeight<i32, StringRestrict>>();

        left::<GallicWeight<i32, crate::weights::float_weight::TropicalWeight, GallicLeft>>();
        right::<GallicWeight<i32, crate::weights::float_weight::TropicalWeight, GallicRight>>();
        idempotent::<GallicWeight<i32, crate::weights::float_weight::TropicalWeight, GallicLeft>>();

        // A left string weight does not distribute on the right, and the bit
        // says so, which is why there is no `RightSemiring` impl for it.
        assert_eq!(
            StringWeight::<i32, StringLeft>::properties() & RIGHT_SEMIRING,
            0
        );
    }
}
