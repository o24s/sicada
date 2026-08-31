use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use crate::error::ParseError;
use crate::fst_type::WeightType;
use crate::weight::{
    COMMUTATIVE, Divide, DivideType, IDEMPOTENT, LEFT_SEMIRING, Minus, RIGHT_SEMIRING, Weight,
};
use crate::weights::pair_weight::PairWeight;

/// A semiring that tracks a probability and an associated expected value.
///
/// Derived from:
/// Eisner, J. 2002. Parameter estimation for probabilistic finite-state
/// transducers. In Proceedings of the 40th Annual Meeting of the
/// Association for Computational Linguistics, pages 1-8.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpectationWeight<W1, W2>(pub PairWeight<W1, W2>);

impl<W1, W2> ExpectationWeight<W1, W2> {
    #[inline(always)]
    pub fn new(w1: W1, w2: W2) -> Self {
        Self(PairWeight::new(w1, w2))
    }

    #[inline(always)]
    pub fn value1(&self) -> &W1 {
        self.0.value1()
    }

    #[inline(always)]
    pub fn value2(&self) -> &W2 {
        self.0.value2()
    }
}

// Ensure the cross-product capability exists, specifically `W1 * W2 -> W2`.
// Rust doesn't natively support differing type outputs in our `Weight` trait,
// so we assume `W1` and `W2` are the same for the standard implementation.
// (OpenFst C++ relies on duck typing for `Times(W1, W2) -> W2`. In Rust, we use
// strict traits. The simplest and safest general approach is `W1 == W2`).

impl<W> Weight for ExpectationWeight<W, W>
where
    W: Weight,
{
    type ReverseWeight = ExpectationWeight<W::ReverseWeight, W::ReverseWeight>;

    #[inline(always)]
    fn zero() -> Self {
        Self::new(W::zero(), W::zero())
    }

    #[inline(always)]
    fn one() -> Self {
        Self::new(W::one(), W::zero())
    }

    #[inline(always)]
    fn no_weight() -> Self {
        Self::new(W::no_weight(), W::no_weight())
    }

    #[inline(always)]
    fn type_name() -> WeightType {
        let s = format!("expectation_{}_{}", W::type_name(), W::type_name());
        WeightType::new_dynamic(s)
    }

    #[inline(always)]
    fn properties() -> u64 {
        W::properties() & (LEFT_SEMIRING | RIGHT_SEMIRING | COMMUTATIVE | IDEMPOTENT)
    }

    #[inline(always)]
    fn is_member(&self) -> bool {
        self.0.is_member()
    }

    #[inline(always)]
    fn approx_equal(&self, other: &Self, delta: f32) -> bool {
        PairWeight::approx_equal(&self.0, &other.0, delta)
    }

    #[inline(always)]
    fn quantize(&self, delta: f32) -> Self {
        Self(self.0.quantize(delta))
    }

    #[inline(always)]
    fn reverse(&self) -> Self::ReverseWeight {
        ExpectationWeight(self.0.reverse())
    }

    #[inline]
    fn plus(&self, rhs: &Self) -> Self {
        Self::new(
            W::plus(self.value1(), rhs.value1()),
            W::plus(self.value2(), rhs.value2()),
        )
    }

    #[inline]
    fn times(&self, rhs: &Self) -> Self {
        let p1 = W::times(self.value1(), rhs.value1());

        let c1 = W::times(self.value1(), rhs.value2());
        let c2 = W::times(self.value2(), rhs.value1());
        let p2 = W::plus(&c1, &c2);

        Self::new(p1, p2)
    }
}

// Require `Minus` for `Divide` to satisfy expectation semiring division proofs.
impl<W> Divide for ExpectationWeight<W, W>
where
    W: Weight + Divide + Minus,
{
    fn divide(&self, rhs: &Self, typ: DivideType) -> Self {
        // q1 = x1 / y1
        let q1 = W::divide(self.value1(), rhs.value1(), typ);

        let q2 = if typ == DivideType::Left {
            // q2 = (x2 - y2 * q1) / y1
            let cross = W::times(rhs.value2(), &q1);
            let diff = W::minus(self.value2(), &cross);
            W::divide(&diff, rhs.value1(), typ)
        } else {
            // q2 = (x2 - q1 * y2) / y1
            let cross = W::times(&q1, rhs.value2());
            let diff = W::minus(self.value2(), &cross);
            W::divide(&diff, rhs.value1(), typ)
        };

        Self::new(q1, q2)
    }
}

impl<W1: Hash, W2: Hash> Hash for ExpectationWeight<W1, W2> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<W1: fmt::Display, W2: fmt::Display> fmt::Display for ExpectationWeight<W1, W2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<W> FromStr for ExpectationWeight<W, W>
where
    W: FromStr + Weight,
{
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts = crate::utils::split_composite_weight(s, ',', '(', ')')?;

        if parts.len() != 2 {
            return Err(ParseError::InvalidElementCount {
                expected: 2,
                found: parts.len(),
            });
        }

        let w1 = parts[0].parse::<W>().map_err(|_| {
            ParseError::InvalidFormat(format!(
                "Failed to parse w1 in ExpectationWeight: {}",
                parts[0]
            ))
        })?;

        let w2 = parts[1].parse::<W>().map_err(|_| {
            ParseError::InvalidFormat(format!(
                "Failed to parse w2 in ExpectationWeight: {}",
                parts[1]
            ))
        })?;

        Ok(Self::new(w1, w2))
    }
}

#[cfg(test)]
mod tests {
    /// Eisner's expectation semiring: `plus` is componentwise, `times` carries
    /// the cross terms so that the second component accumulates an expectation
    /// weighted by the first.
    #[test]
    fn it_satisfies_the_axioms_it_claims() {
        use crate::weight::axioms;
        use crate::weights::float_weight::LogWeight;

        type Exp = ExpectationWeight<LogWeight, LogWeight>;
        axioms::check(&[
            Exp::new(LogWeight(1.0), LogWeight(2.0)),
            Exp::new(LogWeight(2.0), LogWeight(1.0)),
        ]);
    }

    /// One is `(One, Zero)`, not `(One, One)`: the expectation of an empty path
    /// is zero, while its probability is one.
    #[test]
    fn one_carries_no_expectation() {
        use crate::weights::float_weight::LogWeight;

        type Exp = ExpectationWeight<LogWeight, LogWeight>;
        assert_eq!(Exp::one(), Exp::new(LogWeight::one(), LogWeight::zero()));
        assert_eq!(Exp::zero(), Exp::new(LogWeight::zero(), LogWeight::zero()));
    }

    use super::*;
    use crate::float_weight::RealWeight;

    type ExpectationReal = ExpectationWeight<RealWeight, RealWeight>;

    #[test]
    fn test_expectation_weight_plus() {
        let w1 = ExpectationReal::new(RealWeight(0.5), RealWeight(10.0));
        let w2 = ExpectationReal::new(RealWeight(0.2), RealWeight(5.0));

        let w3 = ExpectationReal::plus(&w1, &w2);

        // Plus is just pairwise Plus: (0.5 + 0.2, 10.0 + 5.0)
        assert_eq!(w3.value1().0, 0.7);
        assert_eq!(w3.value2().0, 15.0);
    }

    #[test]
    fn test_expectation_weight_times() {
        let w1 = ExpectationReal::new(RealWeight(0.5), RealWeight(10.0));
        let w2 = ExpectationReal::new(RealWeight(0.2), RealWeight(5.0));

        let w3 = ExpectationReal::times(&w1, &w2);

        // Times:
        // p1 = a1 * a2 = 0.5 * 0.2 = 0.1
        // p2 = a1 * b2 + a2 * b1 = 0.5 * 5.0 + 0.2 * 10.0 = 2.5 + 2.0 = 4.5
        assert_eq!(w3.value1().0, 0.1);
        assert_eq!(w3.value2().0, 4.5);
    }

    #[test]
    fn test_expectation_weight_divide() {
        let w_dividend = ExpectationReal::new(RealWeight(0.1), RealWeight(4.5));
        let w_divisor = ExpectationReal::new(RealWeight(0.2), RealWeight(5.0));

        let w_quotient = ExpectationReal::divide(&w_dividend, &w_divisor, DivideType::Left);

        assert_eq!(w_quotient.value1().0, 0.5);
        assert_eq!(w_quotient.value2().0, 10.0);
    }
}
