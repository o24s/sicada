use std::fmt::{Debug, Display};
use std::str::FromStr;

use crate::fst_type::WeightType;

pub const DELTA: f32 = 1.0 / 1024.0;

pub const LEFT_SEMIRING: u64 = 0x0000000000000001;
pub const RIGHT_SEMIRING: u64 = 0x0000000000000002;
pub const SEMIRING: u64 = LEFT_SEMIRING | RIGHT_SEMIRING;
pub const COMMUTATIVE: u64 = 0x0000000000000004;
pub const IDEMPOTENT: u64 = 0x0000000000000008;
pub const PATH: u64 = 0x0000000000000010;

/// Represents an element of a semiring used as a weight in an FST.
/// Mathematically, a semiring requires addition (plus) and multiplication (times),
/// along with their respective identity elements (zero and one).
pub trait Weight: Clone + PartialEq + Debug + Display + FromStr {
    /// The type returned by reversing this weight (used for Reverse FST operations).
    type ReverseWeight: Weight;

    /// The semiring's zero element (identity for `plus`).
    /// Also represents a non-existent or invalid path (e.g., infinity for costs).
    fn zero() -> Self;

    /// The semiring's one element (identity for `times`).
    /// Represents a zero-cost path.
    fn one() -> Self;

    /// Represents an invalid weight (often represented by NaN internally).
    /// Used instead of `Option<Self>` to maintain strict memory layout performance
    /// in graph algorithms.
    fn no_weight() -> Self;

    /// A string representation of the semiring type (e.g., "tropical", "log").
    fn type_name() -> WeightType;

    /// A bitmask of the semiring properties (e.g., LEFT_SEMIRING | COMMUTATIVE).
    fn properties() -> u64;

    /// Semiring addition (e.g., `min(a, b)` in Tropical, or `-log(e^-a + e^-b)` in Log).
    fn plus(&self, rhs: &Self) -> Self;

    /// Semiring multiplication (e.g., `a + b` in Tropical/Log).
    fn times(&self, rhs: &Self) -> Self;

    /// Returns the reversed weight.
    /// For commutative weights, this is typically `self`.
    fn reverse(&self) -> Self::ReverseWeight;

    /// Returns true if this is a valid member of the semiring (not `no_weight()`).
    fn is_member(&self) -> bool;

    /// Returns true if this weight is approximately equal to `other` within `delta`.
    /// For discrete weights (like Strings), `delta` is ignored.
    fn approx_equal(&self, other: &Self, delta: f32) -> bool;

    /// Quantizes the weight to the nearest multiple of `delta`.
    /// Essential for epsilon-removal and equivalence testing with floating-point weights.
    fn quantize(&self, delta: f32) -> Self;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivideType {
    /// Left division: `y \times x = z`  =>  `x = divide(z, y, Left)`
    Left,
    /// Right division: `x \times y = z` =>  `x = divide(z, y, Right)`
    Right,
    /// Any division (for commutative semirings where Left == Right)
    Any,
}
/// A weight that can be written to a file and read back.
///
/// Port of the `Read`/`Write` pair upstream requires of every weight. Split out
/// rather than folded into [`Weight`], as [`Divide`] is: an algorithm that never
/// serializes should not have to care, and a weight with no defined encoding
/// should not have to invent one.
///
/// The encoding is part of the FST file format. `tests/oracles/` holds the
/// bytes OpenFst produces for the ones that matter.
pub trait WeightIo: Sized {
    /// Reads one weight.
    fn read<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self>;

    /// Writes one weight.
    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()>;
}

/// A marker and operational trait for weights that support division (multiplicative inverse).
/// By extracting this from `Weight`, algorithms that require division (e.g., Determinize, Push)
/// can specify `W: Divide` as a bound, guaranteeing memory/logic safety at compile time.
pub trait Divide: Weight {
    /// Semiring division.
    /// Returns `no_weight()` if the specific division is mathematically undefined
    /// (e.g., dividing by zero).
    fn divide(&self, rhs: &Self, typ: DivideType) -> Self;
}

/// A marker and operational trait for weights that support subtraction.
/// Required by algorithms like Expectation Semiring Division.
pub trait Minus: Weight {
    fn minus(&self, rhs: &Self) -> Self;
}

/// Weight forms a left semiring (required for determinize and pushing to initial).
pub trait LeftSemiring: Weight {}

/// Weight forms a right semiring (required for shortest path and pushing to final).
pub trait RightSemiring: Weight {}

/// Weight has the path property, meaning addition is exactly `min` (required for shortest path).
/// Weight is idempotent, meaning `x.plus(x) == x`.
pub trait IdempotentWeight: Weight {}

/// Weight has the path property: `x.plus(y)` is `x` or `y`, never anything
/// else.
///
/// SICADA-DIVERGE: this is stated as *stronger* than idempotence, which
/// upstream leaves as two unrelated bits. It is: taking `y = x` in the path
/// property gives `x.plus(x) == x`. Saying so lets an algorithm that needs the
/// natural order, which is only an order on an idempotent semiring, state the
/// path property alone and get it.
pub trait PathWeight: IdempotentWeight {}

/// Weight is commutative, meaning `x.times(y) == y.times(x)`.
pub trait CommutativeWeight: Weight {}

/// The strict natural order on an idempotent semiring.
///
/// The natural order is defined by `a <= b` iff `a + b == a`; this is its strict
/// version. It is a negative partial order exactly when the semiring is
/// idempotent, which is why the bound is [`IdempotentWeight`]; it is monotonic
/// for `times` on whichever side the semiring distributes, and it is a *total*
/// order exactly when the semiring has the path property.
///
/// See Mohri, M. 2002. Semiring framework and algorithms for shortest-distance
/// problems, *Journal of Automata, Languages and Combinatorics* 7(3): 321-350.
#[inline]
pub fn natural_less<W: IdempotentWeight>(lhs: &W, rhs: &W) -> bool {
    lhs != rhs && lhs.plus(rhs) == *lhs
}

/// Power is the iterated product for arbitrary semirings.
/// Computes `weight ⊗ weight ⊗ ... ⊗ weight` (n times).
pub fn power<W: Weight>(weight: &W, n: usize) -> W {
    let mut result = W::one();
    for _ in 0..n {
        result = result.times(weight);
    }
    result
}

/// A simple accumulator for semiring addition.
/// For floating point weights like LogWeight or RealWeight,
/// Kahan-compensated adders should be used to avoid precision loss in long sums.
#[derive(Debug, Clone)]
pub struct Adder<W: Weight> {
    sum: W,
}

impl<W: Weight> Adder<W> {
    pub fn new() -> Self {
        Self { sum: W::zero() }
    }

    /// Adds a weight to the accumulator.
    #[inline]
    pub fn add(&mut self, w: &W) {
        self.sum = self.sum.plus(w);
    }

    /// Returns a copy of the current accumulated sum.
    #[inline]
    pub fn sum(&self) -> W {
        self.sum.clone()
    }

    /// Resets the accumulator to the given weight.
    #[inline]
    pub fn reset(&mut self, w: W) {
        self.sum = w;
    }
}

impl<W: Weight> Default for Adder<W> {
    fn default() -> Self {
        Self::new()
    }
}

// Conversions
macro_rules! impl_weight_convert {
    ($from:ident, $to:ident, $closure:expr) => {
        impl From<$from> for $to {
            #[inline(always)]
            fn from(w: $from) -> Self {
                $closure(w)
            }
        }
    };
}

pub(crate) use impl_weight_convert;

/// Checks that a weight actually satisfies the semiring axioms it advertises.
///
/// Every weight module's tests call [`axioms::check`] with a handful of
/// representative values. The properties a weight returns from
/// [`Weight::properties`] are claims that algorithms rely on for correctness:
/// `shortest_distance` needs the natural order to be total, `rm_epsilon` needs
/// left distributivity. An unchecked claim is therefore a latent source of wrong
/// answers throughout the library.
#[cfg(any(test, feature = "axioms"))]
pub mod axioms {
    use super::*;

    /// Tolerance for the approximate comparisons. Float semirings are not
    /// exactly associative, since `(a + b) + c` and `a + (b + c)` differ in the
    /// last bits for the log semiring, so the axioms hold up to `approx_equal`.
    const DELTA: f32 = 1e-4;

    fn same<W: Weight>(lhs: &W, rhs: &W, axiom: &str, context: &str) {
        assert!(
            lhs.approx_equal(rhs, DELTA),
            "{axiom} failed for {context}: {lhs} vs {rhs}"
        );
    }

    /// Checks the axioms implied by `W::properties()` over every combination of
    /// `samples`.
    ///
    /// `samples` should be members of the semiring; `zero` and `one` are added
    /// automatically. Cubic in the sample count, so keep it small.
    pub fn check<W: Weight>(samples: &[W]) {
        let mut values = vec![W::zero(), W::one()];
        values.extend(samples.iter().cloned());
        for value in &values {
            assert!(
                value.is_member(),
                "sample {value} is not a member of {}",
                W::type_name()
            );
        }

        let props = W::properties();
        let type_name = W::type_name();

        for a in &values {
            let context = format!("{type_name}: a={a}");
            same(
                &a.plus(&W::zero()),
                a,
                "zero is a right identity for plus",
                &context,
            );
            same(
                &W::zero().plus(a),
                a,
                "zero is a left identity for plus",
                &context,
            );
            same(
                &a.times(&W::one()),
                a,
                "one is a right identity for times",
                &context,
            );
            same(
                &W::one().times(a),
                a,
                "one is a left identity for times",
                &context,
            );
            same(
                &a.times(&W::zero()),
                &W::zero(),
                "zero annihilates on the right",
                &context,
            );
            same(
                &W::zero().times(a),
                &W::zero(),
                "zero annihilates on the left",
                &context,
            );

            if props & IDEMPOTENT != 0 {
                same(&a.plus(a), a, "plus is idempotent", &context);
            }

            for b in &values {
                let context = format!("{type_name}: a={a}, b={b}");
                same(&a.plus(b), &b.plus(a), "plus is commutative", &context);

                if props & COMMUTATIVE != 0 {
                    same(&a.times(b), &b.times(a), "times is commutative", &context);
                }

                if props & PATH != 0 {
                    let sum = a.plus(b);
                    assert!(
                        sum.approx_equal(a, DELTA) || sum.approx_equal(b, DELTA),
                        "the path property failed for {context}: a+b={sum}"
                    );
                }

                for c in &values {
                    let context = format!("{type_name}: a={a}, b={b}, c={c}");
                    same(
                        &a.plus(b).plus(c),
                        &a.plus(&b.plus(c)),
                        "plus is associative",
                        &context,
                    );
                    same(
                        &a.times(b).times(c),
                        &a.times(&b.times(c)),
                        "times is associative",
                        &context,
                    );

                    if props & LEFT_SEMIRING != 0 {
                        same(
                            &a.times(&b.plus(c)),
                            &a.times(b).plus(&a.times(c)),
                            "times distributes over plus on the left",
                            &context,
                        );
                    }
                    if props & RIGHT_SEMIRING != 0 {
                        same(
                            &b.plus(c).times(a),
                            &b.times(a).plus(&c.times(a)),
                            "times distributes over plus on the right",
                            &context,
                        );
                    }
                }
            }
        }
    }

    /// Checks the division axiom: if `a * b == c` then dividing `c` back by `a`
    /// must give something that multiplies with `a` to `c` again.
    ///
    /// Division is partial, since there need not be any `b` with `c == a * b`,
    /// so a non-member result is accepted and simply carries no obligation.
    pub fn check_divide<W: Weight + Divide>(samples: &[W]) {
        let mut values = vec![W::one()];
        values.extend(samples.iter().cloned());
        let props = W::properties();

        for a in &values {
            for b in &values {
                let c = a.times(b);
                if !c.is_member() {
                    continue;
                }
                if props & LEFT_SEMIRING != 0 {
                    let recovered = c.divide(a, DivideType::Left);
                    if recovered.is_member() {
                        assert!(
                            a.times(&recovered).approx_equal(&c, DELTA),
                            "left divide failed for a={a}, b={b}: recovered {recovered}"
                        );
                    }
                }
                if props & RIGHT_SEMIRING != 0 {
                    let recovered = c.divide(b, DivideType::Right);
                    if recovered.is_member() {
                        assert!(
                            recovered.times(b).approx_equal(&c, DELTA),
                            "right divide failed for a={a}, b={b}: recovered {recovered}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::float_weight::{LogWeight, TropicalWeight};

    /// The harness has to actually catch a violated claim, or it proves nothing.
    /// This is the real semiring, a perfectly good commutative semiring, that
    /// claims idempotence and the path property without having either.
    #[derive(Debug, Clone, PartialEq)]
    struct Liar(f32);

    impl std::fmt::Display for Liar {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl std::str::FromStr for Liar {
        type Err = std::num::ParseFloatError;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            s.parse().map(Liar)
        }
    }

    impl Weight for Liar {
        type ReverseWeight = Liar;
        fn zero() -> Self {
            Liar(0.0)
        }
        fn one() -> Self {
            Liar(1.0)
        }
        fn no_weight() -> Self {
            Liar(f32::NAN)
        }
        fn type_name() -> crate::fst_type::WeightType {
            crate::fst_type::WeightType::new("liar")
        }
        fn properties() -> u64 {
            SEMIRING | COMMUTATIVE | IDEMPOTENT | PATH
        }
        fn plus(&self, rhs: &Self) -> Self {
            Liar(self.0 + rhs.0)
        }
        fn times(&self, rhs: &Self) -> Self {
            Liar(self.0 * rhs.0)
        }
        fn reverse(&self) -> Self {
            self.clone()
        }
        fn is_member(&self) -> bool {
            !self.0.is_nan()
        }
        fn approx_equal(&self, other: &Self, delta: f32) -> bool {
            (self.0 - other.0).abs() <= delta
        }
        fn quantize(&self, _delta: f32) -> Self {
            self.clone()
        }
    }

    #[test]
    #[should_panic(expected = "plus is idempotent")]
    fn the_axiom_harness_rejects_a_weight_that_lies() {
        axioms::check(&[Liar(1.0), Liar(2.0)]);
    }

    #[test]
    fn tropical_satisfies_the_axioms_it_claims() {
        assert_eq!(
            TropicalWeight::properties(),
            SEMIRING | COMMUTATIVE | IDEMPOTENT | PATH
        );
        axioms::check(&[
            TropicalWeight(0.5),
            TropicalWeight(2.0),
            TropicalWeight(-1.5),
        ]);
        axioms::check_divide(&[TropicalWeight(0.5), TropicalWeight(2.0)]);
    }

    #[test]
    fn log_satisfies_the_axioms_it_claims() {
        assert_eq!(LogWeight::properties(), SEMIRING | COMMUTATIVE);
        axioms::check(&[LogWeight(0.5), LogWeight(2.0), LogWeight(-1.5)]);
        axioms::check_divide(&[LogWeight(0.5), LogWeight(2.0)]);
    }

    #[test]
    fn power_is_the_iterated_product() {
        let weight = TropicalWeight(1.5);
        assert_eq!(power(&weight, 0), TropicalWeight::one());
        assert_eq!(power(&weight, 1), weight);
        assert_eq!(power(&weight, 3), TropicalWeight(4.5));
        // Power(w, n) == Times(Power(w, n - 1), w), by definition.
        for n in 1..8 {
            assert_eq!(power(&weight, n), power(&weight, n - 1).times(&weight));
        }
    }

    #[test]
    fn the_natural_order_is_strict_and_total_on_a_path_semiring() {
        let values = [
            TropicalWeight::zero(),
            TropicalWeight(3.0),
            TropicalWeight(1.0),
            TropicalWeight::one(),
        ];
        for a in &values {
            assert!(!natural_less(a, a), "the order must be strict");
            for b in &values {
                // Tropical plus is min, so the natural order is the cost order.
                assert_eq!(natural_less(a, b), a.value() < b.value(), "{a} vs {b}");
                // Totality: any two distinct weights are comparable.
                if a != b {
                    assert!(natural_less(a, b) || natural_less(b, a));
                }
            }
        }
    }

    #[test]
    fn the_adder_accumulates_with_plus() {
        let mut adder = Adder::new();
        assert_eq!(adder.sum(), TropicalWeight::zero());
        for value in [3.0, 1.0, 2.0] {
            adder.add(&TropicalWeight(value));
        }
        assert_eq!(adder.sum(), TropicalWeight(1.0), "tropical plus is min");

        adder.reset(TropicalWeight(0.5));
        assert_eq!(adder.sum(), TropicalWeight(0.5));
    }

    #[test]
    fn the_property_bits_match_openfst() {
        // From vendor/openfst/openfst/lib/weight.h.
        assert_eq!(LEFT_SEMIRING, 0x01);
        assert_eq!(RIGHT_SEMIRING, 0x02);
        assert_eq!(SEMIRING, 0x03);
        assert_eq!(COMMUTATIVE, 0x04);
        assert_eq!(IDEMPOTENT, 0x08);
        assert_eq!(PATH, 0x10);
    }
}
