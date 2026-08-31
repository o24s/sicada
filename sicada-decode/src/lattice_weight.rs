//! The semiring a lattice's arcs carry.
//!
//! A lattice arc has to remember *two* costs, not one: what the decoding graph
//! charged for it and what the acoustic model did. They are added together to
//! decide which path is best, but a second pass wants them apart, because
//! rescoring with a different language-model scale, or with a different model
//! entirely, means rebuilding the graph half while leaving the acoustic half
//! alone. A single total would have thrown that away at decode time.
//!
//! This is Kaldi's `LatticeWeightTpl` (`fstext/lattice-weight.h`), ported. The
//! type name is Kaldi's too, `lattice4` / `lattice8`, because an FST file header
//! records it and a lattice written here should be one Kaldi can read.
//!
//! ⊕ picks the argument with the smaller total, breaking a tie on the graph
//! cost. ⊗ adds componentwise. It therefore has the path property and is
//! idempotent, commutative, and a full semiring: the same shape as the tropical
//! semiring, carrying one more number along for the ride.

use std::fmt;
use std::str::FromStr;

use sicada::fst_type::WeightType;
use sicada::weight::{
    COMMUTATIVE, CommutativeWeight, Divide, IDEMPOTENT, IdempotentWeight, LEFT_SEMIRING,
    LeftSemiring, PATH, PathWeight, RIGHT_SEMIRING, RightSemiring, Weight,
};

macro_rules! define_lattice_weight {
    ($name:ident, $float:ty, $type_name:expr) => {
        /// A graph cost and an acoustic cost, kept apart.
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $name {
            /// What the decoding graph charged: the language model, the
            /// pronunciation, the transition probabilities.
            pub graph: $float,
            /// What the acoustic model charged.
            pub acoustic: $float,
        }

        impl $name {
            /// A weight from its two halves.
            #[inline(always)]
            pub const fn new(graph: $float, acoustic: $float) -> Self {
                Self { graph, acoustic }
            }

            /// The two added together, which is the quantity ⊕ compares on.
            #[inline(always)]
            pub fn total(&self) -> $float {
                self.graph + self.acoustic
            }

            /// The total with the acoustic half rescaled, which is what a
            /// second pass asks for, and why the halves are kept apart.
            #[inline(always)]
            pub fn total_scaled(&self, acoustic_scale: $float) -> $float {
                self.graph + acoustic_scale * self.acoustic
            }

            /// Ordering in the semiring: `Greater` means better, because a
            /// smaller cost is a higher probability.
            ///
            /// Upstream's comment on the tie-break is worth keeping: the
            /// mathematically natural comparison is
            /// `value1 - value2 < value1' - value2'`, but the totals are equal
            /// here, so adding them to both sides and halving leaves the
            /// simpler `value1 < value1'`.
            #[inline]
            fn compare(&self, other: &Self) -> std::cmp::Ordering {
                use std::cmp::Ordering::*;
                let (mine, theirs) = (self.total(), other.total());
                if mine < theirs {
                    Greater
                } else if mine > theirs {
                    Less
                } else if self.graph < other.graph {
                    Greater
                } else if self.graph > other.graph {
                    Less
                } else {
                    Equal
                }
            }
        }

        /// Two `no_weight()`s compare equal, which raw float equality would
        /// not give. Every float semiring in sicada does the same, and it is
        /// what makes `Eq` honest: a weight has to equal itself before it can
        /// be a key, and `determinize` keys its subsets on weights.
        impl PartialEq for $name {
            #[inline(always)]
            fn eq(&self, other: &Self) -> bool {
                let same = |a: $float, b: $float| {
                    if a.is_nan() { b.is_nan() } else { a == b }
                };
                same(self.graph, other.graph) && same(self.acoustic, other.acoustic)
            }
        }

        impl Eq for $name {}

        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                // One NaN, so that two weights that compare equal hash equal.
                let bits = |value: $float| {
                    if value.is_nan() {
                        <$float>::NAN.to_bits()
                    } else {
                        value.to_bits()
                    }
                };
                bits(self.graph).hash(state);
                bits(self.acoustic).hash(state);
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                // Kaldi's text form, so a lattice printed here reads the same.
                write!(f, "{},{}", self.graph, self.acoustic)
            }
        }

        impl FromStr for $name {
            type Err = String;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let (graph, acoustic) = s.split_once(',').ok_or_else(|| {
                    format!("{}: expected `graph,acoustic`, got {s:?}", $type_name)
                })?;
                Ok(Self::new(
                    graph
                        .trim()
                        .parse()
                        .map_err(|_| format!("{}: {graph:?} is not a number", $type_name))?,
                    acoustic
                        .trim()
                        .parse()
                        .map_err(|_| format!("{}: {acoustic:?} is not a number", $type_name))?,
                ))
            }
        }

        /// The bytes Kaldi's `LatticeWeightTpl::Write` produces: the two costs,
        /// **always as `f32`**, whatever `$float` is.
        ///
        /// Upstream's reason, kept: "Always read/write as float, even if T is
        /// double, so we can use OpenFst-style read/write and still maintain
        /// compatibility when compiling with different FloatTypes." A lattice
        /// written in double precision is therefore readable in single, and the
        /// file does not record which was used.
        impl sicada::weight::WeightIo for $name {
            #[inline]
            fn read<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
                let graph: f32 = sicada::utils::io::read_scalar(reader)?;
                let acoustic: f32 = sicada::utils::io::read_scalar(reader)?;
                Ok(Self::new(graph as $float, acoustic as $float))
            }

            #[inline]
            fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
                sicada::utils::io::write_scalar(writer, self.graph as f32)?;
                sicada::utils::io::write_scalar(writer, self.acoustic as f32)
            }
        }

        impl Weight for $name {
            type ReverseWeight = Self;

            #[inline(always)]
            fn zero() -> Self {
                Self::new(<$float>::INFINITY, <$float>::INFINITY)
            }

            #[inline(always)]
            fn one() -> Self {
                Self::new(0.0, 0.0)
            }

            #[inline(always)]
            fn no_weight() -> Self {
                Self::new(<$float>::NAN, <$float>::NAN)
            }

            #[inline]
            fn type_name() -> WeightType {
                WeightType::new($type_name)
            }

            #[inline(always)]
            fn properties() -> u64 {
                LEFT_SEMIRING | RIGHT_SEMIRING | COMMUTATIVE | PATH | IDEMPOTENT
            }

            #[inline(always)]
            fn plus(&self, rhs: &Self) -> Self {
                if !self.is_member() || !rhs.is_member() {
                    return Self::no_weight();
                }
                if self.compare(rhs).is_ge() {
                    *self
                } else {
                    *rhs
                }
            }

            #[inline(always)]
            fn times(&self, rhs: &Self) -> Self {
                if !self.is_member() || !rhs.is_member() {
                    return Self::no_weight();
                }
                Self::new(self.graph + rhs.graph, self.acoustic + rhs.acoustic)
            }

            #[inline(always)]
            fn reverse(&self) -> Self::ReverseWeight {
                *self
            }

            /// Upstream's membership test, kept whole: no NaN, no `-inf`, and
            /// `+inf` in both halves or neither.
            ///
            /// The last clause is the one that matters. A semiring has exactly
            /// one zero; if `(inf, 3.0)` were a member it would be a second
            /// one, indistinguishable from `(inf, inf)` under ⊕ and ⊗ but not
            /// under `==`, and every algorithm that compares against `zero()`
            /// would start missing it.
            #[inline]
            fn is_member(&self) -> bool {
                if self.graph.is_nan() || self.acoustic.is_nan() {
                    return false;
                }
                if self.graph == <$float>::NEG_INFINITY || self.acoustic == <$float>::NEG_INFINITY {
                    return false;
                }
                (self.graph == <$float>::INFINITY) == (self.acoustic == <$float>::INFINITY)
            }

            #[inline]
            fn approx_equal(&self, other: &Self, delta: f32) -> bool {
                let close = |a: $float, b: $float| a == b || (a - b).abs() <= delta as $float;
                close(self.graph, other.graph) && close(self.acoustic, other.acoustic)
            }

            #[inline]
            fn quantize(&self, delta: f32) -> Self {
                // The halves are quantised together or not at all: rounding
                // one of an infinite pair would leave a weight that is not a
                // member.
                let total = self.total();
                if total.is_nan() {
                    return Self::new(total, total);
                }
                if total.is_infinite() {
                    return Self::new(total, total);
                }
                let delta = delta as $float;
                let round = |value: $float| (value / delta + 0.5).floor() * delta;
                Self::new(round(self.graph), round(self.acoustic))
            }
        }

        impl Divide for $name {
            #[inline]
            fn divide(&self, rhs: &Self, _side: sicada::weight::DivideType) -> Self {
                if !self.is_member() || !rhs.is_member() {
                    return Self::no_weight();
                }
                // Anything over zero is undefined, and zero over anything is
                // zero. Otherwise `inf - inf` would produce NaN in a place
                // where the answer is known.
                if *rhs == Self::zero() {
                    return Self::no_weight();
                }
                if *self == Self::zero() {
                    return Self::zero();
                }
                Self::new(self.graph - rhs.graph, self.acoustic - rhs.acoustic)
            }
        }

        impl LeftSemiring for $name {}
        impl RightSemiring for $name {}
        impl CommutativeWeight for $name {}
        impl IdempotentWeight for $name {}
        impl PathWeight for $name {}
    };
}

define_lattice_weight!(LatticeWeight, f32, "lattice4");
define_lattice_weight!(LatticeWeight64, f64, "lattice8");

/// An arc carrying a [`LatticeWeight`], with the label and state-id types of
/// the decoding graph it came from.
pub type LatticeArc<A> = sicada::arc::ArcTpl<
    LatticeWeight,
    <A as sicada::arc::Arc>::Label,
    <A as sicada::arc::Arc>::StateId,
>;

#[cfg(test)]
mod tests {
    use super::*;
    use sicada::weight::{DivideType, axioms};

    fn samples() -> Vec<LatticeWeight> {
        vec![
            LatticeWeight::new(0.0, 0.0),
            LatticeWeight::new(1.0, 0.0),
            LatticeWeight::new(0.0, 1.0),
            LatticeWeight::new(2.5, -1.5),
            LatticeWeight::new(-0.75, 3.25),
            LatticeWeight::zero(),
        ]
    }

    /// A weight's `properties()` bits are a claim every algorithm downstream
    /// acts on, so they are checked rather than asserted.
    #[test]
    fn it_is_the_semiring_it_says_it_is() {
        axioms::check(&samples());
        axioms::check_divide(&samples());
    }

    #[test]
    fn it_is_the_semiring_it_says_it_is_in_double_precision() {
        axioms::check(&[
            LatticeWeight64::new(0.0, 0.0),
            LatticeWeight64::new(1.0, 0.0),
            LatticeWeight64::new(0.0, 1.0),
            LatticeWeight64::new(2.5, -1.5),
            LatticeWeight64::zero(),
        ]);
    }

    /// The name goes into an FST file header, and a lattice written here is
    /// meant to be one Kaldi reads.
    #[test]
    fn its_type_name_is_kaldis() {
        assert_eq!(LatticeWeight::type_name().as_str(), "lattice4");
        assert_eq!(LatticeWeight64::type_name().as_str(), "lattice8");
    }

    #[test]
    fn plus_takes_the_smaller_total() {
        let cheap = LatticeWeight::new(1.0, 1.0);
        let dear = LatticeWeight::new(5.0, 0.0);
        assert_eq!(cheap.plus(&dear), cheap);
        assert_eq!(dear.plus(&cheap), cheap);
    }

    /// The tie-break keeps ⊕ a function rather than a coin toss: two paths of
    /// the same total but a different split have to resolve the same way
    /// whichever order they arrive in.
    #[test]
    fn a_tie_on_the_total_is_broken_on_the_graph_cost() {
        let graph_heavy = LatticeWeight::new(3.0, 1.0);
        let acoustic_heavy = LatticeWeight::new(1.0, 3.0);
        assert_eq!(graph_heavy.total(), acoustic_heavy.total());
        assert_eq!(graph_heavy.plus(&acoustic_heavy), acoustic_heavy);
        assert_eq!(acoustic_heavy.plus(&graph_heavy), acoustic_heavy);
    }

    #[test]
    fn times_keeps_the_halves_apart() {
        let a = LatticeWeight::new(1.0, 2.0);
        let b = LatticeWeight::new(0.5, 0.25);
        assert_eq!(a.times(&b), LatticeWeight::new(1.5, 2.25));
        assert_eq!(a.times(&b).total_scaled(0.1), 1.5 + 0.225);
    }

    /// A half-infinite weight would be a second zero: equal to `zero()` under
    /// every operation but not under `==`.
    #[test]
    fn only_a_wholly_infinite_weight_is_a_member() {
        assert!(LatticeWeight::zero().is_member());
        assert!(!LatticeWeight::new(f32::INFINITY, 3.0).is_member());
        assert!(!LatticeWeight::new(3.0, f32::INFINITY).is_member());
        assert!(!LatticeWeight::new(f32::NEG_INFINITY, 0.0).is_member());
        assert!(!LatticeWeight::no_weight().is_member());
    }

    #[test]
    fn dividing_by_zero_has_no_answer() {
        let w = LatticeWeight::new(1.0, 2.0);
        assert!(
            !w.divide(&LatticeWeight::zero(), DivideType::Any)
                .is_member()
        );
        assert_eq!(
            LatticeWeight::zero().divide(&w, DivideType::Any),
            LatticeWeight::zero()
        );
    }

    /// `Eq` demands reflexivity, and `determinize` keys subsets on weights, so
    /// a weight that did not equal itself would quietly split a subset in two.
    #[test]
    fn a_weight_equals_itself_even_when_it_is_not_one() {
        use std::collections::HashSet;

        assert_eq!(LatticeWeight::no_weight(), LatticeWeight::no_weight());
        let mut seen = HashSet::new();
        assert!(seen.insert(LatticeWeight::no_weight()));
        assert!(!seen.insert(LatticeWeight::no_weight()));
        assert!(seen.insert(LatticeWeight::new(1.0, 2.0)));
        assert!(!seen.insert(LatticeWeight::new(1.0, 2.0)));
    }

    /// Upstream writes both halves as `f32` whatever the precision, so that a
    /// lattice written in double is readable in single. The file cannot say
    /// which it was.
    #[test]
    fn it_writes_two_floats_whatever_its_precision() {
        use sicada::weight::WeightIo;

        let mut bytes = Vec::new();
        LatticeWeight::new(1.5, -2.25).write(&mut bytes).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&1.5f32.to_le_bytes());
        expected.extend_from_slice(&(-2.25f32).to_le_bytes());
        assert_eq!(bytes, expected);

        let mut wide = Vec::new();
        LatticeWeight64::new(1.5, -2.25).write(&mut wide).unwrap();
        assert_eq!(wide, expected, "the same bytes in double precision");

        for weight in samples() {
            let mut bytes = Vec::new();
            weight.write(&mut bytes).unwrap();
            assert_eq!(LatticeWeight::read(&mut bytes.as_slice()).unwrap(), weight);
        }
    }

    #[test]
    fn it_reads_back_what_it_prints() {
        for weight in samples() {
            let text = weight.to_string();
            let parsed: LatticeWeight = text.parse().expect(&text);
            assert_eq!(parsed, weight, "{text}");
        }
        assert!("1.0".parse::<LatticeWeight>().is_err());
        assert!("a,b".parse::<LatticeWeight>().is_err());
    }
}
