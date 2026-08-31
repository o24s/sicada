//! The semiring of a *compact* lattice: a cost and the alignment that earned it.
//!
//! A lattice has one arc per frame, so a word spans as many arcs as it took
//! frames to say, and the same word sequence appears once for every way of
//! lining it up against the audio. That is not what anyone wants to read, to
//! rescore, or to count *n*-best over.
//!
//! A compact lattice has one arc per *word*, with the frames it spanned moved
//! into the weight: [`CompactLatticeWeight`] is a [`LatticeWeight`] paired with
//! the sequence of input labels the word consumed. Determinizing over this
//! semiring is what collapses the alignments: two arcs with the same word merge,
//! and ⊕ keeps the better alignment rather than both.
//!
//! This is Kaldi's `CompactLatticeWeightTpl` (`fstext/lattice-weight.h`).
//!
//! It is the same idea as OpenFst's gallic weight, a weight paired with a
//! string, and sicada has one of those. The reason not to use it: the gallic
//! types' ⊕ resolves a disagreement between two label sequences by taking a
//! common prefix, refusing, or keeping a union, and the answer here is none of
//! those. Two alignments of the same word are both correct; the better-scoring
//! one wins outright, and its whole sequence survives. That is a different ⊕,
//! so it is a different semiring.

use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use sicada::arc::ArcLabel;
use sicada::fst_type::WeightType;
use sicada::utils::io::{FstScalar, read_scalar, write_scalar};
use sicada::weight::{
    Divide, DivideType, IDEMPOTENT, IdempotentWeight, LEFT_SEMIRING, LeftSemiring, PATH,
    PathWeight, RIGHT_SEMIRING, RightSemiring, Weight, WeightIo,
};
use smallvec::SmallVec;

use crate::lattice_weight::LatticeWeight;

/// The alignment a compact-lattice arc carries.
///
/// SICADA-OPT: upstream stores this in a `std::vector`, which heap-allocates
/// every time ⊗ concatenates two of them, and determinization does little else.
/// Most are short: an arc's own string is one label long, and a word's is
/// however many frames it took to say.
pub type Alignment<L> = SmallVec<[L; 8]>;

/// A cost and the input labels that earned it.
#[derive(Debug, Clone, Default)]
pub struct CompactLatticeWeight<L: ArcLabel> {
    weight: LatticeWeight,
    alignment: Alignment<L>,
}

impl<L: ArcLabel> CompactLatticeWeight<L> {
    /// A weight from its cost and its alignment.
    ///
    /// The empty alignment is forced when the cost is `zero()`: a semiring has
    /// exactly one zero, and `(zero, [5])` would be a second one. It would
    /// absorb under ⊗ and lose under ⊕ exactly as `(zero, [])` does, but be
    /// unequal to it, so every algorithm that compares against `zero()` would
    /// miss it.
    #[inline]
    pub fn new(weight: LatticeWeight, alignment: Alignment<L>) -> Self {
        if weight == LatticeWeight::zero() {
            return Self::zero();
        }
        Self { weight, alignment }
    }

    /// A weight with no alignment yet.
    #[inline]
    pub fn from_weight(weight: LatticeWeight) -> Self {
        Self::new(weight, Alignment::new())
    }

    /// The cost half.
    #[inline(always)]
    pub fn weight(&self) -> &LatticeWeight {
        &self.weight
    }

    /// The input labels this weight spans, in order.
    #[inline(always)]
    pub fn alignment(&self) -> &[L] {
        &self.alignment
    }

    /// Ordering in the semiring: `Greater` means better.
    ///
    /// The cost decides; a tie goes to the *shorter* alignment, and then to the
    /// lexicographically smaller one. Upstream's reason for preferring the
    /// shorter one is worth keeping: it makes ⊕ a function of its arguments
    /// rather than of the order they arrived in, which determinization relies
    /// on to converge.
    #[inline]
    fn compare(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        match compare_lattice_weights(&self.weight, &other.weight) {
            Equal => {}
            ordering => return ordering,
        }
        match other.alignment.len().cmp(&self.alignment.len()) {
            Equal => {}
            ordering => return ordering,
        }
        // Both lengths are equal, so this is an ordinary lexicographic
        // comparison, reversed, since smaller labels are "greater" here for the
        // same reason smaller costs are.
        other.alignment.cmp(&self.alignment)
    }
}

/// [`LatticeWeight`]'s own ordering, which it keeps private.
#[inline]
fn compare_lattice_weights(lhs: &LatticeWeight, rhs: &LatticeWeight) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    let (mine, theirs) = (lhs.total(), rhs.total());
    if mine < theirs {
        Greater
    } else if mine > theirs {
        Less
    } else if lhs.graph < rhs.graph {
        Greater
    } else if lhs.graph > rhs.graph {
        Less
    } else {
        Equal
    }
}

impl<L: ArcLabel> PartialEq for CompactLatticeWeight<L> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight && self.alignment == other.alignment
    }
}

impl<L: ArcLabel> Eq for CompactLatticeWeight<L> {}

impl<L: ArcLabel> Hash for CompactLatticeWeight<L> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.weight.hash(state);
        self.alignment.as_slice().hash(state);
    }
}

impl<L: ArcLabel> fmt::Display for CompactLatticeWeight<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Kaldi's text form: the two costs, then the alignment joined by `_`.
        write!(f, "{},", self.weight)?;
        for (index, label) in self.alignment.iter().enumerate() {
            if index > 0 {
                write!(f, "_")?;
            }
            write!(f, "{label}")?;
        }
        Ok(())
    }
}

impl<L: ArcLabel> FromStr for CompactLatticeWeight<L> {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (graph, rest) = s.split_once(',').ok_or_else(|| {
            format!(
                "{}: expected `graph,acoustic,alignment`, got {s:?}",
                Self::type_name()
            )
        })?;
        let (acoustic, labels) = rest.split_once(',').ok_or_else(|| {
            format!(
                "{}: expected `graph,acoustic,alignment`, got {s:?}",
                Self::type_name()
            )
        })?;
        let weight: LatticeWeight = format!("{graph},{acoustic}").parse()?;

        let mut alignment = Alignment::new();
        for label in labels.split('_').filter(|piece| !piece.is_empty()) {
            alignment.push(
                label
                    .trim()
                    .parse()
                    .map_err(|_| format!("{}: {label:?} is not a label", Self::type_name()))?,
            );
        }
        Ok(Self::new(weight, alignment))
    }
}

impl<L: ArcLabel> Weight for CompactLatticeWeight<L> {
    type ReverseWeight = Self;

    #[inline]
    fn zero() -> Self {
        Self {
            weight: LatticeWeight::zero(),
            alignment: Alignment::new(),
        }
    }

    #[inline]
    fn one() -> Self {
        Self {
            weight: LatticeWeight::one(),
            alignment: Alignment::new(),
        }
    }

    #[inline]
    fn no_weight() -> Self {
        Self {
            weight: LatticeWeight::no_weight(),
            alignment: Alignment::new(),
        }
    }

    /// Kaldi's name for this weight, as recorded in an FST file header.
    ///
    /// It carries the *sizes*: `"compact"`, then the inner weight's name
    /// (`"lattice4"` for a pair of `f32`), then the width of one alignment
    /// label in bytes. So a lattice over `i32` labels is `compactlattice44`,
    /// and one over `i64` labels is `compactlattice48`.
    #[inline]
    fn type_name() -> WeightType {
        WeightType::new_dynamic(format!(
            "compact{}{}",
            LatticeWeight::type_name(),
            std::mem::size_of::<L>()
        ))
    }

    #[inline(always)]
    fn properties() -> u64 {
        // Not commutative: ⊗ concatenates alignments, and `a·b` is not `b·a`.
        LEFT_SEMIRING | RIGHT_SEMIRING | PATH | IDEMPOTENT
    }

    #[inline]
    fn plus(&self, rhs: &Self) -> Self {
        if !self.is_member() || !rhs.is_member() {
            return Self::no_weight();
        }
        if self.compare(rhs).is_ge() {
            self.clone()
        } else {
            rhs.clone()
        }
    }

    #[inline]
    fn times(&self, rhs: &Self) -> Self {
        if !self.is_member() || !rhs.is_member() {
            return Self::no_weight();
        }
        let weight = self.weight.times(&rhs.weight);
        if weight == LatticeWeight::zero() {
            return Self::zero();
        }
        let mut alignment = Alignment::with_capacity(self.alignment.len() + rhs.alignment.len());
        alignment.extend_from_slice(&self.alignment);
        alignment.extend_from_slice(&rhs.alignment);
        Self { weight, alignment }
    }

    #[inline]
    fn reverse(&self) -> Self::ReverseWeight {
        let mut alignment = self.alignment.clone();
        alignment.reverse();
        Self {
            weight: self.weight.reverse(),
            alignment,
        }
    }

    #[inline]
    fn is_member(&self) -> bool {
        // The zero is unique, so an alignment attached to one is not a weight.
        self.weight.is_member()
            && (self.weight != LatticeWeight::zero() || self.alignment.is_empty())
    }

    #[inline]
    fn approx_equal(&self, other: &Self, delta: f32) -> bool {
        self.weight.approx_equal(&other.weight, delta) && self.alignment == other.alignment
    }

    #[inline]
    fn quantize(&self, delta: f32) -> Self {
        Self {
            weight: self.weight.quantize(delta),
            alignment: self.alignment.clone(),
        }
    }
}

/// The bytes Kaldi's `CompactLatticeWeightTpl::Write` produces: the cost, then
/// the alignment's length as an `i32`, then the labels.
impl<L: ArcLabel + FstScalar> WeightIo for CompactLatticeWeight<L> {
    fn read<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let weight = LatticeWeight::read(reader)?;
        let size: i32 = read_scalar(reader)?;
        if size < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: an alignment of {size} labels", Self::type_name()),
            ));
        }
        let mut alignment = Alignment::with_capacity(size as usize);
        for _ in 0..size {
            alignment.push(read_scalar(reader)?);
        }
        Ok(Self::new(weight, alignment))
    }

    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.weight.write(writer)?;
        write_scalar(writer, self.alignment.len() as i32)?;
        for &label in &self.alignment {
            write_scalar(writer, label)?;
        }
        Ok(())
    }
}

impl<L: ArcLabel> Divide for CompactLatticeWeight<L> {
    /// Undoes a ⊗ from one side: the costs subtract and the alignment gives
    /// back the part `rhs` did not contribute.
    ///
    /// SICADA-DIVERGE: upstream aborts the process on every case this returns
    /// [`Weight::no_weight`] for: dividing by zero, an alignment `rhs` is not a
    /// prefix or suffix of, or `DivideType::Any`, which has no answer when ⊗
    /// does not commute. sicada already has a value for "this division has no
    /// result", and the algorithms that divide already test for it, so there is
    /// nothing to gain by stopping.
    fn divide(&self, rhs: &Self, side: DivideType) -> Self {
        if !self.is_member() || !rhs.is_member() {
            return Self::no_weight();
        }
        if rhs.weight == LatticeWeight::zero() {
            return Self::no_weight();
        }
        if self.weight == LatticeWeight::zero() {
            return Self::zero();
        }
        if rhs.alignment.len() > self.alignment.len() {
            return Self::no_weight();
        }

        let weight = self.weight.divide(&rhs.weight, side);
        if !weight.is_member() {
            return Self::no_weight();
        }
        let split = self.alignment.len() - rhs.alignment.len();
        let alignment = match side {
            DivideType::Left => {
                if self.alignment[..rhs.alignment.len()] != rhs.alignment[..] {
                    return Self::no_weight();
                }
                Alignment::from_slice(&self.alignment[rhs.alignment.len()..])
            }
            DivideType::Right => {
                if self.alignment[split..] != rhs.alignment[..] {
                    return Self::no_weight();
                }
                Alignment::from_slice(&self.alignment[..split])
            }
            // Which end to take the alignment off is exactly what `Any` does
            // not say, and ⊗ here does not commute.
            DivideType::Any => return Self::no_weight(),
        };
        Self::new(weight, alignment)
    }
}

impl<L: ArcLabel> LeftSemiring for CompactLatticeWeight<L> {}
impl<L: ArcLabel> RightSemiring for CompactLatticeWeight<L> {}
impl<L: ArcLabel> IdempotentWeight for CompactLatticeWeight<L> {}
impl<L: ArcLabel> PathWeight for CompactLatticeWeight<L> {}

/// An arc of a compact lattice: a word on both sides, the cost and the
/// alignment in the weight.
pub type CompactLatticeArc<A> = sicada::arc::ArcTpl<
    CompactLatticeWeight<<A as sicada::arc::Arc>::Label>,
    <A as sicada::arc::Arc>::Label,
    <A as sicada::arc::Arc>::StateId,
>;

#[cfg(test)]
mod tests {
    use super::*;
    use sicada::weight::axioms;

    type W = CompactLatticeWeight<i32>;

    fn aligned(graph: f32, acoustic: f32, labels: &[i32]) -> W {
        W::new(
            LatticeWeight::new(graph, acoustic),
            Alignment::from_slice(labels),
        )
    }

    fn samples() -> Vec<W> {
        vec![
            aligned(0.0, 0.0, &[]),
            aligned(1.0, 0.5, &[7]),
            aligned(0.25, 2.0, &[7, 8]),
            aligned(2.0, -1.0, &[9]),
            aligned(1.0, 0.5, &[8]),
            W::zero(),
        ]
    }

    #[test]
    fn it_is_the_semiring_it_says_it_is() {
        axioms::check(&samples());
        axioms::check_divide(&samples());
    }

    /// The claim it deliberately does *not* make. ⊗ concatenates, so the two
    /// orders differ, and an algorithm that assumed otherwise would reorder a
    /// word's frames.
    #[test]
    fn it_does_not_claim_to_commute() {
        assert_eq!(W::properties() & sicada::weight::COMMUTATIVE, 0);
        let a = aligned(0.0, 0.0, &[1]);
        let b = aligned(0.0, 0.0, &[2]);
        assert_ne!(a.times(&b), b.times(&a));
        assert_eq!(a.times(&b).alignment(), &[1, 2]);
    }

    /// The whole point: two alignments of the same word, and the better one
    /// wins outright rather than being merged with the loser.
    #[test]
    fn plus_keeps_the_better_alignment_whole() {
        let cheap = aligned(1.0, 1.0, &[5, 5, 6]);
        let dear = aligned(1.0, 3.0, &[5, 6, 6]);
        assert_eq!(cheap.plus(&dear), cheap);
        assert_eq!(dear.plus(&cheap), cheap);
        assert_eq!(cheap.plus(&dear).alignment(), &[5, 5, 6]);
    }

    /// A tie has to resolve the same way whichever order the two arrive in, or
    /// determinization would not converge.
    #[test]
    fn a_tie_prefers_the_shorter_alignment() {
        let short = aligned(1.0, 1.0, &[5]);
        let long = aligned(1.0, 1.0, &[5, 5]);
        assert_eq!(short.plus(&long), short);
        assert_eq!(long.plus(&short), short);

        let low = aligned(1.0, 1.0, &[4, 9]);
        let high = aligned(1.0, 1.0, &[5, 5]);
        assert_eq!(low.plus(&high), low);
        assert_eq!(high.plus(&low), low);
    }

    /// A second zero would be absorbing under ⊗ and losing under ⊕ just as the
    /// real one is, but unequal to it, so `== zero()` would start missing it.
    #[test]
    fn the_zero_is_unique() {
        assert_eq!(
            W::new(LatticeWeight::zero(), Alignment::from_slice(&[5])),
            W::zero()
        );
        assert!(W::zero().is_member());
        assert!(
            !W {
                weight: LatticeWeight::zero(),
                alignment: Alignment::from_slice(&[5]),
            }
            .is_member(),
            "one built behind `new`'s back is not a weight"
        );
        assert_eq!(aligned(1.0, 1.0, &[3]).times(&W::zero()), W::zero());
    }

    #[test]
    fn dividing_takes_the_alignment_off_the_named_end() {
        let whole = aligned(3.0, 3.0, &[1, 2, 3]);
        let head = aligned(1.0, 1.0, &[1]);
        let tail = aligned(1.0, 1.0, &[3]);

        let rest = whole.divide(&head, DivideType::Left);
        assert_eq!(rest.alignment(), &[2, 3]);
        assert_eq!(head.times(&rest), whole);

        let start = whole.divide(&tail, DivideType::Right);
        assert_eq!(start.alignment(), &[1, 2]);
        assert_eq!(start.times(&tail), whole);

        // The alignment has to actually be there to be taken off.
        assert!(!whole.divide(&tail, DivideType::Left).is_member());
        assert!(!whole.divide(&head, DivideType::Right).is_member());
        // And `Any` cannot say which end.
        assert!(!whole.divide(&head, DivideType::Any).is_member());
    }

    #[test]
    fn reversing_reverses_the_alignment() {
        let w = aligned(1.0, 2.0, &[1, 2, 3]);
        assert_eq!(w.reverse().alignment(), &[3, 2, 1]);
        assert_eq!(w.reverse().reverse(), w);
    }

    #[test]
    fn it_reads_back_what_it_prints() {
        for weight in samples() {
            let text = weight.to_string();
            let parsed: W = text.parse().expect(&text);
            assert_eq!(parsed, weight, "{text}");
        }
        assert_eq!(aligned(1.0, 2.0, &[3, 4]).to_string(), "1,2,3_4");
        assert!("1,2".parse::<W>().is_err());
    }

    /// The name goes into an FST file header, and it is Kaldi's, sizes and
    /// all: `compact` + the cost pair's name + how wide one label is.
    #[test]
    fn its_type_name_is_kaldis() {
        assert_eq!(W::type_name().as_str(), "compactlattice44");
        assert_eq!(
            CompactLatticeWeight::<i64>::type_name().as_str(),
            "compactlattice48"
        );
    }

    /// A lattice written here should be one Kaldi reads, which means the bytes
    /// are its bytes: the two costs, then the alignment's length, then the
    /// labels.
    #[test]
    fn it_writes_the_bytes_upstream_writes() {
        let mut bytes = Vec::new();
        aligned(1.0, 2.0, &[7, 8]).write(&mut bytes).unwrap();

        let mut expected = Vec::new();
        expected.extend_from_slice(&1.0f32.to_le_bytes());
        expected.extend_from_slice(&2.0f32.to_le_bytes());
        expected.extend_from_slice(&2i32.to_le_bytes());
        expected.extend_from_slice(&7i32.to_le_bytes());
        expected.extend_from_slice(&8i32.to_le_bytes());
        assert_eq!(bytes, expected);

        for weight in samples() {
            let mut bytes = Vec::new();
            weight.write(&mut bytes).unwrap();
            let read = W::read(&mut bytes.as_slice()).unwrap();
            assert_eq!(read, weight);
        }
    }

    #[test]
    fn it_can_be_a_key() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        assert!(seen.insert(aligned(1.0, 2.0, &[3])));
        assert!(!seen.insert(aligned(1.0, 2.0, &[3])));
        assert!(seen.insert(aligned(1.0, 2.0, &[4])));
    }
}
