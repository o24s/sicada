use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::str::FromStr;

use crate::fst_type::{ArcType, WeightType};
use crate::weight::Weight;

/// A trait defining the mathematical and behavioral properties required for an FST Label.
pub trait ArcLabel:
    Copy + Clone + PartialEq + Eq + PartialOrd + Ord + Hash + Debug + Display + FromStr + 'static
{
    /// Returns the symbol representing the epsilon (empty) transition.
    /// In OpenFst, this is universally represented as `0`.
    fn epsilon() -> Self;

    /// Returns the symbol representing an invalid or absent label.
    /// In OpenFst, this is typically represented as `-1`.
    fn no_label() -> Self;

    /// The label as a symbol table key, or `None` if it does not fit one.
    ///
    /// A [`SymbolTable`](crate::symbol_table::SymbolTable) keys on `i64`, which
    /// every label type used in practice fits; the `Option` is for a `u64`
    /// label above `i64::MAX`, which no table can hold.
    fn to_i64(self) -> Option<i64>;

    /// A label from a symbol table key, or `None` if the key does not fit this
    /// label type, such as a negative key for an unsigned label or one too
    /// large.
    fn from_i64(key: i64) -> Option<Self>;
}

impl ArcLabel for i32 {
    #[inline(always)]
    fn epsilon() -> Self {
        0
    }
    #[inline(always)]
    fn no_label() -> Self {
        -1
    }
    #[inline(always)]
    fn to_i64(self) -> Option<i64> {
        Some(self as i64)
    }

    #[inline(always)]
    fn from_i64(key: i64) -> Option<Self> {
        Self::try_from(key).ok()
    }
}

impl ArcLabel for i64 {
    #[inline(always)]
    fn epsilon() -> Self {
        0
    }
    #[inline(always)]
    fn no_label() -> Self {
        -1
    }
    #[inline(always)]
    fn to_i64(self) -> Option<i64> {
        Some(self)
    }

    #[inline(always)]
    fn from_i64(key: i64) -> Option<Self> {
        Some(key)
    }
}

impl ArcLabel for u32 {
    #[inline(always)]
    fn epsilon() -> Self {
        0
    }
    #[inline(always)]
    fn no_label() -> Self {
        u32::MAX
    }
    #[inline(always)]
    fn to_i64(self) -> Option<i64> {
        Some(self as i64)
    }

    #[inline(always)]
    fn from_i64(key: i64) -> Option<Self> {
        Self::try_from(key).ok()
    }
}

impl ArcLabel for usize {
    #[inline(always)]
    fn epsilon() -> Self {
        0
    }
    #[inline(always)]
    fn no_label() -> Self {
        usize::MAX
    }
    #[inline(always)]
    fn to_i64(self) -> Option<i64> {
        i64::try_from(self).ok()
    }

    #[inline(always)]
    fn from_i64(key: i64) -> Option<Self> {
        Self::try_from(key).ok()
    }
}

/// A trait defining the properties required for an FST State ID.
pub trait ArcStateId: Copy + PartialEq + Eq + PartialOrd + Ord + Hash + Debug {
    /// Returns the symbol representing an invalid or absent state ID.
    fn no_state() -> Self;

    fn as_usize(&self) -> usize;
    fn from_usize(n: usize) -> Self;
}

impl ArcStateId for i32 {
    #[inline(always)]
    fn no_state() -> Self {
        -1
    }
    #[inline(always)]
    fn as_usize(&self) -> usize {
        debug_assert!(*self >= 0, "Attempted to use negative state ID as index");
        *self as usize
    }
    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        n as i32
    }
}

impl ArcStateId for i8 {
    #[inline(always)]
    fn no_state() -> Self {
        -1
    }
    #[inline(always)]
    fn as_usize(&self) -> usize {
        debug_assert!(*self >= 0, "Attempted to use negative state ID as index");
        *self as usize
    }
    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        n as i8
    }
}

impl ArcStateId for u32 {
    #[inline(always)]
    fn no_state() -> Self {
        u32::MAX
    }
    #[inline(always)]
    fn as_usize(&self) -> usize {
        *self as usize
    }
    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        n as u32
    }
}

impl ArcStateId for usize {
    #[inline(always)]
    fn no_state() -> Self {
        usize::MAX
    }
    #[inline(always)]
    fn as_usize(&self) -> usize {
        *self
    }
    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        n
    }
}

pub trait Arc: Clone + PartialEq + Debug {
    type Weight: Weight;
    type Label: ArcLabel;
    type StateId: ArcStateId;

    /// The arc of this one's FST read backwards.
    ///
    /// SICADA-DIVERGE: upstream's algorithms that walk an FST in reverse take
    /// the reverse arc as a second template argument
    /// (`ShortestPath<Arc, RevArc>`), and so did this port. It is determined by
    /// the arc, having the same labels and state ids and the weight's reverse,
    /// but a *type parameter* that appears in no argument cannot be inferred, so
    /// every call site had to spell it: `shortest_path::<StdArc, StdArc, _,
    /// _>(…)`. Naming it here makes fifteen signatures shorter and every one of
    /// their call sites turbofish-free.
    type Reverse: Arc<
            Label = Self::Label,
            StateId = Self::StateId,
            Weight = <Self::Weight as Weight>::ReverseWeight,
        >;

    fn new(
        ilabel: Self::Label,
        olabel: Self::Label,
        weight: Self::Weight,
        nextstate: Self::StateId,
    ) -> Self;

    fn ilabel(&self) -> Self::Label;
    fn olabel(&self) -> Self::Label;
    fn weight(&self) -> &Self::Weight;
    fn nextstate(&self) -> Self::StateId;

    /// Returns the strongly-typed name of the arc, used for FST file headers.
    fn type_name() -> ArcType;
}

/// An arc: two labels, a weight, and where it leads.
///
/// `#[repr(C)]` is part of the file format, not a hint. `ConstFst` and
/// `CompactFst` write their arc arrays to disk as a block of memory and read
/// them back the same way, so the field order and padding here have to match
/// what OpenFst's `ArcTpl` produces, which is declaration order, since that
/// struct is standard-layout. Rust is otherwise free to reorder fields.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ArcTpl<W, L = i32, S = i32> {
    pub ilabel: L,
    pub olabel: L,
    pub weight: W,
    pub nextstate: S,
}

impl<W, L, S> Arc for ArcTpl<W, L, S>
where
    W: Weight,
    L: ArcLabel,
    S: ArcStateId,
{
    type Weight = W;
    type Label = L;
    type StateId = S;
    type Reverse = ArcTpl<W::ReverseWeight, L, S>;

    #[inline(always)]
    fn new(ilabel: L, olabel: L, weight: W, nextstate: S) -> Self {
        Self {
            ilabel,
            olabel,
            weight,
            nextstate,
        }
    }

    #[inline(always)]
    fn ilabel(&self) -> Self::Label {
        self.ilabel
    }
    #[inline(always)]
    fn olabel(&self) -> Self::Label {
        self.olabel
    }
    #[inline(always)]
    fn weight(&self) -> &Self::Weight {
        &self.weight
    }
    #[inline(always)]
    fn nextstate(&self) -> Self::StateId {
        self.nextstate
    }

    #[inline]
    fn type_name() -> ArcType {
        let w_type = W::type_name();

        // OpenFst historically maps "tropical" to "standard" for Arc type names.
        if w_type == WeightType::TROPICAL {
            ArcType::STANDARD
        } else {
            // Converts the WeightType string representation dynamically into an ArcType.
            ArcType::new_dynamic(w_type.to_string())
        }
    }
}

pub type StdArc = ArcTpl<crate::float_weight::TropicalWeight>;
pub type Std64Arc = ArcTpl<crate::float_weight::TropicalWeight64>;
pub type LogArc = ArcTpl<crate::float_weight::LogWeight>;
pub type Log64Arc = ArcTpl<crate::float_weight::Log64Weight>;
pub type RealArc = ArcTpl<crate::float_weight::RealWeight>;
pub type Real64Arc = ArcTpl<crate::float_weight::Real64Weight>;
pub type MinMaxArc = ArcTpl<crate::float_weight::MinMaxWeight>;
pub type MinMax64Arc = ArcTpl<crate::float_weight::MinMaxWeight64>;
pub type SignedLogArc = ArcTpl<crate::signed_log_weight::SignedLogWeight>;
pub type SignedLog64Arc = ArcTpl<crate::signed_log_weight::SignedLog64Weight>;

// SICADA-DIVERGE: upstream names one alias per (weight, rank) pair
// (`Power3TropicalArc`, `LexicographicMinMaxTropicalArc`, and so on) because a
// C++ template argument list is unpleasant to repeat at every use. The generic
// wrappers below (`PowerArc<A, N>`, `GallicArc<A, G>`, `ReverseArc<A>`) and
// `ArcTpl<W>` say the same thing without the combinatorics, so the aliases the
// old bindings carried are gone rather than commented out: a name that resolves
// to nothing is worse than no name. Write `PowerArc<StdArc, 3>` for what
// upstream calls `Power3TropicalArc`, and `ArcTpl<LexicographicWeight>` for
// `LexicographicArc`.

// ---------------------------------------------------------------------------
// Arc wrappers
// ---------------------------------------------------------------------------
//
// Upstream derives these from `ArcTpl` purely to give each a distinct `Type()`
// string, which is the name an FST file header records. A Rust type alias cannot
// change an associated function, so each is a newtype that delegates everything
// except the name.

/// Emits the `Arc` impl body, which is pure delegation for every wrapper.
macro_rules! delegate_arc {
    ($weight:ty, $base:ty, $reverse:ty, $type_name:expr) => {
        type Weight = $weight;
        type Label = <$base as Arc>::Label;
        type StateId = <$base as Arc>::StateId;
        // Each wrapper names its own reverse rather than flattening to an
        // `ArcTpl` over the reversed weight. The two would satisfy the same
        // bound, but not answer `type_name()` the same way: a flattened
        // `GallicArc` reports its *weight*'s name, `"right_gallic"`, where the
        // arc's name is `"right_gallic_standard"`. An FST file header records
        // that string, so the structure has to survive the reversal.
        type Reverse = $reverse;

        #[inline(always)]
        fn new(
            ilabel: Self::Label,
            olabel: Self::Label,
            weight: Self::Weight,
            nextstate: Self::StateId,
        ) -> Self {
            Self {
                inner: ArcTpl::new(ilabel, olabel, weight, nextstate),
            }
        }

        #[inline(always)]
        fn ilabel(&self) -> Self::Label {
            self.inner.ilabel
        }
        #[inline(always)]
        fn olabel(&self) -> Self::Label {
            self.inner.olabel
        }
        #[inline(always)]
        fn weight(&self) -> &Self::Weight {
            &self.inner.weight
        }
        #[inline(always)]
        fn nextstate(&self) -> Self::StateId {
            self.inner.nextstate
        }

        #[inline]
        fn type_name() -> ArcType {
            let base = <$base as Arc>::type_name();
            ArcType::new_dynamic($type_name(base.as_str()))
        }
    };
}

/// An arc whose weight is the reverse of `A`'s.
///
/// Traversing an FST backwards has to reverse a non-commutative weight along
/// with the direction; this is the arc type that results.
#[derive(Debug, Clone, PartialEq)]
pub struct ReverseArc<A: Arc> {
    inner: ArcTpl<<A::Weight as Weight>::ReverseWeight, A::Label, A::StateId>,
}

/// Reversing twice is the identity, so `ReverseArc<A>`'s own reverse is `A` --
/// which holds exactly when the weight's reversal is an involution. Every
/// algorithm that walks an FST backwards already states that bound, so
/// requiring it here narrows nothing they could have used.
impl<A: Arc> Arc for ReverseArc<A>
where
    <A::Weight as Weight>::ReverseWeight: Weight<ReverseWeight = A::Weight>,
{
    delegate_arc!(
        <A::Weight as Weight>::ReverseWeight,
        A,
        A,
        |base: &str| format!("reverse_{base}")
    );
}

/// An arc carrying the `N`-fold Cartesian power of `A`'s weight.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerArc<A: Arc, const N: usize> {
    inner: ArcTpl<crate::weights::power_weight::PowerWeight<A::Weight, N>, A::Label, A::StateId>,
}

impl<A: Arc, const N: usize> Arc for PowerArc<A, N> {
    delegate_arc!(
        crate::weights::power_weight::PowerWeight<A::Weight, N>,
        A,
        PowerArc<A::Reverse, N>,
        |base: &str| format!("{base}_^{N}")
    );
}

/// An arc carrying a gallic weight: `A`'s weight paired with a label sequence.
///
/// This is the arc type determinization and weight factoring work over, where a
/// transducer's output labels have to travel with its costs.
#[derive(Debug, Clone, PartialEq)]
pub struct GallicArc<A: Arc, G: crate::weights::string_weight::GallicTypeMarker> {
    inner: ArcTpl<
        crate::weights::string_weight::GallicWeight<A::Label, A::Weight, G>,
        A::Label,
        A::StateId,
    >,
}

// No `where GallicWeight<..>: Weight` clause here, deliberately. It was
// redundant, since the blanket impl covers every `L: ArcLabel, W: Weight,
// G: GallicTypeMarker`, and it was harmful: given the bound as a where-clause
// the compiler answers `ReverseWeight` from the clause rather than from the
// impl, so it never learns that the reverse of a gallic weight is again a
// gallic weight, and `Reverse = GallicArc<..>` fails to typecheck.
impl<A: Arc, G: crate::weights::string_weight::GallicTypeMarker> Arc for GallicArc<A, G> {
    delegate_arc!(
        crate::weights::string_weight::GallicWeight<A::Label, A::Weight, G>,
        A,
        GallicArc<A::Reverse, G::Reverse>,
        |base: &str| format!("{}{base}", G::ARC_PREFIX)
    );
}

/// The arc type of an empty FST archive.
///
/// Uninhabited, since [`ErrorWeight`](crate::weights::error_weight::ErrorWeight)
/// is; it exists only so that such an archive has a type name to record.
pub type ErrorArc = ArcTpl<crate::weights::error_weight::ErrorWeight>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::float_weight::{LogWeight, TropicalWeight};

    /// A wrapper arc's reverse keeps the wrapper, and so keeps its name. The
    /// flattened form, an `ArcTpl` over the reversed weight, satisfies the same
    /// bound but reports the *weight*'s name, `"right_gallic"`, which is not
    /// what an FST file header would carry for that arc.
    #[test]
    fn a_reversed_wrapper_arc_keeps_its_type_name() {
        use crate::weights::string_weight::{GallicLeft, GallicRight};

        type Left = GallicArc<StdArc, GallicLeft>;
        assert_eq!(Left::type_name().as_str(), "left_gallic_standard");
        assert_eq!(
            <Left as Arc>::Reverse::type_name().as_str(),
            "right_gallic_standard",
            "reversing a left-gallic arc gives a right-gallic one, not a bare weight"
        );

        type Right = GallicArc<StdArc, GallicRight>;
        assert_eq!(
            <Right as Arc>::Reverse::type_name().as_str(),
            "left_gallic_standard"
        );

        assert_eq!(
            <PowerArc<StdArc, 3> as Arc>::Reverse::type_name().as_str(),
            "standard_^3"
        );
        assert_eq!(
            <ReverseArc<StdArc> as Arc>::Reverse::type_name().as_str(),
            "standard",
            "reversing twice is the identity"
        );
    }

    /// The name in an FST file header comes from the arc type, so these strings
    /// are part of the binary format.
    #[test]
    fn arc_type_names_match_openfst() {
        assert_eq!(StdArc::type_name().as_str(), "standard");
        assert_eq!(LogArc::type_name().as_str(), "log");
        assert_eq!(Log64Arc::type_name().as_str(), "log64");
        assert_eq!(RealArc::type_name().as_str(), "real");
        assert_eq!(MinMaxArc::type_name().as_str(), "minmax");
        // Upstream builds this from its two components:
        // "signed_log_" + W1::Type() + "_" + W2::Type().
        assert_eq!(
            SignedLogArc::type_name().as_str(),
            "signed_log_tropical_log"
        );
        assert_eq!(
            SignedLog64Arc::type_name().as_str(),
            "signed_log_tropical_log64"
        );
    }

    /// Tropical is the odd one out: its arc is called "standard", not
    /// "tropical". Getting this wrong makes every standard FST unreadable.
    #[test]
    fn the_tropical_arc_is_called_standard() {
        assert_eq!(TropicalWeight::type_name().as_str(), "tropical");
        assert_eq!(StdArc::type_name().as_str(), "standard");
        // Every other arc takes its weight's name unchanged.
        assert_eq!(
            LogArc::type_name().as_str(),
            LogWeight::type_name().as_str()
        );
    }

    #[test]
    fn the_wrapper_arcs_decorate_the_base_name() {
        assert_eq!(
            ReverseArc::<StdArc>::type_name().as_str(),
            "reverse_standard"
        );
        assert_eq!(PowerArc::<StdArc, 3>::type_name().as_str(), "standard_^3");
        assert_eq!(
            GallicArc::<StdArc, crate::weights::string_weight::GallicLeft>::type_name().as_str(),
            "left_gallic_standard"
        );
        assert_eq!(
            GallicArc::<StdArc, crate::weights::string_weight::GallicMin>::type_name().as_str(),
            "min_gallic_standard"
        );
    }

    #[test]
    fn an_arc_carries_its_four_fields() {
        let arc = StdArc::new(1, 2, TropicalWeight(3.5), 4);
        assert_eq!(arc.ilabel(), 1);
        assert_eq!(arc.olabel(), 2);
        assert_eq!(arc.weight(), &TropicalWeight(3.5));
        assert_eq!(arc.nextstate(), 4);
    }

    #[test]
    fn a_wrapper_arc_carries_its_fields_too() {
        let arc = ReverseArc::<StdArc>::new(1, 2, TropicalWeight(3.5), 4);
        assert_eq!(arc.ilabel(), 1);
        assert_eq!(arc.olabel(), 2);
        assert_eq!(arc.nextstate(), 4);
    }

    /// Epsilon is zero and the absent label is -1, universally in OpenFst.
    #[test]
    fn the_special_labels_are_what_openfst_uses() {
        assert_eq!(<i32 as ArcLabel>::epsilon(), 0);
        assert_eq!(<i32 as ArcLabel>::no_label(), -1);
        assert_eq!(<i64 as ArcLabel>::epsilon(), 0);
        assert_eq!(<i64 as ArcLabel>::no_label(), -1);
        // Unsigned label types cannot hold -1, so they use their maximum.
        assert_eq!(<u32 as ArcLabel>::epsilon(), 0);
        assert_ne!(<u32 as ArcLabel>::no_label(), 0);
    }

    /// An arc is the unit of storage in every FST, so its size is a memory
    /// budget, not an implementation detail.
    #[test]
    fn the_standard_arc_is_four_words() {
        assert_eq!(size_of::<StdArc>(), 16);
        assert_eq!(align_of::<StdArc>(), 4);
    }
}
