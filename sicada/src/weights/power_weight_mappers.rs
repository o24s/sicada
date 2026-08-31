//! Conversions to and from [`PowerWeight`] and [`SparsePowerWeight`].
//!
//! Port of OpenFst's `power-weight-mappers.h`. These are weight-level functions
//! meant to be handed to arc mapping, so they are written against a small trait,
//! [`ComponentWeightVector`], that both power weight flavours implement. It is
//! the same duck typing upstream gets from templates.
//!
//! `TransformPowerWeightMapper` has no counterpart here. It exists upstream only
//! to wrap a callable in a class with an `operator()`, which a Rust closure
//! already is; pass the closure directly.

use crate::weight::Weight;
use crate::weights::power_weight::PowerWeight;
use crate::weights::sparse_power_weight::SparsePowerWeight;

use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

/// A weight that behaves as a vector of component weights addressed by an index.
///
/// Implemented by [`PowerWeight`] (dense, rank fixed at compile time) and
/// [`SparsePowerWeight`] (sparse, with a default value for absent components).
pub trait ComponentWeightVector: Weight {
    /// How a component is addressed.
    type Index: Copy;
    /// The weight stored in each component.
    type Component: Weight;

    /// Builds a vector holding `weight` at `index` and `default_weight` elsewhere.
    fn from_component(
        index: Self::Index,
        weight: Self::Component,
        default_weight: Self::Component,
    ) -> Self;

    /// Returns the component at `index`.
    fn component(&self, index: Self::Index) -> Self::Component;
}

impl<W: Weight, const N: usize> ComponentWeightVector for PowerWeight<W, N> {
    type Index = usize;
    type Component = W;

    #[inline]
    fn from_component(index: usize, weight: W, default_weight: W) -> Self {
        Self::from_component(index, weight, default_weight)
    }

    #[inline]
    fn component(&self, index: usize) -> W {
        self.value(index).clone()
    }
}

impl<W, K> ComponentWeightVector for SparsePowerWeight<W, K>
where
    W: Weight,
    K: Copy + Ord + Hash + FromStr + fmt::Display + fmt::Debug + 'static,
{
    type Index = K;
    type Component = W;

    #[inline]
    fn from_component(key: K, weight: W, default_weight: W) -> Self {
        Self::from_component(key, weight, default_weight)
    }

    #[inline]
    fn component(&self, key: K) -> W {
        self.inner.value(key).clone()
    }
}

/// Maps a weight into one component of a power weight, leaving the rest zero.
///
/// The component conversion is expressed with `Into` rather than a bespoke
/// conversion trait; upstream reinterprets the raw stored value via
/// `ToPowerWeight::Weight(w.Value())`, which is `weight.h`'s `WeightConvert` and
/// which specializes to the identity for equal types, exactly as Rust's
/// reflexive `From<T> for T` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToPowerWeightMapper<To: ComponentWeightVector> {
    index: To::Index,
}

impl<To: ComponentWeightVector> ToPowerWeightMapper<To> {
    /// Maps into component `index`.
    pub fn new(index: To::Index) -> Self {
        Self { index }
    }

    /// Applies the mapping.
    pub fn map<From>(&self, weight: &From) -> To
    where
        From: Weight + Clone + Into<To::Component>,
    {
        To::from_component(
            self.index,
            weight.clone().into(),
            <To::Component as Weight>::zero(),
        )
    }
}

/// Maps one component of a power weight back out to a plain weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FromPowerWeightMapper<From: ComponentWeightVector> {
    index: From::Index,
}

impl<From: ComponentWeightVector> FromPowerWeightMapper<From> {
    /// Reads component `index`.
    pub fn new(index: From::Index) -> Self {
        Self { index }
    }

    /// Applies the mapping.
    pub fn map<To>(&self, weight: &From) -> To
    where
        To: Weight,
        From::Component: Into<To>,
    {
        weight.component(self.index).into()
    }
}

/// Projects one component of a power weight onto another component, filling the
/// rest with `default_weight`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectPowerWeightMapper<P: ComponentWeightVector> {
    from_index: P::Index,
    to_index: P::Index,
    default_weight: P::Component,
}

impl<P: ComponentWeightVector> ProjectPowerWeightMapper<P> {
    /// Projects `from_index` onto `to_index`, filling with `Zero`.
    pub fn new(from_index: P::Index, to_index: P::Index) -> Self {
        Self::with_default(from_index, to_index, <P::Component as Weight>::zero())
    }

    /// Projects `from_index` onto `to_index`, filling with `default_weight`.
    pub fn with_default(
        from_index: P::Index,
        to_index: P::Index,
        default_weight: P::Component,
    ) -> Self {
        Self {
            from_index,
            to_index,
            default_weight,
        }
    }

    /// Applies the mapping.
    pub fn map(&self, weight: &P) -> P {
        P::from_component(
            self.to_index,
            weight.component(self.from_index),
            self.default_weight.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::float_weight::TropicalWeight;

    type Power3 = PowerWeight<TropicalWeight, 3>;
    type Sparse = SparsePowerWeight<TropicalWeight, i64>;

    fn tropical(value: f32) -> TropicalWeight {
        TropicalWeight(value)
    }

    #[test]
    fn to_power_weight_fills_one_component() {
        let mapper = ToPowerWeightMapper::<Power3>::new(1);
        let mapped = mapper.map(&tropical(2.5));
        assert_eq!(
            mapped,
            Power3::new([
                TropicalWeight::zero(),
                tropical(2.5),
                TropicalWeight::zero()
            ])
        );
    }

    #[test]
    fn to_power_weight_works_for_the_sparse_flavour() {
        let mapper = ToPowerWeightMapper::<Sparse>::new(7);
        let mapped = mapper.map(&tropical(1.25));
        assert_eq!(mapped.inner.value(7), &tropical(1.25));
        assert_eq!(mapped.inner.value(0), &TropicalWeight::zero());
    }

    #[test]
    fn from_power_weight_reads_one_component() {
        let weight = Power3::new([tropical(1.0), tropical(2.0), tropical(3.0)]);
        let mapper = FromPowerWeightMapper::<Power3>::new(2);
        let mapped: TropicalWeight = mapper.map(&weight);
        assert_eq!(mapped, tropical(3.0));
    }

    #[test]
    fn from_power_weight_reads_an_absent_sparse_component_as_the_default() {
        let weight = Sparse::from_component(3, tropical(4.0), TropicalWeight::one());
        let mapper = FromPowerWeightMapper::<Sparse>::new(9);
        let mapped: TropicalWeight = mapper.map(&weight);
        assert_eq!(mapped, TropicalWeight::one());
    }

    #[test]
    fn round_trips_through_a_power_weight() {
        let original = tropical(0.75);
        let to = ToPowerWeightMapper::<Power3>::new(2);
        let from = FromPowerWeightMapper::<Power3>::new(2);
        let back: TropicalWeight = from.map(&to.map(&original));
        assert_eq!(back, original);
    }

    #[test]
    fn project_moves_a_component_and_fills_the_rest() {
        let weight = Power3::new([tropical(1.0), tropical(2.0), tropical(3.0)]);
        let mapper = ProjectPowerWeightMapper::<Power3>::new(0, 2);
        assert_eq!(
            mapper.map(&weight),
            Power3::new([
                TropicalWeight::zero(),
                TropicalWeight::zero(),
                tropical(1.0)
            ])
        );
    }

    #[test]
    fn project_honours_a_non_zero_fill() {
        let weight = Power3::new([tropical(1.0), tropical(2.0), tropical(3.0)]);
        let mapper = ProjectPowerWeightMapper::<Power3>::with_default(1, 0, TropicalWeight::one());
        assert_eq!(
            mapper.map(&weight),
            Power3::new([tropical(2.0), TropicalWeight::one(), TropicalWeight::one()])
        );
    }

    #[test]
    fn project_is_idempotent_when_the_indices_match() {
        let weight = Power3::from_component(1, tropical(5.0), TropicalWeight::zero());
        let mapper = ProjectPowerWeightMapper::<Power3>::new(1, 1);
        assert_eq!(mapper.map(&weight), weight);
    }

    #[test]
    #[should_panic(expected = "component index 3 out of range for rank 3")]
    fn a_component_index_past_the_rank_panics() {
        let _ = Power3::from_component(3, tropical(1.0), TropicalWeight::zero());
    }
}
