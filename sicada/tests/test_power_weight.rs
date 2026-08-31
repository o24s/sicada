//! Power weights: a fixed-rank vector of weights, componentwise.
//!
//! The mappers move a plain weight into one component and back out, which is
//! what lets an FST over one semiring be lifted into a vector of them --
//! upstream's `power-weight-mappers.h`.

use sicada::algorithms::arc_map::{ArcMapper, MapFinalAction, arc_map_to};
use sicada::arc::{Arc, PowerArc, StdArc};
use sicada::fst::{Fst, MutableFst};
use sicada::fsts::vector_fst::VectorFst;
use sicada::weight::Weight;
use sicada::weights::float_weight::TropicalWeight;
use sicada::weights::power_weight::PowerWeight;
use sicada::weights::power_weight_mappers::{
    FromPowerWeightMapper, ProjectPowerWeightMapper, ToPowerWeightMapper,
};

/// Three tropical weights in a row: `PowerArc` is the generic form of what
/// upstream names `Power3Arc`.
type Power3Arc = PowerArc<StdArc, 3>;
/// The weight it carries.
type Power3 = PowerWeight<TropicalWeight, 3>;

/// ⊕ and ⊗ are the component weight's, applied position by position.
#[test]
fn a_power_weight_works_componentwise() {
    let lhs = PowerWeight::new([
        TropicalWeight(1.0),
        TropicalWeight(2.0),
        TropicalWeight(3.0),
    ]);
    let rhs = PowerWeight::new([
        TropicalWeight(3.0),
        TropicalWeight(1.0),
        TropicalWeight(2.0),
    ]);

    // Tropical ⊕ is min.
    let sum = lhs.plus(&rhs);
    assert_eq!(
        sum.elements,
        [
            TropicalWeight(1.0),
            TropicalWeight(1.0),
            TropicalWeight(2.0)
        ]
    );

    // Tropical ⊗ is +.
    let product = lhs.times(&rhs);
    assert_eq!(
        product.elements,
        [
            TropicalWeight(4.0),
            TropicalWeight(3.0),
            TropicalWeight(5.0)
        ]
    );
}

/// Lifts every arc's weight into one component of a power weight.
struct IntoComponent {
    mapper: ToPowerWeightMapper<Power3>,
}

impl ArcMapper<StdArc, Power3Arc> for IntoComponent {
    fn map(&mut self, arc: &StdArc) -> Power3Arc {
        Power3Arc::new(
            arc.ilabel(),
            arc.olabel(),
            self.mapper.map(arc.weight()),
            arc.nextstate(),
        )
    }

    fn final_action(&self) -> MapFinalAction {
        MapFinalAction::NoSuperfinal
    }

    fn properties(&self, props: u64) -> u64 {
        props
    }
}

/// Reads one component back out into a plain weight.
struct OutOfComponent {
    mapper: FromPowerWeightMapper<Power3>,
}

impl ArcMapper<Power3Arc, StdArc> for OutOfComponent {
    fn map(&mut self, arc: &Power3Arc) -> StdArc {
        StdArc::new(
            arc.ilabel(),
            arc.olabel(),
            self.mapper.map(arc.weight()),
            arc.nextstate(),
        )
    }

    fn final_action(&self) -> MapFinalAction {
        MapFinalAction::NoSuperfinal
    }

    fn properties(&self, props: u64) -> u64 {
        props
    }
}

/// A weight goes into component 1, is projected onto component 0, and comes
/// back out unchanged. The components it is not in hold Zero throughout.
#[test]
fn a_weight_travels_through_a_component_and_back() {
    let mut plain = VectorFst::<StdArc>::new();
    let s0 = plain.add_state();
    let s1 = plain.add_state();
    plain.set_start(s0);
    plain.set_final(s1, TropicalWeight(0.5));
    plain.add_arc(s0, StdArc::new(1, 2, TropicalWeight(2.0), s1));

    let mut lifted = VectorFst::<Power3Arc>::new();
    arc_map_to(
        &plain,
        &mut lifted,
        &mut IntoComponent {
            mapper: ToPowerWeightMapper::new(1),
        },
    )
    .expect("an FST over power weights");

    let weight = lifted
        .arcs(s0)
        .next()
        .expect("the one arc")
        .weight()
        .clone();
    assert_eq!(weight.elements[1], TropicalWeight(2.0));
    assert_eq!(
        weight.elements[0],
        TropicalWeight::zero(),
        "the components it was not put in stay Zero"
    );

    let projected = ProjectPowerWeightMapper::<Power3>::new(1, 0).map(&weight);
    assert_eq!(projected.elements[0], TropicalWeight(2.0));
    assert_eq!(projected.elements[1], TropicalWeight::zero());

    let mut back = VectorFst::<StdArc>::new();
    arc_map_to(
        &lifted,
        &mut back,
        &mut OutOfComponent {
            mapper: FromPowerWeightMapper::new(1),
        },
    )
    .expect("an FST over plain weights");
    assert_eq!(
        back.arcs(s0).next().expect("the one arc").weight(),
        &TropicalWeight(2.0)
    );
}
