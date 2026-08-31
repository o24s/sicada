//! Rewriting every arc of an FST through a mapper.
//!
//! Port of OpenFst's `arc-map.h`. A great many operations amount to "do the
//! same thing to each arc": invert the weights, drop them, quantize them, make
//! the input side epsilon. This is the shape they all share. What varies
//! between them is a [`ArcMapper`], and what makes the pattern non-trivial is
//! the final weights: some mappings turn a final weight into something an FST
//! cannot store as a final weight, and then a superfinal state is needed.

use std::marker::PhantomData;

use crate::arc::{Arc, ArcLabel, ArcStateId, GallicArc};
use crate::error::OpenFstError;
use crate::fst::{Fst, MutableFst};
use crate::properties::{
    K_ADD_SUPER_FINAL_PROPERTIES, K_COPY_PROPERTIES, K_ERROR, K_FST_PROPERTIES,
    K_O_LABEL_INVARIANT_PROPERTIES, K_WEIGHT_INVARIANT_PROPERTIES, project_properties,
};
use crate::weight::{Divide, DivideType, Weight};
use crate::weights::string_weight::{
    GallicTypeMarker, GallicWeight, StringWeight, StringWeightValue,
};

/// What a mapper needs done about final weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapFinalAction {
    /// The mapped final weight is still a final weight. A mapper that could
    /// produce labels here is a mistake.
    NoSuperfinal,
    /// The mapped final weight may come out as an arc, in which case it goes
    /// to a superfinal state, added only if some state actually needs it.
    AllowSuperfinal,
    /// The mapped final weight always goes to a superfinal state, which is
    /// therefore always added.
    RequireSuperfinal,
}

/// What a mapper needs done about symbol tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapSymbolsAction {
    /// The mapping changes what the labels mean, so the tables are dropped.
    Clear,
    /// The labels keep their meanings, so the tables carry over.
    Copy,
    /// The mapper sets them itself.
    Noop,
}

/// Turns one arc into another.
///
/// A final weight is offered as an arc `(epsilon, epsilon, weight, no_state)`,
/// and what the mapper does with it is governed by
/// [`final_action`](Self::final_action).
pub trait ArcMapper<From: Arc, To: Arc> {
    /// Maps one arc.
    fn map(&mut self, arc: &From) -> To;

    /// What to do about final weights.
    fn final_action(&self) -> MapFinalAction;

    /// What to do about the input symbol table.
    fn input_symbols_action(&self) -> MapSymbolsAction {
        MapSymbolsAction::Copy
    }

    /// What to do about the output symbol table.
    fn output_symbols_action(&self) -> MapSymbolsAction {
        MapSymbolsAction::Copy
    }

    /// The properties the result has, given those of the input.
    fn properties(&self, props: u64) -> u64;
}

/// A final weight, offered to a mapper in the shape of an arc.
fn final_as_arc<From: Arc>(weight: From::Weight) -> From {
    From::new(
        From::Label::epsilon(),
        From::Label::epsilon(),
        weight,
        From::StateId::no_state(),
    )
}

/// Whether a mapped final weight came back as something that has to become an
/// arc rather than a final weight.
fn became_an_arc<To: Arc>(arc: &To) -> bool {
    arc.ilabel() != To::Label::epsilon() || arc.olabel() != To::Label::epsilon()
}

/// Rewrites every arc of `fst` in place.
///
/// SICADA-DIVERGE: upstream reports a mapper that produces labels where it
/// promised not to by setting `K_ERROR` and carrying on to the next state, so
/// the FST ends up half mapped and marked broken. Here it is an error.
pub fn arc_map<A, F, M>(fst: &mut F, mapper: &mut M) -> Result<(), OpenFstError>
where
    A: Arc,
    F: MutableFst<A>,
    M: ArcMapper<A, A>,
{
    if mapper.input_symbols_action() == MapSymbolsAction::Clear {
        fst.set_input_symbols(None);
    }
    if mapper.output_symbols_action() == MapSymbolsAction::Clear {
        fst.set_output_symbols(None);
    }
    if fst.start().is_none() {
        return Ok(());
    }

    let props = fst.properties(K_FST_PROPERTIES, false);
    let final_action = mapper.final_action();
    let zero = A::Weight::zero();

    let mut superfinal = None;
    if final_action == MapFinalAction::RequireSuperfinal {
        let s = fst.add_state();
        fst.set_final(s, A::Weight::one());
        superfinal = Some(s);
    }

    let states: Vec<A::StateId> = fst.states().collect();
    for state in states {
        // The mapper is borrowed by the closure, so the error it may raise is
        // collected rather than returned from inside.
        let mut label_error = false;
        fst.mutate_arcs(state, |arc| *arc = mapper.map(arc));

        if Some(state) == superfinal {
            continue;
        }
        let mapped = mapper.map(&final_as_arc::<A>(fst.final_weight(state)));
        match final_action {
            MapFinalAction::NoSuperfinal => {
                if became_an_arc(&mapped) {
                    label_error = true;
                } else {
                    fst.set_final(state, mapped.weight().clone());
                }
            }
            MapFinalAction::AllowSuperfinal => {
                if became_an_arc(&mapped) {
                    let target = match superfinal {
                        Some(s) => s,
                        None => {
                            let s = fst.add_state();
                            fst.set_final(s, A::Weight::one());
                            superfinal = Some(s);
                            s
                        }
                    };
                    fst.add_arc(
                        state,
                        A::new(
                            mapped.ilabel(),
                            mapped.olabel(),
                            mapped.weight().clone(),
                            target,
                        ),
                    );
                    fst.set_final(state, zero.clone());
                } else {
                    fst.set_final(state, mapped.weight().clone());
                }
            }
            MapFinalAction::RequireSuperfinal => {
                let target = superfinal.expect("a superfinal state was added above");
                if became_an_arc(&mapped) || *mapped.weight() != zero {
                    fst.add_arc(
                        state,
                        A::new(
                            mapped.ilabel(),
                            mapped.olabel(),
                            mapped.weight().clone(),
                            target,
                        ),
                    );
                }
                fst.set_final(state, zero.clone());
            }
        }
        if label_error {
            return Err(OpenFstError::InvalidOperation(
                "ArcMap: the mapper produced a labelled arc from a final weight, but said it \
                 would not need a superfinal state"
                    .to_string(),
            ));
        }
    }

    fst.set_properties(mapper.properties(props), K_FST_PROPERTIES);
    Ok(())
}

/// Rewrites every arc of `ifst` into `ofst`, which may have a different arc
/// type.
pub fn arc_map_to<From, To, F1, F2, M>(
    ifst: &F1,
    ofst: &mut F2,
    mapper: &mut M,
) -> Result<(), OpenFstError>
where
    From: Arc,
    To: Arc<StateId = From::StateId>,
    F1: Fst<From>,
    F2: MutableFst<To>,
    M: ArcMapper<From, To>,
{
    ofst.delete_all_states();
    match mapper.input_symbols_action() {
        MapSymbolsAction::Copy => ofst.set_input_symbols(ifst.input_symbols()),
        MapSymbolsAction::Clear => ofst.set_input_symbols(None),
        MapSymbolsAction::Noop => {}
    }
    match mapper.output_symbols_action() {
        MapSymbolsAction::Copy => ofst.set_output_symbols(ifst.output_symbols()),
        MapSymbolsAction::Clear => ofst.set_output_symbols(None),
        MapSymbolsAction::Noop => {}
    }

    let iprops = ifst.properties(K_COPY_PROPERTIES, false);
    let Some(start) = ifst.start() else {
        return Ok(());
    };

    let final_action = mapper.final_action();
    let zero = To::Weight::zero();
    if let Some(num_states) = ifst.num_states_if_known() {
        ofst.reserve_states(num_states + usize::from(final_action != MapFinalAction::NoSuperfinal));
    }
    for _ in ifst.states() {
        ofst.add_state();
    }
    let mut superfinal = None;
    if final_action == MapFinalAction::RequireSuperfinal {
        let s = ofst.add_state();
        ofst.set_final(s, To::Weight::one());
        superfinal = Some(s);
    }

    for state in ifst.states() {
        if state == start {
            ofst.set_start(state);
        }
        ofst.reserve_arcs(
            state,
            ifst.num_arcs(state) + usize::from(final_action != MapFinalAction::NoSuperfinal),
        );
        for arc in ifst.arcs(state) {
            let mapped = mapper.map(&arc);
            ofst.add_arc(state, mapped);
        }

        let mapped = mapper.map(&final_as_arc::<From>(ifst.final_weight(state)));
        match final_action {
            MapFinalAction::NoSuperfinal => {
                if became_an_arc(&mapped) {
                    return Err(OpenFstError::InvalidOperation(
                        "ArcMap: the mapper produced a labelled arc from a final weight, but \
                         said it would not need a superfinal state"
                            .to_string(),
                    ));
                }
                ofst.set_final(state, mapped.weight().clone());
            }
            MapFinalAction::AllowSuperfinal => {
                if became_an_arc(&mapped) {
                    let target = match superfinal {
                        Some(s) => s,
                        None => {
                            let s = ofst.add_state();
                            ofst.set_final(s, To::Weight::one());
                            superfinal = Some(s);
                            s
                        }
                    };
                    ofst.add_arc(
                        state,
                        To::new(
                            mapped.ilabel(),
                            mapped.olabel(),
                            mapped.weight().clone(),
                            target,
                        ),
                    );
                    ofst.set_final(state, zero.clone());
                } else {
                    ofst.set_final(state, mapped.weight().clone());
                }
            }
            MapFinalAction::RequireSuperfinal => {
                let target = superfinal.expect("a superfinal state was added above");
                if became_an_arc(&mapped) || *mapped.weight() != zero {
                    ofst.add_arc(
                        state,
                        To::new(
                            mapped.ilabel(),
                            mapped.olabel(),
                            mapped.weight().clone(),
                            target,
                        ),
                    );
                }
                ofst.set_final(state, zero.clone());
            }
        }
    }

    let oprops = ofst.properties(K_FST_PROPERTIES, false);
    ofst.set_properties(mapper.properties(iprops) | oprops, K_FST_PROPERTIES);
    Ok(())
}

/// Leaves every arc as it is.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityArcMapper;

impl<A: Arc> ArcMapper<A, A> for IdentityArcMapper {
    #[inline]
    fn map(&mut self, arc: &A) -> A {
        arc.clone()
    }

    fn final_action(&self) -> MapFinalAction {
        MapFinalAction::NoSuperfinal
    }

    fn properties(&self, props: u64) -> u64 {
        props
    }
}

/// Makes the input side of every arc epsilon.
#[derive(Debug, Clone, Copy, Default)]
pub struct InputEpsilonMapper;

impl<A: Arc> ArcMapper<A, A> for InputEpsilonMapper {
    #[inline]
    fn map(&mut self, arc: &A) -> A {
        A::new(
            A::Label::epsilon(),
            arc.olabel(),
            arc.weight().clone(),
            arc.nextstate(),
        )
    }

    fn final_action(&self) -> MapFinalAction {
        MapFinalAction::NoSuperfinal
    }

    fn input_symbols_action(&self) -> MapSymbolsAction {
        MapSymbolsAction::Clear
    }

    fn properties(&self, props: u64) -> u64 {
        crate::properties::project_properties(props, false)
            & crate::properties::K_SET_ARC_PROPERTIES
            | crate::properties::K_I_EPSILONS
            | crate::properties::K_I_LABEL_SORTED
    }
}

/// Makes the output side of every arc epsilon.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutputEpsilonMapper;

impl<A: Arc> ArcMapper<A, A> for OutputEpsilonMapper {
    #[inline]
    fn map(&mut self, arc: &A) -> A {
        A::new(
            arc.ilabel(),
            A::Label::epsilon(),
            arc.weight().clone(),
            arc.nextstate(),
        )
    }

    fn final_action(&self) -> MapFinalAction {
        MapFinalAction::NoSuperfinal
    }

    fn output_symbols_action(&self) -> MapSymbolsAction {
        MapSymbolsAction::Clear
    }

    fn properties(&self, props: u64) -> u64 {
        crate::properties::project_properties(props, true) & crate::properties::K_SET_ARC_PROPERTIES
            | crate::properties::K_O_EPSILONS
            | crate::properties::K_O_LABEL_SORTED
    }
}

/// Leaves arcs alone but forces every final weight through a superfinal state.
///
/// The result has exactly one final state, as an operation wanting a single
/// accepting point, such as concatenation or closure, requires.
#[derive(Debug, Clone, Copy, Default)]
pub struct SuperFinalMapper;

impl<A: Arc> ArcMapper<A, A> for SuperFinalMapper {
    #[inline]
    fn map(&mut self, arc: &A) -> A {
        arc.clone()
    }

    fn final_action(&self) -> MapFinalAction {
        MapFinalAction::RequireSuperfinal
    }

    fn properties(&self, props: u64) -> u64 {
        props & crate::properties::K_ADD_SUPER_FINAL_PROPERTIES
    }
}

/// Multiplies a weight onto every arc and final weight, on the left.
#[derive(Debug, Clone)]
pub struct TimesMapper<W> {
    weight: W,
}

impl<W: Weight> TimesMapper<W> {
    /// Multiplies by `weight`.
    pub fn new(weight: W) -> Self {
        Self { weight }
    }
}

impl<A: Arc> ArcMapper<A, A> for TimesMapper<A::Weight> {
    #[inline]
    fn map(&mut self, arc: &A) -> A {
        A::new(
            arc.ilabel(),
            arc.olabel(),
            self.weight.times(arc.weight()),
            arc.nextstate(),
        )
    }

    fn final_action(&self) -> MapFinalAction {
        MapFinalAction::NoSuperfinal
    }

    fn properties(&self, props: u64) -> u64 {
        props & crate::properties::K_WEIGHT_INVARIANT_PROPERTIES
    }
}

/// Adds a weight to every arc and final weight.
#[derive(Debug, Clone)]
pub struct PlusMapper<W> {
    weight: W,
}

impl<W: Weight> PlusMapper<W> {
    /// Adds `weight`.
    pub fn new(weight: W) -> Self {
        Self { weight }
    }
}

impl<A: Arc> ArcMapper<A, A> for PlusMapper<A::Weight> {
    #[inline]
    fn map(&mut self, arc: &A) -> A {
        A::new(
            arc.ilabel(),
            arc.olabel(),
            self.weight.plus(arc.weight()),
            arc.nextstate(),
        )
    }

    fn final_action(&self) -> MapFinalAction {
        MapFinalAction::NoSuperfinal
    }

    fn properties(&self, props: u64) -> u64 {
        props & crate::properties::K_WEIGHT_INVARIANT_PROPERTIES
    }
}

/// Raises every weight to a power.
#[derive(Debug, Clone, Copy)]
pub struct PowerMapper {
    power: f64,
}

impl PowerMapper {
    /// Raises to `power`.
    pub fn new(power: f64) -> Self {
        Self { power }
    }
}

impl<A: Arc> ArcMapper<A, A> for PowerMapper
where
    A::Weight: Weight,
{
    fn map(&mut self, arc: &A) -> A {
        A::new(
            arc.ilabel(),
            arc.olabel(),
            power(arc.weight(), self.power),
            arc.nextstate(),
        )
    }

    fn final_action(&self) -> MapFinalAction {
        MapFinalAction::NoSuperfinal
    }

    fn properties(&self, props: u64) -> u64 {
        props & crate::properties::K_WEIGHT_INVARIANT_PROPERTIES
    }
}

/// A weight multiplied by itself `n` times.
///
/// SICADA-DIVERGE: upstream's `Power` is a free function over `size_t`
/// exponents. A fractional power is only meaningful for the float weights,
/// where it is a multiplication in the log domain; the general case is repeated
/// multiplication, which is what this does once the exponent is rounded.
fn power<W: Weight>(weight: &W, n: f64) -> W {
    let n = n.max(0.0).round() as u64;
    let mut result = W::one();
    for _ in 0..n {
        result = result.times(weight);
    }
    result
}

/// Replaces every weight by its inverse.
#[derive(Debug, Clone, Copy, Default)]
pub struct InvertWeightMapper;

impl<A: Arc> ArcMapper<A, A> for InvertWeightMapper
where
    A::Weight: Divide,
{
    #[inline]
    fn map(&mut self, arc: &A) -> A {
        A::new(
            arc.ilabel(),
            arc.olabel(),
            A::Weight::one().divide(arc.weight(), DivideType::Any),
            arc.nextstate(),
        )
    }

    fn final_action(&self) -> MapFinalAction {
        MapFinalAction::NoSuperfinal
    }

    fn properties(&self, props: u64) -> u64 {
        props & crate::properties::K_WEIGHT_INVARIANT_PROPERTIES
    }
}

/// Drops the weights, leaving an unweighted FST.
#[derive(Debug, Clone, Copy, Default)]
pub struct RmWeightMapper;

impl<A: Arc> ArcMapper<A, A> for RmWeightMapper {
    #[inline]
    fn map(&mut self, arc: &A) -> A {
        let weight = if *arc.weight() == A::Weight::zero() {
            A::Weight::zero()
        } else {
            A::Weight::one()
        };
        A::new(arc.ilabel(), arc.olabel(), weight, arc.nextstate())
    }

    fn final_action(&self) -> MapFinalAction {
        MapFinalAction::NoSuperfinal
    }

    fn properties(&self, props: u64) -> u64 {
        (props & crate::properties::K_WEIGHT_INVARIANT_PROPERTIES) | crate::properties::K_UNWEIGHTED
    }
}

/// Rounds every weight to a multiple of `delta`.
#[derive(Debug, Clone, Copy)]
pub struct QuantizeMapper {
    delta: f32,
}

impl Default for QuantizeMapper {
    fn default() -> Self {
        Self {
            delta: crate::weight::DELTA,
        }
    }
}

impl QuantizeMapper {
    /// Rounds to a multiple of `delta`.
    pub fn new(delta: f32) -> Self {
        Self { delta }
    }
}

impl<A: Arc> ArcMapper<A, A> for QuantizeMapper {
    #[inline]
    fn map(&mut self, arc: &A) -> A {
        A::new(
            arc.ilabel(),
            arc.olabel(),
            arc.weight().quantize(self.delta),
            arc.nextstate(),
        )
    }

    fn final_action(&self) -> MapFinalAction {
        MapFinalAction::NoSuperfinal
    }

    fn properties(&self, props: u64) -> u64 {
        props & crate::properties::K_WEIGHT_INVARIANT_PROPERTIES
    }
}

/// Replaces every weight by its reverse, and swaps the arc's sides.
///
/// The reverse of a weight lives in a different semiring in general, which is
/// why this maps between arc types.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReverseWeightMapper;

impl<From, To> ArcMapper<From, To> for ReverseWeightMapper
where
    From: Arc,
    To: Arc<
            Label = From::Label,
            StateId = From::StateId,
            Weight = <From::Weight as Weight>::ReverseWeight,
        >,
{
    #[inline]
    fn map(&mut self, arc: &From) -> To {
        To::new(
            arc.ilabel(),
            arc.olabel(),
            arc.weight().reverse(),
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

/// Keeps the arcs and copies the symbol tables, for a mapper that only changes
/// the weight type.
#[derive(Debug, Clone, Copy, Default)]
/// SICADA-DIVERGE: upstream's `WeightConvert` is a template that fails to
/// compile unless specialized, which is how it reports "these two weights are
/// unrelated". `From` says the same thing, the compiler's message names the two
/// types rather than a template, and every pair `weights/float_weight.rs`
/// already relates through `impl From` is usable here without a second trait
/// having to repeat them.
pub struct WeightConvertMapper<To> {
    _marker: std::marker::PhantomData<To>,
}

impl<To> WeightConvertMapper<To> {
    /// Converts weights into `To`'s weight type.
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<A1, A2> ArcMapper<A1, A2> for WeightConvertMapper<A2>
where
    A1: Arc,
    A2: Arc<Label = A1::Label, StateId = A1::StateId>,
    A2::Weight: From<A1::Weight>,
{
    #[inline]
    fn map(&mut self, arc: &A1) -> A2 {
        A2::new(
            arc.ilabel(),
            arc.olabel(),
            A2::Weight::from(arc.weight().clone()),
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

/// Moves an arc's output label into its weight, making an acceptor over the
/// gallic semiring.
///
/// The gallic semiring pairs a label sequence with a weight, so an arc's output
/// side becomes part of what a shortest-distance computation sums over. That is
/// what lets labels be pushed the way weights are.
#[derive(Debug, Clone, Copy, Default)]
pub struct ToGallicMapper<G> {
    _marker: PhantomData<G>,
}

impl<G> ToGallicMapper<G> {
    /// Creates the mapper.
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<A, G> ArcMapper<A, GallicArc<A, G>> for ToGallicMapper<G>
where
    A: Arc,
    G: GallicTypeMarker,
{
    fn map(&mut self, arc: &A) -> GallicArc<A, G> {
        let epsilon = A::Label::epsilon();
        let no_state = A::StateId::no_state();
        let gallic = |labels: StringWeight<A::Label, G::StringType>, weight: A::Weight| {
            GallicWeight::<A::Label, A::Weight, G>::from_parts(labels, weight)
        };
        if arc.nextstate() == no_state {
            // A final weight, offered as an arc. A state that is not final has
            // nothing to carry across.
            return if *arc.weight() == A::Weight::zero() {
                GallicArc::new(epsilon, epsilon, GallicWeight::zero(), no_state)
            } else {
                GallicArc::new(
                    epsilon,
                    epsilon,
                    gallic(StringWeight::one(), arc.weight().clone()),
                    no_state,
                )
            };
        }
        // The result is an acceptor over the input side; the output label moves
        // into the weight, and an epsilon output contributes no label at all.
        let labels = if arc.olabel() == epsilon {
            StringWeight::one()
        } else {
            StringWeight::new(vec![arc.olabel()])
        };
        GallicArc::new(
            arc.ilabel(),
            arc.ilabel(),
            gallic(labels, arc.weight().clone()),
            arc.nextstate(),
        )
    }

    fn final_action(&self) -> MapFinalAction {
        MapFinalAction::NoSuperfinal
    }

    fn input_symbols_action(&self) -> MapSymbolsAction {
        MapSymbolsAction::Copy
    }

    fn output_symbols_action(&self) -> MapSymbolsAction {
        // The output labels are now inside the weights, so a table keyed on
        // them no longer describes anything the FST has.
        MapSymbolsAction::Clear
    }

    fn properties(&self, props: u64) -> u64 {
        project_properties(props, true) & K_WEIGHT_INVARIANT_PROPERTIES
    }
}

/// Moves the label out of a gallic weight and back onto the arc's output side.
///
/// SICADA-DIVERGE: upstream reports a weight carrying more than one label, which
/// no single arc can represent, by setting a `mutable bool error_` from a
/// `const` member function and returning an arc built from the values it just
/// failed to extract. The failure is carried here and turned into an error by
/// [`arc_map`], which is the only thing that can act on it.
#[derive(Debug, Clone)]
pub struct FromGallicMapper<L, G> {
    /// The input label to give an arc that came from a final weight carrying a
    /// label.
    superfinal_label: L,
    /// Whether a weight was met that no arc can carry.
    error: bool,
    _marker: PhantomData<G>,
}

impl<L: ArcLabel, G> FromGallicMapper<L, G> {
    /// Uses epsilon as the label of an arc made from a labelled final weight.
    pub fn new() -> Self {
        Self::with_superfinal_label(L::epsilon())
    }

    /// Uses `superfinal_label` for such an arc.
    pub fn with_superfinal_label(superfinal_label: L) -> Self {
        Self {
            superfinal_label,
            error: false,
            _marker: PhantomData,
        }
    }

    /// Whether a weight was met that no single arc can carry.
    pub fn error(&self) -> bool {
        self.error
    }
}

impl<L: ArcLabel, G> Default for FromGallicMapper<L, G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A, G> ArcMapper<GallicArc<A, G>, A> for FromGallicMapper<A::Label, G>
where
    A: Arc,
    G: GallicTypeMarker,
{
    fn map(&mut self, arc: &GallicArc<A, G>) -> A {
        let epsilon = A::Label::epsilon();
        let no_state = A::StateId::no_state();
        if arc.nextstate() == no_state && *arc.weight() == GallicWeight::zero() {
            // A state that was not final stays not final.
            return A::new(arc.ilabel(), epsilon, A::Weight::zero(), no_state);
        }
        // A gallic weight an arc can carry holds at most one label.
        let (label, weight) = match &arc.weight().labels().value {
            StringWeightValue::Labels(labels) if labels.len() <= 1 => (
                labels.first().copied().unwrap_or(epsilon),
                arc.weight().weight().clone(),
            ),
            _ => {
                self.error = true;
                (epsilon, A::Weight::zero())
            }
        };
        if arc.ilabel() != arc.olabel() {
            self.error = true;
        }
        if arc.ilabel() == epsilon && label != epsilon && arc.nextstate() == no_state {
            // A final weight carrying a label has to become a real arc, and
            // needs an input label to go on it.
            A::new(self.superfinal_label, label, weight, arc.nextstate())
        } else {
            A::new(arc.ilabel(), label, weight, arc.nextstate())
        }
    }

    fn final_action(&self) -> MapFinalAction {
        MapFinalAction::AllowSuperfinal
    }

    fn input_symbols_action(&self) -> MapSymbolsAction {
        MapSymbolsAction::Copy
    }

    fn output_symbols_action(&self) -> MapSymbolsAction {
        MapSymbolsAction::Clear
    }

    fn properties(&self, props: u64) -> u64 {
        let out = props
            & K_O_LABEL_INVARIANT_PROPERTIES
            & K_WEIGHT_INVARIANT_PROPERTIES
            & K_ADD_SUPER_FINAL_PROPERTIES;
        if self.error { out | K_ERROR } else { out }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AtomicRc;
    use crate::arc::StdArc;
    use crate::fst::ExpandedFst as _;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::{K_ACCEPTOR, K_NOT_ACCEPTOR, K_UNWEIGHTED};
    use crate::symbol_table::SymbolTable;
    use crate::weights::float_weight::TropicalWeight;

    fn fst() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 2, TropicalWeight(1.0), 1));
        fst.add_arc(1, StdArc::new(3, 4, TropicalWeight(2.0), 2));
        fst.set_final(2, TropicalWeight(3.0));
        fst
    }

    fn arcs(fst: &StdVectorFst) -> Vec<(i32, i32, f32, i32)> {
        (0..fst.num_states() as i32)
            .flat_map(|s| {
                fst.arcs(s)
                    .map(|a| (a.ilabel(), a.olabel(), a.weight().value(), a.nextstate()))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn the_identity_mapper_changes_nothing() {
        let mut fst = fst();
        let before = arcs(&fst);
        arc_map(&mut fst, &mut IdentityArcMapper).unwrap();
        assert_eq!(arcs(&fst), before);
        assert_eq!(fst.final_weight(2), TropicalWeight(3.0));
    }

    #[test]
    fn the_epsilon_mappers_blank_one_side_and_drop_its_table() {
        let mut table = SymbolTable::new("input".to_string());
        table.add_symbol("<eps>", 0);

        let mut with_tables = fst();
        with_tables.set_input_symbols(Some(AtomicRc::new(table.clone())));
        with_tables.set_output_symbols(Some(AtomicRc::new(table)));

        arc_map(&mut with_tables, &mut InputEpsilonMapper).unwrap();
        assert_eq!(
            arcs(&with_tables),
            vec![(0, 2, 1.0, 1), (0, 4, 2.0, 2)],
            "the input side is blanked"
        );
        assert!(
            with_tables.input_symbols().is_none(),
            "its table went with it"
        );
        assert!(with_tables.output_symbols().is_some());

        let mut other = fst();
        arc_map(&mut other, &mut OutputEpsilonMapper).unwrap();
        assert_eq!(arcs(&other), vec![(1, 0, 1.0, 1), (3, 0, 2.0, 2)]);
    }

    #[test]
    fn removing_weights_leaves_an_unweighted_fst() {
        let mut fst = fst();
        arc_map(&mut fst, &mut RmWeightMapper).unwrap();
        assert_eq!(arcs(&fst), vec![(1, 2, 0.0, 1), (3, 4, 0.0, 2)]);
        assert_eq!(fst.final_weight(2), TropicalWeight::one());
        assert_ne!(fst.properties(K_UNWEIGHTED, false) & K_UNWEIGHTED, 0);
    }

    /// In the tropical semiring times is addition, so this adds a constant to
    /// every arc, and to the final weights, which is the point of the final
    /// action.
    #[test]
    fn multiplying_reaches_the_final_weights_too() {
        let mut fst = fst();
        arc_map(&mut fst, &mut TimesMapper::new(TropicalWeight(10.0))).unwrap();
        assert_eq!(arcs(&fst), vec![(1, 2, 11.0, 1), (3, 4, 12.0, 2)]);
        assert_eq!(fst.final_weight(2), TropicalWeight(13.0));
        // A non-final state stays non-final: Zero times anything is Zero.
        assert_eq!(fst.final_weight(0), TropicalWeight::zero());
    }

    /// Tropical plus is min, so this caps every weight.
    #[test]
    fn adding_caps_every_weight() {
        let mut fst = fst();
        arc_map(&mut fst, &mut PlusMapper::new(TropicalWeight(1.5))).unwrap();
        assert_eq!(arcs(&fst), vec![(1, 2, 1.0, 1), (3, 4, 1.5, 2)]);
        assert_eq!(fst.final_weight(2), TropicalWeight(1.5));
    }

    #[test]
    fn quantizing_rounds_every_weight() {
        let mut fst = StdVectorFst::new();
        fst.add_state();
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight(1.234_567), 0));
        fst.set_final(0, TropicalWeight(2.765_432));

        arc_map(&mut fst, &mut QuantizeMapper::new(0.5)).unwrap();
        assert_eq!(fst.arcs(0).next().unwrap().weight().value(), 1.0);
        assert_eq!(fst.final_weight(0).value(), 3.0);
    }

    /// Requiring a superfinal state moves every final weight onto an arc, so
    /// the result has exactly one final state.
    #[test]
    fn the_superfinal_mapper_leaves_exactly_one_final_state() {
        let mut fst = fst();
        fst.set_final(1, TropicalWeight(5.0));

        arc_map(&mut fst, &mut SuperFinalMapper).unwrap();

        let finals: Vec<i32> = (0..fst.num_states() as i32)
            .filter(|&s| fst.final_weight(s) != TropicalWeight::zero())
            .collect();
        assert_eq!(finals.len(), 1);
        let superfinal = finals[0];
        assert_eq!(fst.final_weight(superfinal), TropicalWeight::one());

        // The states that were final now reach it, carrying what they weighed.
        let to_superfinal: Vec<f32> = (0..fst.num_states() as i32)
            .flat_map(|s| {
                fst.arcs(s)
                    .filter(|a| a.nextstate() == superfinal)
                    .map(|a| a.weight().value())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(to_superfinal.len(), 2);
        assert!(to_superfinal.contains(&5.0));
        assert!(to_superfinal.contains(&3.0));
    }

    /// Mapping into a second FST leaves the input alone and reproduces its
    /// shape in the output.
    #[test]
    fn mapping_into_another_fst_leaves_the_input_alone() {
        let ifst = fst();
        let mut ofst = StdVectorFst::new();
        ofst.add_state();
        arc_map_to(&ifst, &mut ofst, &mut RmWeightMapper).unwrap();

        assert_eq!(ofst.num_states(), 3, "the output starts from nothing");
        assert_eq!(ofst.start(), Some(0));
        assert_eq!(arcs(&ofst), vec![(1, 2, 0.0, 1), (3, 4, 0.0, 2)]);
        assert_eq!(ofst.final_weight(2), TropicalWeight::one());
        // The input is untouched.
        assert_eq!(arcs(&ifst), vec![(1, 2, 1.0, 1), (3, 4, 2.0, 2)]);
    }

    /// A tropical weight is its own reverse, so this changes nothing, but it
    /// exercises the path that maps between arc types.
    #[test]
    fn reversing_weights_maps_between_arc_types() {
        let ifst = fst();
        let mut ofst = StdVectorFst::new();
        arc_map_to(&ifst, &mut ofst, &mut ReverseWeightMapper).unwrap();
        assert_eq!(arcs(&ofst), arcs(&ifst));
        assert_eq!(ofst.final_weight(2), TropicalWeight(3.0));
    }

    #[test]
    fn an_fst_with_no_start_state_is_left_alone() {
        let mut fst = StdVectorFst::new();
        fst.add_state();
        arc_map(&mut fst, &mut RmWeightMapper).unwrap();
        assert_eq!(fst.num_states(), 1);

        let ifst = StdVectorFst::new();
        let mut ofst = StdVectorFst::new();
        ofst.add_state();
        arc_map_to(&ifst, &mut ofst, &mut IdentityArcMapper).unwrap();
        assert_eq!(ofst.num_states(), 0);
    }

    /// The acceptor bit cannot survive a mapping that blanks one side.
    #[test]
    fn label_properties_are_given_up_when_labels_change() {
        let mut fst = StdVectorFst::new();
        for _ in 0..2 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.set_final(1, TropicalWeight::one());
        fst.add_arc(0, StdArc::new(5, 5, TropicalWeight::one(), 1));
        assert_ne!(fst.properties(K_ACCEPTOR, true) & K_ACCEPTOR, 0);

        arc_map(&mut fst, &mut InputEpsilonMapper).unwrap();
        assert_eq!(
            fst.properties(K_ACCEPTOR, false) & K_ACCEPTOR,
            0,
            "it is no longer an acceptor and must not say it is"
        );
        // Blanking one side made it a transducer, which a scan confirms.
        assert_ne!(fst.properties(K_NOT_ACCEPTOR, true) & K_NOT_ACCEPTOR, 0);
    }
}
