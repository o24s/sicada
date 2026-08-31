use std::ops::{Deref, DerefMut};

use crate::arc::{Arc, ArcLabel};
use crate::error::OpenFstError;
use crate::fst::Fst;
use crate::weight::Weight;
use std::fmt::Write as _;

// BINARY PROPERTIES
pub const K_EXPANDED: u64 = 0x0000000000000001;
pub const K_MUTABLE: u64 = 0x0000000000000002;
pub const K_ERROR: u64 = 0x0000000000000004;

// TRINARY PROPERTIES
pub const K_ACCEPTOR: u64 = 0x0000000000010000;
pub const K_NOT_ACCEPTOR: u64 = 0x0000000000020000;
pub const K_I_DETERMINISTIC: u64 = 0x0000000000040000;
pub const K_NON_I_DETERMINISTIC: u64 = 0x0000000000080000;
pub const K_O_DETERMINISTIC: u64 = 0x0000000000100000;
pub const K_NON_O_DETERMINISTIC: u64 = 0x0000000000200000;
pub const K_EPSILONS: u64 = 0x0000000000400000;
pub const K_NO_EPSILONS: u64 = 0x0000000000800000;
pub const K_I_EPSILONS: u64 = 0x0000000001000000;
pub const K_NO_I_EPSILONS: u64 = 0x0000000002000000;
pub const K_O_EPSILONS: u64 = 0x0000000004000000;
pub const K_NO_O_EPSILONS: u64 = 0x0000000008000000;
pub const K_I_LABEL_SORTED: u64 = 0x0000000010000000;
pub const K_NOT_I_LABEL_SORTED: u64 = 0x0000000020000000;
pub const K_O_LABEL_SORTED: u64 = 0x0000000040000000;
pub const K_NOT_O_LABEL_SORTED: u64 = 0x0000000080000000;
pub const K_WEIGHTED: u64 = 0x0000000100000000;
pub const K_UNWEIGHTED: u64 = 0x0000000200000000;
pub const K_CYCLIC: u64 = 0x0000000400000000;
pub const K_ACYCLIC: u64 = 0x0000000800000000;
pub const K_INITIAL_CYCLIC: u64 = 0x0000001000000000;
pub const K_INITIAL_ACYCLIC: u64 = 0x0000002000000000;
pub const K_TOP_SORTED: u64 = 0x0000004000000000;
pub const K_NOT_TOP_SORTED: u64 = 0x0000008000000000;
pub const K_ACCESSIBLE: u64 = 0x0000010000000000;
pub const K_NOT_ACCESSIBLE: u64 = 0x0000020000000000;
pub const K_CO_ACCESSIBLE: u64 = 0x0000040000000000;
pub const K_NOT_CO_ACCESSIBLE: u64 = 0x0000080000000000;
pub const K_STRING: u64 = 0x0000100000000000;
pub const K_NOT_STRING: u64 = 0x0000200000000000;
pub const K_WEIGHTED_CYCLES: u64 = 0x0000400000000000;
pub const K_UNWEIGHTED_CYCLES: u64 = 0x0000800000000000;

// COMPOSITE PROPERTIES
pub const K_NULL_PROPERTIES: u64 = K_ACCEPTOR
    | K_I_DETERMINISTIC
    | K_O_DETERMINISTIC
    | K_NO_EPSILONS
    | K_NO_I_EPSILONS
    | K_NO_O_EPSILONS
    | K_I_LABEL_SORTED
    | K_O_LABEL_SORTED
    | K_UNWEIGHTED
    | K_ACYCLIC
    | K_INITIAL_ACYCLIC
    | K_TOP_SORTED
    | K_ACCESSIBLE
    | K_CO_ACCESSIBLE
    | K_STRING
    | K_UNWEIGHTED_CYCLES;

pub const K_COMPILED_STRING_PROPERTIES: u64 = K_ACCEPTOR
    | K_STRING
    | K_UNWEIGHTED
    | K_I_DETERMINISTIC
    | K_O_DETERMINISTIC
    | K_I_LABEL_SORTED
    | K_O_LABEL_SORTED
    | K_ACYCLIC
    | K_INITIAL_ACYCLIC
    | K_UNWEIGHTED_CYCLES
    | K_TOP_SORTED
    | K_ACCESSIBLE
    | K_CO_ACCESSIBLE;

pub const K_COPY_PROPERTIES: u64 = K_ERROR
    | K_ACCEPTOR
    | K_NOT_ACCEPTOR
    | K_I_DETERMINISTIC
    | K_NON_I_DETERMINISTIC
    | K_O_DETERMINISTIC
    | K_NON_O_DETERMINISTIC
    | K_EPSILONS
    | K_NO_EPSILONS
    | K_I_EPSILONS
    | K_NO_I_EPSILONS
    | K_O_EPSILONS
    | K_NO_O_EPSILONS
    | K_I_LABEL_SORTED
    | K_NOT_I_LABEL_SORTED
    | K_O_LABEL_SORTED
    | K_NOT_O_LABEL_SORTED
    | K_WEIGHTED
    | K_UNWEIGHTED
    | K_CYCLIC
    | K_ACYCLIC
    | K_INITIAL_CYCLIC
    | K_INITIAL_ACYCLIC
    | K_TOP_SORTED
    | K_NOT_TOP_SORTED
    | K_ACCESSIBLE
    | K_NOT_ACCESSIBLE
    | K_CO_ACCESSIBLE
    | K_NOT_CO_ACCESSIBLE
    | K_STRING
    | K_NOT_STRING
    | K_WEIGHTED_CYCLES
    | K_UNWEIGHTED_CYCLES;

pub const K_INTRINSIC_PROPERTIES: u64 = K_EXPANDED
    | K_MUTABLE
    | K_ACCEPTOR
    | K_NOT_ACCEPTOR
    | K_I_DETERMINISTIC
    | K_NON_I_DETERMINISTIC
    | K_O_DETERMINISTIC
    | K_NON_O_DETERMINISTIC
    | K_EPSILONS
    | K_NO_EPSILONS
    | K_I_EPSILONS
    | K_NO_I_EPSILONS
    | K_O_EPSILONS
    | K_NO_O_EPSILONS
    | K_I_LABEL_SORTED
    | K_NOT_I_LABEL_SORTED
    | K_O_LABEL_SORTED
    | K_NOT_O_LABEL_SORTED
    | K_WEIGHTED
    | K_UNWEIGHTED
    | K_CYCLIC
    | K_ACYCLIC
    | K_INITIAL_CYCLIC
    | K_INITIAL_ACYCLIC
    | K_TOP_SORTED
    | K_NOT_TOP_SORTED
    | K_ACCESSIBLE
    | K_NOT_ACCESSIBLE
    | K_CO_ACCESSIBLE
    | K_NOT_CO_ACCESSIBLE
    | K_STRING
    | K_NOT_STRING
    | K_WEIGHTED_CYCLES
    | K_UNWEIGHTED_CYCLES;

pub const K_EXTRINSIC_PROPERTIES: u64 = K_ERROR;

pub const K_SET_START_PROPERTIES: u64 = K_EXPANDED
    | K_MUTABLE
    | K_ERROR
    | K_ACCEPTOR
    | K_NOT_ACCEPTOR
    | K_I_DETERMINISTIC
    | K_NON_I_DETERMINISTIC
    | K_O_DETERMINISTIC
    | K_NON_O_DETERMINISTIC
    | K_EPSILONS
    | K_NO_EPSILONS
    | K_I_EPSILONS
    | K_NO_I_EPSILONS
    | K_O_EPSILONS
    | K_NO_O_EPSILONS
    | K_I_LABEL_SORTED
    | K_NOT_I_LABEL_SORTED
    | K_O_LABEL_SORTED
    | K_NOT_O_LABEL_SORTED
    | K_WEIGHTED
    | K_UNWEIGHTED
    | K_CYCLIC
    | K_ACYCLIC
    | K_TOP_SORTED
    | K_NOT_TOP_SORTED
    | K_CO_ACCESSIBLE
    | K_NOT_CO_ACCESSIBLE
    | K_WEIGHTED_CYCLES
    | K_UNWEIGHTED_CYCLES;

pub const K_SET_FINAL_PROPERTIES: u64 = K_EXPANDED
    | K_MUTABLE
    | K_ERROR
    | K_ACCEPTOR
    | K_NOT_ACCEPTOR
    | K_I_DETERMINISTIC
    | K_NON_I_DETERMINISTIC
    | K_O_DETERMINISTIC
    | K_NON_O_DETERMINISTIC
    | K_EPSILONS
    | K_NO_EPSILONS
    | K_I_EPSILONS
    | K_NO_I_EPSILONS
    | K_O_EPSILONS
    | K_NO_O_EPSILONS
    | K_I_LABEL_SORTED
    | K_NOT_I_LABEL_SORTED
    | K_O_LABEL_SORTED
    | K_NOT_O_LABEL_SORTED
    | K_CYCLIC
    | K_ACYCLIC
    | K_INITIAL_CYCLIC
    | K_INITIAL_ACYCLIC
    | K_TOP_SORTED
    | K_NOT_TOP_SORTED
    | K_ACCESSIBLE
    | K_NOT_ACCESSIBLE
    | K_WEIGHTED_CYCLES
    | K_UNWEIGHTED_CYCLES;

pub const K_ADD_STATE_PROPERTIES: u64 = K_EXPANDED
    | K_MUTABLE
    | K_ERROR
    | K_ACCEPTOR
    | K_NOT_ACCEPTOR
    | K_I_DETERMINISTIC
    | K_NON_I_DETERMINISTIC
    | K_O_DETERMINISTIC
    | K_NON_O_DETERMINISTIC
    | K_EPSILONS
    | K_NO_EPSILONS
    | K_I_EPSILONS
    | K_NO_I_EPSILONS
    | K_O_EPSILONS
    | K_NO_O_EPSILONS
    | K_I_LABEL_SORTED
    | K_NOT_I_LABEL_SORTED
    | K_O_LABEL_SORTED
    | K_NOT_O_LABEL_SORTED
    | K_WEIGHTED
    | K_UNWEIGHTED
    | K_CYCLIC
    | K_ACYCLIC
    | K_INITIAL_CYCLIC
    | K_INITIAL_ACYCLIC
    | K_TOP_SORTED
    | K_NOT_TOP_SORTED
    | K_NOT_ACCESSIBLE
    | K_NOT_CO_ACCESSIBLE
    | K_NOT_STRING
    | K_WEIGHTED_CYCLES
    | K_UNWEIGHTED_CYCLES;

pub const K_ADD_ARC_PROPERTIES: u64 = K_EXPANDED
    | K_MUTABLE
    | K_ERROR
    | K_NOT_ACCEPTOR
    | K_NON_I_DETERMINISTIC
    | K_NON_O_DETERMINISTIC
    | K_EPSILONS
    | K_I_EPSILONS
    | K_O_EPSILONS
    | K_NOT_I_LABEL_SORTED
    | K_NOT_O_LABEL_SORTED
    | K_WEIGHTED
    | K_CYCLIC
    | K_INITIAL_CYCLIC
    | K_NOT_TOP_SORTED
    | K_ACCESSIBLE
    | K_CO_ACCESSIBLE
    | K_WEIGHTED_CYCLES;

pub const K_SET_ARC_PROPERTIES: u64 = K_EXPANDED | K_MUTABLE | K_ERROR;

/// Recomputes the property bits after one arc is replaced by another.
///
/// Port of the body of upstream's `MutableArcIterator<VectorFst>::SetValue`.
/// What the old arc justified is cleared, what the new one forces is set, and
/// everything else an arc could have affected (sortedness, determinism, top
/// sortedness) is dropped to unknown, since changing a label can break any of
/// them and only a rescan could tell.
///
/// Clearing a bit without setting its opposite leaves the property unknown
/// rather than false, which makes this safe when the old arc was not the only
/// reason the bit was set.
#[inline]
pub fn set_arc_properties<A: Arc>(inprops: u64, old_arc: &A, new_arc: &A) -> u64 {
    let epsilon = A::Label::epsilon();
    let mut props = inprops;

    if old_arc.ilabel() != old_arc.olabel() {
        props &= !K_NOT_ACCEPTOR;
    }
    if old_arc.ilabel() == epsilon {
        props &= !K_I_EPSILONS;
        if old_arc.olabel() == epsilon {
            props &= !K_EPSILONS;
        }
    }
    if old_arc.olabel() == epsilon {
        props &= !K_O_EPSILONS;
    }
    if !is_trivial_weight(old_arc.weight()) {
        props &= !K_WEIGHTED;
    }

    if new_arc.ilabel() != new_arc.olabel() {
        props |= K_NOT_ACCEPTOR;
        props &= !K_ACCEPTOR;
    }
    if new_arc.ilabel() == epsilon {
        props |= K_I_EPSILONS;
        props &= !K_NO_I_EPSILONS;
        if new_arc.olabel() == epsilon {
            props |= K_EPSILONS;
            props &= !K_NO_EPSILONS;
        }
    }
    if new_arc.olabel() == epsilon {
        props |= K_O_EPSILONS;
        props &= !K_NO_O_EPSILONS;
    }
    if !is_trivial_weight(new_arc.weight()) {
        props |= K_WEIGHTED;
        props &= !K_UNWEIGHTED;
    }

    props
        & (K_SET_ARC_PROPERTIES
            | K_ACCEPTOR
            | K_NOT_ACCEPTOR
            | K_EPSILONS
            | K_NO_EPSILONS
            | K_I_EPSILONS
            | K_NO_I_EPSILONS
            | K_O_EPSILONS
            | K_NO_O_EPSILONS
            | K_WEIGHTED
            | K_UNWEIGHTED)
}

/// Whether a weight is `Zero` or `One`, the two that leave an FST unweighted.
#[inline]
fn is_trivial_weight<W: Weight>(weight: &W) -> bool {
    *weight == W::zero() || *weight == W::one()
}

pub const K_DELETE_STATES_PROPERTIES: u64 = K_EXPANDED
    | K_MUTABLE
    | K_ERROR
    | K_ACCEPTOR
    | K_I_DETERMINISTIC
    | K_O_DETERMINISTIC
    | K_NO_EPSILONS
    | K_NO_I_EPSILONS
    | K_NO_O_EPSILONS
    | K_I_LABEL_SORTED
    | K_O_LABEL_SORTED
    | K_UNWEIGHTED
    | K_ACYCLIC
    | K_INITIAL_ACYCLIC
    | K_TOP_SORTED
    | K_UNWEIGHTED_CYCLES;

pub const K_DELETE_ARCS_PROPERTIES: u64 = K_EXPANDED
    | K_MUTABLE
    | K_ERROR
    | K_ACCEPTOR
    | K_I_DETERMINISTIC
    | K_O_DETERMINISTIC
    | K_NO_EPSILONS
    | K_NO_I_EPSILONS
    | K_NO_O_EPSILONS
    | K_I_LABEL_SORTED
    | K_O_LABEL_SORTED
    | K_UNWEIGHTED
    | K_ACYCLIC
    | K_INITIAL_ACYCLIC
    | K_TOP_SORTED
    | K_NOT_ACCESSIBLE
    | K_NOT_CO_ACCESSIBLE
    | K_UNWEIGHTED_CYCLES;

pub const K_STATE_SORT_PROPERTIES: u64 = K_EXPANDED
    | K_MUTABLE
    | K_ERROR
    | K_ACCEPTOR
    | K_NOT_ACCEPTOR
    | K_I_DETERMINISTIC
    | K_NON_I_DETERMINISTIC
    | K_O_DETERMINISTIC
    | K_NON_O_DETERMINISTIC
    | K_EPSILONS
    | K_NO_EPSILONS
    | K_I_EPSILONS
    | K_NO_I_EPSILONS
    | K_O_EPSILONS
    | K_NO_O_EPSILONS
    | K_I_LABEL_SORTED
    | K_NOT_I_LABEL_SORTED
    | K_O_LABEL_SORTED
    | K_NOT_O_LABEL_SORTED
    | K_WEIGHTED
    | K_UNWEIGHTED
    | K_CYCLIC
    | K_ACYCLIC
    | K_INITIAL_CYCLIC
    | K_INITIAL_ACYCLIC
    | K_ACCESSIBLE
    | K_NOT_ACCESSIBLE
    | K_CO_ACCESSIBLE
    | K_NOT_CO_ACCESSIBLE
    | K_WEIGHTED_CYCLES
    | K_UNWEIGHTED_CYCLES;

pub const K_ARC_SORT_PROPERTIES: u64 = K_EXPANDED
    | K_MUTABLE
    | K_ERROR
    | K_ACCEPTOR
    | K_NOT_ACCEPTOR
    | K_I_DETERMINISTIC
    | K_NON_I_DETERMINISTIC
    | K_O_DETERMINISTIC
    | K_NON_O_DETERMINISTIC
    | K_EPSILONS
    | K_NO_EPSILONS
    | K_I_EPSILONS
    | K_NO_I_EPSILONS
    | K_O_EPSILONS
    | K_NO_O_EPSILONS
    | K_WEIGHTED
    | K_UNWEIGHTED
    | K_CYCLIC
    | K_ACYCLIC
    | K_INITIAL_CYCLIC
    | K_INITIAL_ACYCLIC
    | K_TOP_SORTED
    | K_NOT_TOP_SORTED
    | K_ACCESSIBLE
    | K_NOT_ACCESSIBLE
    | K_CO_ACCESSIBLE
    | K_NOT_CO_ACCESSIBLE
    | K_STRING
    | K_NOT_STRING
    | K_WEIGHTED_CYCLES
    | K_UNWEIGHTED_CYCLES;

pub const K_I_LABEL_INVARIANT_PROPERTIES: u64 = K_EXPANDED
    | K_MUTABLE
    | K_ERROR
    | K_O_DETERMINISTIC
    | K_NON_O_DETERMINISTIC
    | K_O_EPSILONS
    | K_NO_O_EPSILONS
    | K_O_LABEL_SORTED
    | K_NOT_O_LABEL_SORTED
    | K_WEIGHTED
    | K_UNWEIGHTED
    | K_CYCLIC
    | K_ACYCLIC
    | K_INITIAL_CYCLIC
    | K_INITIAL_ACYCLIC
    | K_TOP_SORTED
    | K_NOT_TOP_SORTED
    | K_ACCESSIBLE
    | K_NOT_ACCESSIBLE
    | K_CO_ACCESSIBLE
    | K_NOT_CO_ACCESSIBLE
    | K_STRING
    | K_NOT_STRING
    | K_WEIGHTED_CYCLES
    | K_UNWEIGHTED_CYCLES;

pub const K_O_LABEL_INVARIANT_PROPERTIES: u64 = K_EXPANDED
    | K_MUTABLE
    | K_ERROR
    | K_I_DETERMINISTIC
    | K_NON_I_DETERMINISTIC
    | K_I_EPSILONS
    | K_NO_I_EPSILONS
    | K_I_LABEL_SORTED
    | K_NOT_I_LABEL_SORTED
    | K_WEIGHTED
    | K_UNWEIGHTED
    | K_CYCLIC
    | K_ACYCLIC
    | K_INITIAL_CYCLIC
    | K_INITIAL_ACYCLIC
    | K_TOP_SORTED
    | K_NOT_TOP_SORTED
    | K_ACCESSIBLE
    | K_NOT_ACCESSIBLE
    | K_CO_ACCESSIBLE
    | K_NOT_CO_ACCESSIBLE
    | K_STRING
    | K_NOT_STRING
    | K_WEIGHTED_CYCLES
    | K_UNWEIGHTED_CYCLES;

pub const K_WEIGHT_INVARIANT_PROPERTIES: u64 = K_EXPANDED
    | K_MUTABLE
    | K_ERROR
    | K_ACCEPTOR
    | K_NOT_ACCEPTOR
    | K_I_DETERMINISTIC
    | K_NON_I_DETERMINISTIC
    | K_O_DETERMINISTIC
    | K_NON_O_DETERMINISTIC
    | K_EPSILONS
    | K_NO_EPSILONS
    | K_I_EPSILONS
    | K_NO_I_EPSILONS
    | K_O_EPSILONS
    | K_NO_O_EPSILONS
    | K_I_LABEL_SORTED
    | K_NOT_I_LABEL_SORTED
    | K_O_LABEL_SORTED
    | K_NOT_O_LABEL_SORTED
    | K_CYCLIC
    | K_ACYCLIC
    | K_INITIAL_CYCLIC
    | K_INITIAL_ACYCLIC
    | K_TOP_SORTED
    | K_NOT_TOP_SORTED
    | K_ACCESSIBLE
    | K_NOT_ACCESSIBLE
    | K_CO_ACCESSIBLE
    | K_NOT_CO_ACCESSIBLE
    | K_STRING
    | K_NOT_STRING;

pub const K_ADD_SUPER_FINAL_PROPERTIES: u64 = K_EXPANDED
    | K_MUTABLE
    | K_ERROR
    | K_ACCEPTOR
    | K_NOT_ACCEPTOR
    | K_NON_I_DETERMINISTIC
    | K_NON_O_DETERMINISTIC
    | K_EPSILONS
    | K_I_EPSILONS
    | K_O_EPSILONS
    | K_NOT_I_LABEL_SORTED
    | K_NOT_O_LABEL_SORTED
    | K_WEIGHTED
    | K_UNWEIGHTED
    | K_CYCLIC
    | K_ACYCLIC
    | K_INITIAL_CYCLIC
    | K_INITIAL_ACYCLIC
    | K_NOT_TOP_SORTED
    | K_NOT_ACCESSIBLE
    | K_CO_ACCESSIBLE
    | K_NOT_CO_ACCESSIBLE
    | K_NOT_STRING
    | K_WEIGHTED_CYCLES
    | K_UNWEIGHTED_CYCLES;

pub const K_RM_SUPER_FINAL_PROPERTIES: u64 = K_EXPANDED
    | K_MUTABLE
    | K_ERROR
    | K_ACCEPTOR
    | K_NOT_ACCEPTOR
    | K_I_DETERMINISTIC
    | K_O_DETERMINISTIC
    | K_NO_EPSILONS
    | K_NO_I_EPSILONS
    | K_NO_O_EPSILONS
    | K_I_LABEL_SORTED
    | K_O_LABEL_SORTED
    | K_WEIGHTED
    | K_UNWEIGHTED
    | K_CYCLIC
    | K_ACYCLIC
    | K_INITIAL_CYCLIC
    | K_INITIAL_ACYCLIC
    | K_TOP_SORTED
    | K_ACCESSIBLE
    | K_CO_ACCESSIBLE
    | K_NOT_CO_ACCESSIBLE
    | K_STRING
    | K_WEIGHTED_CYCLES
    | K_UNWEIGHTED_CYCLES;

pub const K_BINARY_PROPERTIES: u64 = 0x0000000000000007;
pub const K_TRINARY_PROPERTIES: u64 = 0x0000ffffffff0000;
pub const K_POS_TRINARY_PROPERTIES: u64 = K_TRINARY_PROPERTIES & 0x5555555555555555;
pub const K_NEG_TRINARY_PROPERTIES: u64 = K_TRINARY_PROPERTIES & 0xaaaaaaaaaaaaaaaa;
pub const K_FST_PROPERTIES: u64 = K_BINARY_PROPERTIES | K_TRINARY_PROPERTIES;

// PROPERTY FUNCTIONS

#[inline]
pub fn set_start_properties(inprops: u64) -> u64 {
    let mut outprops = inprops & K_SET_START_PROPERTIES;
    if (inprops & K_ACYCLIC) != 0 {
        outprops |= K_INITIAL_ACYCLIC;
    }
    outprops
}

#[inline]
pub fn set_final_properties<W: Weight>(inprops: u64, old_weight: &W, new_weight: &W) -> u64 {
    let mut outprops = inprops;
    let zero = W::zero();
    let one = W::one();

    if *old_weight != zero && *old_weight != one {
        outprops &= !K_WEIGHTED;
    }
    if *new_weight != zero && *new_weight != one {
        outprops |= K_WEIGHTED;
        outprops &= !K_UNWEIGHTED;
    }
    outprops &= K_SET_FINAL_PROPERTIES | K_WEIGHTED | K_UNWEIGHTED;
    outprops
}

#[inline]
pub fn add_state_properties(inprops: u64) -> u64 {
    inprops & K_ADD_STATE_PROPERTIES
}

#[inline]
pub fn add_arc_properties<A: Arc>(
    inprops: u64,
    s: A::StateId,
    arc: &A,
    prev_arc: Option<&A>,
) -> u64 {
    let mut outprops = inprops;
    let zero = A::Weight::zero();
    let one = A::Weight::one();

    if arc.ilabel() != arc.olabel() {
        outprops |= K_NOT_ACCEPTOR;
        outprops &= !K_ACCEPTOR;
    }
    if arc.ilabel() == A::Label::epsilon() {
        outprops |= K_I_EPSILONS;
        outprops &= !K_NO_I_EPSILONS;
        if arc.olabel() == A::Label::epsilon() {
            outprops |= K_EPSILONS;
            outprops &= !K_NO_EPSILONS;
        }
    }
    if arc.olabel() == A::Label::epsilon() {
        outprops |= K_O_EPSILONS;
        outprops &= !K_NO_O_EPSILONS;
    }
    if let Some(prev) = prev_arc {
        if prev.ilabel() > arc.ilabel() {
            outprops |= K_NOT_I_LABEL_SORTED;
            outprops &= !K_I_LABEL_SORTED;
        }
        if prev.olabel() > arc.olabel() {
            outprops |= K_NOT_O_LABEL_SORTED;
            outprops &= !K_O_LABEL_SORTED;
        }
    }
    if *arc.weight() != zero && *arc.weight() != one {
        outprops |= K_WEIGHTED;
        outprops &= !K_UNWEIGHTED;
    }
    if arc.nextstate() <= s {
        outprops |= K_NOT_TOP_SORTED;
        outprops &= !K_TOP_SORTED;
    }

    outprops &= K_ADD_ARC_PROPERTIES
        | K_ACCEPTOR
        | K_NO_EPSILONS
        | K_NO_I_EPSILONS
        | K_NO_O_EPSILONS
        | K_I_LABEL_SORTED
        | K_O_LABEL_SORTED
        | K_UNWEIGHTED
        | K_TOP_SORTED;

    if (outprops & K_TOP_SORTED) != 0 {
        outprops |= K_ACYCLIC | K_INITIAL_ACYCLIC;
    }
    outprops
}

#[inline]
pub fn delete_states_properties(inprops: u64) -> u64 {
    inprops & K_DELETE_STATES_PROPERTIES
}

#[inline]
pub fn delete_all_states_properties(inprops: u64, static_props: u64) -> u64 {
    let outprops = inprops & K_ERROR;
    outprops | K_NULL_PROPERTIES | static_props
}

#[inline]
pub fn delete_arcs_properties(inprops: u64) -> u64 {
    inprops & K_DELETE_ARCS_PROPERTIES
}

pub fn closure_properties(inprops: u64, _star: bool, delayed: bool) -> u64 {
    let mut outprops = (K_ERROR | K_ACCEPTOR | K_UNWEIGHTED | K_ACCESSIBLE) & inprops;
    if (inprops & K_UNWEIGHTED) != 0 {
        outprops |= K_UNWEIGHTED_CYCLES;
    }
    if !delayed {
        outprops |=
            (K_EXPANDED | K_MUTABLE | K_CO_ACCESSIBLE | K_NOT_TOP_SORTED | K_NOT_STRING) & inprops;
    }
    if !delayed || (inprops & K_ACCESSIBLE) != 0 {
        outprops |= (K_NOT_ACCEPTOR
            | K_NON_I_DETERMINISTIC
            | K_NON_O_DETERMINISTIC
            | K_NOT_I_LABEL_SORTED
            | K_NOT_O_LABEL_SORTED
            | K_WEIGHTED
            | K_WEIGHTED_CYCLES
            | K_NOT_ACCESSIBLE
            | K_NOT_CO_ACCESSIBLE)
            & inprops;
        if (inprops & K_WEIGHTED) != 0
            && (inprops & K_ACCESSIBLE) != 0
            && (inprops & K_CO_ACCESSIBLE) != 0
        {
            outprops |= K_WEIGHTED_CYCLES;
        }
    }
    outprops
}

pub fn complement_properties(inprops: u64) -> u64 {
    let mut outprops = K_ACCEPTOR
        | K_UNWEIGHTED
        | K_UNWEIGHTED_CYCLES
        | K_NO_EPSILONS
        | K_NO_I_EPSILONS
        | K_NO_O_EPSILONS
        | K_I_DETERMINISTIC
        | K_O_DETERMINISTIC
        | K_ACCESSIBLE;

    outprops |= (K_ERROR | K_I_LABEL_SORTED | K_O_LABEL_SORTED | K_INITIAL_CYCLIC) & inprops;

    if (inprops & K_ACCESSIBLE) != 0 {
        outprops |= K_NOT_I_LABEL_SORTED | K_NOT_O_LABEL_SORTED | K_CYCLIC;
    }
    outprops
}

pub fn compose_properties(inprops1: u64, inprops2: u64) -> u64 {
    let mut outprops = K_ERROR & (inprops1 | inprops2);

    if (inprops1 & K_ACCEPTOR) != 0 && (inprops2 & K_ACCEPTOR) != 0 {
        outprops |= K_ACCEPTOR | K_ACCESSIBLE;
        outprops |=
            (K_NO_EPSILONS | K_NO_I_EPSILONS | K_NO_O_EPSILONS | K_ACYCLIC | K_INITIAL_ACYCLIC)
                & inprops1
                & inprops2;
        if (K_NO_I_EPSILONS & inprops1 & inprops2) != 0 {
            outprops |= (K_I_DETERMINISTIC | K_O_DETERMINISTIC) & inprops1 & inprops2;
        }
    } else {
        outprops |= K_ACCESSIBLE;
        outprops |=
            (K_ACCEPTOR | K_NO_I_EPSILONS | K_ACYCLIC | K_INITIAL_ACYCLIC) & inprops1 & inprops2;
        if (K_NO_I_EPSILONS & inprops1 & inprops2) != 0 {
            outprops |= K_I_DETERMINISTIC & inprops1 & inprops2;
        }
    }
    outprops
}

pub fn concat_properties(inprops1: u64, inprops2: u64, delayed: bool) -> u64 {
    let mut outprops =
        (K_ACCEPTOR | K_UNWEIGHTED | K_UNWEIGHTED_CYCLES | K_ACYCLIC) & inprops1 & inprops2;
    outprops |= K_ERROR & (inprops1 | inprops2);

    let empty1 = delayed;
    let empty2 = delayed;

    if !delayed {
        outprops |= (K_EXPANDED | K_MUTABLE | K_NOT_TOP_SORTED | K_NOT_STRING) & inprops1;
        outprops |= (K_NOT_TOP_SORTED | K_NOT_STRING) & inprops2;
    }
    if !empty1 {
        outprops |= (K_INITIAL_ACYCLIC | K_INITIAL_CYCLIC) & inprops1;
    }

    if !delayed || (inprops1 & K_ACCESSIBLE) != 0 {
        outprops |= (K_NOT_ACCEPTOR
            | K_NON_I_DETERMINISTIC
            | K_NON_O_DETERMINISTIC
            | K_EPSILONS
            | K_I_EPSILONS
            | K_O_EPSILONS
            | K_NOT_I_LABEL_SORTED
            | K_NOT_O_LABEL_SORTED
            | K_WEIGHTED
            | K_WEIGHTED_CYCLES
            | K_CYCLIC
            | K_NOT_ACCESSIBLE
            | K_NOT_CO_ACCESSIBLE)
            & inprops1;
    }

    if (inprops1 & (K_ACCESSIBLE | K_CO_ACCESSIBLE)) == (K_ACCESSIBLE | K_CO_ACCESSIBLE) && !empty1
    {
        outprops |= K_ACCESSIBLE & inprops2;
        if !empty2 {
            outprops |= K_CO_ACCESSIBLE & inprops2;
        }
        if !delayed || (inprops2 & K_ACCESSIBLE) != 0 {
            outprops |= (K_NOT_ACCEPTOR
                | K_NON_I_DETERMINISTIC
                | K_NON_O_DETERMINISTIC
                | K_EPSILONS
                | K_I_EPSILONS
                | K_O_EPSILONS
                | K_NOT_I_LABEL_SORTED
                | K_NOT_O_LABEL_SORTED
                | K_WEIGHTED
                | K_WEIGHTED_CYCLES
                | K_CYCLIC
                | K_NOT_ACCESSIBLE
                | K_NOT_CO_ACCESSIBLE)
                & inprops2;
        }
    }
    outprops
}

pub fn determinize_properties(
    inprops: u64,
    has_subsequential_label: bool,
    distinct_psubsequential_labels: bool,
) -> u64 {
    let mut outprops = K_ACCESSIBLE;

    if (K_ACCEPTOR & inprops) != 0
        || ((K_NO_I_EPSILONS & inprops) != 0 && distinct_psubsequential_labels)
        || (has_subsequential_label && distinct_psubsequential_labels)
    {
        outprops |= K_I_DETERMINISTIC;
    }

    outprops |= (K_ERROR | K_ACCEPTOR | K_ACYCLIC | K_INITIAL_ACYCLIC | K_CO_ACCESSIBLE | K_STRING)
        & inprops;

    if (inprops & K_NO_I_EPSILONS) != 0 && distinct_psubsequential_labels {
        outprops |= K_NO_EPSILONS & inprops;
    }

    if (inprops & K_ACCESSIBLE) != 0 {
        outprops |= (K_I_EPSILONS | K_O_EPSILONS | K_CYCLIC) & inprops;
    }

    if (inprops & K_ACCEPTOR) != 0 {
        outprops |= (K_NO_I_EPSILONS | K_NO_O_EPSILONS) & inprops;
    }

    if (inprops & K_NO_I_EPSILONS) != 0 && has_subsequential_label {
        outprops |= K_NO_I_EPSILONS;
    }

    outprops
}

pub fn factor_weight_properties(inprops: u64) -> u64 {
    let mut outprops = (K_EXPANDED
        | K_MUTABLE
        | K_ERROR
        | K_ACCEPTOR
        | K_ACYCLIC
        | K_ACCESSIBLE
        | K_CO_ACCESSIBLE)
        & inprops;

    if (inprops & K_ACCESSIBLE) != 0 {
        outprops |= (K_NOT_ACCEPTOR
            | K_NON_I_DETERMINISTIC
            | K_NON_O_DETERMINISTIC
            | K_EPSILONS
            | K_I_EPSILONS
            | K_O_EPSILONS
            | K_CYCLIC
            | K_NOT_I_LABEL_SORTED
            | K_NOT_O_LABEL_SORTED)
            & inprops;
    }
    outprops
}

pub fn invert_properties(inprops: u64) -> u64 {
    let mut outprops = (K_EXPANDED
        | K_MUTABLE
        | K_ERROR
        | K_ACCEPTOR
        | K_NOT_ACCEPTOR
        | K_EPSILONS
        | K_NO_EPSILONS
        | K_WEIGHTED
        | K_UNWEIGHTED
        | K_WEIGHTED_CYCLES
        | K_UNWEIGHTED_CYCLES
        | K_CYCLIC
        | K_ACYCLIC
        | K_INITIAL_CYCLIC
        | K_INITIAL_ACYCLIC
        | K_TOP_SORTED
        | K_NOT_TOP_SORTED
        | K_ACCESSIBLE
        | K_NOT_ACCESSIBLE
        | K_CO_ACCESSIBLE
        | K_NOT_CO_ACCESSIBLE
        | K_STRING
        | K_NOT_STRING)
        & inprops;

    if (K_I_DETERMINISTIC & inprops) != 0 {
        outprops |= K_O_DETERMINISTIC;
    }
    if (K_NON_I_DETERMINISTIC & inprops) != 0 {
        outprops |= K_NON_O_DETERMINISTIC;
    }
    if (K_O_DETERMINISTIC & inprops) != 0 {
        outprops |= K_I_DETERMINISTIC;
    }
    if (K_NON_O_DETERMINISTIC & inprops) != 0 {
        outprops |= K_NON_I_DETERMINISTIC;
    }

    if (K_I_EPSILONS & inprops) != 0 {
        outprops |= K_O_EPSILONS;
    }
    if (K_NO_I_EPSILONS & inprops) != 0 {
        outprops |= K_NO_O_EPSILONS;
    }
    if (K_O_EPSILONS & inprops) != 0 {
        outprops |= K_I_EPSILONS;
    }
    if (K_NO_O_EPSILONS & inprops) != 0 {
        outprops |= K_NO_I_EPSILONS;
    }

    if (K_I_LABEL_SORTED & inprops) != 0 {
        outprops |= K_O_LABEL_SORTED;
    }
    if (K_NOT_I_LABEL_SORTED & inprops) != 0 {
        outprops |= K_NOT_O_LABEL_SORTED;
    }
    if (K_O_LABEL_SORTED & inprops) != 0 {
        outprops |= K_I_LABEL_SORTED;
    }
    if (K_NOT_O_LABEL_SORTED & inprops) != 0 {
        outprops |= K_NOT_I_LABEL_SORTED;
    }

    outprops
}

pub fn project_properties(inprops: u64, project_input: bool) -> u64 {
    let mut outprops = K_ACCEPTOR;
    outprops |= (K_EXPANDED
        | K_MUTABLE
        | K_ERROR
        | K_WEIGHTED
        | K_UNWEIGHTED
        | K_WEIGHTED_CYCLES
        | K_UNWEIGHTED_CYCLES
        | K_CYCLIC
        | K_ACYCLIC
        | K_INITIAL_CYCLIC
        | K_INITIAL_ACYCLIC
        | K_TOP_SORTED
        | K_NOT_TOP_SORTED
        | K_ACCESSIBLE
        | K_NOT_ACCESSIBLE
        | K_CO_ACCESSIBLE
        | K_NOT_CO_ACCESSIBLE
        | K_STRING
        | K_NOT_STRING)
        & inprops;

    if project_input {
        outprops |= (K_I_DETERMINISTIC
            | K_NON_I_DETERMINISTIC
            | K_I_EPSILONS
            | K_NO_I_EPSILONS
            | K_I_LABEL_SORTED
            | K_NOT_I_LABEL_SORTED)
            & inprops;

        if (K_I_DETERMINISTIC & inprops) != 0 {
            outprops |= K_O_DETERMINISTIC;
        }
        if (K_NON_I_DETERMINISTIC & inprops) != 0 {
            outprops |= K_NON_O_DETERMINISTIC;
        }

        if (K_I_EPSILONS & inprops) != 0 {
            outprops |= K_O_EPSILONS | K_EPSILONS;
        }
        if (K_NO_I_EPSILONS & inprops) != 0 {
            outprops |= K_NO_O_EPSILONS | K_NO_EPSILONS;
        }

        if (K_I_LABEL_SORTED & inprops) != 0 {
            outprops |= K_O_LABEL_SORTED;
        }
        if (K_NOT_I_LABEL_SORTED & inprops) != 0 {
            outprops |= K_NOT_O_LABEL_SORTED;
        }
    } else {
        outprops |= (K_O_DETERMINISTIC
            | K_NON_O_DETERMINISTIC
            | K_O_EPSILONS
            | K_NO_O_EPSILONS
            | K_O_LABEL_SORTED
            | K_NOT_O_LABEL_SORTED)
            & inprops;

        if (K_O_DETERMINISTIC & inprops) != 0 {
            outprops |= K_I_DETERMINISTIC;
        }
        if (K_NON_O_DETERMINISTIC & inprops) != 0 {
            outprops |= K_NON_I_DETERMINISTIC;
        }

        if (K_O_EPSILONS & inprops) != 0 {
            outprops |= K_I_EPSILONS | K_EPSILONS;
        }
        if (K_NO_O_EPSILONS & inprops) != 0 {
            outprops |= K_NO_I_EPSILONS | K_NO_EPSILONS;
        }

        if (K_O_LABEL_SORTED & inprops) != 0 {
            outprops |= K_I_LABEL_SORTED;
        }
        if (K_NOT_O_LABEL_SORTED & inprops) != 0 {
            outprops |= K_NOT_I_LABEL_SORTED;
        }
    }
    outprops
}

pub fn rand_gen_properties(inprops: u64, weighted: bool) -> u64 {
    let mut outprops = K_ACYCLIC | K_INITIAL_ACYCLIC | K_ACCESSIBLE | K_UNWEIGHTED_CYCLES;
    outprops |= inprops & K_ERROR;

    if weighted {
        outprops |= K_TOP_SORTED;
        outprops |= (K_ACCEPTOR
            | K_NO_EPSILONS
            | K_NO_I_EPSILONS
            | K_NO_O_EPSILONS
            | K_I_DETERMINISTIC
            | K_O_DETERMINISTIC
            | K_I_LABEL_SORTED
            | K_O_LABEL_SORTED)
            & inprops;
    } else {
        outprops |= K_UNWEIGHTED;
        outprops |= (K_ACCEPTOR | K_I_LABEL_SORTED | K_O_LABEL_SORTED) & inprops;
    }
    outprops
}

pub fn replace_properties(
    inprops: &[u64],
    root: usize,
    epsilon_on_call: bool,
    epsilon_on_return: bool,
    out_epsilon_on_call: bool,
    out_epsilon_on_return: bool,
    replace_transducer: bool,
    no_empty_fsts: bool,
    all_ilabel_sorted: bool,
    all_olabel_sorted: bool,
    all_negative_or_dense: bool,
) -> u64 {
    if inprops.is_empty() {
        return K_NULL_PROPERTIES;
    }

    let mut outprops = 0;
    for inprop in inprops {
        outprops |= K_ERROR & inprop;
    }

    let mut access_props = if no_empty_fsts {
        K_ACCESSIBLE | K_CO_ACCESSIBLE
    } else {
        0
    };
    for inprop in inprops {
        access_props &= inprop & (K_ACCESSIBLE | K_CO_ACCESSIBLE);
    }

    if access_props == (K_ACCESSIBLE | K_CO_ACCESSIBLE) {
        outprops |= access_props;
        if (inprops.get(root).copied().unwrap_or(0) & K_INITIAL_CYCLIC) != 0 {
            outprops |= K_INITIAL_CYCLIC;
        }

        let mut props = 0;
        let mut string = true;
        for inprop in inprops {
            if replace_transducer {
                props |= K_NOT_ACCEPTOR & inprop;
            }
            props |= (K_NON_I_DETERMINISTIC
                | K_NON_O_DETERMINISTIC
                | K_EPSILONS
                | K_I_EPSILONS
                | K_O_EPSILONS
                | K_WEIGHTED
                | K_WEIGHTED_CYCLES
                | K_CYCLIC
                | K_NOT_TOP_SORTED
                | K_NOT_STRING)
                & inprop;

            if (inprop & K_STRING) == 0 {
                string = false;
            }
        }
        outprops |= props;
        if string {
            outprops |= K_STRING;
        }
    }

    let mut acceptor = !replace_transducer;
    let mut ideterministic = !epsilon_on_call && epsilon_on_return;
    let mut no_iepsilons = !epsilon_on_call && !epsilon_on_return;
    let mut acyclic = true;
    let mut unweighted = true;

    for (i, inprop) in inprops.iter().enumerate() {
        if (inprop & K_ACCEPTOR) == 0 {
            acceptor = false;
        }
        if (inprop & K_I_DETERMINISTIC) == 0 {
            ideterministic = false;
        }
        if (inprop & K_NO_I_EPSILONS) == 0 {
            no_iepsilons = false;
        }
        if (inprop & K_ACYCLIC) == 0 {
            acyclic = false;
        }
        if (inprop & K_UNWEIGHTED) == 0 {
            unweighted = false;
        }
        if i != root && (inprop & K_NO_I_EPSILONS) == 0 {
            ideterministic = false;
        }
    }

    if acceptor {
        outprops |= K_ACCEPTOR;
    }
    if ideterministic {
        outprops |= K_I_DETERMINISTIC;
    }
    if no_iepsilons {
        outprops |= K_NO_I_EPSILONS;
    }
    if acyclic {
        outprops |= K_ACYCLIC;
    }
    if unweighted {
        outprops |= K_UNWEIGHTED;
    }

    if (inprops.get(root).copied().unwrap_or(0) & K_INITIAL_ACYCLIC) != 0 {
        outprops |= K_INITIAL_ACYCLIC;
    }

    if all_ilabel_sorted && epsilon_on_return && (!epsilon_on_call || all_negative_or_dense) {
        outprops |= K_I_LABEL_SORTED;
    }

    if all_olabel_sorted && out_epsilon_on_return && (!out_epsilon_on_call || all_negative_or_dense)
    {
        outprops |= K_O_LABEL_SORTED;
    }

    outprops
}

pub fn relabel_properties(inprops: u64) -> u64 {
    const OUTPROPS: u64 = K_EXPANDED
        | K_MUTABLE
        | K_ERROR
        | K_WEIGHTED
        | K_UNWEIGHTED
        | K_WEIGHTED_CYCLES
        | K_UNWEIGHTED_CYCLES
        | K_CYCLIC
        | K_ACYCLIC
        | K_INITIAL_CYCLIC
        | K_INITIAL_ACYCLIC
        | K_TOP_SORTED
        | K_NOT_TOP_SORTED
        | K_ACCESSIBLE
        | K_NOT_ACCESSIBLE
        | K_CO_ACCESSIBLE
        | K_NOT_CO_ACCESSIBLE
        | K_STRING
        | K_NOT_STRING;
    OUTPROPS & inprops
}

pub fn reverse_properties(inprops: u64, has_superinitial: bool) -> u64 {
    let mut outprops = (K_EXPANDED
        | K_MUTABLE
        | K_ERROR
        | K_ACCEPTOR
        | K_NOT_ACCEPTOR
        | K_EPSILONS
        | K_I_EPSILONS
        | K_O_EPSILONS
        | K_UNWEIGHTED
        | K_CYCLIC
        | K_ACYCLIC
        | K_WEIGHTED_CYCLES
        | K_UNWEIGHTED_CYCLES)
        & inprops;
    if has_superinitial {
        outprops |= K_WEIGHTED & inprops;
    }
    outprops
}

pub fn reweight_properties(inprops: u64, added_start_epsilon: bool) -> u64 {
    let mut outprops = inprops & K_WEIGHT_INVARIANT_PROPERTIES;
    outprops &= !K_CO_ACCESSIBLE;

    if added_start_epsilon {
        outprops &= !(K_NO_EPSILONS | K_NO_I_EPSILONS | K_NO_O_EPSILONS | K_INITIAL_CYCLIC);
        outprops |= K_EPSILONS | K_I_EPSILONS | K_O_EPSILONS | K_INITIAL_ACYCLIC;
    }
    outprops
}

pub fn rm_epsilon_properties(inprops: u64, delayed: bool) -> u64 {
    let mut outprops = K_NO_EPSILONS;
    outprops |= (K_ERROR | K_ACCEPTOR | K_ACYCLIC | K_INITIAL_ACYCLIC) & inprops;

    if (inprops & K_ACCEPTOR) != 0 {
        outprops |= K_NO_I_EPSILONS | K_NO_O_EPSILONS;
    }
    if !delayed {
        outprops |= K_EXPANDED | K_MUTABLE;
        outprops |= K_TOP_SORTED & inprops;
    }
    if !delayed || (inprops & K_ACCESSIBLE) != 0 {
        outprops |= K_NOT_ACCEPTOR & inprops;
    }
    outprops
}

pub fn shortest_path_properties(props: u64, tree: bool) -> u64 {
    let mut outprops = props | K_ACYCLIC | K_INITIAL_ACYCLIC | K_ACCESSIBLE | K_UNWEIGHTED_CYCLES;
    if !tree {
        outprops |= K_CO_ACCESSIBLE;
    }
    outprops
}

pub fn synchronize_properties(inprops: u64) -> u64 {
    let mut outprops = (K_ERROR
        | K_ACCEPTOR
        | K_ACYCLIC
        | K_ACCESSIBLE
        | K_CO_ACCESSIBLE
        | K_UNWEIGHTED
        | K_UNWEIGHTED_CYCLES)
        & inprops;

    if (inprops & K_ACCESSIBLE) != 0 {
        outprops |= (K_CYCLIC | K_NOT_CO_ACCESSIBLE | K_WEIGHTED | K_WEIGHTED_CYCLES) & inprops;
    }
    outprops
}

pub fn union_properties(inprops1: u64, inprops2: u64, delayed: bool) -> u64 {
    let mut outprops = (K_ACCEPTOR | K_UNWEIGHTED | K_UNWEIGHTED_CYCLES | K_ACYCLIC | K_ACCESSIBLE)
        & inprops1
        & inprops2;

    outprops |= K_ERROR & (inprops1 | inprops2);
    outprops |= K_INITIAL_ACYCLIC;

    let empty1 = delayed;
    let empty2 = delayed;

    if !delayed {
        outprops |= (K_EXPANDED | K_MUTABLE | K_NOT_TOP_SORTED) & inprops1;
        outprops |= K_NOT_TOP_SORTED & inprops2;
    }

    if !empty1 && !empty2 {
        outprops |= K_EPSILONS | K_I_EPSILONS | K_O_EPSILONS;
        outprops |= K_CO_ACCESSIBLE & inprops1 & inprops2;
    }

    if !delayed || (inprops1 & K_ACCESSIBLE) != 0 {
        outprops |= (K_NOT_ACCEPTOR
            | K_NON_I_DETERMINISTIC
            | K_NON_O_DETERMINISTIC
            | K_EPSILONS
            | K_I_EPSILONS
            | K_O_EPSILONS
            | K_NOT_I_LABEL_SORTED
            | K_NOT_O_LABEL_SORTED
            | K_WEIGHTED
            | K_WEIGHTED_CYCLES
            | K_CYCLIC
            | K_NOT_ACCESSIBLE)
            & inprops1;
    }

    if !delayed || (inprops2 & K_ACCESSIBLE) != 0 {
        outprops |= (K_NOT_ACCEPTOR
            | K_NON_I_DETERMINISTIC
            | K_NON_O_DETERMINISTIC
            | K_EPSILONS
            | K_I_EPSILONS
            | K_O_EPSILONS
            | K_NOT_I_LABEL_SORTED
            | K_NOT_O_LABEL_SORTED
            | K_WEIGHTED
            | K_WEIGHTED_CYCLES
            | K_CYCLIC
            | K_NOT_ACCESSIBLE
            | K_NOT_CO_ACCESSIBLE)
            & inprops2;
    }

    outprops
}

/// Names of the property bits, indexed by bit position.
///
/// Mirrors upstream's `internal::PropertyNames`. Bits 3..15 are reserved and
/// have no name.
pub const PROPERTY_NAMES: [&str; 48] = [
    // Binary.
    "expanded",
    "mutable",
    "error",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    // Ternary.
    "acceptor",
    "not acceptor",
    "input deterministic",
    "non input deterministic",
    "output deterministic",
    "non output deterministic",
    "input/output epsilons",
    "no input/output epsilons",
    "input epsilons",
    "no input epsilons",
    "output epsilons",
    "no output epsilons",
    "input label sorted",
    "not input label sorted",
    "output label sorted",
    "not output label sorted",
    "weighted",
    "unweighted",
    "cyclic",
    "acyclic",
    "cyclic at initial state",
    "acyclic at initial state",
    "top sorted",
    "not top sorted",
    "accessible",
    "not accessible",
    "coaccessible",
    "not coaccessible",
    "string",
    "not string",
    "weighted cycles",
    "unweighted cycles",
];

/// The name of the property held in bit `bit`, or `""` if it has none.
pub fn property_name(bit: usize) -> &'static str {
    PROPERTY_NAMES.get(bit).copied().unwrap_or("")
}

pub mod internal {
    use super::*;

    /// Constructs a fully expanded known-properties mask based on the input properties.
    pub fn known_properties(props: u64) -> u64 {
        K_BINARY_PROPERTIES
            | (props & K_TRINARY_PROPERTIES)
            | ((props & K_POS_TRINARY_PROPERTIES) << 1)
            | ((props & K_NEG_TRINARY_PROPERTIES) >> 1)
    }

    /// The bits on which two property sets disagree, restricted to the bits both
    /// sides actually claim to know.
    pub fn incompatible_properties(props1: u64, props2: u64) -> u64 {
        let known_props = known_properties(props1) & known_properties(props2);
        (props1 & known_props) ^ (props2 & known_props)
    }

    /// Tests compatibility between two sets of properties.
    pub fn compat_properties(props1: u64, props2: u64) -> bool {
        incompatible_properties(props1, props2) == 0
    }

    /// Describes every disagreement between two property sets, one per line.
    ///
    /// SICADA-DIVERGE: upstream's `CompatProperties` writes this straight to the
    /// global log and returns only a bool, so a caller cannot report or test what
    /// actually mismatched. Returning the text leaves that choice to the caller,
    /// the same reasoning as `compat_symbols_with_warn` taking a writer.
    pub fn describe_incompatible_properties(props1: u64, props2: u64) -> String {
        let incompat_props = incompatible_properties(props1, props2);
        let mut description = String::new();
        for bit in 0..u64::BITS as usize {
            let mask = 1u64 << bit;
            if mask & incompat_props == 0 {
                continue;
            }
            let _ = writeln!(
                description,
                "{}: {} vs {}",
                property_name(bit),
                props1 & mask != 0,
                props2 & mask != 0
            );
        }
        description
    }
}

/// A marker trait for expected FST properties.
pub trait FstProperty {
    const MASK: u64;
    const EXPECTED: u64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Acceptor;
impl FstProperty for Acceptor {
    const MASK: u64 = K_ACCEPTOR;
    const EXPECTED: u64 = K_ACCEPTOR;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DetEpsFreeAcceptor;
impl FstProperty for DetEpsFreeAcceptor {
    const MASK: u64 = K_ACCEPTOR | K_I_DETERMINISTIC | K_NO_EPSILONS;
    const EXPECTED: u64 = K_ACCEPTOR | K_I_DETERMINISTIC | K_NO_EPSILONS;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnweightedDetEpsFreeAcceptor;
impl FstProperty for UnweightedDetEpsFreeAcceptor {
    const MASK: u64 = K_ACCEPTOR | K_I_DETERMINISTIC | K_NO_EPSILONS | K_UNWEIGHTED;
    const EXPECTED: u64 = K_ACCEPTOR | K_I_DETERMINISTIC | K_NO_EPSILONS | K_UNWEIGHTED;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Acyclic;
impl FstProperty for Acyclic {
    const MASK: u64 = K_ACYCLIC;
    const EXPECTED: u64 = K_ACYCLIC;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StringFst;
impl FstProperty for StringFst {
    const MASK: u64 = K_STRING;
    const EXPECTED: u64 = K_STRING;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Verified<F, const P: u64> {
    inner: F,
}

impl<F, const P: u64> Verified<F, P> {
    #[inline(always)]
    pub fn assume(inner: F) -> Self {
        Self { inner }
    }

    #[inline(always)]
    pub fn into_inner(self) -> F {
        self.inner
    }
}

impl<F, const P: u64> Deref for Verified<F, P> {
    type Target = F;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<F, const P: u64> DerefMut for Verified<F, P> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

pub trait VerifyExt<A: Arc>: Sized {
    fn verify<const P: u64>(self) -> Result<Verified<Self, P>, OpenFstError>;
}

impl<A: Arc, F: Fst<A>> VerifyExt<A> for F {
    fn verify<const P: u64>(self) -> Result<Verified<Self, P>, OpenFstError> {
        let actual = self.properties(P, true) & P;
        if actual == P {
            Ok(Verified::assume(self))
        } else {
            Err(OpenFstError::PropertyVerificationFailed {
                mask: P,
                expected: P,
                actual,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::internal::{compat_properties, known_properties};
    use super::*;
    use crate::arc::StdArc;
    use crate::weights::float_weight::TropicalWeight;

    /// Every property bit, pinned to the value OpenFst assigns it.
    ///
    /// These bits are written into the FST file header, so a change here breaks
    /// binary compatibility with files OpenFst produced. The expected values were
    /// taken by compiling the constant definitions out of
    /// `vendor/openfst/openfst/lib/properties.h` and printing them; that program
    /// is `tests/oracles/property-bits.cc`.
    #[test]
    fn property_bits_match_openfst() {
        const EXPECTED: &[(u64, u64, &str)] = &[
            (K_EXPANDED, 0x0000000000000001, "kExpanded"),
            (K_MUTABLE, 0x0000000000000002, "kMutable"),
            (K_ERROR, 0x0000000000000004, "kError"),
            (K_ACCEPTOR, 0x0000000000010000, "kAcceptor"),
            (K_NOT_ACCEPTOR, 0x0000000000020000, "kNotAcceptor"),
            (K_I_DETERMINISTIC, 0x0000000000040000, "kIDeterministic"),
            (
                K_NON_I_DETERMINISTIC,
                0x0000000000080000,
                "kNonIDeterministic",
            ),
            (K_O_DETERMINISTIC, 0x0000000000100000, "kODeterministic"),
            (
                K_NON_O_DETERMINISTIC,
                0x0000000000200000,
                "kNonODeterministic",
            ),
            (K_EPSILONS, 0x0000000000400000, "kEpsilons"),
            (K_NO_EPSILONS, 0x0000000000800000, "kNoEpsilons"),
            (K_I_EPSILONS, 0x0000000001000000, "kIEpsilons"),
            (K_NO_I_EPSILONS, 0x0000000002000000, "kNoIEpsilons"),
            (K_O_EPSILONS, 0x0000000004000000, "kOEpsilons"),
            (K_NO_O_EPSILONS, 0x0000000008000000, "kNoOEpsilons"),
            (K_I_LABEL_SORTED, 0x0000000010000000, "kILabelSorted"),
            (K_NOT_I_LABEL_SORTED, 0x0000000020000000, "kNotILabelSorted"),
            (K_O_LABEL_SORTED, 0x0000000040000000, "kOLabelSorted"),
            (K_NOT_O_LABEL_SORTED, 0x0000000080000000, "kNotOLabelSorted"),
            (K_WEIGHTED, 0x0000000100000000, "kWeighted"),
            (K_UNWEIGHTED, 0x0000000200000000, "kUnweighted"),
            (K_CYCLIC, 0x0000000400000000, "kCyclic"),
            (K_ACYCLIC, 0x0000000800000000, "kAcyclic"),
            (K_INITIAL_CYCLIC, 0x0000001000000000, "kInitialCyclic"),
            (K_INITIAL_ACYCLIC, 0x0000002000000000, "kInitialAcyclic"),
            (K_TOP_SORTED, 0x0000004000000000, "kTopSorted"),
            (K_NOT_TOP_SORTED, 0x0000008000000000, "kNotTopSorted"),
            (K_ACCESSIBLE, 0x0000010000000000, "kAccessible"),
            (K_NOT_ACCESSIBLE, 0x0000020000000000, "kNotAccessible"),
            (K_CO_ACCESSIBLE, 0x0000040000000000, "kCoAccessible"),
            (K_NOT_CO_ACCESSIBLE, 0x0000080000000000, "kNotCoAccessible"),
            (K_STRING, 0x0000100000000000, "kString"),
            (K_NOT_STRING, 0x0000200000000000, "kNotString"),
            (K_WEIGHTED_CYCLES, 0x0000400000000000, "kWeightedCycles"),
            (K_UNWEIGHTED_CYCLES, 0x0000800000000000, "kUnweightedCycles"),
            (K_NULL_PROPERTIES, 0x0000956a5a950000, "kNullProperties"),
            (
                K_COMPILED_STRING_PROPERTIES,
                0x0000956a50150000,
                "kCompiledStringProperties",
            ),
            (K_COPY_PROPERTIES, 0x0000ffffffff0004, "kCopyProperties"),
            (
                K_INTRINSIC_PROPERTIES,
                0x0000ffffffff0003,
                "kIntrinsicProperties",
            ),
            (
                K_EXTRINSIC_PROPERTIES,
                0x0000000000000004,
                "kExtrinsicProperties",
            ),
            (
                K_SET_START_PROPERTIES,
                0x0000cccfffff0007,
                "kSetStartProperties",
            ),
            (
                K_SET_FINAL_PROPERTIES,
                0x0000c3fcffff0007,
                "kSetFinalProperties",
            ),
            (
                K_ADD_STATE_PROPERTIES,
                0x0000eaffffff0007,
                "kAddStateProperties",
            ),
            (
                K_ADD_ARC_PROPERTIES,
                0x00004595a56a0007,
                "kAddArcProperties",
            ),
            (
                K_SET_ARC_PROPERTIES,
                0x0000000000000007,
                "kSetArcProperties",
            ),
            (
                K_DELETE_STATES_PROPERTIES,
                0x0000806a5a950007,
                "kDeleteStatesProperties",
            ),
            (
                K_DELETE_ARCS_PROPERTIES,
                0x00008a6a5a950007,
                "kDeleteArcsProperties",
            ),
            (
                K_STATE_SORT_PROPERTIES,
                0x0000cf3fffff0007,
                "kStateSortProperties",
            ),
            (
                K_ARC_SORT_PROPERTIES,
                0x0000ffff0fff0007,
                "kArcSortProperties",
            ),
            (
                K_I_LABEL_INVARIANT_PROPERTIES,
                0x0000ffffcc300007,
                "kILabelInvariantProperties",
            ),
            (
                K_O_LABEL_INVARIANT_PROPERTIES,
                0x0000ffff330c0007,
                "kOLabelInvariantProperties",
            ),
            (
                K_WEIGHT_INVARIANT_PROPERTIES,
                0x00003ffcffff0007,
                "kWeightInvariantProperties",
            ),
            (
                K_ADD_SUPER_FINAL_PROPERTIES,
                0x0000eebfa56b0007,
                "kAddSuperFinalProperties",
            ),
            (
                K_RM_SUPER_FINAL_PROPERTIES,
                0x0000dd7f5a970007,
                "kRmSuperFinalProperties",
            ),
            (K_BINARY_PROPERTIES, 0x0000000000000007, "kBinaryProperties"),
            (
                K_TRINARY_PROPERTIES,
                0x0000ffffffff0000,
                "kTrinaryProperties",
            ),
            (
                K_POS_TRINARY_PROPERTIES,
                0x0000555555550000,
                "kPosTrinaryProperties",
            ),
            (
                K_NEG_TRINARY_PROPERTIES,
                0x0000aaaaaaaa0000,
                "kNegTrinaryProperties",
            ),
            (K_FST_PROPERTIES, 0x0000ffffffff0007, "kFstProperties"),
        ];
        assert_eq!(EXPECTED.len(), 59, "properties.h defines 59 constants");
        for &(actual, expected, name) in EXPECTED {
            assert_eq!(
                actual, expected,
                "{name} is {actual:#018x}, OpenFst uses {expected:#018x}"
            );
        }
    }

    /// A trinary property occupies two adjacent bits, the positive one below the
    /// negative one. `known_properties` relies on that layout to widen a
    /// one-sided mask.
    #[test]
    fn trinary_properties_are_adjacent_bit_pairs() {
        assert_eq!(K_POS_TRINARY_PROPERTIES << 1, K_NEG_TRINARY_PROPERTIES);
        assert_eq!(
            K_POS_TRINARY_PROPERTIES | K_NEG_TRINARY_PROPERTIES,
            K_TRINARY_PROPERTIES
        );
        assert_eq!(
            K_POS_TRINARY_PROPERTIES & K_NEG_TRINARY_PROPERTIES,
            0,
            "the two halves must not overlap"
        );
        assert_eq!(
            K_BINARY_PROPERTIES & K_TRINARY_PROPERTIES,
            0,
            "a property is either binary or trinary"
        );
        assert_eq!(
            K_BINARY_PROPERTIES | K_TRINARY_PROPERTIES,
            K_FST_PROPERTIES,
            "every property is accounted for"
        );
    }

    #[test]
    fn known_properties_widens_one_sided_trinary_bits() {
        // Knowing the positive half implies knowing the negative half, and back.
        assert_ne!(known_properties(K_ACCEPTOR) & K_NOT_ACCEPTOR, 0);
        assert_ne!(known_properties(K_NOT_ACCEPTOR) & K_ACCEPTOR, 0);
        // Binary properties are always considered known.
        assert_eq!(
            known_properties(0) & K_BINARY_PROPERTIES,
            K_BINARY_PROPERTIES
        );
        // Nothing is claimed about a trinary property that was never set.
        assert_eq!(known_properties(0) & K_TRINARY_PROPERTIES, 0);
    }

    #[test]
    fn compat_properties_only_compares_what_both_sides_know() {
        assert!(compat_properties(K_ACCEPTOR, K_ACCEPTOR));
        assert!(!compat_properties(K_ACCEPTOR, K_NOT_ACCEPTOR));
        // One side says nothing about being an acceptor, so there is no conflict.
        assert!(compat_properties(K_ACCEPTOR, 0));
        assert!(compat_properties(0, K_NOT_ACCEPTOR));
        // Unrelated properties do not interfere.
        assert!(compat_properties(K_ACCEPTOR | K_ACYCLIC, K_ACCEPTOR));
    }

    #[test]
    fn set_start_implies_initial_acyclic_for_an_acyclic_fst() {
        assert_ne!(set_start_properties(K_ACYCLIC) & K_INITIAL_ACYCLIC, 0);
        assert_eq!(set_start_properties(K_CYCLIC) & K_INITIAL_ACYCLIC, 0);
        // Everything else is narrowed to the mask. kInitialAcyclic is added back
        // after masking, so it is the one bit allowed to escape it.
        assert_eq!(
            set_start_properties(!0) & !(K_SET_START_PROPERTIES | K_INITIAL_ACYCLIC),
            0
        );
    }

    #[test]
    fn set_final_tracks_weightedness() {
        let zero = TropicalWeight::zero();
        let one = TropicalWeight::one();
        let other = TropicalWeight(2.5);

        // Making a state non-trivially weighted marks the FST weighted.
        let props = set_final_properties(K_UNWEIGHTED, &zero, &other);
        assert_ne!(props & K_WEIGHTED, 0);
        assert_eq!(props & K_UNWEIGHTED, 0);

        // Clearing a non-trivial weight only retracts the claim; it cannot prove
        // the FST unweighted, since other arcs may still carry weight.
        let props = set_final_properties(K_WEIGHTED, &other, &one);
        assert_eq!(props & K_WEIGHTED, 0);
        assert_eq!(props & K_UNWEIGHTED, 0);

        // Zero and One are both trivial, so nothing changes.
        let props = set_final_properties(K_UNWEIGHTED, &zero, &one);
        assert_ne!(props & K_UNWEIGHTED, 0);
    }

    #[test]
    fn delete_all_states_keeps_only_errors_and_static_properties() {
        // An emptied FST takes on the properties of the empty FST, so anything
        // that contradicts them is dropped; kError and the static properties of
        // the concrete FST type survive.
        let props = delete_all_states_properties(K_ERROR | K_NOT_ACCEPTOR | K_WEIGHTED, K_EXPANDED);
        assert_ne!(props & K_ERROR, 0);
        assert_ne!(props & K_EXPANDED, 0);
        assert_eq!(props & K_NOT_ACCEPTOR, 0);
        assert_eq!(props & K_WEIGHTED, 0);

        assert_eq!(
            delete_all_states_properties(K_NOT_ACCEPTOR, 0),
            K_NULL_PROPERTIES
        );
        // The empty FST is trivially an epsilon-free, unweighted acyclic acceptor.
        assert_ne!(K_NULL_PROPERTIES & K_ACCEPTOR, 0);
        assert_ne!(K_NULL_PROPERTIES & K_ACYCLIC, 0);
        assert_ne!(K_NULL_PROPERTIES & K_UNWEIGHTED, 0);
    }

    #[test]
    fn add_state_and_delete_masks_only_narrow() {
        for props in [0, K_ACCEPTOR, K_ACYCLIC | K_TOP_SORTED, !0] {
            assert_eq!(add_state_properties(props) & !props, 0);
            assert_eq!(delete_states_properties(props) & !props, 0);
            assert_eq!(delete_arcs_properties(props) & !props, 0);
        }
    }

    fn arc(ilabel: i32, olabel: i32, weight: f32, nextstate: i32) -> StdArc {
        use crate::arc::Arc as _;
        StdArc::new(ilabel, olabel, TropicalWeight(weight), nextstate)
    }

    #[test]
    fn add_arc_detects_a_transducer() {
        let start = K_ACCEPTOR | K_NO_EPSILONS | K_NO_I_EPSILONS | K_NO_O_EPSILONS;
        let props = add_arc_properties(start, 0, &arc(1, 2, 0.0, 1), None);
        assert_ne!(props & K_NOT_ACCEPTOR, 0);
        assert_eq!(props & K_ACCEPTOR, 0);
    }

    #[test]
    fn add_arc_classifies_epsilons() {
        let start = K_NO_EPSILONS | K_NO_I_EPSILONS | K_NO_O_EPSILONS;

        // Input epsilon only.
        let props = add_arc_properties(start, 0, &arc(0, 1, 0.0, 1), None);
        assert_ne!(props & K_I_EPSILONS, 0);
        assert_eq!(props & K_NO_I_EPSILONS, 0);
        assert_eq!(props & K_EPSILONS, 0, "not a full epsilon arc");
        assert_ne!(props & K_NO_EPSILONS, 0);

        // Output epsilon only.
        let props = add_arc_properties(start, 0, &arc(1, 0, 0.0, 1), None);
        assert_ne!(props & K_O_EPSILONS, 0);
        assert_eq!(props & K_NO_O_EPSILONS, 0);

        // Both, which is a true epsilon arc.
        let props = add_arc_properties(start, 0, &arc(0, 0, 0.0, 1), None);
        assert_ne!(props & K_EPSILONS, 0);
        assert_eq!(props & K_NO_EPSILONS, 0);
    }

    #[test]
    fn add_arc_detects_unsorted_labels() {
        let start = K_I_LABEL_SORTED | K_O_LABEL_SORTED;
        let previous = arc(5, 5, 0.0, 1);

        let props = add_arc_properties(start, 0, &arc(3, 7, 0.0, 1), Some(&previous));
        assert_ne!(props & K_NOT_I_LABEL_SORTED, 0);
        assert_eq!(props & K_I_LABEL_SORTED, 0);
        assert_ne!(
            props & K_O_LABEL_SORTED,
            0,
            "output labels are still sorted"
        );

        // Equal labels keep the ordering claim.
        let props = add_arc_properties(start, 0, &arc(5, 5, 0.0, 1), Some(&previous));
        assert_ne!(props & K_I_LABEL_SORTED, 0);
        assert_ne!(props & K_O_LABEL_SORTED, 0);
    }

    #[test]
    fn add_arc_detects_a_back_edge_and_the_topological_consequences() {
        let start = K_TOP_SORTED | K_ACYCLIC | K_INITIAL_ACYCLIC;

        // A self loop is not a forward edge.
        let props = add_arc_properties(start, 3, &arc(1, 1, 0.0, 3), None);
        assert_ne!(props & K_NOT_TOP_SORTED, 0);
        assert_eq!(props & K_TOP_SORTED, 0);
        assert_eq!(props & K_ACYCLIC, 0, "acyclicity can no longer be claimed");

        // A forward edge keeps topological order, which implies acyclicity.
        let props = add_arc_properties(start, 3, &arc(1, 1, 0.0, 4), None);
        assert_ne!(props & K_TOP_SORTED, 0);
        assert_ne!(props & K_ACYCLIC, 0);
        assert_ne!(props & K_INITIAL_ACYCLIC, 0);
    }

    #[test]
    fn add_arc_tracks_weightedness() {
        let props = add_arc_properties(K_UNWEIGHTED, 0, &arc(1, 1, 2.5, 1), None);
        assert_ne!(props & K_WEIGHTED, 0);
        assert_eq!(props & K_UNWEIGHTED, 0);

        // One is the trivial weight and does not make the FST weighted.
        let props = add_arc_properties(K_UNWEIGHTED, 0, &arc(1, 1, 0.0, 1), None);
        assert_ne!(props & K_UNWEIGHTED, 0);
        assert_eq!(props & K_WEIGHTED, 0);
    }
    /// Every property propagation function, checked against the values the C++
    /// implementation produces.
    ///
    /// These rows were taken by extracting the functions out of
    /// `vendor/openfst/openfst/lib/properties.cc` and running both
    /// implementations over the same pseudo-random inputs; all 400 rows of that
    /// run agreed, and a sample is pinned here. See
    /// tests/oracles/property-functions.cc.
    #[test]
    fn property_propagation_matches_openfst() {
        #[allow(clippy::type_complexity)]
        const CASES: &[(u64, u64, bool, bool, [u64; 17])] = &[
            (
                0x00000d6569fa0005,
                0x0000c497a6b00003,
                true,
                false,
                [
                    0x00004d01202a0005,
                    0x00008106ea950004,
                    0x0000010000000004,
                    0x00000805216a0004,
                    0x0000052401000004,
                    0x00000504216a0005,
                    0x00000d6596ee0005,
                    0x00000d65a5690005,
                    0x0000816848900004,
                    0x00000d6500000005,
                    0x0000000501420005,
                    0x00000964657a0005,
                    0x0000002000820004,
                    0x00008d6d69fa0005,
                    0x00000d0500000004,
                    0x00000025216a0004,
                    0x0000000040000004,
                ],
            ),
            (
                0x0000af875e8f0004,
                0x0000dc1bcc6a0004,
                true,
                true,
                [
                    0x0000cb03000b0004,
                    0x00008106fa950004,
                    0x0000010000000004,
                    0x00008a07040a0004,
                    0x000005040e850004,
                    0x00000504040b0004,
                    0x0000af875bb30004,
                    0x0000af875abd0004,
                    0x000081685a850004,
                    0x0000af8700000004,
                    0x0000800704030004,
                    0x00002ba4554f0004,
                    0x000000000a830004,
                    0x0000afaf5e8f0004,
                    0x00008d0700010004,
                    0x00008227040a0004,
                    0x0000000000000004,
                ],
            ),
            (
                0x0000f901f7b30007,
                0x0000a0b2aa750000,
                false,
                true,
                [
                    0x00004901a0230004,
                    0x00008106fa950004,
                    0x0000010002110004,
                    0x0000e881a5230007,
                    0x0000110007850004,
                    0x00000100a5230007,
                    0x0000f901fd8f0007,
                    0x0000f901f57d0007,
                    0x0000812a50010004,
                    0x0000f90100000007,
                    0x0000c00005030007,
                    0x00003900f7b30007,
                    0x000000000a830007,
                    0x0000fd29f7b30007,
                    0x0000c90100010004,
                    0x0000c0a1a5630007,
                    0x0000002010000004,
                ],
            ),
            (
                0x0000c090ad4d0006,
                0x0000ec0e24d40005,
                true,
                true,
                [
                    0x0000000000010004,
                    0x000081120a950004,
                    0x0000010000000004,
                    0x0000800000000004,
                    0x0000010008050004,
                    0x0000000000010006,
                    0x0000c090a7710006,
                    0x0000c090a57d0006,
                    0x0000816808050004,
                    0x0000c09000000006,
                    0x0000c00005410006,
                    0x000000a0a54d0006,
                    0x000000000a810004,
                    0x0000c1b8ad4d0006,
                    0x0000800000010004,
                    0x0000802000000004,
                    0x0000000000000004,
                ],
            ),
            (
                0x0000ad48a79f0003,
                0x000089a12e700002,
                true,
                true,
                [
                    0x00000900a00b0000,
                    0x00008106aa950000,
                    0x0000010002000000,
                    0x00008800a50a0000,
                    0x0000050807850000,
                    0x00000508a50b0003,
                    0x0000ad48adb70003,
                    0x0000ad48affd0003,
                    0x0000816802950000,
                    0x0000ad4800000003,
                    0x0000800805030003,
                    0x00002968a55f0003,
                    0x000000080a830000,
                    0x0000ad68a79f0003,
                    0x00008d0800010000,
                    0x00008921a56a0000,
                    0x0000002000000000,
                ],
            ),
            (
                0x0000170384290000,
                0x0000bc5645610007,
                true,
                true,
                [
                    0x0000c30380290000,
                    0x00008106aa950000,
                    0x0000010000010004,
                    0x0000020384290004,
                    0x0000150004050000,
                    0x0000050084290000,
                    0x0000170321290000,
                    0x0000170300290000,
                    0x0000816800010000,
                    0x0000170300000000,
                    0x0000000304010000,
                    0x0000132085690000,
                    0x000000000a810000,
                    0x0000972b84290000,
                    0x0000050300010000,
                    0x0000022384290004,
                    0x0000000000000004,
                ],
            ),
            (
                0x0000f019de080005,
                0x0000490aca480005,
                false,
                false,
                [
                    0x0000600180080005,
                    0x000081125a950004,
                    0x0000010802000004,
                    0x0000601984080005,
                    0x0000110800000004,
                    0x0000000800000005,
                    0x0000f0197b200005,
                    0x0000f019ffc10005,
                    0x0000812a50000004,
                    0x0000f01900000005,
                    0x0000c00804000005,
                    0x00003018de080005,
                    0x0000000800800007,
                    0x0000f539de080005,
                    0x0000800800000004,
                    0x0000482985480005,
                    0x0000000000000004,
                ],
            ),
            (
                0x0000658ed7cb0007,
                0x000031a297470001,
                true,
                false,
                [
                    0x0000e582800b0007,
                    0x00008106fa950004,
                    0x0000010002010004,
                    0x00004006854b0004,
                    0x0000050c07050004,
                    0x0000050c854b0007,
                    0x0000658e7de30007,
                    0x0000658e5fe90007,
                    0x0000816852810004,
                    0x0000658e00000007,
                    0x0000400e05430007,
                    0x000021acd54b0007,
                    0x000000080a830004,
                    0x0000e5aed7cb0007,
                    0x0000450e00010004,
                    0x00004126854b0004,
                    0x0000002040000004,
                ],
            ),
            (
                0x00007131b14b0000,
                0x0000a746ae7e0003,
                true,
                false,
                [
                    0x00006101a00b0000,
                    0x00008116ba950000,
                    0x0000010000000000,
                    0x00004001a14a0000,
                    0x0000112001050000,
                    0x00000100a14b0000,
                    0x00007131e4630000,
                    0x00007131f5690000,
                    0x0000816810010000,
                    0x0000713100000000,
                    0x0000400101430000,
                    0x00003120b54b0000,
                    0x000000200a830000,
                    0x0000f139b14b0000,
                    0x0000410100010000,
                    0x00004325a56a0000,
                    0x0000000040000000,
                ],
            ),
            (
                0x0000d996b9660004,
                0x0000f3a129fb0004,
                true,
                false,
                [
                    0x0000c982a0220004,
                    0x00008116ba950004,
                    0x0000010000000004,
                    0x0000c804a1620004,
                    0x0000110401000004,
                    0x00000104a1620004,
                    0x0000d996e65a0004,
                    0x0000d996f5550004,
                    0x0000816818040004,
                    0x0000d99600000004,
                    0x0000c00601420004,
                    0x000019a4b5660004,
                    0x0000000000820004,
                    0x0000d9beb9660004,
                    0x0000c90600000004,
                    0x0000c325a16a0004,
                    0x0000002040000004,
                ],
            ),
            (
                0x00006ebf264d0000,
                0x0000eddf5fac0001,
                false,
                true,
                [
                    0x0000800200010000,
                    0x000081120a950000,
                    0x0000010802040000,
                    0x00006abf24480000,
                    0x0000052802050000,
                    0x0000040800010000,
                    0x00006ebf89710000,
                    0x00006ebf05410000,
                    0x0000812a00010000,
                    0x00006ebf00000000,
                    0x0000400e04410000,
                    0x00002abc264d0000,
                    0x000000280a810003,
                    0x0000efbf264d0000,
                    0x0000040a00010000,
                    0x00004eaf25680000,
                    0x0000000010000000,
                ],
            ),
            (
                0x000042b3da0e0007,
                0x000057a0843a0005,
                false,
                true,
                [
                    0x0000800200000004,
                    0x000081125a950004,
                    0x0000012000000004,
                    0x000042b1800a0007,
                    0x0000012000040004,
                    0x0000000000000007,
                    0x000042b37a320007,
                    0x000042b3fa810007,
                    0x0000812a50000004,
                    0x000042b300000007,
                    0x0000400200020007,
                    0x000002b0da0e0007,
                    0x0000002000820007,
                    0x0000c7bbda0e0007,
                    0x0000000200000004,
                    0x000042a1856a0007,
                    0x0000002010000004,
                ],
            ),
            (
                0x00002d71cf410007,
                0x00006615ef060002,
                true,
                true,
                [
                    0x0000490180010004,
                    0x00008116ea950004,
                    0x0000010002000004,
                    0x0000080185400004,
                    0x000005200f050004,
                    0x0000050085410007,
                    0x00002d713f410007,
                    0x00002d710fc10007,
                    0x000081684a010004,
                    0x00002d7100000007,
                    0x0000000105410007,
                    0x00002960c5410007,
                    0x000000200a810004,
                    0x0000ad79cf410007,
                    0x00000d0100010004,
                    0x0000002185400004,
                    0x0000000000000004,
                ],
            ),
            (
                0x0000845e02170003,
                0x0000a315cd700000,
                false,
                false,
                [
                    0x0000840200030003,
                    0x000081120a950000,
                    0x0000010000000000,
                    0x0000a01400020003,
                    0x0000050802050000,
                    0x0000040800010003,
                    0x0000845e08170003,
                    0x0000845e00150003,
                    0x0000812a00010000,
                    0x0000845e00000003,
                    0x0000800e00030003,
                    0x0000005c02170003,
                    0x000000480a830003,
                    0x0000857e02170003,
                    0x0000840a00010000,
                    0x0000822585620003,
                    0x0000000000000000,
                ],
            ),
            (
                0x000062de9f350001,
                0x00001ef4d7170000,
                true,
                true,
                [
                    0x0000800200010000,
                    0x000081121a950000,
                    0x0000010002150000,
                    0x0000000000010000,
                    0x000001080a050000,
                    0x0000000800010001,
                    0x000062de6f1d0001,
                    0x000062de5fd50001,
                    0x000081681a150000,
                    0x000062de00000001,
                    0x0000400e05010001,
                    0x000022ec95750001,
                    0x000000080a810000,
                    0x0000e3fe9f350001,
                    0x0000000a00010000,
                    0x0000002000010000,
                    0x0000002000000000,
                ],
            ),
            (
                0x0000c53f03c80000,
                0x0000a4dad0ab0000,
                false,
                false,
                [
                    0x0000c50300080000,
                    0x00008116aa950000,
                    0x0000010800000000,
                    0x0000e4bf816a0000,
                    0x0000052c01000000,
                    0x0000050c01480000,
                    0x0000c53f0ce00000,
                    0x0000c53f00010000,
                    0x0000812a00000000,
                    0x0000c53f00000000,
                    0x0000c00e01400000,
                    0x0000013c03c80000,
                    0x0000002800800003,
                    0x0000c53f03c80000,
                    0x0000c50f00000000,
                    0x0000c4af856a0000,
                    0x0000000000000000,
                ],
            ),
        ];

        for &(a, b, f1, f2, expected) in CASES {
            let actual = [
                closure_properties(a, f1, f2),
                complement_properties(a),
                compose_properties(a, b),
                concat_properties(a, b, f1),
                determinize_properties(a, f1, f2),
                factor_weight_properties(a),
                invert_properties(a),
                project_properties(a, f1),
                rand_gen_properties(a, f1),
                relabel_properties(a),
                reverse_properties(a, f1),
                reweight_properties(a, f1),
                rm_epsilon_properties(a, f1),
                shortest_path_properties(a, f1),
                synchronize_properties(a),
                union_properties(a, b, f1),
                replace_properties(&[a, b, a ^ b], 1, f1, f2, !f1, !f2, f1, f2, !f1, !f2, f1),
            ];
            const NAMES: [&str; 17] = [
                "closure",
                "complement",
                "compose",
                "concat",
                "determinize",
                "factor_weight",
                "invert",
                "project",
                "rand_gen",
                "relabel",
                "reverse",
                "reweight",
                "rm_epsilon",
                "shortest_path",
                "synchronize",
                "union",
                "replace",
            ];
            for i in 0..17 {
                assert_eq!(
                    actual[i], expected[i],
                    "{}({a:#018x}, {b:#018x}, {f1}, {f2})",
                    NAMES[i]
                );
            }
        }
    }

    #[test]
    fn every_property_bit_has_a_name() {
        for bit in 0..u64::BITS as usize {
            let named = !property_name(bit).is_empty();
            let is_a_property = K_FST_PROPERTIES & (1u64 << bit) != 0;
            assert_eq!(
                named,
                is_a_property,
                "bit {bit} is {}named but {}a property",
                if named { "" } else { "un" },
                if is_a_property { "" } else { "not " }
            );
        }
    }

    #[test]
    fn incompatible_properties_are_described_by_name() {
        let description = super::internal::describe_incompatible_properties(
            K_ACCEPTOR | K_ACYCLIC,
            K_NOT_ACCEPTOR | K_ACYCLIC,
        );
        assert!(
            description.contains("acceptor: true vs false"),
            "{description}"
        );
        assert!(
            description.contains("not acceptor: false vs true"),
            "{description}"
        );
        assert!(!description.contains("cyclic"), "{description}");

        assert!(
            super::internal::describe_incompatible_properties(K_ACCEPTOR, K_ACCEPTOR).is_empty()
        );
    }
}
