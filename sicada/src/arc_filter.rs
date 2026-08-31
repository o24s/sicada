use crate::arc::{Arc, ArcLabel};
use crate::data_structures::compact_set::{CompactSet, CompactSetKey};

/// A trait for filtering arcs during FST traversal.
pub trait ArcFilter<A: Arc> {
    fn call(&self, arc: &A) -> bool;
}

/// True for all arcs.
#[derive(Debug, Clone, Default)]
pub struct AnyArcFilter;

impl<A: Arc> ArcFilter<A> for AnyArcFilter {
    #[inline(always)]
    fn call(&self, _arc: &A) -> bool {
        true
    }
}

/// True for (input and output) epsilon arcs.
#[derive(Debug, Clone, Default)]
pub struct EpsilonArcFilter;

impl<A: Arc> ArcFilter<A> for EpsilonArcFilter {
    #[inline(always)]
    fn call(&self, arc: &A) -> bool {
        arc.ilabel() == <A::Label as ArcLabel>::epsilon()
            && arc.olabel() == <A::Label as ArcLabel>::epsilon()
    }
}

/// True for input epsilon arcs.
#[derive(Debug, Clone, Default)]
pub struct InputEpsilonArcFilter;

impl<A: Arc> ArcFilter<A> for InputEpsilonArcFilter {
    #[inline(always)]
    fn call(&self, arc: &A) -> bool {
        arc.ilabel() == <A::Label as ArcLabel>::epsilon()
    }
}

/// True for output epsilon arcs.
#[derive(Debug, Clone, Default)]
pub struct OutputEpsilonArcFilter;

impl<A: Arc> ArcFilter<A> for OutputEpsilonArcFilter {
    #[inline(always)]
    fn call(&self, arc: &A) -> bool {
        arc.olabel() == <A::Label as ArcLabel>::epsilon()
    }
}

/// True if the specified label matches (or doesn't match) depending on `keep_match`.
#[derive(Debug, Clone)]
pub struct LabelArcFilter<L> {
    label: L,
    match_input: bool,
    keep_match: bool,
}

impl<L> LabelArcFilter<L> {
    /// Matches the specified input label and keeps the match.
    pub fn new(label: L) -> Self {
        Self {
            label,
            match_input: true,
            keep_match: true,
        }
    }

    /// Full constructor specifying all options.
    pub fn with_options(label: L, match_input: bool, keep_match: bool) -> Self {
        Self {
            label,
            match_input,
            keep_match,
        }
    }
}

impl<A: Arc> ArcFilter<A> for LabelArcFilter<A::Label> {
    #[inline]
    fn call(&self, arc: &A) -> bool {
        let match_found = if self.match_input {
            arc.ilabel() == self.label
        } else {
            arc.olabel() == self.label
        };

        if self.keep_match {
            match_found
        } else {
            !match_found
        }
    }
}

/// True if any of the specified labels match (or don't match) depending on `keep_match`.
///
/// Backed by a [`CompactSet`], as upstream is: a filter's labels almost always
/// cluster in a narrow range, which that structure answers without a lookup at
/// all. A hash set would pay to hash every arc's label.
#[derive(Debug, Clone)]
pub struct MultiLabelArcFilter<L: CompactSetKey> {
    labels: CompactSet<L>,
    match_input: bool,
    keep_match: bool,
}

impl<L: CompactSetKey> Default for MultiLabelArcFilter<L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<L: CompactSetKey> MultiLabelArcFilter<L> {
    /// Matches the specified input labels and keeps the match.
    pub fn new() -> Self {
        Self {
            labels: CompactSet::new(),
            match_input: true,
            keep_match: true,
        }
    }

    /// Full constructor specifying all options.
    pub fn with_options(match_input: bool, keep_match: bool) -> Self {
        Self {
            labels: CompactSet::new(),
            match_input,
            keep_match,
        }
    }

    /// Adds a label to the filter set.
    pub fn add_label(&mut self, label: L) {
        self.labels.insert(label);
    }
}

impl<A: Arc> ArcFilter<A> for MultiLabelArcFilter<A::Label>
where
    A::Label: CompactSetKey,
{
    #[inline]
    fn call(&self, arc: &A) -> bool {
        let target_label = if self.match_input {
            arc.ilabel()
        } else {
            arc.olabel()
        };

        let match_found = self.labels.is_member(target_label);

        if self.keep_match {
            match_found
        } else {
            !match_found
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::weight::Weight;
    use crate::weights::float_weight::TropicalWeight;

    fn arc(ilabel: i32, olabel: i32) -> StdArc {
        StdArc::new(ilabel, olabel, TropicalWeight::one(), 0)
    }

    #[test]
    fn any_accepts_everything() {
        assert!(AnyArcFilter.call(&arc(0, 0)));
        assert!(AnyArcFilter.call(&arc(1, 2)));
    }

    /// The three epsilon filters differ in which side they look at, which is
    /// easy to swap by accident and changes what epsilon removal removes.
    #[test]
    fn the_epsilon_filters_look_at_the_side_they_name() {
        // Only a fully epsilon arc passes EpsilonArcFilter.
        assert!(EpsilonArcFilter.call(&arc(0, 0)));
        assert!(!EpsilonArcFilter.call(&arc(0, 1)));
        assert!(!EpsilonArcFilter.call(&arc(1, 0)));

        assert!(InputEpsilonArcFilter.call(&arc(0, 1)));
        assert!(InputEpsilonArcFilter.call(&arc(0, 0)));
        assert!(!InputEpsilonArcFilter.call(&arc(1, 0)));

        assert!(OutputEpsilonArcFilter.call(&arc(1, 0)));
        assert!(OutputEpsilonArcFilter.call(&arc(0, 0)));
        assert!(!OutputEpsilonArcFilter.call(&arc(0, 1)));
    }

    /// Epsilon is compared against the label type's own epsilon rather than a
    /// literal zero, so a filter works for any label type. An earlier version
    /// bound these filters to `Arc<Label = i32>` and compared against 0.
    #[test]
    fn the_epsilon_filters_do_not_assume_an_i32_label() {
        type WideArc = crate::arc::ArcTpl<TropicalWeight, i64, i32>;
        let epsilon: WideArc = Arc::new(0, 0, TropicalWeight::one(), 0);
        let labelled: WideArc = Arc::new(1, 1, TropicalWeight::one(), 0);
        assert!(EpsilonArcFilter.call(&epsilon));
        assert!(!EpsilonArcFilter.call(&labelled));
    }

    #[test]
    fn a_label_filter_can_match_either_side_and_invert() {
        let input_keep = LabelArcFilter::new(5);
        assert!(input_keep.call(&arc(5, 9)));
        assert!(!input_keep.call(&arc(9, 5)));

        let output_keep = LabelArcFilter::with_options(5, false, true);
        assert!(output_keep.call(&arc(9, 5)));
        assert!(!output_keep.call(&arc(5, 9)));

        let input_drop = LabelArcFilter::with_options(5, true, false);
        assert!(!input_drop.call(&arc(5, 9)));
        assert!(input_drop.call(&arc(9, 5)));
    }

    #[test]
    fn a_multi_label_filter_matches_any_of_its_labels() {
        let mut filter = MultiLabelArcFilter::new();
        filter.add_label(1);
        filter.add_label(3);

        assert!(filter.call(&arc(1, 0)));
        assert!(filter.call(&arc(3, 0)));
        assert!(!filter.call(&arc(2, 0)));
        assert!(!filter.call(&arc(0, 1)), "it looks at the input side");
    }

    #[test]
    fn a_multi_label_filter_can_invert_and_match_the_output_side() {
        let mut filter = MultiLabelArcFilter::with_options(false, false);
        filter.add_label(1);
        assert!(
            !filter.call(&arc(0, 1)),
            "1 matches, and matches are dropped"
        );
        assert!(filter.call(&arc(1, 2)), "the output label 2 does not match");
    }

    #[test]
    fn an_empty_multi_label_filter_matches_nothing() {
        let filter: MultiLabelArcFilter<i32> = MultiLabelArcFilter::new();
        assert!(!filter.call(&arc(0, 0)));
        assert!(!filter.call(&arc(7, 7)));
    }
}
