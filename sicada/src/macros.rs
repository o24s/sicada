/// Creates a linear FST from a string or a list of labels.
///
/// A shorthand for [`compile_labels`](crate::string::compile_labels) with the
/// byte token type, for the common case of a literal in a test or an example.
///
/// # Examples
/// ```
/// use sicada::{fst_linear, fst::ExpandedFst, vector_fst::StdVectorFst};
///
/// let fst = fst_linear!(StdVectorFst, "hello");
/// assert_eq!(fst.num_states(), 6);
///
/// let from_labels = fst_linear!(StdVectorFst, [1, 2, 3]);
/// assert_eq!(from_labels.num_states(), 4);
/// ```
#[macro_export]
macro_rules! fst_linear {
    ($fst_type:ty, [ $($label:expr),* $(,)? ]) => {{
        let mut fst = <$fst_type>::new();
        $crate::string::compile_labels(
            &[$($label),*],
            &mut fst,
            $crate::weight::Weight::one(),
        );
        fst
    }};

    ($fst_type:ty, $text:expr) => {{
        let mut fst = <$fst_type>::new();
        let labels: Vec<_> = $text.bytes().map(::std::convert::Into::into).collect();
        $crate::string::compile_labels(
            &labels,
            &mut fst,
            $crate::weight::Weight::one(),
        );
        fst
    }};
}
