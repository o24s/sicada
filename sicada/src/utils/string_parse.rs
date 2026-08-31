//! Parsing of OpenFst's composite weight text representation.
//!
//! Port of the `CompositeWeightReader` / `CompositeWeightWriter` pair in
//! `weight.cc`, which gives a nested weight such as a pair of pairs an
//! unambiguous textual form.
//!
//! SICADA-DIVERGE: upstream reads the separator and the enclosing parentheses
//! from two process-wide flags, `--fst_weight_separator` and
//! `--fst_weight_parentheses`, so how a weight prints depends on global state
//! that any part of the program can change and that no caller can override for
//! one call. Here they are parameters. The defaults are unchanged, `,` with no
//! parentheses, so the text a weight produces is the same.

use crate::error::SplitCompositeError;

/// Splits a string by a separator character, but ignores separators enclosed in parentheses.
/// Automatically strips outer parentheses if they enclose the entire string.
///
/// Example: `"(1.0, 2.0), (3.0, 4.0)"` -> `["(1.0, 2.0)", "(3.0, 4.0)"]`
/// Example: `"(1.0, 2.0)"` -> `["1.0", "2.0"]`
pub fn split_composite_weight(
    mut s: &str,
    separator: char,
    open_paren: char,
    close_paren: char,
) -> Result<Vec<&str>, SplitCompositeError> {
    s = s.trim();

    // Strip outer parentheses if the entire string is wrapped in one matching pair.
    if s.starts_with(open_paren) && s.ends_with(close_paren) {
        let mut depth = 0;
        let mut is_single_group = true;
        for (i, c) in s.char_indices() {
            if c == open_paren {
                depth += 1;
            } else if c == close_paren {
                depth -= 1;
                // If depth hits 0 before the last character, it's not a single enclosing group.
                // e.g., "(1.0), (2.0)"
                if depth == 0 && i != s.len() - close_paren.len_utf8() {
                    is_single_group = false;
                    break;
                }
            }
        }
        if is_single_group && depth == 0 {
            s = s[open_paren.len_utf8()..s.len() - close_paren.len_utf8()].trim();
        }
    }

    let mut parts = Vec::new();
    let mut depth = 0;
    let mut start = 0;

    for (i, c) in s.char_indices() {
        if c == open_paren {
            depth += 1;
        } else if c == close_paren {
            if depth == 0 {
                return Err(SplitCompositeError::UnmatchedCloseParenthesis { offset: i });
            }
            depth -= 1;
        } else if c == separator && depth == 0 {
            parts.push(s[start..i].trim());
            start = i + c.len_utf8();
        }
    }

    if depth != 0 {
        return Err(SplitCompositeError::UnmatchedOpenParenthesis);
    }

    parts.push(s[start..].trim());
    Ok(parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_separator_inside_parentheses_does_not_split() {
        // This is the whole reason the reader exists: a nested composite weight
        // contains the separator, and the outer split must not see it.
        assert_eq!(
            split_composite_weight("(1,2),(3,4)", ',', '(', ')').unwrap(),
            vec!["(1,2)", "(3,4)"]
        );
        assert_eq!(
            split_composite_weight("((1,2),3),4", ',', '(', ')').unwrap(),
            vec!["((1,2),3)", "4"]
        );
    }

    #[test]
    fn outer_parentheses_are_stripped_only_when_they_wrap_everything() {
        assert_eq!(
            split_composite_weight("(1,2)", ',', '(', ')').unwrap(),
            vec!["1", "2"]
        );
        // Here the parentheses do not enclose the whole string, so they stay.
        assert_eq!(
            split_composite_weight("(1),(2)", ',', '(', ')').unwrap(),
            vec!["(1)", "(2)"]
        );
    }

    #[test]
    fn whitespace_around_elements_is_trimmed() {
        assert_eq!(
            split_composite_weight("  1 , 2  ", ',', '(', ')').unwrap(),
            vec!["1", "2"]
        );
        assert_eq!(
            split_composite_weight(" ( 1 , 2 ) ", ',', '(', ')').unwrap(),
            vec!["1", "2"]
        );
    }

    #[test]
    fn an_empty_element_is_preserved_rather_than_dropped() {
        // The caller decides whether an empty element is an error; splitting
        // must not silently change the arity.
        assert_eq!(
            split_composite_weight("1,,2", ',', '(', ')').unwrap(),
            vec!["1", "", "2"]
        );
        assert_eq!(split_composite_weight("", ',', '(', ')').unwrap(), vec![""]);
    }

    #[test]
    fn a_different_separator_and_bracket_pair_work() {
        assert_eq!(
            split_composite_weight("[1|2]", '|', '[', ']').unwrap(),
            vec!["1", "2"]
        );
    }

    #[test]
    fn unbalanced_parentheses_are_reported_with_their_position() {
        assert_eq!(
            split_composite_weight("1,2)", ',', '(', ')'),
            Err(SplitCompositeError::UnmatchedCloseParenthesis { offset: 3 })
        );
        assert_eq!(
            split_composite_weight("(1,(2)", ',', '(', ')'),
            Err(SplitCompositeError::UnmatchedOpenParenthesis)
        );
    }

    #[test]
    fn test_split_composite_weight() {
        let text = "1.0, 2.0, 3.0";
        let parts = split_composite_weight(text, ',', '(', ')').unwrap();
        assert_eq!(parts, vec!["1.0", "2.0", "3.0"]);

        let wrapped_text = "(1.0, 2.0)";
        let parts2 = split_composite_weight(wrapped_text, ',', '(', ')').unwrap();
        assert_eq!(parts2, vec!["1.0", "2.0"]);

        let nested_text = "(1.0, 2.0), (3.0, 4.0), 5.0";
        let nested_parts = split_composite_weight(nested_text, ',', '(', ')').unwrap();
        assert_eq!(nested_parts, vec!["(1.0, 2.0)", "(3.0, 4.0)", "5.0"]);

        let error_text = "(1.0, 2.0";
        assert_eq!(
            split_composite_weight(error_text, ',', '(', ')'),
            Err(SplitCompositeError::UnmatchedOpenParenthesis)
        );

        let error_text2 = "1.0, 2.0)";
        assert_eq!(
            split_composite_weight(error_text2, ',', '(', ')'),
            Err(SplitCompositeError::UnmatchedCloseParenthesis { offset: 8 })
        );
    }
}
