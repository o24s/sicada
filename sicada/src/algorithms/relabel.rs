//! Renaming the labels of an FST.
//!
//! Port of OpenFst's `relabel.h`. Relabelling changes what the arcs are called
//! without changing the shape of the FST, which is how two FSTs built against
//! different symbol tables are brought into a common alphabet before anything
//! is composed.

use rustc_hash::FxHashMap;

use crate::AtomicRc;
use crate::arc::{Arc, ArcLabel};
use crate::error::OpenFstError;
use crate::fst::MutableFst;
use crate::properties::{K_FST_PROPERTIES, relabel_properties};
use crate::symbol_table::{K_NO_SYMBOL, SymbolTable};

/// Renames the labels of `fst` in place.
///
/// `ipairs` and `opairs` give the old-to-new mapping for each side; a label
/// with no entry keeps the name it has.
///
/// SICADA-DIVERGE: upstream signals a destination of `kNoLabel`, which is what
/// the symbol-table form produces for a symbol missing from the target table,
/// by setting `K_ERROR` on the FST and returning, leaving the arcs it had
/// already rewritten rewritten. Here it is an error, and the check runs before
/// anything is changed, so a failed relabelling leaves the FST as it was.
pub fn relabel<A: Arc, F: MutableFst<A>>(
    fst: &mut F,
    ipairs: &[(A::Label, A::Label)],
    opairs: &[(A::Label, A::Label)],
) -> Result<(), OpenFstError> {
    let props = fst.properties(K_FST_PROPERTIES, false);
    let input_map: FxHashMap<A::Label, A::Label> = ipairs.iter().copied().collect();
    let output_map: FxHashMap<A::Label, A::Label> = opairs.iter().copied().collect();

    let no_label = A::Label::no_label();
    for (side, map) in [("Input", &input_map), ("Output", &output_map)] {
        if let Some((from, _)) = map.iter().find(|(_, to)| **to == no_label) {
            return Err(OpenFstError::InvalidOperation(format!(
                "Relabel: {side} symbol ID {from} missing from target vocabulary"
            )));
        }
    }

    let states: Vec<A::StateId> = fst.states().collect();
    for state in states {
        fst.mutate_arcs(state, |arc| {
            let ilabel = *input_map.get(&arc.ilabel()).unwrap_or(&arc.ilabel());
            let olabel = *output_map.get(&arc.olabel()).unwrap_or(&arc.olabel());
            if ilabel != arc.ilabel() || olabel != arc.olabel() {
                *arc = A::new(ilabel, olabel, arc.weight().clone(), arc.nextstate());
            }
        });
    }

    fst.set_properties(relabel_properties(props), K_FST_PROPERTIES);
    Ok(())
}

/// How a symbol missing from the target table is handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingSymbol<'a> {
    /// Leave the label alone. Upstream's behaviour when no unknown symbol is
    /// given: it warns and maps the label to `kNoLabel`, which `Relabel` then
    /// rejects, so in practice this is the case that fails.
    #[default]
    Refuse,
    /// Map it to this symbol of the target table.
    MapTo(&'a str),
}

/// Renames the labels of `fst` so that they mean the same symbols they did,
/// under new symbol tables.
///
/// A label is looked up in `old`, and the symbol it stands for is looked up in
/// `new` to get its new number. `attach` says whether the new table should
/// replace the FST's own.
pub struct RelabelSide<'a> {
    /// The table the FST's labels currently refer to.
    pub old: &'a SymbolTable,
    /// The table they should refer to instead.
    pub new: &'a AtomicRc<SymbolTable>,
    /// What to do with a symbol the new table does not have.
    pub missing: MissingSymbol<'a>,
    /// Whether to attach the new table to the FST.
    pub attach: bool,
}

/// The old-to-new label pairs a side calls for.
fn pairs<A: Arc>(side: &RelabelSide<'_>) -> Result<Vec<(A::Label, A::Label)>, OpenFstError> {
    {
        let unknown = match side.missing {
            MissingSymbol::Refuse => None,
            MissingSymbol::MapTo(symbol) => {
                let label = side.new.find_key(symbol);
                if label == K_NO_SYMBOL {
                    return Err(OpenFstError::SymbolTable(format!(
                        "Relabel: the symbol '{symbol}' offered for unknown symbols is itself \
                         missing from the target table"
                    )));
                }
                Some(label)
            }
        };

        let mut pairs = Vec::new();
        for item in side.old.iter() {
            let new_label = match side.new.find_key(&item.symbol) {
                K_NO_SYMBOL => match unknown {
                    Some(label) => label,
                    None => {
                        return Err(OpenFstError::SymbolTable(format!(
                            "Relabel: symbol '{}' (ID {}) is missing from the target table",
                            item.symbol, item.label
                        )));
                    }
                },
                label => label,
            };
            let (Some(from), Some(to)) = (
                A::Label::from_i64(item.label),
                A::Label::from_i64(new_label),
            ) else {
                return Err(OpenFstError::SymbolTable(format!(
                    "Relabel: symbol '{}' has a label that does not fit the arc's label type",
                    item.symbol
                )));
            };
            pairs.push((from, to));
        }
        Ok(pairs)
    }
}

/// Renames the labels of `fst` under new symbol tables.
///
/// Either side may be left out, in which case its labels are untouched.
pub fn relabel_tables<A: Arc, F: MutableFst<A>>(
    fst: &mut F,
    input: Option<RelabelSide<'_>>,
    output: Option<RelabelSide<'_>>,
) -> Result<(), OpenFstError> {
    let ipairs = match &input {
        Some(side) => pairs::<A>(side)?,
        None => Vec::new(),
    };
    let opairs = match &output {
        Some(side) => pairs::<A>(side)?,
        None => Vec::new(),
    };

    // Everything that could fail has failed by now, so the FST is only touched
    // once the whole relabelling is known to be possible.
    relabel(fst, &ipairs, &opairs)?;

    if let Some(side) = input
        && side.attach
    {
        fst.set_input_symbols(Some(AtomicRc::clone(side.new)));
    }
    if let Some(side) = output
        && side.attach
    {
        fst.set_output_symbols(Some(AtomicRc::clone(side.new)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::fst::{ExpandedFst as _, Fst as _};
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::{K_ACCEPTOR, K_NOT_ACCEPTOR};
    use crate::weight::Weight;
    use crate::weights::float_weight::TropicalWeight;

    fn transducer() -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.set_final(2, TropicalWeight::one());
        fst.add_arc(0, StdArc::new(1, 10, TropicalWeight(1.0), 1));
        fst.add_arc(1, StdArc::new(2, 20, TropicalWeight(2.0), 2));
        fst
    }

    fn arcs(fst: &StdVectorFst) -> Vec<(i32, i32)> {
        (0..fst.num_states() as i32)
            .flat_map(|s| {
                fst.arcs(s)
                    .map(|a| (a.ilabel(), a.olabel()))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn labels_with_a_mapping_are_renamed_and_the_rest_are_left_alone() {
        let mut fst = transducer();
        relabel(&mut fst, &[(1, 100)], &[(20, 200)]).unwrap();
        assert_eq!(arcs(&fst), vec![(100, 10), (2, 200)]);
    }

    /// Only the labels change: the shape of the FST does not.
    #[test]
    fn relabelling_leaves_the_structure_alone() {
        let mut fst = transducer();
        let before: Vec<(i32, i32)> = (0..fst.num_states() as i32)
            .flat_map(|s| {
                fst.arcs(s)
                    .map(move |a| (s, a.nextstate()))
                    .collect::<Vec<_>>()
            })
            .collect();

        relabel(&mut fst, &[(1, 5), (2, 6)], &[(10, 50), (20, 60)]).unwrap();

        assert_eq!(fst.num_states(), 3);
        assert_eq!(fst.start(), Some(0));
        assert_eq!(fst.final_weight(2), TropicalWeight::one());
        let after: Vec<(i32, i32)> = (0..fst.num_states() as i32)
            .flat_map(|s| {
                fst.arcs(s)
                    .map(move |a| (s, a.nextstate()))
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(after, before);
    }

    /// Relabelling can turn a transducer into an acceptor, so the property bits
    /// it used to claim about labels cannot survive.
    #[test]
    fn label_properties_are_given_up() {
        let mut fst = transducer();
        assert_ne!(fst.properties(K_NOT_ACCEPTOR, true) & K_NOT_ACCEPTOR, 0);

        relabel(&mut fst, &[], &[(10, 1), (20, 2)]).unwrap();
        // Now an acceptor in fact.
        assert_eq!(arcs(&fst), vec![(1, 1), (2, 2)]);
        let props = fst.properties(K_FST_PROPERTIES, false);
        assert_eq!(
            props & (K_ACCEPTOR | K_NOT_ACCEPTOR),
            0,
            "acceptor-ness is no longer claimed either way"
        );
        // And a scan agrees with what it now is.
        assert_ne!(fst.properties(K_ACCEPTOR, true) & K_ACCEPTOR, 0);
    }

    /// A mapping to the no-label marker cannot be carried out, and the FST is
    /// left as it was rather than half rewritten.
    #[test]
    fn a_mapping_to_no_label_is_refused_before_anything_changes() {
        let mut fst = transducer();
        let before = arcs(&fst);
        assert!(relabel(&mut fst, &[(2, -1)], &[]).is_err());
        assert_eq!(arcs(&fst), before);

        assert!(relabel(&mut fst, &[], &[(20, -1)]).is_err());
        assert_eq!(arcs(&fst), before);
    }

    fn table(name: &str, symbols: &[(&str, i64)]) -> AtomicRc<SymbolTable> {
        let mut table = SymbolTable::new(name.to_string());
        for &(symbol, label) in symbols {
            table.add_symbol(symbol, label);
        }
        AtomicRc::new(table)
    }

    #[test]
    fn symbol_tables_decide_the_new_numbering() {
        let old = table("old", &[("<eps>", 0), ("a", 1), ("b", 2)]);
        let new = table("new", &[("<eps>", 0), ("b", 7), ("a", 9)]);

        let mut fst = transducer();
        relabel_tables(
            &mut fst,
            Some(RelabelSide {
                old: &old,
                new: &new,
                missing: MissingSymbol::Refuse,
                attach: true,
            }),
            None,
        )
        .unwrap();

        // "a" was 1 and is now 9; "b" was 2 and is now 7.
        assert_eq!(arcs(&fst), vec![(9, 10), (7, 20)]);
        assert_eq!(
            fst.input_symbols().unwrap().find_symbol(9),
            Some("a"),
            "the new table is attached"
        );
    }

    #[test]
    fn a_symbol_the_new_table_lacks_is_refused_or_mapped_to_the_unknown_one() {
        let old = table("old", &[("<eps>", 0), ("a", 1), ("b", 2)]);
        let new = table("new", &[("<eps>", 0), ("a", 9), ("<unk>", 99)]);

        let mut fst = transducer();
        let before = arcs(&fst);
        assert!(
            relabel_tables(
                &mut fst,
                Some(RelabelSide {
                    old: &old,
                    new: &new,
                    missing: MissingSymbol::Refuse,
                    attach: true,
                }),
                None,
            )
            .is_err(),
            "'b' is missing from the new table"
        );
        assert_eq!(arcs(&fst), before, "nothing was changed");

        relabel_tables(
            &mut fst,
            Some(RelabelSide {
                old: &old,
                new: &new,
                missing: MissingSymbol::MapTo("<unk>"),
                attach: false,
            }),
            None,
        )
        .unwrap();
        assert_eq!(arcs(&fst), vec![(9, 10), (99, 20)]);
        assert!(fst.input_symbols().is_none(), "the table was not attached");
    }

    #[test]
    fn an_unknown_symbol_that_is_itself_missing_is_refused() {
        let old = table("old", &[("<eps>", 0), ("a", 1), ("b", 2)]);
        let new = table("new", &[("<eps>", 0), ("a", 9)]);

        let mut fst = transducer();
        assert!(
            relabel_tables(
                &mut fst,
                Some(RelabelSide {
                    old: &old,
                    new: &new,
                    missing: MissingSymbol::MapTo("<unk>"),
                    attach: true,
                }),
                None,
            )
            .is_err()
        );
    }
}
