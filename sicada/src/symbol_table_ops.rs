//! Operations over symbol tables.
//!
//! Port of OpenFst's `symbol-table-ops.h` and `.cc`.

use std::collections::BTreeMap;
use std::convert::TryInto;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use rustc_hash::FxHashSet;

use crate::arc::Arc;
use crate::error::OpenFstError;
use crate::fst::Fst;
use crate::fst_header::{FstHeader, flags};
use crate::symbol_table::{K_NO_SYMBOL, SymbolTable};

/// Returns a minimal symbol table containing only symbols referenced by the
/// passed FST. Symbols preserve their original numbering.
pub fn prune_symbol_table<A, F>(
    fst: &F,
    syms: &SymbolTable,
    input: bool,
) -> Result<SymbolTable, OpenFstError>
where
    A: Arc,
    F: Fst<A>,
    A::Label: TryInto<i64>,
{
    let mut seen = FxHashSet::default();
    seen.insert(0); // Always keep epsilon

    for state in fst.states() {
        for arc in fst.arcs(state) {
            let sym_label = if input { arc.ilabel() } else { arc.olabel() };

            let sym_i64: i64 = sym_label.try_into().map_err(|_| {
                OpenFstError::SymbolTable("Failed to cast Arc::Label to i64".to_string())
            })?;
            seen.insert(sym_i64);
        }
    }

    let mut pruned = SymbolTable::new(format!("{}_pruned", syms.name()));
    for item in syms.iter() {
        if seen.contains(&item.label) {
            pruned.add_symbol(&item.symbol, item.label);
        }
    }

    Ok(pruned)
}

/// Relabels a symbol table to make it a contiguous mapping.
pub fn compact_symbol_table(syms: &SymbolTable) -> SymbolTable {
    let sorted: BTreeMap<i64, String> = syms.iter().map(|item| (item.label, item.symbol)).collect();

    let mut compact = SymbolTable::new(format!("{}_compact", syms.name()));
    for (new_key, (_, symbol)) in sorted.into_iter().enumerate() {
        compact.add_symbol(&symbol, new_key as i64);
    }
    compact
}

/// Merges two SymbolTables, with `left` symbols taking precedence in ID assignment.
///
/// Returns a tuple containing the merged `SymbolTable` and a boolean indicating
/// whether symbols from the right table needed to be implicitly relabeled.
pub fn merge_symbol_table(left: &SymbolTable, right: &SymbolTable) -> (SymbolTable, bool) {
    let mut merged = SymbolTable::new(format!("merge_{}_{}", left.name(), right.name()));

    let mut left_has_all = true;
    let mut right_has_all = true;
    let mut relabel = false;

    for litem in left.iter() {
        merged.add_symbol(&litem.symbol, litem.label);
        if right_has_all {
            let key = right.find_key(&litem.symbol);
            if key == K_NO_SYMBOL {
                right_has_all = false;
            } else if key != litem.label {
                // SICADA-BUGFIX: upstream leaves `right_has_all` set here, so a
                // right table that happens to contain every symbol of left, but
                // with different labels for some, short-circuits to returning
                // right unchanged. That discards left's assignments,
                // which the function's own contract promises never to modify, and
                // the `relabel` flag it sets cannot repair it: the caller
                // relabels the right FST against a table that *is* right's own.
                right_has_all = false;
                relabel = true;
            }
        }
    }

    if right_has_all {
        return (right.clone(), relabel);
    }

    let mut conflicts = Vec::new();
    for ritem in right.iter() {
        let key = merged.find_key(&ritem.symbol);
        if key != K_NO_SYMBOL {
            if key != ritem.label {
                relabel = true;
            }
            continue;
        }

        left_has_all = false;
        if merged.find_symbol(ritem.label).is_some() {
            // SICADA-BUGFIX: this symbol's label is already taken, so it is
            // appended with a fresh one below. That is a reassignment of a right
            // symbol, which is exactly what `relabel` is supposed to report.
            // Upstream leaves the flag alone here, so the caller does not
            // relabel the right FST and the two sides disagree on this symbol.
            relabel = true;
            conflicts.push(ritem.symbol.clone());
            continue;
        }

        merged.add_symbol(&ritem.symbol, ritem.label);
    }

    if left_has_all {
        return (left.clone(), relabel);
    }

    for conflict in conflicts {
        merged.add_symbol_auto(&conflict);
    }

    (merged, relabel)
}

/// Reads one symbol table out of an FST file without loading the FST.
///
/// Returns `None` when the file carries no table of the requested side.
///
/// SICADA-DIVERGE: upstream returns a null pointer both when the file has no
/// such table and when reading one failed, logging the difference. Here the
/// first is `Ok(None)` and the second an `Err`, so a caller can tell them apart.
pub fn fst_read_symbols(
    source: impl AsRef<Path>,
    input_symbols: bool,
) -> Result<Option<SymbolTable>, OpenFstError> {
    let mut reader = BufReader::new(File::open(source.as_ref())?);
    let header = FstHeader::read(&mut reader)?;

    // The tables follow the header in order, so the output table can only be
    // reached by reading past the input one.
    if header.flags & flags::HAS_ISYMBOLS != 0 {
        let isymbols = SymbolTable::read(&mut reader)?;
        if input_symbols {
            return Ok(Some(isymbols));
        }
    }
    if header.flags & flags::HAS_OSYMBOLS != 0 {
        let osymbols = SymbolTable::read(&mut reader)?;
        if !input_symbols {
            return Ok(Some(osymbols));
        }
    }
    Ok(None)
}

/// Adds a contiguous range of symbols to a symbol table using a simple prefix.
///
/// Returns `Err` if the inserted symbol string clashes with any currently present.
pub fn add_auxiliary_symbols(
    prefix: &str,
    start_label: i64,
    nlabels: i64,
    syms: &mut SymbolTable,
) -> Result<(), OpenFstError> {
    for i in 0..nlabels {
        let index = i + start_label;
        let symbol_str = format!("{}{}", prefix, i);
        if index != syms.add_symbol(&symbol_str, index) {
            return Err(OpenFstError::SymbolTable(format!(
                "AddAuxiliarySymbols: Symbol table clash for symbol '{}' at index {}",
                symbol_str, index
            )));
        }
    }
    Ok(())
}
