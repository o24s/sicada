use std::io::Cursor;

use sicada::arc::{Arc, StdArc};
use sicada::fst::MutableFst;
use sicada::symbol_table::{K_NO_SYMBOL, SymbolTable, compat_symbols, compat_symbols_with_warn};
use sicada::symbol_table_ops::{
    add_auxiliary_symbols, compact_symbol_table, merge_symbol_table, prune_symbol_table,
};
use sicada::vector_fst::VectorFst;
use sicada::weight::Weight;

#[test]
fn test_basic_operations() {
    let mut syms = SymbolTable::new("test");

    assert_eq!(syms.add_symbol("epsilon", 0), 0);
    assert_eq!(syms.add_symbol("a", 1), 1);
    assert_eq!(syms.add_symbol("b", 2), 2);

    assert_eq!(syms.num_symbols(), 3);
    assert_eq!(syms.available_key(), 3);

    assert_eq!(syms.find_symbol(1), Some("a"));
    assert_eq!(syms.find_key("b"), 2);
    assert_eq!(syms.find_key("c"), K_NO_SYMBOL);
    assert_eq!(syms.find_symbol(99), None);

    assert!(syms.member_key(0));
    assert!(syms.member_symbol("a"));
    assert!(!syms.member_key(3));

    assert_eq!(syms.add_symbol("a", 1), 1);

    assert_eq!(syms.add_symbol_auto("c"), 3);
    assert_eq!(syms.available_key(), 4);
}

#[test]
fn test_dense_and_sparse_keys() {
    let mut syms = SymbolTable::new("test");

    syms.add_symbol("0", 0);
    syms.add_symbol("1", 1);
    syms.add_symbol("2", 2);

    syms.add_symbol("100", 100);
    syms.add_symbol("minus", -5);

    assert_eq!(syms.num_symbols(), 5);
    assert_eq!(syms.find_symbol(100), Some("100"));
    assert_eq!(syms.find_symbol(-5), Some("minus"));
    assert_eq!(syms.available_key(), 101);

    syms.remove_symbol(1);
    assert_eq!(syms.num_symbols(), 4);
    assert_eq!(syms.find_symbol(1), None);
    assert_eq!(syms.find_symbol(2), Some("2"));
    assert_eq!(syms.find_symbol(100), Some("100"));
}

#[test]
fn test_text_io() {
    let mut syms = SymbolTable::new("text_io");
    syms.add_symbol("epsilon", 0);
    syms.add_symbol("a", 1);
    syms.add_symbol("foo", 10);

    let mut out = Vec::new();
    syms.write_text(&mut out, "\t").unwrap();

    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("epsilon\t0\n"));
    assert!(text.contains("a\t1\n"));
    assert!(text.contains("foo\t10\n"));

    let mut cursor = Cursor::new(text);
    let parsed_syms = SymbolTable::read_text(&mut cursor, "parsed", "\t").unwrap();

    assert_eq!(parsed_syms.num_symbols(), 3);
    assert_eq!(parsed_syms.find_key("foo"), 10);
}

#[test]
fn test_binary_io() {
    let mut syms = SymbolTable::new("bin_io");
    syms.add_symbol("eps", 0);
    syms.add_symbol("sparse", 999);

    let mut out = Vec::new();
    syms.write(&mut out).unwrap();

    let mut cursor = Cursor::new(out);
    let parsed_syms = SymbolTable::read(&mut cursor).unwrap();

    assert_eq!(parsed_syms.name(), "bin_io");
    assert_eq!(parsed_syms.num_symbols(), 2);
    assert_eq!(parsed_syms.find_symbol(0), Some("eps"));
    assert_eq!(parsed_syms.find_symbol(999), Some("sparse"));
}

#[test]
fn test_checksums() {
    let mut syms = SymbolTable::new("hash");
    syms.add_symbol("a", 1);

    let sum1 = syms.check_sum().to_string();
    let l_sum1 = syms.labeled_check_sum().to_string();

    syms.add_symbol("b", 2);

    let sum2 = syms.check_sum().to_string();
    let l_sum2 = syms.labeled_check_sum().to_string();

    assert_ne!(sum1, sum2);
    assert_ne!(l_sum1, l_sum2);
}

#[test]
fn test_copy_on_write() {
    let mut syms1 = SymbolTable::new("cow");
    syms1.add_symbol("a", 1);

    let mut syms2 = syms1.clone();
    syms2.add_symbol("b", 2);

    assert_eq!(syms1.num_symbols(), 1);
    assert_eq!(syms2.num_symbols(), 2);
}

#[test]
fn test_compact_symbol_table() {
    let mut syms = SymbolTable::new("sparse");
    syms.add_symbol("x", 10);
    syms.add_symbol("y", 20);
    syms.add_symbol("z", 30);

    let compacted = compact_symbol_table(&syms);

    assert_eq!(compacted.num_symbols(), 3);
    assert_eq!(compacted.find_key("x"), 0);
    assert_eq!(compacted.find_key("y"), 1);
    assert_eq!(compacted.find_key("z"), 2);
}

#[test]
fn test_merge_symbol_table() {
    let mut left = SymbolTable::new("left");
    left.add_symbol("a", 1);
    left.add_symbol("b", 2);

    let mut right = SymbolTable::new("right");
    right.add_symbol("b", 2);
    right.add_symbol("c", 3);
    right.add_symbol("a", 99);

    let (merged, relabeled) = merge_symbol_table(&left, &right);

    assert!(relabeled);
    assert_eq!(merged.find_key("a"), 1);
    assert_eq!(merged.find_key("b"), 2);
    assert_eq!(merged.find_key("c"), 3);
}

#[test]
fn test_add_auxiliary_symbols() {
    let mut syms = SymbolTable::new("aux");
    syms.add_symbol("a", 1);

    let res = add_auxiliary_symbols("aux_", 10, 3, &mut syms);
    assert!(res.is_ok());
    assert_eq!(syms.find_key("aux_0"), 10);
    assert_eq!(syms.find_key("aux_1"), 11);
    assert_eq!(syms.find_key("aux_2"), 12);

    syms.add_symbol("clash_0", 99);
    let clash_res = add_auxiliary_symbols("clash_", 20, 1, &mut syms);
    assert!(clash_res.is_err());
}

#[test]
fn test_prune_symbol_table() {
    let mut syms = SymbolTable::new("full");
    syms.add_symbol("eps", 0);
    syms.add_symbol("a", 1);
    syms.add_symbol("b", 2);
    syms.add_symbol("c", 3);

    let mut fst = VectorFst::<StdArc>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, Weight::one());

    fst.add_arc(s0, StdArc::new(1, 1, Weight::one(), s1));
    fst.add_arc(s1, StdArc::new(3, 3, Weight::one(), s1));

    let pruned = prune_symbol_table(&fst, &syms, true).unwrap();

    assert_eq!(pruned.num_symbols(), 3);
    assert!(pruned.member_key(0));
    assert!(pruned.member_key(1));
    assert!(pruned.member_key(3));
    assert!(!pruned.member_key(2));
}

#[test]
fn test_compat_symbols_with_warn() {
    let mut syms1 = SymbolTable::new("t1");
    syms1.add_symbol("a", 1);

    let mut syms2 = SymbolTable::new("t2");
    syms2.add_symbol("a", 1);

    let mut syms3 = SymbolTable::new("t3");
    syms3.add_symbol("a", 1);
    syms3.add_symbol("b", 2);

    assert!(compat_symbols(Some(&mut syms1), Some(&mut syms2)));

    let mut buf = Vec::new();
    let is_compat = compat_symbols_with_warn(Some(&mut syms1), Some(&mut syms3), &mut buf).unwrap();

    assert!(!is_compat);
    let output = String::from_utf8(buf).unwrap();
    assert!(output.contains("WARNING: CompatSymbols"));
    assert!(output.contains("Table sizes are 1 and 2"));
}

/// Two tables that upstream's XOR-32 `CheckSummer` cannot tell apart must still
/// get distinct checksums here.
///
/// Upstream folds byte `i` of the concatenated `symbol\0` stream into slot
/// `i % 32`, so swapping two bytes 32 positions apart leaves the digest
/// unchanged. Eight three-character symbols fill exactly 32 bytes, which puts
/// the ninth symbol's first byte at slot 0 alongside the first symbol's.
/// See the SICADA-DIVERGE note on `CheckSummer` in src/symbol_table.rs.
#[test]
fn test_checksum_distinguishes_upstream_collision() {
    fn table_of(symbols: &[&str]) -> SymbolTable {
        let mut syms = SymbolTable::new("collision");
        for (i, symbol) in symbols.iter().enumerate() {
            syms.add_symbol(symbol, i as i64);
        }
        syms
    }

    let filler = ["bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh"];
    let mut first: Vec<&str> = vec!["aaa"];
    first.extend_from_slice(&filler);
    first.push("ppp");
    let mut second: Vec<&str> = vec!["paa"];
    second.extend_from_slice(&filler);
    second.push("app");

    // The two streams differ, but agree byte-for-byte after XOR-folding mod 32.
    let stream = |symbols: &[&str]| {
        let mut bytes = Vec::new();
        for symbol in symbols {
            bytes.extend_from_slice(symbol.as_bytes());
            bytes.push(0);
        }
        bytes
    };
    let xor32 = |bytes: &[u8]| {
        let mut digest = [0u8; 32];
        for (i, byte) in bytes.iter().enumerate() {
            digest[i % 32] ^= byte;
        }
        digest
    };
    assert_ne!(stream(&first), stream(&second));
    assert_eq!(
        xor32(&stream(&first)),
        xor32(&stream(&second)),
        "test setup no longer reproduces the upstream collision"
    );

    let mut first = table_of(&first);
    let mut second = table_of(&second);
    assert_ne!(
        first.check_sum().to_string(),
        second.check_sum().to_string()
    );
    assert_ne!(
        first.labeled_check_sum().to_string(),
        second.labeled_check_sum().to_string()
    );
    assert!(!compat_symbols(Some(&mut first), Some(&mut second)));
}

/// A symbol table written by OpenFst must be readable here, byte for byte.
///
/// The reference bytes come from running SymbolTableImpl::Write's logic with the
/// ReadType/WriteType overloads from util.h; see
/// tests/oracles/symbol-table-golden.cc. sicada previously wrote a 64-bit
/// string length where the format calls for 32, so tables round tripped within
/// sicada and were unreadable by OpenFst.
#[test]
fn reads_and_writes_the_openfst_symbol_table_format() {
    const GOLDEN_HEX: &str = "74fbb27e040000007465737465000000000000000400000000000000050000003c6570733e00000000000000000100000061010000000000000001000000620200000000000000060000007370617273656400000000000000";

    let golden: Vec<u8> = (0..GOLDEN_HEX.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&GOLDEN_HEX[i..i + 2], 16).unwrap())
        .collect();
    assert_eq!(golden.len(), 89);

    let table = SymbolTable::read(&mut Cursor::new(golden.clone())).unwrap();
    assert_eq!(table.name(), "test");
    assert_eq!(table.num_symbols(), 4);
    assert_eq!(table.find_symbol(0), Some("<eps>"));
    assert_eq!(table.find_symbol(1), Some("a"));
    assert_eq!(table.find_symbol(2), Some("b"));
    assert_eq!(table.find_symbol(100), Some("sparse"));
    assert_eq!(table.find_key("sparse"), 100i64);

    // And writing it back reproduces the same bytes.
    let mut written = Vec::new();
    table.write(&mut written).unwrap();
    assert_eq!(written, golden);
}

#[test]
fn a_symbol_table_with_a_negative_declared_size_is_rejected() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&2125658996i32.to_le_bytes()); // magic
    bytes.extend_from_slice(&0i32.to_le_bytes()); // empty name
    bytes.extend_from_slice(&0i64.to_le_bytes()); // available_key
    bytes.extend_from_slice(&(-1i64).to_le_bytes()); // size
    assert!(SymbolTable::read(&mut Cursor::new(bytes)).is_err());
}

/// Symbols with negative labels must affect the labelled checksum.
///
/// Upstream drops them: its loop skips every key below `dense_key_limit_`, which
/// is meant to avoid repeating the dense range but catches negatives as well. Its
/// own comment calls that a bug. The consequence is that `compat_symbols` treats
/// two tables that differ only in their negative labels as equivalent.
#[test]
fn negative_labels_take_part_in_the_labeled_checksum() {
    let mut without = SymbolTable::new("t");
    without.add_symbol("<eps>", 0);
    without.add_symbol("a", 1);

    let mut with = SymbolTable::new("t");
    with.add_symbol("<eps>", 0);
    with.add_symbol("a", 1);
    with.add_symbol("phi", -2);

    assert_ne!(
        without.labeled_check_sum().to_string(),
        with.labeled_check_sum().to_string(),
        "a negatively labelled symbol must change the checksum"
    );
    assert!(!compat_symbols(Some(&mut without), Some(&mut with)));

    // And two tables whose negative labels differ must also be distinguished.
    let mut other = SymbolTable::new("t");
    other.add_symbol("<eps>", 0);
    other.add_symbol("a", 1);
    other.add_symbol("phi", -3);
    assert_ne!(
        with.labeled_check_sum().to_string(),
        other.labeled_check_sum().to_string()
    );
}

/// A right table that happens to contain every symbol of left must not win when
/// their labels disagree.
///
/// Upstream short-circuits to returning right unchanged here, discarding left's
/// assignments, which its own contract promises never to modify, and the relabel
/// flag it sets cannot repair that, since the caller would relabel the right FST
/// against right's own table.
#[test]
fn merging_keeps_the_left_assignments_when_right_is_a_superset() {
    let mut left = SymbolTable::new("left");
    left.add_symbol("a", 1);
    left.add_symbol("b", 2);

    let mut right = SymbolTable::new("right");
    right.add_symbol("a", 99); // same symbol, different label
    right.add_symbol("b", 2);
    right.add_symbol("c", 3);

    let (merged, relabeled) = merge_symbol_table(&left, &right);
    assert!(relabeled, "the caller has to be told to relabel");
    assert_eq!(merged.find_key("a"), 1, "left's assignment must survive");
    assert_eq!(merged.find_key("b"), 2);
    assert_eq!(merged.find_key("c"), 3);
    // And nothing from left was dropped.
    assert!(merged.member_key(1));
    assert!(merged.member_key(2));
}

/// When right really is a superset with matching labels, returning it is right,
/// and no relabelling is needed.
#[test]
fn merging_returns_right_when_it_agrees_on_every_label() {
    let mut left = SymbolTable::new("left");
    left.add_symbol("a", 1);

    let mut right = SymbolTable::new("right");
    right.add_symbol("a", 1);
    right.add_symbol("b", 2);

    let (merged, relabeled) = merge_symbol_table(&left, &right);
    assert!(!relabeled);
    assert_eq!(merged.find_key("a"), 1);
    assert_eq!(merged.find_key("b"), 2);
}

/// A symbol that exists only in right, whose label is already taken by left,
/// has to be appended with a fresh label rather than overwriting.
#[test]
fn merging_reassigns_a_conflicting_right_symbol() {
    let mut left = SymbolTable::new("left");
    left.add_symbol("a", 1);

    let mut right = SymbolTable::new("right");
    right.add_symbol("z", 1); // label 1 is left's "a"

    let (merged, relabeled) = merge_symbol_table(&left, &right);
    assert!(relabeled);
    assert_eq!(merged.find_key("a"), 1, "left keeps label 1");
    let z = merged.find_key("z");
    assert_ne!(z, 1);
    assert_ne!(z, K_NO_SYMBOL, "z must still be present");
}

/// Reading one table out of an FST file without loading the FST.
#[test]
fn symbol_tables_can_be_read_back_out_of_a_file() {
    use sicada::fst_header::{FstHeader, flags};
    use sicada::symbol_table_ops::fst_read_symbols;
    use std::io::Write as _;

    let mut input = SymbolTable::new("in");
    input.add_symbol("<eps>", 0);
    input.add_symbol("a", 1);
    let mut output = SymbolTable::new("out");
    output.add_symbol("<eps>", 0);
    output.add_symbol("x", 1);
    output.add_symbol("y", 2);

    let mut bytes = Vec::new();
    FstHeader {
        fst_type: "vector".to_string(),
        arc_type: "standard".to_string(),
        version: 2,
        flags: flags::HAS_ISYMBOLS | flags::HAS_OSYMBOLS,
        properties: 0,
        start: 0,
        num_states: 0,
        num_arcs: 0,
    }
    .write(&mut bytes)
    .unwrap();
    input.write(&mut bytes).unwrap();
    output.write(&mut bytes).unwrap();

    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&bytes).unwrap();
    file.flush().unwrap();

    let read_input = fst_read_symbols(file.path(), true).unwrap().unwrap();
    assert_eq!(read_input.name(), "in");
    assert_eq!(read_input.find_symbol(1), Some("a"));

    // The output table sits after the input one, so reaching it means reading
    // past the first.
    let read_output = fst_read_symbols(file.path(), false).unwrap().unwrap();
    assert_eq!(read_output.name(), "out");
    assert_eq!(read_output.find_symbol(2), Some("y"));
}

#[test]
fn reading_symbols_from_a_file_without_them_answers_none() {
    use sicada::fst_header::FstHeader;
    use sicada::symbol_table_ops::fst_read_symbols;
    use std::io::Write as _;

    let mut bytes = Vec::new();
    FstHeader {
        fst_type: "vector".to_string(),
        arc_type: "standard".to_string(),
        version: 2,
        flags: 0,
        properties: 0,
        start: 0,
        num_states: 0,
        num_arcs: 0,
    }
    .write(&mut bytes)
    .unwrap();

    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&bytes).unwrap();
    file.flush().unwrap();

    assert!(fst_read_symbols(file.path(), true).unwrap().is_none());
    assert!(fst_read_symbols(file.path(), false).unwrap().is_none());
}
