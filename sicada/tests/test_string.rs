use sicada::fst::ExpandedFst;
use sicada::fst_linear;
use sicada::string::{StringCompiler, StringPrinter, TokenType, string_fst_to_output_labels};
use sicada::vector_fst::StdVectorFst;
use sicada::weight::Weight;
use sicada::weights::float_weight::TropicalWeight;

#[test]
fn a_literal_compiles_to_one_arc_per_byte() {
    let fst = fst_linear!(StdVectorFst, "hello");
    assert_eq!(fst.num_states(), 6);

    let (text, weight) = StringPrinter::default().print_weighted(&fst).unwrap();
    assert_eq!(text, b"hello");
    assert_eq!(weight, TropicalWeight::one());
}

#[test]
fn a_list_of_labels_compiles_to_one_arc_each() {
    let fst = fst_linear!(StdVectorFst, [10, 20, 30]);
    let (labels, _) = string_fst_to_output_labels(&fst).unwrap();
    assert_eq!(labels, vec![10, 20, 30]);
}

/// The compiler and the printer are inverses across the public API too, not
/// only inside the crate.
#[test]
fn a_symbol_string_survives_the_round_trip_through_the_public_api() {
    let mut syms = sicada::symbol_table::SymbolTable::new("words");
    syms.add_symbol("<eps>", 0);
    syms.add_symbol("hello", 1);
    syms.add_symbol("world", 2);

    let compiler = StringCompiler::<i32>::new(TokenType::Symbol).with_symbols(&syms);
    let mut fst = StdVectorFst::new();
    compiler.compile(b"hello world", &mut fst).unwrap();
    assert_eq!(fst.num_states(), 3);

    let printer = StringPrinter::new(TokenType::Symbol).with_symbols(&syms);
    assert_eq!(printer.print(&fst).unwrap(), b"hello world");
}
