//! Compiling a string into a linear FST, and reading one back out.
//!
//! Port of OpenFst's `string.h`. A string is an FST with one path: one arc per
//! token, in order, ending at a single final state. Three ways of cutting a
//! string into tokens are supported, namely bytes, UTF-8 characters, and symbols
//! looked up in a [`SymbolTable`], and they are what [`TokenType`] names.

use std::fmt;

use crate::arc::{Arc, ArcLabel, ArcStateId};
use crate::error::OpenFstError;
use crate::fst::{Fst, MutableFst};
use crate::properties::K_COMPILED_STRING_PROPERTIES;
use crate::symbol_table::{K_NO_SYMBOL, SymbolTable};
use crate::utils::io::parse_int64;
use crate::utils::labels::{
    byte_string_to_labels, labels_to_byte_string, labels_to_utf8_string, utf8_string_to_labels,
};
use crate::weight::Weight;

/// The default separator between symbols, matching upstream's
/// `--fst_field_separator`.
pub const DEFAULT_SEPARATOR: &str = "\t ";

/// How a string is cut into tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TokenType {
    /// One token per symbol-table entry, separated by whitespace.
    Symbol,
    /// One token per byte.
    #[default]
    Byte,
    /// One token per UTF-8 character.
    Utf8,
}

impl fmt::Display for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Symbol => "symbol",
            Self::Byte => "byte",
            Self::Utf8 => "utf8",
        })
    }
}

/// What to do about a symbol the table does not have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnknownSymbol<L> {
    /// Refuse the string.
    #[default]
    Refuse,
    /// Use this label instead.
    Map(L),
}

/// Only the last character of `sep` separates symbols, as upstream does. The
/// rest of the string is ignored on output, though every character of it
/// separates on input.
fn output_separator(sep: &str) -> &str {
    match sep.char_indices().next_back() {
        Some((at, _)) => &sep[at..],
        None => "",
    }
}

/// The label one symbol stands for.
///
/// Without a table the token has to be a label written out as an integer, which
/// is how an FST stays readable without one.
fn symbol_to_label<L: ArcLabel>(
    token: &str,
    syms: Option<&SymbolTable>,
    unknown: UnknownSymbol<L>,
) -> Result<L, OpenFstError> {
    let key = match syms {
        Some(syms) => match syms.find_key(token) {
            K_NO_SYMBOL => match unknown {
                UnknownSymbol::Map(label) => return Ok(label),
                UnknownSymbol::Refuse => {
                    return Err(OpenFstError::SymbolTable(format!(
                        "symbol \"{token}\" is not mapped to any label in symbol table {}",
                        syms.name()
                    )));
                }
            },
            key => key,
        },
        None => parse_int64(token, 10).ok_or_else(|| {
            OpenFstError::InvalidOperation(format!("bad label integer \"{token}\""))
        })?,
    };
    L::from_i64(key).ok_or_else(|| {
        OpenFstError::InvalidOperation(format!("label {key} does not fit the arc's label type"))
    })
}

/// Cuts `text` into labels.
///
/// A newline always separates symbols, whatever `sep` says, and empty tokens
/// are dropped, so a run of separators acts as one separator.
///
/// SICADA-DIVERGE: upstream takes the string as bytes and, for the byte and
/// UTF-8 token types, hands them to functions that accept unrestricted
/// Thompson/Pike UTF-8 rather than the standard. That is why this takes `&[u8]`
/// rather than `&str`.
pub fn string_to_labels<L>(
    text: &[u8],
    token_type: TokenType,
    syms: Option<&SymbolTable>,
    unknown: UnknownSymbol<L>,
    sep: &str,
) -> Result<Vec<L>, OpenFstError>
where
    L: ArcLabel + From<u8> + TryFrom<u32>,
{
    let mut labels = Vec::new();
    match token_type {
        TokenType::Byte => {
            labels.reserve(text.len());
            byte_string_to_labels(text, &mut labels);
        }
        TokenType::Utf8 => utf8_string_to_labels(text, &mut labels)?,
        TokenType::Symbol => {
            let text = std::str::from_utf8(text).map_err(|e| {
                OpenFstError::InvalidOperation(format!("symbol string is not UTF-8: {e}"))
            })?;
            for token in text.split(|c: char| c == '\n' || sep.contains(c)) {
                if token.is_empty() {
                    continue;
                }
                labels.push(symbol_to_label(token, syms, unknown)?);
            }
        }
    }
    Ok(labels)
}

/// Writes labels back out as a string.
///
/// With [`TokenType::Symbol`] and no table the labels are written as integers,
/// so a string FST can be printed without one.
pub fn labels_to_string<L: ArcLabel + Into<i64>>(
    labels: &[L],
    token_type: TokenType,
    syms: Option<&SymbolTable>,
    sep: &str,
    omit_epsilon: bool,
) -> Result<Vec<u8>, OpenFstError> {
    let mut out = Vec::new();
    match token_type {
        TokenType::Byte => labels_to_byte_string(labels, &mut out)?,
        TokenType::Utf8 => labels_to_utf8_string(labels, &mut out)?,
        TokenType::Symbol => {
            let sep = output_separator(sep);
            let mut delim = "";
            for label in labels {
                if omit_epsilon && *label == L::epsilon() {
                    continue;
                }
                out.extend_from_slice(delim.as_bytes());
                match syms {
                    Some(syms) => {
                        let key = label.to_i64().ok_or_else(|| {
                            OpenFstError::SymbolTable(format!(
                                "label {label} does not fit a symbol table key"
                            ))
                        })?;
                        let symbol = syms.find_symbol(key).ok_or_else(|| {
                            OpenFstError::SymbolTable(format!(
                                "label {label} is not mapped onto any symbol in symbol table {}",
                                syms.name()
                            ))
                        })?;
                        out.extend_from_slice(symbol.as_bytes());
                    }
                    None => out.extend_from_slice(label.to_string().as_bytes()),
                }
                delim = sep;
            }
        }
    }
    Ok(out)
}

/// Compiles strings into linear FSTs.
///
/// SICADA-DIVERGE: upstream reads the separator from a global command-line flag
/// (`--fst_field_separator`) by default, so the same call compiles differently
/// depending on how the process was started. It is a field here, defaulting to
/// the same characters.
#[derive(Debug, Clone)]
pub struct StringCompiler<'a, L> {
    /// How to cut a string into tokens.
    pub token_type: TokenType,
    /// The table to look symbols up in, when the token type is
    /// [`TokenType::Symbol`].
    pub symbols: Option<&'a SymbolTable>,
    /// What to do about a symbol the table does not have.
    pub unknown: UnknownSymbol<L>,
    /// Which characters separate symbols, besides the newline that always
    /// does.
    pub separator: String,
}

impl<L> Default for StringCompiler<'_, L> {
    fn default() -> Self {
        Self {
            token_type: TokenType::Byte,
            symbols: None,
            unknown: UnknownSymbol::Refuse,
            separator: DEFAULT_SEPARATOR.to_string(),
        }
    }
}

impl<'a, L: ArcLabel + From<u8> + TryFrom<u32>> StringCompiler<'a, L> {
    /// A compiler cutting strings the given way.
    pub fn new(token_type: TokenType) -> Self {
        Self {
            token_type,
            ..Default::default()
        }
    }

    /// Looks symbols up in `symbols`.
    pub fn with_symbols(mut self, symbols: &'a SymbolTable) -> Self {
        self.symbols = Some(symbols);
        self
    }

    /// Uses `label` for a symbol the table does not have, rather than refusing
    /// the string.
    pub fn with_unknown_label(mut self, label: L) -> Self {
        self.unknown = UnknownSymbol::Map(label);
        self
    }

    /// Separates symbols on any of these characters, besides the newline.
    pub fn with_separator(mut self, separator: impl Into<String>) -> Self {
        self.separator = separator.into();
        self
    }

    /// The labels `text` stands for.
    pub fn labels(&self, text: &[u8]) -> Result<Vec<L>, OpenFstError> {
        string_to_labels(
            text,
            self.token_type,
            self.symbols,
            self.unknown,
            &self.separator,
        )
    }

    /// Replaces `fst` with the linear FST accepting `text` at weight
    /// [`Weight::one`].
    pub fn compile<A, F>(&self, text: &[u8], fst: &mut F) -> Result<(), OpenFstError>
    where
        A: Arc<Label = L>,
        F: MutableFst<A>,
    {
        self.compile_weighted(text, fst, A::Weight::one())
    }

    /// Replaces `fst` with the linear FST accepting `text`, whose final weight
    /// is `weight`.
    pub fn compile_weighted<A, F>(
        &self,
        text: &[u8],
        fst: &mut F,
        weight: A::Weight,
    ) -> Result<(), OpenFstError>
    where
        A: Arc<Label = L>,
        F: MutableFst<A>,
    {
        let labels = self.labels(text)?;
        compile_labels(&labels, fst, weight);
        Ok(())
    }
}

/// Replaces `fst` with the linear FST accepting exactly `labels`.
///
/// The same label goes on both sides of every arc, so the result is an
/// acceptor.
pub fn compile_labels<A, F>(labels: &[A::Label], fst: &mut F, weight: A::Weight)
where
    A: Arc,
    F: MutableFst<A>,
{
    fst.delete_all_states();
    let mut state = fst.add_state();
    fst.set_start(state);
    for label in labels {
        let next = fst.add_state();
        fst.add_arc(state, A::new(*label, *label, A::Weight::one(), next));
        state = next;
    }
    fst.set_final(state, weight);
    fst.set_properties(K_COMPILED_STRING_PROPERTIES, K_COMPILED_STRING_PROPERTIES);
}

/// The output labels along the one path of a string FST, and the weight of that
/// path.
///
/// SICADA-DIVERGE: upstream's diagnostic for a state with several outgoing arcs
/// names the state it has already moved on to, because it reassigns `s` before
/// building the message. This names the state that actually has them.
///
/// Upstream documents that this "may loop for non-string FSTs" and leaves it at
/// that; a cycle whose states are all non-final makes it spin. Every arc is
/// followed at most once here, since a state reached twice would be a state
/// with two arcs coming out of it somewhere along the path, so a cyclic input is
/// an error rather than a hang.
pub fn string_fst_to_output_labels<A: Arc, F: Fst<A>>(
    fst: &F,
) -> Result<(Vec<A::Label>, A::Weight), OpenFstError> {
    let mut labels = Vec::new();
    let mut path_weight = A::Weight::one();
    let mut state = fst.start().ok_or(OpenFstError::NoStartState)?;
    let zero = A::Weight::zero();

    let mut final_weight = fst.final_weight(state);
    let mut steps = 0usize;
    while final_weight == zero {
        let mut arcs = fst.arcs(state);
        let arc = arcs.next().ok_or(OpenFstError::DoesNotReachFinalState)?;
        if arcs.next().is_some() {
            return Err(OpenFstError::MultipleOutgoingArcs {
                state: state.as_usize(),
            });
        }
        labels.push(arc.olabel());
        path_weight = path_weight.times(arc.weight());
        state = arc.nextstate();
        final_weight = fst.final_weight(state);

        steps += 1;
        if fst.num_states_if_known().is_some_and(|n| steps > n) {
            return Err(OpenFstError::InvalidOperation(
                "StringFstToOutputLabels: the FST is not a string; the path is cyclic".into(),
            ));
        }
    }
    if fst.num_arcs(state) != 0 {
        return Err(OpenFstError::FinalStateHasOutgoingArcs {
            state: state.as_usize(),
        });
    }
    Ok((labels, path_weight.times(&final_weight)))
}

/// Prints string FSTs.
#[derive(Debug, Clone)]
pub struct StringPrinter<'a> {
    /// How to write the labels out.
    pub token_type: TokenType,
    /// The table to look labels up in, when the token type is
    /// [`TokenType::Symbol`].
    pub symbols: Option<&'a SymbolTable>,
    /// Whether to leave epsilons out. Only applies to
    /// [`TokenType::Symbol`], since the other two have no epsilon to write.
    pub omit_epsilon: bool,
    /// The characters separating symbols; only the last one is written.
    pub separator: String,
}

impl Default for StringPrinter<'_> {
    fn default() -> Self {
        Self {
            token_type: TokenType::Byte,
            symbols: None,
            omit_epsilon: true,
            separator: DEFAULT_SEPARATOR.to_string(),
        }
    }
}

impl<'a> StringPrinter<'a> {
    /// A printer writing labels the given way.
    pub fn new(token_type: TokenType) -> Self {
        Self {
            token_type,
            ..Default::default()
        }
    }

    /// Looks labels up in `symbols`.
    pub fn with_symbols(mut self, symbols: &'a SymbolTable) -> Self {
        self.symbols = Some(symbols);
        self
    }

    /// Writes epsilons out rather than leaving them out.
    pub fn keeping_epsilons(mut self) -> Self {
        self.omit_epsilon = false;
        self
    }

    /// Separates symbols with the last character of `separator`.
    pub fn with_separator(mut self, separator: impl Into<String>) -> Self {
        self.separator = separator.into();
        self
    }

    /// Writes `labels` out.
    pub fn labels_to_string<L: ArcLabel + Into<i64>>(
        &self,
        labels: &[L],
    ) -> Result<Vec<u8>, OpenFstError> {
        labels_to_string(
            labels,
            self.token_type,
            self.symbols,
            &self.separator,
            self.omit_epsilon,
        )
    }

    /// The string a string FST spells on its output side.
    pub fn print<A: Arc, F: Fst<A>>(&self, fst: &F) -> Result<Vec<u8>, OpenFstError>
    where
        A::Label: Into<i64>,
    {
        Ok(self.print_weighted(fst)?.0)
    }

    /// As [`print`](Self::print), with the weight of the path.
    pub fn print_weighted<A: Arc, F: Fst<A>>(
        &self,
        fst: &F,
    ) -> Result<(Vec<u8>, A::Weight), OpenFstError>
    where
        A::Label: Into<i64>,
    {
        let (labels, weight) = string_fst_to_output_labels(fst)?;
        Ok((self.labels_to_string(&labels)?, weight))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::StdArc;
    use crate::fst::ExpandedFst as _;
    use crate::fsts::vector_fst::StdVectorFst;
    use crate::properties::K_ACCEPTOR;
    use crate::weights::float_weight::TropicalWeight;

    fn symbols() -> SymbolTable {
        let mut syms = SymbolTable::new("test");
        syms.add_symbol("<eps>", 0);
        syms.add_symbol("cat", 1);
        syms.add_symbol("sat", 2);
        syms.add_symbol("mat", 3);
        syms
    }

    fn compile(text: &[u8], compiler: &StringCompiler<'_, i32>) -> StdVectorFst {
        let mut fst = StdVectorFst::new();
        compiler.compile(text, &mut fst).unwrap();
        fst
    }

    /// One arc per token, in order, and one final state at the end.
    #[test]
    fn a_string_becomes_one_path() {
        let fst = compile(b"abc", &StringCompiler::default());
        assert_eq!(fst.num_states(), 4);
        assert_eq!(fst.start(), Some(0));
        for (state, byte) in [(0, b'a'), (1, b'b'), (2, b'c')] {
            let arcs: Vec<StdArc> = fst.arcs(state).collect();
            assert_eq!(arcs.len(), 1);
            assert_eq!(arcs[0].ilabel(), byte as i32);
            assert_eq!(
                arcs[0].olabel(),
                byte as i32,
                "a compiled string is an acceptor"
            );
            assert_eq!(*arcs[0].weight(), TropicalWeight::one());
            assert_eq!(arcs[0].nextstate(), state + 1);
        }
        assert_eq!(fst.final_weight(3), TropicalWeight::one());
        assert_ne!(fst.properties(K_ACCEPTOR, false) & K_ACCEPTOR, 0);
    }

    /// The empty string is one state, which is both the start and final.
    #[test]
    fn the_empty_string_is_a_single_final_state() {
        let fst = compile(b"", &StringCompiler::default());
        assert_eq!(fst.num_states(), 1);
        assert_eq!(fst.start(), Some(0));
        assert_eq!(fst.final_weight(0), TropicalWeight::one());
    }

    /// The weight goes on the final state, so the path has it exactly once.
    #[test]
    fn a_weight_goes_on_the_final_state() {
        let mut fst = StdVectorFst::new();
        StringCompiler::<i32>::default()
            .compile_weighted(b"ab", &mut fst, TropicalWeight(2.5))
            .unwrap();
        assert_eq!(fst.final_weight(2), TropicalWeight(2.5));
        for state in 0..2 {
            for arc in fst.arcs(state) {
                assert_eq!(*arc.weight(), TropicalWeight::one());
            }
        }
        let printer = StringPrinter::default();
        let (text, weight) = printer.print_weighted(&fst).unwrap();
        assert_eq!(text, b"ab");
        assert_eq!(weight, TropicalWeight(2.5));
    }

    /// Compiling and printing are inverses, for each way of cutting a string.
    #[test]
    fn compiling_then_printing_gives_the_string_back() {
        let syms = symbols();
        for (token_type, text) in [
            (TokenType::Byte, &b"the cat sat"[..]),
            (TokenType::Utf8, "a\u{00e9}\u{6f22}\u{1F600}".as_bytes()),
            (TokenType::Symbol, &b"cat sat mat"[..]),
        ] {
            let mut compiler = StringCompiler::<i32>::new(token_type);
            let mut printer = StringPrinter::new(token_type);
            if token_type == TokenType::Symbol {
                compiler = compiler.with_symbols(&syms);
                printer = printer.with_symbols(&syms);
            }
            let fst = compile(text, &compiler);
            assert_eq!(printer.print(&fst).unwrap(), text, "{token_type}");
        }
    }

    /// UTF-8 gives one label per character, bytes give one per byte.
    #[test]
    fn the_token_type_decides_how_many_arcs_there_are() {
        let text = "\u{6f22}\u{5b57}".as_bytes();
        assert_eq!(text.len(), 6);
        assert_eq!(
            compile(text, &StringCompiler::<i32>::new(TokenType::Byte)).num_states(),
            7
        );
        assert_eq!(
            compile(text, &StringCompiler::<i32>::new(TokenType::Utf8)).num_states(),
            3
        );
    }

    /// Any of the separator characters splits, a newline always splits, and a
    /// run of them counts once.
    #[test]
    fn symbols_split_on_the_separator_and_on_newlines() {
        let syms = symbols();
        let compiler = StringCompiler::<i32>::new(TokenType::Symbol).with_symbols(&syms);
        for text in [
            &b"cat sat mat"[..],
            &b"cat\tsat\tmat"[..],
            &b"cat\nsat\nmat"[..],
            &b"  cat \t\n sat\t mat  "[..],
        ] {
            assert_eq!(compiler.labels(text).unwrap(), vec![1, 2, 3], "{text:?}");
        }
    }

    /// Only the last character of the separator is written back out, though
    /// every one of them separates on the way in.
    #[test]
    fn only_the_last_separator_character_is_written() {
        let syms = symbols();
        let printer = StringPrinter::new(TokenType::Symbol)
            .with_symbols(&syms)
            .with_separator("\t ");
        assert_eq!(
            printer.labels_to_string(&[1i32, 2, 3]).unwrap(),
            b"cat sat mat"
        );

        let printer = printer.with_separator("-+");
        assert_eq!(
            printer.labels_to_string(&[1i32, 2, 3]).unwrap(),
            b"cat+sat+mat"
        );
    }

    /// Without a table, symbols are the labels written as integers, which is
    /// what makes a string FST printable with no table at all.
    #[test]
    fn symbols_fall_back_to_integers_without_a_table() {
        let compiler = StringCompiler::<i32>::new(TokenType::Symbol);
        assert_eq!(compiler.labels(b"3 1 4 -1").unwrap(), vec![3, 1, 4, -1]);

        let printer = StringPrinter::new(TokenType::Symbol);
        assert_eq!(printer.labels_to_string(&[3i32, 1, 4]).unwrap(), b"3 1 4");

        assert!(compiler.labels(b"not-an-integer").is_err());
    }

    /// A symbol the table does not have is refused, unless a stand-in was
    /// named.
    #[test]
    fn an_unknown_symbol_is_refused_unless_one_was_chosen_for_it() {
        let syms = symbols();
        let compiler = StringCompiler::<i32>::new(TokenType::Symbol).with_symbols(&syms);
        let err = compiler.labels(b"cat dog").unwrap_err();
        assert!(format!("{err}").contains("dog"), "{err}");

        let lenient = compiler.clone().with_unknown_label(99);
        assert_eq!(lenient.labels(b"cat dog").unwrap(), vec![1, 99]);
    }

    /// Epsilons are left out by default, and kept when asked for.
    #[test]
    fn epsilons_are_left_out_unless_they_are_wanted() {
        let syms = symbols();
        let printer = StringPrinter::new(TokenType::Symbol).with_symbols(&syms);
        assert_eq!(printer.labels_to_string(&[1i32, 0, 2]).unwrap(), b"cat sat");
        assert_eq!(
            printer
                .keeping_epsilons()
                .labels_to_string(&[1i32, 0, 2])
                .unwrap(),
            b"cat <eps> sat"
        );
    }

    /// Everything an FST has to be for this to make sense, and what happens
    /// when it is not.
    #[test]
    fn an_fst_that_is_not_a_string_is_refused() {
        // No start state.
        let empty = StdVectorFst::new();
        assert!(string_fst_to_output_labels(&empty).is_err());

        // A state with two ways out.
        let mut forked = StdVectorFst::new();
        for _ in 0..3 {
            forked.add_state();
        }
        forked.set_start(0);
        forked.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        forked.add_arc(0, StdArc::new(2, 2, TropicalWeight::one(), 2));
        forked.set_final(1, TropicalWeight::one());
        let Err(err) = string_fst_to_output_labels(&forked) else {
            panic!("a forked path is not a string")
        };
        assert!(
            format!("{err}").contains('0'),
            "the state named must be the one with the arcs, not the one after it: {err}"
        );

        // A path that stops without reaching a final state.
        let mut stuck = StdVectorFst::new();
        for _ in 0..2 {
            stuck.add_state();
        }
        stuck.set_start(0);
        stuck.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        assert!(string_fst_to_output_labels(&stuck).is_err());

        // A final state that carries on.
        let mut trailing = StdVectorFst::new();
        for _ in 0..2 {
            trailing.add_state();
        }
        trailing.set_start(0);
        trailing.set_final(0, TropicalWeight::one());
        trailing.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        assert!(string_fst_to_output_labels(&trailing).is_err());
    }

    /// A cycle of non-final states with one arc each is where upstream spins
    /// forever.
    #[test]
    fn a_cyclic_path_is_an_error_rather_than_a_hang() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(0, StdArc::new(1, 1, TropicalWeight::one(), 1));
        fst.add_arc(1, StdArc::new(2, 2, TropicalWeight::one(), 2));
        fst.add_arc(2, StdArc::new(3, 3, TropicalWeight::one(), 0));

        let Err(err) = string_fst_to_output_labels(&fst) else {
            panic!("a cycle is not a string")
        };
        assert!(format!("{err}").contains("cyclic"), "{err}");
    }

    /// The output side is what is printed, so an FST whose two sides differ
    /// prints the output one.
    #[test]
    fn the_output_side_is_what_is_printed() {
        let mut fst = StdVectorFst::new();
        for _ in 0..3 {
            fst.add_state();
        }
        fst.set_start(0);
        fst.add_arc(
            0,
            StdArc::new(b'a' as i32, b'x' as i32, TropicalWeight::one(), 1),
        );
        fst.add_arc(
            1,
            StdArc::new(b'b' as i32, b'y' as i32, TropicalWeight::one(), 2),
        );
        fst.set_final(2, TropicalWeight::one());
        assert_eq!(StringPrinter::default().print(&fst).unwrap(), b"xy");
    }

    /// A label that is not a byte cannot be written as one.
    #[test]
    fn a_label_too_large_for_a_byte_is_refused() {
        let printer = StringPrinter::new(TokenType::Byte);
        assert!(printer.labels_to_string(&[300i32]).is_err());
    }

    #[test]
    fn a_token_type_names_itself_the_way_upstream_does() {
        assert_eq!(TokenType::Byte.to_string(), "byte");
        assert_eq!(TokenType::Utf8.to_string(), "utf8");
        assert_eq!(TokenType::Symbol.to_string(), "symbol");
    }
}
