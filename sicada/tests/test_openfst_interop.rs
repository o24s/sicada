//! Every file format sicada names, read from files OpenFst itself wrote.
//!
//! The golden tests elsewhere pin the bytes sicada produces against upstream's
//! serialization code extracted and compiled on its own. That leaves one step
//! unchecked: whether a file the real library wrote actually reads, symbol
//! tables and all. The fixtures under `tests/fixtures/` were written by
//! `tests/oracles/interop-fixtures.cc`, which links `libfst.a`.
//!
//! Seventeen formats, which fall into three shapes:
//!
//! - `vector` and `const` lay the arcs out one after another.
//! - The ten compact formats store only the fields their compactor can rebuild
//!   an arc from, so the layout differs per format and a reader that has it
//!   wrong still hands back states and arcs, just the wrong ones. Five
//!   compactors at each of the two widths the offsets can have.
//! - `edit`, `arc_lookahead`, `ilabel_lookahead` and `olabel_lookahead` put a
//!   second structure after the first: a base FST and the edits over it, or a
//!   base FST and a look-ahead index beside it.
//!
//! Twelve of them are also written back out and compared byte for byte, which
//! is the other direction of the claim: what OpenFst reads of its own files, it
//! reads of these. The five that are not say why where they are tested, and for
//! the one whose bytes differ by a field rather than by padding, that upstream
//! still reads the result is established by `tests/oracles/interop-readback.cc`
//! rather than asserted here, since checking it needs the C++ library.

use std::io::Cursor;

use sicada::arc::{Arc as _, StdArc};
use sicada::fst::{ExpandedFst, Fst, FstReadOptions, FstWriteOptions};
use sicada::fst_type::FstType;
use sicada::fsts::any_fst::AnyFst;
use sicada::fsts::compact_fst::{
    AcceptorCompactor, ArcCompactor, CompactFst, CompactStringFst, StringCompactor, Unsigned,
    UnweightedAcceptorCompactor, UnweightedCompactor, WeightedStringCompactor,
};
use sicada::fsts::const_fst::ConstFst;
use sicada::fsts::edit_fst::EditFst;
use sicada::fsts::matcher_fst::MatcherFst;
use sicada::fsts::vector_fst::VectorFst;
use sicada::weight::Weight;
use sicada::weights::float_weight::TropicalWeight;

fn fixture(name: &str) -> Vec<u8> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/").to_string() + name;
    std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// What `interop-fixtures.cc` builds, in the order a reader sees it.
fn assert_is_the_fixture<F: Fst<StdArc> + ExpandedFst<StdArc>>(fst: &F) {
    assert_eq!(fst.num_states(), 3);
    assert_eq!(fst.start(), Some(0));

    let arcs = |s: i32| {
        fst.arcs(s)
            .map(|a| (a.ilabel(), a.olabel(), a.weight().0, a.nextstate()))
            .collect::<Vec<_>>()
    };
    assert_eq!(arcs(0), vec![(1, 10, 0.5, 1), (2, 20, 1.5, 2)]);
    assert_eq!(arcs(1), vec![(3, 30, 2.25, 2)]);
    assert_eq!(arcs(2), vec![]);

    assert_eq!(fst.final_weight(0), TropicalWeight::zero());
    assert_eq!(fst.final_weight(1), TropicalWeight::zero());
    assert_eq!(fst.final_weight(2), TropicalWeight(0.75));

    let isyms = fst.input_symbols().expect("the file carries input symbols");
    assert_eq!(isyms.name(), "in");
    for (label, symbol) in [(0, "<eps>"), (1, "a"), (2, "b"), (3, "c")] {
        assert_eq!(isyms.find_symbol(label), Some(symbol));
    }
    let osyms = fst
        .output_symbols()
        .expect("the file carries output symbols");
    assert_eq!(osyms.name(), "out");
    for (label, symbol) in [(0, "<eps>"), (10, "X"), (20, "Y"), (30, "Z")] {
        assert_eq!(osyms.find_symbol(label), Some(symbol));
    }
}

#[test]
fn a_vector_fst_openfst_wrote_reads_back_and_writes_back_identically() {
    let bytes = fixture("openfst-vector.fst");
    let fst: VectorFst<StdArc> = VectorFst::read(&mut bytes.as_slice(), &FstReadOptions::default())
        .expect("OpenFst's vector FST does not read");
    assert_is_the_fixture(&fst);

    let mut out = Vec::new();
    fst.write(&mut out, &FstWriteOptions::default()).unwrap();
    assert_eq!(
        out, bytes,
        "the bytes written back differ from the ones OpenFst wrote"
    );
}

#[test]
fn a_const_fst_openfst_wrote_reads_back_and_writes_back_identically() {
    fn check<U: Unsigned>(name: &str) {
        let bytes = fixture(name);
        let fst = ConstFst::<StdArc, U>::read(&mut Cursor::new(&bytes), &FstReadOptions::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_is_the_fixture(&fst);

        let mut out = Vec::new();
        ConstFst::<StdArc, U>::write_fst(&fst, &mut out, &FstWriteOptions::default()).unwrap();
        assert_eq!(
            out, bytes,
            "the bytes written back differ from the ones OpenFst wrote"
        );
    }
    // The offsets into the arc store are what the unsigned parameter sizes, and
    // upstream gives the wider one its own name.
    check::<u32>("openfst-const.fst");
}

/// The 64-bit const format, which cannot be compared byte for byte.
///
/// `ConstState` puts a four-byte weight in front of the first offset, so at a
/// width of eight there are four bytes of padding between them, and upstream
/// writes the state array to the stream as it stands in memory. Those four
/// bytes per state are whatever the allocator last left there: two runs of
/// `tests/oracles/interop-fixtures.cc` produce different `openfst-const64.fst`
/// files, while every other fixture it writes is reproducible. sicada writes
/// zeros. What can be checked is that the file reads, so this checks that.
#[test]
fn a_const64_fst_openfst_wrote_reads_back() {
    let bytes = fixture("openfst-const64.fst");
    let fst = ConstFst::<StdArc, u64>::read(&mut Cursor::new(&bytes), &FstReadOptions::default())
        .expect("OpenFst's 64-bit const FST does not read");
    assert_is_the_fixture(&fst);

    // Written back and read again, it is still the same FST: the padding is the
    // only thing sicada does differently, and nothing reads it.
    let mut out = Vec::new();
    ConstFst::<StdArc, u64>::write_fst(&fst, &mut out, &FstWriteOptions::default()).unwrap();
    assert_eq!(out.len(), bytes.len(), "the same size, padding aside");
    let again = ConstFst::<StdArc, u64>::read(&mut Cursor::new(&out), &FstReadOptions::default())
        .expect("what sicada wrote does not read");
    assert_is_the_fixture(&again);
}

/// `(ilabel, olabel, weight, nextstate)` for one state, in the order stored.
fn arcs_of<F: Fst<StdArc>>(fst: &F, state: i32) -> Vec<(i32, i32, f32, i32)> {
    fst.arcs(state)
        .map(|a| (a.ilabel(), a.olabel(), a.weight().0, a.nextstate()))
        .collect()
}

/// The final weight of each state, `None` where the state is not final.
fn finals<F: Fst<StdArc> + ExpandedFst<StdArc>>(fst: &F) -> Vec<Option<f32>> {
    (0..fst.num_states() as i32)
        .map(|s| {
            let w = fst.final_weight(s);
            (w != TropicalWeight::zero()).then_some(w.0)
        })
        .collect()
}

/// Reads a compact fixture and checks that writing it back reproduces the file.
fn round_trip<C, U>(name: &str) -> CompactFst<'static, StdArc, C, U>
where
    C: ArcCompactor<StdArc> + Default,
    U: Unsigned,
{
    let bytes = fixture(name);
    let fst = CompactFst::<StdArc, C, U>::read(
        &mut Cursor::new(bytes.clone()),
        &FstReadOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{name}: {e}"));

    let mut out = Vec::new();
    fst.write(&mut out, &FstWriteOptions::default())
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(out, bytes, "{name} did not come back out byte for byte");
    fst
}

/// A chain of three arcs, rebuilt from labels alone: the weight is One and the
/// next state is the one after this, neither of them on disk.
#[test]
fn a_compact_string_fst_written_by_openfst_reads_here() {
    fn check<F: Fst<StdArc> + ExpandedFst<StdArc>>(fst: &F) {
        assert_eq!(fst.num_states(), 4);
        assert_eq!(fst.start(), Some(0));
        for state in 0..3 {
            assert_eq!(
                arcs_of(fst, state),
                vec![(state + 1, state + 1, 0.0, state + 1)]
            );
        }
        assert_eq!(finals(fst), vec![None, None, None, Some(0.0)]);
    }
    check(&round_trip::<StringCompactor<StdArc>, u32>(
        "openfst-compact-string.fst",
    ));
    check(&round_trip::<StringCompactor<StdArc>, u64>(
        "openfst-compact64-string.fst",
    ));
}

/// The same chain with a weight per arc, which this format does store.
#[test]
fn a_compact_weighted_string_fst_written_by_openfst_reads_here() {
    fn check<F: Fst<StdArc> + ExpandedFst<StdArc>>(fst: &F) {
        assert_eq!(fst.num_states(), 4);
        assert_eq!(arcs_of(fst, 0), vec![(1, 1, 0.5, 1)]);
        assert_eq!(arcs_of(fst, 1), vec![(2, 2, 1.5, 2)]);
        assert_eq!(arcs_of(fst, 2), vec![(3, 3, 2.25, 3)]);
        assert_eq!(finals(fst), vec![None, None, None, Some(0.75)]);
    }
    check(&round_trip::<WeightedStringCompactor<StdArc>, u32>(
        "openfst-compact-weighted-string.fst",
    ));
    check(&round_trip::<WeightedStringCompactor<StdArc>, u64>(
        "openfst-compact64-weighted-string.fst",
    ));
}

/// Branching, so the next state is stored; one label per arc, so only one is.
#[test]
fn a_compact_acceptor_fst_written_by_openfst_reads_here() {
    fn check<F: Fst<StdArc> + ExpandedFst<StdArc>>(fst: &F) {
        assert_eq!(fst.num_states(), 3);
        assert_eq!(arcs_of(fst, 0), vec![(1, 1, 0.5, 1), (2, 2, 1.5, 2)]);
        assert_eq!(arcs_of(fst, 1), vec![(3, 3, 2.25, 2)]);
        assert_eq!(arcs_of(fst, 2), vec![]);
        assert_eq!(finals(fst), vec![None, None, Some(0.75)]);
    }
    check(&round_trip::<AcceptorCompactor<StdArc>, u32>(
        "openfst-compact-acceptor.fst",
    ));
    check(&round_trip::<AcceptorCompactor<StdArc>, u64>(
        "openfst-compact64-acceptor.fst",
    ));
}

/// One label and a next state per arc, and no weight at all.
#[test]
fn a_compact_unweighted_acceptor_fst_written_by_openfst_reads_here() {
    fn check<F: Fst<StdArc> + ExpandedFst<StdArc>>(fst: &F) {
        assert_eq!(fst.num_states(), 3);
        assert_eq!(arcs_of(fst, 0), vec![(1, 1, 0.0, 1), (2, 2, 0.0, 2)]);
        assert_eq!(arcs_of(fst, 1), vec![(3, 3, 0.0, 2)]);
        assert_eq!(finals(fst), vec![None, None, Some(0.0)]);
    }
    check(&round_trip::<UnweightedAcceptorCompactor<StdArc>, u32>(
        "openfst-compact-unweighted-acceptor.fst",
    ));
    check(&round_trip::<UnweightedAcceptorCompactor<StdArc>, u64>(
        "openfst-compact64-unweighted-acceptor.fst",
    ));
}

/// Both labels and a next state, and no weight: the transducer of the pair.
#[test]
fn a_compact_unweighted_fst_written_by_openfst_reads_here() {
    fn check<F: Fst<StdArc> + ExpandedFst<StdArc>>(fst: &F) {
        assert_eq!(fst.num_states(), 3);
        assert_eq!(arcs_of(fst, 0), vec![(1, 10, 0.0, 1), (2, 20, 0.0, 2)]);
        assert_eq!(arcs_of(fst, 1), vec![(3, 30, 0.0, 2)]);
        assert_eq!(finals(fst), vec![None, None, Some(0.0)]);
    }
    check(&round_trip::<UnweightedCompactor<StdArc>, u32>(
        "openfst-compact-unweighted.fst",
    ));
    check(&round_trip::<UnweightedCompactor<StdArc>, u64>(
        "openfst-compact64-unweighted.fst",
    ));
}

/// The five compact formats are five layouts, and reading one as another has to
/// fail rather than produce arcs from whatever the bytes happen to say.
#[test]
fn a_compact_file_does_not_read_as_the_wrong_compactor() {
    let bytes = fixture("openfst-compact-acceptor.fst");
    let wrong =
        CompactStringFst::<StdArc>::read(&mut Cursor::new(bytes), &FstReadOptions::default());
    assert!(
        wrong.is_err(),
        "an acceptor file read as a string FST has to be refused"
    );
}

/// The edit format keeps the base FST and the edits over it apart, and the
/// reader has to apply the second to the first: state 1 is final only because
/// an edit made it so, and state 2's second arc is only in the edits.
#[test]
fn an_edit_fst_written_by_openfst_reads_here() {
    let bytes = fixture("openfst-edit.fst");
    let fst = EditFst::<StdArc, AnyFst<StdArc>>::read(
        &mut Cursor::new(bytes),
        &FstReadOptions::default(),
    )
    .expect("the edit fixture reads");

    assert_eq!(fst.num_states(), 3);
    assert_eq!(fst.start(), Some(0));
    assert_eq!(arcs_of(&fst, 0), vec![(1, 10, 0.5, 1), (2, 20, 1.5, 2)]);
    assert_eq!(arcs_of(&fst, 1), vec![(3, 30, 2.25, 2)]);
    assert_eq!(arcs_of(&fst, 2), vec![(2, 20, 4.0, 0)], "the added arc");
    assert_eq!(
        finals(&fst),
        vec![None, Some(3.5), Some(0.75)],
        "state 1 is final only through an edit"
    );
}

/// The matcher format stores a look-ahead index beside the FST. What the states
/// and arcs are is the wrapped FST's answer, so reading one back has to give
/// the base FST unchanged, sorted the way the index was built over.
#[test]
fn a_matcher_fst_written_by_openfst_reads_here() {
    let bytes = fixture("openfst-matcher-arc.fst");
    let fst = MatcherFst::<StdArc>::read(
        &mut Cursor::new(bytes.clone()),
        &FstReadOptions::default(),
        FstType::ARC_LOOKAHEAD,
    )
    .expect("the matcher fixture reads");

    assert_eq!(fst.num_states(), 3);
    assert_eq!(fst.start(), Some(0));
    assert_eq!(arcs_of(&fst, 0), vec![(1, 10, 0.5, 1), (2, 20, 1.5, 2)]);
    assert_eq!(arcs_of(&fst, 1), vec![(3, 30, 2.25, 2)]);
    assert_eq!(finals(&fst), vec![None, None, Some(0.75)]);

    // Written back, the two differ in three fields of the *outer* header and
    // nowhere else. Upstream leaves the start state and the two counts at the
    // placeholders it wrote them with, since the nested FST carries its own
    // header and these are never read back; sicada fills them in from the
    // wrapped FST. `tests/oracles/interop-readback.cc` is how "never read back"
    // was established: OpenFst reads the file this test writes and reports the
    // same start state, arcs, weights and symbol table.
    let mut out = Vec::new();
    fst.write(&mut out, &FstWriteOptions::default())
        .expect("it writes");

    let (theirs, header_len) = outer_header(&bytes);
    let (ours, our_len) = outer_header(&out);
    assert_eq!(header_len, our_len, "the outer headers are the same length");
    assert_eq!(
        bytes[header_len..],
        out[header_len..],
        "everything past the outer header is identical"
    );
    assert_eq!(theirs, (-1, 0, 0), "upstream leaves the placeholders");
    assert_eq!(ours, (0, 3, -1), "sicada fills them from the wrapped FST");
}

/// `(start, num_states, num_arcs)` from an FST header, and where the header
/// ends. Only the flagless case is handled, which is what a matcher FST writes:
/// its symbol tables belong to the FST nested inside it.
fn outer_header(bytes: &[u8]) -> ((i64, i64, i64), usize) {
    let mut at = 0usize;
    let i32_at = |at: &mut usize| {
        let v = i32::from_le_bytes(bytes[*at..*at + 4].try_into().unwrap());
        *at += 4;
        v
    };
    let i64_at = |at: &mut usize| {
        let v = i64::from_le_bytes(bytes[*at..*at + 8].try_into().unwrap());
        *at += 8;
        v
    };
    assert_eq!(i32_at(&mut at), 2_125_659_606, "the OpenFst magic number");
    for _ in 0..2 {
        let len = i32_at(&mut at) as usize;
        at += len;
    }
    i32_at(&mut at); // version
    assert_eq!(i32_at(&mut at), 0, "no symbol tables in the outer header");
    i64_at(&mut at); // properties
    let start = i64_at(&mut at);
    let states = i64_at(&mut at);
    let arcs = i64_at(&mut at);
    ((start, states, arcs), at)
}

/// A label look-ahead file is refused, and that is the right answer rather than
/// a gap.
///
/// Upstream's `ilabel_lookahead_flags` leave `kLookAheadKeepRelabelData` off,
/// so the file it writes has the labels renumbered to the index and no map back
/// to the originals. sicada keeps the map and looks a label up per question, so
/// there is nothing it can say about a file whose labels are already gone.
/// Reading it as though the numbers were labels would give a matcher that
/// quietly answers "no" to every real label.
#[test]
fn a_label_lookahead_file_without_its_label_map_is_refused() {
    for (name, fst_type) in [
        ("openfst-matcher-ilabel.fst", FstType::ILABEL_LOOKAHEAD),
        ("openfst-matcher-olabel.fst", FstType::OLABEL_LOOKAHEAD),
    ] {
        let read = MatcherFst::<StdArc>::read(
            &mut Cursor::new(fixture(name)),
            &FstReadOptions::default(),
            fst_type,
        );
        let Err(refused) = read else {
            panic!("{name}: a file with no label map cannot be read");
        };
        assert!(
            refused.to_string().contains("label map"),
            "{name}: the reason should name the missing map, got: {refused}"
        );
    }
}

/// The symbol tables ride in the header of every format that has one, so a
/// reader that gets the body right can still lose them.
#[test]
fn the_symbol_tables_survive_the_formats_that_carry_them() {
    let bytes = fixture("openfst-edit.fst");
    let fst = EditFst::<StdArc, AnyFst<StdArc>>::read(
        &mut Cursor::new(bytes),
        &FstReadOptions::default(),
    )
    .expect("the edit fixture reads");

    let isyms = fst.input_symbols().expect("input symbols");
    let osyms = fst.output_symbols().expect("output symbols");
    for (label, symbol) in [(0, "<eps>"), (1, "a"), (2, "b"), (3, "c")] {
        assert_eq!(isyms.find_symbol(label), Some(symbol));
    }
    for (label, symbol) in [(0, "<eps>"), (10, "X"), (20, "Y"), (30, "Z")] {
        assert_eq!(osyms.find_symbol(label), Some(symbol));
    }
}
