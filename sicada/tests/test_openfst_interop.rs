//! Files written by OpenFst itself, read here.
//!
//! The golden tests elsewhere pin the bytes sicada produces against upstream's
//! serialization code extracted and compiled on its own. That leaves one step
//! unchecked: whether a file the real library wrote actually reads, symbol
//! tables and all. The fixtures under `tests/fixtures/` were written by
//! `tests/oracles/interop-fixtures.cc`, which links `libfst.a`.
//!
//! Each is also written back out and compared byte for byte. Identical output
//! is the other direction of the claim: whatever OpenFst can read of its own
//! files, it can read of these.

use sicada::arc::{Arc as _, StdArc};
use sicada::fst::{ExpandedFst, Fst, FstReadOptions, FstWriteOptions};
use sicada::fsts::const_fst::ConstFst;
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
    let bytes = fixture("openfst-const.fst");
    let fst = ConstFst::<StdArc, u32>::read(
        &mut std::io::Cursor::new(&bytes),
        &FstReadOptions::default(),
    )
    .expect("OpenFst's const FST does not read");
    assert_is_the_fixture(&fst);

    let mut out = Vec::new();
    ConstFst::<StdArc, u32>::write_fst(&fst, &mut out, &FstWriteOptions::default()).unwrap();
    assert_eq!(
        out, bytes,
        "the bytes written back differ from the ones OpenFst wrote"
    );
}
