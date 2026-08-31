//! A `CompactFst`'s whole life: compacted from a `VectorFst`, written, and read
//! back through the type name in its header.

use std::fs::File;
use std::io::{BufWriter, Write as _};

use sicada::arc::{Arc, StdArc};
use sicada::cache::CacheOptions;
use sicada::fst::{ExpandedFst, Fst, FstReadOptions, FstWriteOptions, MutableFst};
use sicada::fsts::any_fst::AnyFst;
use sicada::fsts::compact_fst::CompactStringFst;
use sicada::fsts::vector_fst::StdVectorFst;
use sicada::weight::Weight;
use sicada::weights::float_weight::TropicalWeight;
use tempfile::NamedTempFile;

#[test]
fn a_compact_string_fst_round_trips_through_a_file() {
    let file = NamedTempFile::new().expect("a temporary file");
    let path = file.path();

    // A linear unweighted string, of the shape the string compactor packs.
    let mut source = StdVectorFst::new();
    let s0 = source.add_state();
    let s1 = source.add_state();
    let s2 = source.add_state();
    source.set_start(s0);
    source.set_final(s2, TropicalWeight::one());
    source.add_arc(s0, StdArc::new(1, 1, TropicalWeight::one(), s1));
    source.add_arc(s1, StdArc::new(2, 2, TropicalWeight::one(), s2));

    let compact =
        CompactStringFst::<StdArc>::new(&source, Default::default(), CacheOptions::default())
            .expect("a compact FST");
    assert!(compact.fst_type().starts_with("compact"));
    assert_eq!(compact.num_states(), 3);
    assert_eq!(
        compact.arcs(s0).next().expect("the first arc").ilabel(),
        1,
        "the packed arcs read back as they went in"
    );

    {
        let mut writer = BufWriter::new(File::create(path).expect("a writable file"));
        compact
            .write(&mut writer, &FstWriteOptions::default())
            .expect("the FST written");
        writer.flush().expect("flushed");
    }

    let any = AnyFst::<StdArc>::read_from_file(path, &FstReadOptions::default())
        .expect("an FST of some type");
    assert!(any.fst_type().starts_with("compact"));
    assert_eq!(any.num_states(), 3);
    assert_eq!(any.arcs(s0).next().expect("the first arc").ilabel(), 1);
}
