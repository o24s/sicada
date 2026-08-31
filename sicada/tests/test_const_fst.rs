//! A `ConstFst`'s whole life: built from a `VectorFst`, written, read back
//! both directly and through the type name in its header.

use std::fs::File;
use std::io::BufWriter;

use sicada::algorithms::shortest_distance::{SHORTEST_DELTA, shortest_distance};
use sicada::arc::Arc as _;
use sicada::arc::StdArc;
use sicada::fst::{ExpandedFst, Fst, FstReadOptions, FstWriteOptions, MutableFst};
use sicada::fsts::any_fst::AnyFst;
use sicada::fsts::const_fst::ConstFst;
use sicada::fsts::vector_fst::StdVectorFst;
use sicada::weights::float_weight::TropicalWeight;
use tempfile::NamedTempFile;

#[test]
fn a_const_fst_round_trips_through_a_file() {
    let file = NamedTempFile::new().expect("a temporary file");
    let path = file.path();

    let mut source = StdVectorFst::new();
    let s0 = source.add_state();
    let s1 = source.add_state();
    source.set_start(s0);
    source.set_final(s1, TropicalWeight(2.0));
    source.add_arc(s0, StdArc::new(1, 2, TropicalWeight(1.5), s1));

    let const_fst = ConstFst::<StdArc, u32>::from_fst(&source).expect("a const FST");
    assert!(const_fst.fst_type().starts_with("const"));
    assert_eq!(const_fst.num_states(), 2);

    {
        let mut writer = BufWriter::new(File::create(path).expect("a writable file"));
        const_fst
            .write(
                &mut writer,
                &FstWriteOptions {
                    align: true,
                    ..Default::default()
                },
            )
            .expect("the FST written");
    }

    // And read as itself.
    let loaded = ConstFst::<StdArc, u32>::read_from_file(path, &FstReadOptions::default())
        .expect("a const FST");

    // Read without being told what it is: the header names the type.
    let any = AnyFst::<StdArc>::read_from_file(path, &FstReadOptions::default())
        .expect("an FST of some type");
    assert!(any.fst_type().starts_with("const"));
    assert_eq!(any.num_states(), 2);

    let total = shortest_distance(&loaded, SHORTEST_DELTA).expect("a shortest distance");
    assert_eq!(
        total,
        TropicalWeight(1.5 + 2.0),
        "the one path costs the arc plus the final weight"
    );
}
