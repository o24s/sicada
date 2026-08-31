# Oracles

The C++ programs that produced the byte sequences and decision tables pinned in
this crate's tests. Each one is a verbatim extraction from
`vendor/openfst/openfst/lib/`, reduced to the part that emits the value, so the
constant in the Rust test can be traced back to what OpenFst actually writes.

The golden ones are verbatim extractions compiled on their own.
`interop-fixtures.cc` and `hlg-reference.cc` instead link `libfst.a` and run the
real algorithms, so they need OpenFst built as the README describes.

Nothing here is built by cargo. They are run by hand when a format or a
decision table needs to be established or re-checked, and each file carries its
own build line:

```sh
g++ -std=c++17 -O2 -o /tmp/vfg tests/oracles/vector-fst-golden.cc && /tmp/vfg
```

| file | pins |
| --- | --- |
| `compose-filter-decisions.cc` | the six compose filters' decisions over every input combination |
| `const-fst-golden.cc` | `ConstFst`'s on-disk layout and bytes |
| `encode-table-golden.cc` | the encode table's serialized form |
| `fst-header-golden.cc` | the FST file header |
| `io-golden.cc` | the scalar, string and vector encodings |
| `property-bits.cc` | the value of every `kProperty` constant |
| `property-functions.cc` | how the property functions propagate bits |
| `symbol-table-golden.cc` | the symbol table's serialized form |
| `vector-fst-golden.cc` | `VectorFst`'s body bytes |
| `interop-fixtures.cc` | writes `tests/fixtures/`, read back by `test_openfst_interop.rs` |
| `hlg-reference.cc` | the HLG in `tests/fixtures/`, compared against in `test_hlg.rs` |

A change to any of these is a change to the file format, and breaks
compatibility with files OpenFst produced.
