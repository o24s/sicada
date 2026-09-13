# sicada

Weighted finite-state transducers in Rust. This workspace contains two crates:

- **`sicada`** is an independent implementation of OpenFst's `openfst/lib` and
  is compatible with the OpenFst binary file format.
- **`sicada-decode`** provides frame-synchronous speech decoding, lattices over
  Kaldi's semirings, CTC topologies, and exact forced alignment without beam
  pruning.

Both crates are pre-1.0 and their APIs may change. The remainder of this README
describes `sicada`.

## Differences from OpenFst

- No FFI is used. The library has no build script and links no C++ code. Only
  `sicada-bench` links OpenFst for comparative benchmarks.
- FSTs can be exchanged with OpenFst through the compatible binary format.
- Algorithms are generic over an arc type rather than a weight type. Required
  semiring properties are expressed with trait bounds.
- Runtime type discovery uses a closed enum instead of OpenFst's dynamic type
  registry and `dlopen` plugin mechanism.

## Differences from rustfst

- rustfst and sicada are independent implementations of OpenFst semantics.
- sicada uses generic associated types for iterators, so `Fst` is not
  object-safe and there is no `Box<dyn Fst>`.
- sicada takes inputs by reference and writes to a mutable output argument, so
  Rust can infer every type parameter. In rustfst 1.3.1,
  `compose(owned, &borrowed)` produces `E0283` unless all six type parameters are
  specified ([rustfst#235](https://github.com/garvys-org/rustfst/issues/235)).
  The trade-off is that sicada operations cannot be nested as expressions.

## Benchmarks

These results come from a single build in which all four implementations were
linked into one executable and measured in alternating rounds. Each table shows
the best round. Ratios divide the other implementation's time by sicada's, so a
ratio above 1.00x means sicada was faster. The comparison used OpenFst
`1.8.5-377-ge6bbae9`, rustfst 1.3.1, and arcweight 0.3.0 on x86_64 Linux with
GCC 15.2 and rustc 1.97. C++ used `-O3`; Rust used the `release` profile.

Before timing, each benchmark [checks the number of states and arcs and the
semiring sum over all paths](https://github.com/o24s/sicada/blob/main/sicada-bench/src/bin/ab.rs).

### Data structures

rustfst and arcweight do not expose these data structures, so only OpenFst is
included in this comparison. The C++ benchmark uses the relevant upstream
implementations verbatim.

| | sicada | OpenFst | OpenFst / sicada |
| --- | ---: | ---: | ---: |
| `heap/1k` | 15.6 µs | 29.3 µs | 1.88x |
| `heap/100k` | 6.00 ms | 9.12 ms | 1.52x |
| `heap-insert/1k` | 3.1 µs | 7.8 µs | 2.52x |
| `heap-insert-pop/1k` | 15.0 µs | 25.6 µs | 1.71x |
| `union-find/1k` | 4.7 µs | 4.7 µs | 1.00x |
| `union-find/100k` | 2.04 ms | 2.00 ms | 0.98x |
| `arc-arena/10000x4` | 59.8 µs | 58.6 µs | 0.98x |
| `arc-arena/1000x64` | 62.9 µs | 72.8 µs | 1.16x |
| `compact-set/64` | 49.7 µs | 119.7 µs | 2.41x |
| `compact-set/4096` | 291.2 µs | 513.9 µs | 1.76x |

### Algorithms

Two rows show clear regressions rather than measurement noise. For cyclic
`shortest-distance`, arcweight reaches 0.68x because sicada follows OpenFst and
decomposes the graph into strongly connected components before selecting a
queue. That setup cost is recovered on acyclic inputs. For
`shortest-path/10000x4-acyclic`, rustfst reaches 0.73x because sicada first runs
a depth-first search to obtain the topological order used by the general
distance algorithm, while rustfst uses a dedicated shortest-path search. Ratios
within a few percent of 1.00x vary between runs and should be treated as ties.

| | sicada | OpenFst | rustfst | arcweight | best other / sicada | worst other / sicada |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `shortest-distance/10000x4` | 1.71 ms | 1.98 ms | 1.73 ms | 1.16 ms | 0.68x | 1.16x |
| `shortest-path/10000x4` | 1.69 ms | 1.92 ms | 1.61 ms | 2.72 ms | 0.95x | 1.61x |
| `connect/10000x4` | 633.7 µs | 981.6 µs | 1.13 ms | 1.70 ms | 1.55x | 2.69x |
| `arcsort/10000x4` | 221.9 µs | 1.02 ms | 516.5 µs | 426.2 µs | 1.92x | 4.58x |
| `shortest-distance/2000x16` | 577.4 µs | 637.4 µs | 816.5 µs | 551.4 µs | 0.96x | 1.41x |
| `shortest-path/2000x16` | 553.6 µs | 609.7 µs | 613.1 µs | 1.29 ms | 1.10x | 2.33x |
| `connect/2000x16` | 290.5 µs | 316.2 µs | 318.7 µs | 784.4 µs | 1.09x | 2.70x |
| `arcsort/2000x16` | 291.1 µs | 675.0 µs | 279.9 µs | 405.1 µs | 0.96x | 2.32x |
| `shortest-distance/10000x4-acyclic` | 48.9 µs | 63.2 µs | 51.9 µs | 1.00 ms | 1.06x | 20.56x |
| `shortest-path/10000x4-acyclic` | 49.9 µs | 51.5 µs | 36.3 µs | 1.96 ms | 0.73x | 39.35x |
| `connect/10000x4-acyclic` | 470.2 µs | 750.4 µs | 853.5 µs | 1.01 ms | 1.60x | 2.15x |
| `arcsort/10000x4-acyclic` | 219.6 µs | 924.2 µs | 518.7 µs | 432.5 µs | 1.97x | 4.21x |
| `topsort/10000x4-acyclic` | 677.9 µs | 1.11 ms | 1.43 ms | 1.20 ms | 1.63x | 2.11x |
| `rmepsilon/1000x4` | 243.9 µs | 335.2 µs | 435.2 µs | - ¹ | 1.37x | 1.78x |
| `determinize/1000x4` | 1.41 ms | 2.83 ms | - ² | 1.90 ms | 1.35x | 2.00x |
| `minimize/1000x4` | 3.15 ms | 8.38 ms | 12.18 ms | - ³ | 2.66x | 3.86x |
| `compose/1000x4` | 573.2 µs | 880.8 µs | 1.28 ms | - ⁴ | 1.54x | 2.24x |
| `compose/dense-1000x4` | 241.9 µs | 399.1 µs | 533.0 µs | 295.9 µs | 1.22x | 2.20x |
| `rmepsilon/3000x4` | 729.4 µs | 983.1 µs | 1.32 ms | - ¹ | 1.35x | 1.82x |
| `determinize/3000x4` | 5.85 ms | 11.71 ms | - ² | 7.70 ms | 1.32x | 2.00x |
| `minimize/3000x4` | 11.22 ms | 31.95 ms | 50.44 ms | - ³ | 2.85x | 4.49x |
| `compose/3000x4` | 2.44 ms | 3.45 ms | 5.37 ms | - ⁴ | 1.42x | 2.21x |
| `compose/dense-3000x4` | 873.3 µs | 1.35 ms | 1.93 ms | 1.12 ms | 1.28x | 2.21x |

Some implementations are omitted from individual rows because their output
structure differed from the other three. The total path weight still agrees in
these tropical-semiring benchmarks because ⊕ is `min`, which absorbs duplicate
paths and dominated parallel arcs. With the log semiring, cases 1 and 4 would
also change the total weight. The counts below come from
[`diag`](https://github.com/o24s/sicada/blob/main/sicada-bench/src/bin/diag.rs),
which reports the same three validation metrics:

1. arcweight's `remove_epsilons` [appends the closure's arcs](https://github.com/aaronstevenwhite/arcweight/blob/bec1c8ee9863c914c512d9e601783095917063bd/src/algorithms/rmepsilon.rs#L390-L419)
   without combining parallel arcs. A state reached both directly and through
   an epsilon path therefore retains both arcs: 1437 arcs versus sicada's 1432
   on `1000x4`, with 283 states in both results. Each extra arc duplicates a
   label and destination at a higher weight, so tropical ⊕ discards it.
   `DIAG_EPS=1 diag` confirms that all five differences have this form.
2. rustfst's `determinize` [rebuilds each subset from a `HashMap`](https://github.com/garvys-org/rustfst/blob/8e1391df1ef3dfb85e309dd4ee8af45251d28c9f/rustfst/src/algorithms/determinize/determinize_fsa_op.rs#L147-L179)
   and [compares subsets as an ordered `Vec`](https://github.com/garvys-org/rustfst/blob/8e1391df1ef3dfb85e309dd4ee8af45251d28c9f/rustfst/src/algorithms/determinize/element.rs#L15-L18). Consequently, the same subset can be represented in different orders and
   become multiple states. Four runs on `1000x4` produced 3064, 3125, 3065, and
   3057 states, compared with sicada's stable 2497. Minimization produces 952
   states in both implementations, confirming language equivalence.
3. arcweight's `minimize` produces 1032 states on `1000x4` where the other three
   produce 952. It is [Brzozowski's algorithm](https://github.com/aaronstevenwhite/arcweight/blob/bec1c8ee9863c914c512d9e601783095917063bd/src/algorithms/minimize.rs#L279-L298), reverse
   and determinize twice, not the weight pushing and encoded minimization OpenFst
   uses. Its documentation specifies no weight-pushing precondition and states
   that the operation preserves the weighted language and returns the unique
   canonical minimal FST.
4. arcweight's `compose` produces 4733 states on `1000x4` where the other three
   produce 307. Its [default filter](https://github.com/aaronstevenwhite/arcweight/blob/bec1c8ee9863c914c512d9e601783095917063bd/src/algorithms/compose.rs#L180-L186) is stateless: an
   epsilon-sequencing filter needs state to record which side may advance on an
   epsilon, but this filter's `FilterState` is `()`. Although `compose` accepts a
   filter parameter, `DefaultComposeFilter` is the only implementation provided
   by the crate.

Graph inputs contain `states` states with `arcs` outgoing arcs per state, labels
in 1..64, and weights quantized to quarters. In `-acyclic` inputs, every arc
points forward. Automaton inputs are acyclic acceptors with an epsilon on one in
eight arcs. The `minimize` benchmark determinizes before minimization, and the
`compose` benchmark sorts both inputs first. Determinization results are passed
through `connect` before comparison. The `compose/dense-*` benchmark also
measures two sicada variants not shown in the table: look-ahead composition that
builds its index for each call (486.6 µs and 1.60 ms), and composition with a
prebuilt index (187.0 µs and 746.2 µs).

### Reproducing

```sh
git submodule update --init
cmake -S vendor/openfst -B /path/to/ofst-build \
      -DCMAKE_BUILD_TYPE=Release -DOPENFST_BUILD_TESTS=OFF \
      -DOPENFST_ENABLE_BIN=OFF -DOPENFST_ENABLE_INSTALL=OFF
cmake --build /path/to/ofst-build -j 4
OPENFST_BUILD_DIR=/path/to/ofst-build cargo run --release -p sicada-bench --bin ab
```

Without `OPENFST_BUILD_DIR`, the OpenFst algorithm columns are omitted. The data
structure benchmarks still run because their C++ implementations are compiled
directly into the benchmark crate.

Only measurements from the same build are comparable. Relinking changes the
placement and alignment of the C++ hot loops by enough to affect these results.

## Building

```sh
cargo build
cargo test
```

The OpenFst submodule is required only for comparative benchmarks and source
reference.

## License

Apache License 2.0.

The library does not link against third-party C++ code. Its algorithm semantics
and binary format are based on OpenFst (Apache License 2.0), whose source is
vendored as a reference submodule and remains under its original license. Four
OpenFst data structures are included in `sicada-bench/cpp/openfst_shim.cc` for
direct benchmark comparison; that file retains the OpenFst copyright notice.
The lattice semirings and decoder structure in `sicada-decode` are based on
Kaldi (Apache License 2.0).
