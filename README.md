# sicada

Weighted finite-state transducers in Rust. The repository holds two crates:

- **`sicada`** implements OpenFst's `openfst/lib` from scratch, and reads and
  writes the same files.
- **`sicada-decode`** sits on top of it: a frame-synchronous decoder, lattices
  over Kaldi's semirings, CTC topologies and an exact forced aligner (no beam).

Both APIs are unstable. The sections below are about `sicada`.

## Differences from OpenFst

- No FFI. The library links nothing and has no build script; only `sicada-bench`
  links against the C++, to measure it.
- The binary file format is the same, so FSTs written by OpenFst can be read here
  and vice versa.
- Algorithms are generic over an arc type rather than over a weight, and carry no
  other type arguments. Most state the semiring properties they rely on as trait
  bounds instead of taking any weight.
- There is no dynamic type registry and no `dlopen` plugin mechanism. Reading an
  FST whose type is only known from its file header dispatches over a closed
  enum instead.

## Differences from rustfst

- rustfst is also a port of OpenFst; the two are independent.
- sicada uses generic associated types for its iterators, so `Fst` is not
  object-safe and there is no `Box<dyn Fst>`.
- Call sites need no type annotations: sicada takes its inputs by reference and
  writes into a `&mut` output, so every type parameter appears in an argument.
  rustfst returns the output and takes inputs through `Borrow`, so in 1.3.1
  `compose(owned, &borrowed)` is `E0283` and wants all six spelled out
  ([rustfst#235](https://github.com/garvys-org/rustfst/issues/235)). The cost is that calls do not nest into expressions.

## Benchmarks

One run, all four implementations built together and measured alternately, best
round of each. Ratios are the other implementation divided by sicada, so **above
1.00x means sicada took less time**. Compared against OpenFst at
`1.8.5-377-ge6bbae9`, rustfst 1.3.1 and arcweight 0.3.0, on x86_64 Linux with
gcc 15.2 and rustc 1.97, at `-O3` and `release`.

Every benchmark [asserts that the implementations produced the same
answer](https://github.com/o24s/sicada/blob/main/sicada-bench/src/bin/ab.rs) before timing them, as states, arcs
and the ⊕-sum over all paths.

### Data structures

rustfst and arcweight do not expose these, so the comparison is against OpenFst
only. The C++ side is upstream's own code extracted verbatim.

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

Two rows are losses rather than ties. On the cyclic `shortest-distance`,
arcweight is at 0.68x: sicada chooses its queue by decomposing the graph into
strongly connected components, as OpenFst does and arcweight does not, and the
acyclic rows are where that decomposition pays for itself. On
`shortest-path/10000x4-acyclic`, rustfst is at 0.73x: sicada reads the acyclic
property and takes a queue in topological order, and most of the time goes into
the depth-first search that produces that order, where rustfst has a search of
its own for shortest paths that does not go through the general distance
algorithm. Rows within a few percent of 1.00x move between runs and are
ties in either direction.

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

Left out of a row because the result differed from the other three. The sum over
all paths agrees everywhere, so these are structural differences: the semiring
here is tropical, where ⊕ is `min`, so a duplicated path or a parallel arc is
absorbed and never reaches the total. Over the log semiring the differences in
1 and 4 would be differences in the answer. The numbers below are from
[`diag`](https://github.com/o24s/sicada/blob/main/sicada-bench/src/bin/diag.rs), which prints the same three:

1. arcweight's `remove_epsilons` [appends the closure's arcs](https://github.com/aaronstevenwhite/arcweight/blob/bec1c8ee9863c914c512d9e601783095917063bd/src/algorithms/rmepsilon.rs#L390-L419)
   without combining parallel ones, so a state reached both directly and along
   an epsilon path keeps two arcs to it: 1437 against 1432 on `1000x4`, out of
   the same 283 states. Every extra arc repeats one of sicada's along the same
   label to the same state carrying the heavier weight, which is the one ⊕
   discards; `DIAG_EPS=1 diag` counts them and finds no other kind.
2. rustfst's `determinize` [rebuilds each subset from a `HashMap`](https://github.com/garvys-org/rustfst/blob/8e1391df1ef3dfb85e309dd4ee8af45251d28c9f/rustfst/src/algorithms/determinize/determinize_fsa_op.rs#L147-L179)
   and [compares subsets as an ordered `Vec`](https://github.com/garvys-org/rustfst/blob/8e1391df1ef3dfb85e309dd4ee8af45251d28c9f/rustfst/src/algorithms/determinize/element.rs#L15-L18), so one subset reached
   twice can become two states. The count is not stable between runs: four runs
   on `1000x4` gave 3064, 3125, 3065 and 3057 states against sicada's 2497. Its
   `minimize` is stable and agrees at 952, so the language is the same.
3. arcweight's `minimize` produces 1032 states on `1000x4` where the other three
   produce 952. It is [Brzozowski's algorithm](https://github.com/aaronstevenwhite/arcweight/blob/bec1c8ee9863c914c512d9e601783095917063bd/src/algorithms/minimize.rs#L279-L298), reverse
   and determinize twice, not the weight pushing and encoded minimization OpenFst
   uses. No precondition is documented: what it says instead is that it preserves
   the weighted language and returns the unique canonical minimal FST, so the
   input is not expected to arrive pushed.
4. arcweight's `compose` produces 4733 states on `1000x4` where the other three
   produce 307. Its [default filter](https://github.com/aaronstevenwhite/arcweight/blob/bec1c8ee9863c914c512d9e601783095917063bd/src/algorithms/compose.rs#L180-L186) is stateless: an
   epsilon-sequencing filter needs somewhere to record which side may advance on
   an epsilon, and this one's `FilterState` is `()`. `compose` does take a filter,
   but `DefaultComposeFilter` is the only one the crate implements, so this is
   not a choice the caller can make differently.

The graph inputs are `states` states with `arcs` arcs each, labels 1..64, weights
in quarters; `-acyclic` means arcs always point forward. The automaton inputs are
acyclic acceptors with one arc in eight unlabelled. `minimize` is determinize
then minimize; `compose` sorts both sides first. All four determinization results
are `connect`ed before comparison. `compose/dense-*` also carries two sicada
variants not shown above: look-ahead composition building its index each time
(486.6 µs and 1.60 ms) and with the index already built (187.0 µs and 746.2 µs).

### Reproducing

```sh
git submodule update --init
cmake -S vendor/openfst -B /path/to/ofst-build \
      -DCMAKE_BUILD_TYPE=Release -DOPENFST_BUILD_TESTS=OFF \
      -DOPENFST_ENABLE_BIN=OFF -DOPENFST_ENABLE_INSTALL=OFF
cmake --build /path/to/ofst-build -j 4
OPENFST_BUILD_DIR=/path/to/ofst-build cargo run --release -p sicada-bench --bin ab
```

Without `OPENFST_BUILD_DIR` the OpenFst algorithm columns are absent; the data
structure rows still run, since that C++ is compiled into the benchmark crate.

Only figures from the same build are comparable: relinking moves the C++ code and
changes hot-loop alignment by more than the differences measured here.

## Building

```sh
cargo build
cargo test
```

The OpenFst submodule is only needed for the benchmarks or to consult the C++.

## License

Apache License 2.0.

This library does not link against any third-party C++ code. Its design,
algorithm semantics, and binary file format are derived from OpenFst (Apache
License 2.0), whose sources are vendored as a submodule for reference; that code
remains under its own license. Four of its data structures are copied into
`sicada-bench/cpp/openfst_shim.cc` so that the benchmarks measure upstream's own
code, and that file carries OpenFst's copyright. `sicada-decode` follows Kaldi
(Apache License 2.0) for its lattice semirings and decoder structure.
