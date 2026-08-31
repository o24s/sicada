# sicada-decode

Decoding an acoustic model's output with [sicada](https://crates.io/crates/sicada).

The inference half of a pair: train on a GPU with k2, then run inference on a
CPU here. It sits outside `sicada` proper, which is a port of OpenFst's library;
the pieces here come from the Kaldi and k2 side.

- `dense` reads the acoustic model's `T × V` score matrix as an FST, so that
  composing a decoding graph against it is an ordinary composition.
- `viterbi` walks that composition one frame at a time without building it.
- `lattice` does the same but keeps the alternatives, over a semiring
  (`lattice_weight`) that holds the graph cost and the acoustic cost apart.
- `compact` collapses the alignments, so each word sequence appears once with
  the best one, over a cost with the frames it spanned attached
  (`compact_lattice_weight`). Both read and write Kaldi's file formats.
- `nbest` reads the answers back out and rescales the two halves against each
  other without decoding again.
- `ctc` builds the graph side for a CTC model.
- `align` is forced alignment: the reference is known, so the graph is one chain
  and the whole band is searched rather than a beam. `occupancy` is the
  forward-backward pass over the same chain.

The API is unstable.

## License

Apache License 2.0. The lattice semirings and the decoder structure follow
Kaldi (Apache License 2.0).
