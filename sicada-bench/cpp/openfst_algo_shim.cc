// Benchmark shim for the algorithm comparisons, built against real OpenFst.
//
// Unlike `openfst_shim.cc`, which extracts a few self-contained data structures
// so that they can be measured without a build of the library, this file
// includes upstream's own headers and links against the library CMake builds
// from `vendor/openfst`. Nothing here is a paraphrase of an algorithm: the
// benchmark calls `fst::ShortestPath` and so on directly.
//
// It is compiled only when the environment variable OPENFST_BUILD_DIR points at
// that build, which has to be made once by hand: it needs cmake and downloads
// abseil.
//
// Both sides of every comparison build the same FST from the same generator and
// return a checksum of the result, which the harness asserts equal before it
// times anything.

#include <cstdint>
#include <vector>

#include "openfst/lib/arcsort.h"
#include "openfst/lib/compose.h"
#include "openfst/lib/determinize.h"
#include "openfst/lib/minimize.h"
#include "openfst/lib/rmepsilon.h"
#include "openfst/lib/connect.h"
#include "openfst/lib/shortest-distance.h"
#include "openfst/lib/shortest-path.h"
#include "openfst/lib/topsort.h"
#include "openfst/lib/vector-fst.h"

namespace {

using fst::StdArc;
using fst::StdVectorFst;
using fst::TropicalWeight;

// The generator the Rust side uses too, so both build the same FST.
struct Xorshift {
  uint64_t state;
  explicit Xorshift(uint64_t seed) : state(seed) {}
  uint64_t Next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  }
};

// Weights are multiples of 1/4 so that every path sum is exact in binary and
// the two implementations cannot disagree by a rounding step.
float WeightOf(Xorshift* rng) {
  return static_cast<float>(rng->Next() % 400) / 4.0f;
}

// A random FST with `states` states and `arcs_per_state` arcs leaving each.
// `acyclic` sends every arc forward, which is what the topological benchmarks
// need; otherwise arcs go anywhere.
StdVectorFst* Build(uint64_t states, uint64_t arcs_per_state, uint64_t seed,
                    bool acyclic) {
  auto* fst = new StdVectorFst();
  for (uint64_t i = 0; i < states; ++i) fst->AddState();
  fst->SetStart(0);
  Xorshift rng(seed);
  for (uint64_t s = 0; s < states; ++s) {
    for (uint64_t k = 0; k < arcs_per_state; ++k) {
      const int32_t label = 1 + static_cast<int32_t>(rng.Next() % 64);
      const float weight = WeightOf(&rng);
      uint64_t next;
      if (acyclic) {
        const uint64_t room = states - s - 1;
        if (room == 0) continue;
        next = s + 1 + rng.Next() % room;
      } else {
        next = rng.Next() % states;
      }
      fst->AddArc(s, StdArc(label, label, TropicalWeight(weight),
                            static_cast<int>(next)));
    }
    if (s % 8 == 0) fst->SetFinal(s, TropicalWeight(WeightOf(&rng)));
  }
  // Both sides compute every property once here, so that an algorithm which
  // picks its strategy from them (AutoQueue reads kTopSorted and kAcyclic) sees
  // the same answer on each and does the same amount of work.
  fst->Properties(fst::kFstProperties, true);
  return fst;
}

// Weights are compared through a fixed number of ticks rather than their bits,
// so that a checksum says "the same answer" and not "the same float".
uint64_t Tick(const TropicalWeight& w) {
  if (w == TropicalWeight::Zero()) return 0;
  return static_cast<uint64_t>(static_cast<int64_t>(w.Value() * 4.0f + 0.5f));
}

// The checksum every result is compared through, in every implementation.
//
// Determinization, minimization and composition each number their output states
// as they please, and no two of the four libraries agree on that, nor need
// they. What they must agree on is how big the result is and what it accepts.
// Mirrors `sicada_bench::shape`; the order of the arcs leaving a state is
// deliberately not in here, since no algorithm but a sort promises anything
// about it.
uint64_t Shape(uint64_t states, uint64_t arcs, const TropicalWeight& total) {
  return (states * 1000003 + arcs) * 31 + Tick(total);
}

// Counts the arcs, and says whether they came out sorted on input labels.
uint64_t CountArcs(const StdVectorFst& fst, bool* sorted) {
  uint64_t arcs = 0;
  *sorted = true;
  for (fst::StateIterator<StdVectorFst> siter(fst); !siter.Done();
       siter.Next()) {
    int32_t previous = 0;
    for (fst::ArcIterator<StdVectorFst> aiter(fst, siter.Value()); !aiter.Done();
         aiter.Next()) {
      ++arcs;
      if (aiter.Value().ilabel < previous) *sorted = false;
      previous = aiter.Value().ilabel;
    }
  }
  return arcs;
}

uint64_t Checksum(const StdVectorFst& fst) {
  bool sorted;
  const uint64_t arcs = CountArcs(fst, &sorted);
  return Shape(fst.NumStates(), arcs, fst::ShortestDistance(fst));
}

// As `Checksum`, and whether the arcs came out sorted on input labels.
uint64_t SortedChecksum(const StdVectorFst& fst) {
  bool sorted;
  const uint64_t arcs = CountArcs(fst, &sorted);
  return Shape(fst.NumStates(), arcs, fst::ShortestDistance(fst)) * 2 +
         (sorted ? 1 : 0);
}

// A cheap value standing for "the result", so that timing cannot elide it.
uint64_t Size(const StdVectorFst& fst) {
  return static_cast<uint64_t>(fst.NumStates());
}

// An acyclic acceptor over a small alphabet, with epsilons.
//
// This is what the epsilon-removal, determinization, minimization and
// composition benchmarks run on: acyclic so that determinization always
// terminates, an acceptor so that minimization takes its own path rather than
// the gallic one, and a small alphabet so that composition finds matches.
StdVectorFst* BuildAcceptor(uint64_t states, uint64_t arcs_per_state,
                            uint64_t seed, bool epsilons) {
  auto* fst = new StdVectorFst();
  for (uint64_t i = 0; i < states; ++i) fst->AddState();
  fst->SetStart(0);
  Xorshift rng(seed);
  for (uint64_t s = 0; s < states; ++s) {
    for (uint64_t k = 0; k < arcs_per_state; ++k) {
      // One arc in eight carries no label, so epsilon removal has work to do.
      const uint64_t draw = rng.Next() % 8;
      const int32_t label =
          (epsilons && draw == 0) ? 0 : static_cast<int32_t>(1 + draw % 7);
      const float weight = WeightOf(&rng);
      const uint64_t room = states - s - 1;
      if (room == 0) continue;
      const uint64_t next = s + 1 + rng.Next() % room;
      fst->AddArc(s, StdArc(label, label, TropicalWeight(weight),
                            static_cast<int>(next)));
    }
    if (s % 8 == 0) fst->SetFinal(s, TropicalWeight(WeightOf(&rng)));
  }
  fst->Properties(fst::kFstProperties, true);
  return fst;
}

// An independent copy of `in`.
//
// `VectorFst`'s copy constructor shares the implementation and defers the
// deep copy to the first mutation, so a benchmark that copies and then runs
// an algorithm which bails out early (`TopSort` on a cyclic FST) never pays for
// the copy at all, while the Rust side's `clone` pays for it up front. The
// `SetStart` forces the copy, so the two sides do the same work. Where the
// algorithm does mutate, this costs nothing extra: the copy it would have
// triggered has already happened.
StdVectorFst DeepCopy(const StdVectorFst& in) {
  StdVectorFst out(in);
  out.SetStart(out.Start());
  return out;
}

}  // namespace

extern "C" {

void* openfst_bench_fst_new(uint64_t states, uint64_t arcs_per_state,
                            uint64_t seed, int acyclic) {
  return Build(states, arcs_per_state, seed, acyclic != 0);
}

void openfst_bench_fst_delete(void* fst) {
  delete static_cast<StdVectorFst*>(fst);
}

uint64_t openfst_bench_fst_checksum(const void* fst) {
  return Checksum(*static_cast<const StdVectorFst*>(fst));
}

uint64_t openfst_bench_shortest_distance(const void* fst, int verify) {
  const auto& in = *static_cast<const StdVectorFst*>(fst);
  std::vector<TropicalWeight> distance;
  fst::ShortestDistance(in, &distance);
  if (!verify) return distance.size();
  uint64_t acc = 0;
  for (const auto& w : distance) acc = acc * 31 + Tick(w);
  return acc * 31 + distance.size();
}

uint64_t openfst_bench_shortest_path(const void* fst, int verify) {
  const auto& in = *static_cast<const StdVectorFst*>(fst);
  StdVectorFst out;
  fst::ShortestPath(in, &out);
  return verify ? Checksum(out) : Size(out);
}

uint64_t openfst_bench_connect(const void* fst, int verify) {
  StdVectorFst out = DeepCopy(*static_cast<const StdVectorFst*>(fst));
  fst::Connect(&out);
  return verify ? Checksum(out) : Size(out);
}

uint64_t openfst_bench_arcsort(const void* fst, int verify) {
  StdVectorFst out = DeepCopy(*static_cast<const StdVectorFst*>(fst));
  fst::ArcSort(&out, fst::ILabelCompare<StdArc>());
  return verify ? SortedChecksum(out) : Size(out);
}

uint64_t openfst_bench_topsort(const void* fst, int verify) {
  StdVectorFst out = DeepCopy(*static_cast<const StdVectorFst*>(fst));
  if (!fst::TopSort(&out)) return 0;
  return verify ? Checksum(out) : Size(out);
}

void* openfst_bench_acceptor_new(uint64_t states, uint64_t arcs_per_state,
                                 uint64_t seed) {
  return BuildAcceptor(states, arcs_per_state, seed, true);
}

// The same without epsilons, which is what look-ahead composition needs of its
// second argument.
void* openfst_bench_dense_acceptor_new(uint64_t states, uint64_t arcs_per_state,
                                       uint64_t seed) {
  return BuildAcceptor(states, arcs_per_state, seed, false);
}

uint64_t openfst_bench_shape_checksum(const void* fst) {
  return Checksum(*static_cast<const StdVectorFst*>(fst));
}

uint64_t openfst_bench_rmepsilon(const void* fst, int verify) {
  StdVectorFst out = DeepCopy(*static_cast<const StdVectorFst*>(fst));
  fst::RmEpsilon(&out);
  return verify ? Checksum(out) : Size(out);
}

uint64_t openfst_bench_determinize(const void* fst, int verify) {
  const auto& in = *static_cast<const StdVectorFst*>(fst);
  StdVectorFst out;
  fst::Determinize(in, &out);
  // Determinization leaves states the result cannot finish from, and the four
  // libraries leave different ones; trimming makes the answers comparable, and
  // all four pay for it.
  fst::Connect(&out);
  return verify ? Checksum(out) : Size(out);
}

uint64_t openfst_bench_minimize(const void* fst, int verify) {
  StdVectorFst out;
  fst::Determinize(*static_cast<const StdVectorFst*>(fst), &out);
  fst::Minimize(&out);
  return verify ? Checksum(out) : Size(out);
}

uint64_t openfst_bench_compose(const void* lhs, const void* rhs, int verify) {
  StdVectorFst left = DeepCopy(*static_cast<const StdVectorFst*>(lhs));
  StdVectorFst right = DeepCopy(*static_cast<const StdVectorFst*>(rhs));
  fst::ArcSort(&left, fst::OLabelCompare<StdArc>());
  fst::ArcSort(&right, fst::ILabelCompare<StdArc>());
  StdVectorFst out;
  fst::Compose(left, right, &out);
  return verify ? Checksum(out) : Size(out);
}

}  // extern "C"
