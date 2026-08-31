// The bytes OpenFst's ConstFst writes, and the layout of the two structures it
// writes them from.
//
// Verbatim extraction of `ConstFstImpl::ConstState` and of `ArcTpl` from
// vendor/openfst/openfst/lib/const-fst.h and arc.h (commit 694dc53). Both are
// written to disk with a raw `strm.write(&value, sizeof(value))`, so their size,
// alignment and field offsets are part of the file format.
//
// Build and run:
//   g++ -std=c++17 -O2 -o /tmp/cfg tests/oracles/const-fst-golden.cc && /tmp/cfg
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace fst {

// float-weight.h / arc.h, reduced to the layout.
struct TropicalWeight {
  float value;
};

template <class W, class L = int32_t, class S = int32_t>
struct ArcTpl {
  L ilabel;
  L olabel;
  W weight;
  S nextstate;
};

using StdArc = ArcTpl<TropicalWeight>;

// const-fst.h, verbatim.
template <class Weight, class Unsigned>
struct ConstState {
  Weight final_weight;
  Unsigned pos;
  Unsigned narcs;
  Unsigned niepsilons;
  Unsigned noepsilons;
};

}  // namespace fst

using namespace fst;

static void Layout(const char* name, size_t size, size_t align,
                   const std::vector<std::pair<const char*, size_t>>& fields) {
  printf("%s size=%zu align=%zu", name, size, align);
  for (const auto& [field, offset] : fields) printf(" %s=%zu", field, offset);
  putchar('\n');
}

// Writes the exact bytes ConstFst::WriteFst puts after the header, for a small
// FST: three states 0 -> 1 -> 2 with state 2 final, all arcs weight One.
static void GoldenBody() {
  using State = ConstState<TropicalWeight, uint32_t>;
  const float kZero = std::numeric_limits<float>::infinity();
  const float kOne = 0.0f;

  State states[3];
  memset(states, 0, sizeof(states));
  states[0] = State{{kZero}, 0, 1, 0, 0};
  states[1] = State{{kZero}, 1, 1, 0, 0};
  states[2] = State{{kOne}, 2, 0, 0, 0};

  StdArc arcs[2];
  memset(arcs, 0, sizeof(arcs));
  arcs[0] = StdArc{1, 1, {kOne}, 1};
  arcs[1] = StdArc{2, 2, {kOne}, 2};

  printf("states ");
  const auto* p = reinterpret_cast<const unsigned char*>(states);
  for (size_t i = 0; i < sizeof(states); ++i) printf("%02x", p[i]);
  putchar('\n');

  printf("arcs ");
  p = reinterpret_cast<const unsigned char*>(arcs);
  for (size_t i = 0; i < sizeof(arcs); ++i) printf("%02x", p[i]);
  putchar('\n');
}

int main() {
  Layout("StdArc", sizeof(StdArc), alignof(StdArc),
         {{"ilabel", offsetof(StdArc, ilabel)},
          {"olabel", offsetof(StdArc, olabel)},
          {"weight", offsetof(StdArc, weight)},
          {"nextstate", offsetof(StdArc, nextstate)}});

  using State32 = ConstState<TropicalWeight, uint32_t>;
  Layout("ConstState<Tropical,u32>", sizeof(State32), alignof(State32),
         {{"final_weight", offsetof(State32, final_weight)},
          {"pos", offsetof(State32, pos)},
          {"narcs", offsetof(State32, narcs)},
          {"niepsilons", offsetof(State32, niepsilons)},
          {"noepsilons", offsetof(State32, noepsilons)}});

  using State64 = ConstState<TropicalWeight, uint64_t>;
  Layout("ConstState<Tropical,u64>", sizeof(State64), alignof(State64),
         {{"final_weight", offsetof(State64, final_weight)},
          {"pos", offsetof(State64, pos)},
          {"narcs", offsetof(State64, narcs)},
          {"niepsilons", offsetof(State64, niepsilons)},
          {"noepsilons", offsetof(State64, noepsilons)}});

  GoldenBody();
  return 0;
}
