// The bytes OpenFst's VectorFst writes after the header.
//
// Verbatim extraction of the body of `VectorFst::WriteFst` from
// vendor/openfst/openfst/lib/vector-fst.h (commit 694dc53), with the FST
// replaced by a table of states. Unlike ConstFst, this format writes each field
// in turn rather than dumping structs, so what is pinned here is the order and
// width of those fields.
//
// Build and run:
//   g++ -std=c++17 -O2 -o /tmp/vfg tests/oracles/vector-fst-golden.cc && /tmp/vfg
#include <cstdint>
#include <cstdio>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace fst {

// util.h: the two primitives everything else is written through.
template <class T>
std::ostream &WriteType(std::ostream &strm, const T value) {
  return strm.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

// float-weight.h: a weight writes its value and nothing else.
struct TropicalWeight {
  float value;
  static TropicalWeight Zero() {
    return TropicalWeight{std::numeric_limits<float>::infinity()};
  }
  static TropicalWeight One() { return TropicalWeight{0.0f}; }
  std::ostream &Write(std::ostream &strm) const { return WriteType(strm, value); }
};

struct Arc {
  int32_t ilabel;
  int32_t olabel;
  TropicalWeight weight;
  int32_t nextstate;
};

struct State {
  TropicalWeight final_weight;
  std::vector<Arc> arcs;
};

}  // namespace fst

using namespace fst;

int main() {
  // 0 -> 1 -> 2, state 2 final with weight One, arcs weighted 0.5 and 1.5.
  std::vector<State> states = {
      {TropicalWeight::Zero(), {Arc{1, 1, {0.5f}, 1}}},
      {TropicalWeight::Zero(), {Arc{2, 2, {1.5f}, 2}}},
      {TropicalWeight::One(), {}},
  };

  std::ostringstream strm;
  // The body of VectorFst::WriteFst, after the header.
  for (const auto &s : states) {
    s.final_weight.Write(strm);
    const int64_t narcs = s.arcs.size();
    WriteType(strm, narcs);
    for (const auto &arc : s.arcs) {
      WriteType(strm, arc.ilabel);
      WriteType(strm, arc.olabel);
      arc.weight.Write(strm);
      WriteType(strm, arc.nextstate);
    }
  }

  const std::string bytes = strm.str();
  printf("body ");
  for (unsigned char c : bytes) printf("%02x", c);
  printf("\nlength %zu\n", bytes.size());
  return 0;
}
