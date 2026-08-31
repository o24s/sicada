// Writes the fixtures in `tests/fixtures/` with the real OpenFst library.
//
// The other oracles here pin bytes that sicada produces against upstream's
// serialization code extracted and compiled on its own. This one is the end of
// that chain: it links libfst.a and writes whole files, so the fixtures are
// what OpenFst actually puts on disk rather than what a transcription of it
// does. `tests/test_openfst_interop.rs` reads them back.
//
// Build and run (from the workspace root, with OpenFst built as the README
// describes):
//
//   g++ -std=c++17 -O2 -I vendor/openfst -I $B/_deps/abseil-cpp-src \
//       -o /tmp/fixtures sicada/tests/oracles/interop-fixtures.cc \
//       -L $B/openfst/lib -lfst \
//       $(find $B/_deps/abseil-cpp-build/absl -name 'lib*.a' -printf '-L%h ' \
//         | tr ' ' '\n' | sort -u | tr '\n' ' ') \
//       $(for i in 1 2 3; do find $B/_deps/abseil-cpp-build/absl -name 'lib*.a' \
//         -printf '%f ' | sed 's/lib\([^ ]*\)\.a/-l\1/g'; done) -lpthread -ldl
//   /tmp/fixtures sicada/tests/fixtures
#include <cstdio>
#include <string>

#include "openfst/lib/const-fst.h"
#include "openfst/lib/symbol-table.h"
#include "openfst/lib/vector-fst.h"

using namespace fst;

// Three states, two of them with arcs, one final weight, and a symbol table on
// each side. Small enough to assert exhaustively, wide enough to cover the
// fields the format has: labels that differ between the sides, a weight that is
// not One, a non-final state written as Zero, and sparse symbol ids.
static void Build(VectorFst<StdArc>* fst, SymbolTable* isyms, SymbolTable* osyms) {
  for (int i = 0; i < 3; ++i) fst->AddState();
  fst->SetStart(0);
  fst->AddArc(0, StdArc(1, 10, TropicalWeight(0.5), 1));
  fst->AddArc(0, StdArc(2, 20, TropicalWeight(1.5), 2));
  fst->AddArc(1, StdArc(3, 30, TropicalWeight(2.25), 2));
  fst->SetFinal(2, TropicalWeight(0.75));
  isyms->AddSymbol("<eps>", 0);
  isyms->AddSymbol("a", 1);
  isyms->AddSymbol("b", 2);
  isyms->AddSymbol("c", 3);
  osyms->AddSymbol("<eps>", 0);
  osyms->AddSymbol("X", 10);
  osyms->AddSymbol("Y", 20);
  osyms->AddSymbol("Z", 30);
  fst->SetInputSymbols(isyms);
  fst->SetOutputSymbols(osyms);
}

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : ".";
  VectorFst<StdArc> fst;
  SymbolTable isyms("in"), osyms("out");
  Build(&fst, &isyms, &osyms);

  if (!fst.Write(dir + "/openfst-vector.fst")) return 1;
  const ConstFst<StdArc> cfst(fst);
  if (!cfst.Write(dir + "/openfst-const.fst")) return 1;
  printf("wrote openfst-vector.fst and openfst-const.fst into %s\n", dir.c_str());
  return 0;
}
