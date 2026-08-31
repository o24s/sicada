// Writes every fixture in `tests/fixtures/` with the real OpenFst library.
//
// The other oracles here pin bytes that sicada produces against upstream's
// serialization code extracted and compiled on its own. This one is the end of
// that chain: it links libfst.a and writes whole files, so the fixtures are
// what OpenFst actually puts on disk rather than what a transcription of it
// does. `tests/test_openfst_interop.rs` reads them back.
//
// Seventeen files, one per format sicada names. The shape of the input differs per format,
// because each compactor only accepts arcs it can rebuild: a string is a chain
// reconstructed from labels alone, an acceptor has one label per arc, and the
// unweighted forms drop the weight. Handing any of them something outside that
// shape is what upstream's compaction rejects, so the fixtures are the shapes
// themselves.
//
// Build and run (from the workspace root, with OpenFst built as the README
// describes, `B` being the build directory):
//
//   g++ -std=c++17 -O2 -I vendor/openfst -I $B/_deps/abseil-cpp-src \
//       -o /tmp/fixtures sicada/tests/oracles/interop-fixtures.cc \
//       -L $B/openfst/lib -lfst \
//       $(find $B/_deps/abseil-cpp-build/absl -name 'lib*.a' -printf '-L%h ' \
//         | tr ' ' '\n' | sort -u | tr '\n' ' ') \
//       -Wl,--start-group \
//       $(find $B/_deps/abseil-cpp-build/absl -name 'lib*.a' -printf '%f ' \
//         | sed 's/lib\([^ ]*\)\.a/-l\1/g') -Wl,--end-group -lpthread -ldl
//   /tmp/fixtures sicada/tests/fixtures
#include <cstdio>
#include <string>

#include "openfst/lib/arcsort.h"
#include "openfst/lib/compact-fst.h"
#include "openfst/lib/const-fst.h"
#include "openfst/lib/edit-fst.h"
#include "openfst/lib/matcher-fst.h"
#include "openfst/lib/symbol-table.h"
#include "openfst/lib/vector-fst.h"

using namespace fst;

// A chain of three arcs, which is what the two string compactors rebuild: the
// element carries the label and the next state is implied.
static void BuildChain(VectorFst<StdArc>* fst, bool weighted) {
  for (int i = 0; i < 4; ++i) fst->AddState();
  fst->SetStart(0);
  const float w[3] = {0.5, 1.5, 2.25};
  for (int i = 0; i < 3; ++i) {
    const TropicalWeight weight = weighted ? TropicalWeight(w[i])
                                           : TropicalWeight::One();
    fst->AddArc(i, StdArc(i + 1, i + 1, weight, i + 1));
  }
  fst->SetFinal(3, weighted ? TropicalWeight(0.75) : TropicalWeight::One());
}

// Branching, so the next state has to be stored rather than implied. `acceptor`
// puts the same label on both sides; `weighted` keeps a weight per arc.
static void BuildBranching(VectorFst<StdArc>* fst, bool acceptor, bool weighted) {
  for (int i = 0; i < 3; ++i) fst->AddState();
  fst->SetStart(0);
  const auto weight = [weighted](float x) {
    return weighted ? TropicalWeight(x) : TropicalWeight::One();
  };
  fst->AddArc(0, StdArc(1, acceptor ? 1 : 10, weight(0.5), 1));
  fst->AddArc(0, StdArc(2, acceptor ? 2 : 20, weight(1.5), 2));
  fst->AddArc(1, StdArc(3, acceptor ? 3 : 30, weight(2.25), 2));
  fst->SetFinal(2, weight(0.75));
}

// Three states, two of them with arcs, one final weight, and a symbol table on
// each side. Small enough to assert exhaustively, wide enough to cover the
// fields the format has: labels that differ between the sides, a weight that is
// not One, a non-final state written as Zero, and sparse symbol ids. The vector,
// const, edit and matcher fixtures are all this FST.
static void BuildBase(VectorFst<StdArc>* fst, SymbolTable* isyms,
                      SymbolTable* osyms) {
  BuildBranching(fst, /*acceptor=*/false, /*weighted=*/true);
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

  // Each group builds its own FST. Writing one is not free of effect on it:
  // converting to a const FST computes properties the source then caches, and a
  // later fixture built from the same object would carry the extra bits. A
  // fixture must not depend on what was written before it.
  SymbolTable isyms("in"), osyms("out");

  // The two formats that lay the arcs out one after another.
  {
    VectorFst<StdArc> base;
    BuildBase(&base, &isyms, &osyms);
    if (!base.Write(dir + "/openfst-vector.fst")) return 1;
    if (!ConstFst<StdArc>(base).Write(dir + "/openfst-const.fst")) return 1;
    if (!ConstFst<StdArc, uint64_t>(base).Write(dir + "/openfst-const64.fst"))
      return 1;
  }

  // The five compact formats, each over the shape its compactor accepts.
  VectorFst<StdArc> chain, wchain, acceptor, uacceptor, unweighted;
  BuildChain(&chain, /*weighted=*/false);
  BuildChain(&wchain, /*weighted=*/true);
  BuildBranching(&acceptor, /*acceptor=*/true, /*weighted=*/true);
  BuildBranching(&uacceptor, /*acceptor=*/true, /*weighted=*/false);
  BuildBranching(&unweighted, /*acceptor=*/false, /*weighted=*/false);

  // Each compact format twice: the offsets into the arc store are what the
  // unsigned parameter sizes, and upstream gives the wider one its own name.
  if (!CompactStringFst<StdArc, uint32_t>(chain)
           .Write(dir + "/openfst-compact-string.fst"))
    return 1;
  if (!CompactStringFst<StdArc, uint64_t>(chain)
           .Write(dir + "/openfst-compact64-string.fst"))
    return 1;
  if (!CompactWeightedStringFst<StdArc, uint32_t>(wchain)
           .Write(dir + "/openfst-compact-weighted-string.fst"))
    return 1;
  if (!CompactWeightedStringFst<StdArc, uint64_t>(wchain)
           .Write(dir + "/openfst-compact64-weighted-string.fst"))
    return 1;
  if (!CompactAcceptorFst<StdArc, uint32_t>(acceptor)
           .Write(dir + "/openfst-compact-acceptor.fst"))
    return 1;
  if (!CompactAcceptorFst<StdArc, uint64_t>(acceptor)
           .Write(dir + "/openfst-compact64-acceptor.fst"))
    return 1;
  if (!CompactUnweightedAcceptorFst<StdArc, uint32_t>(uacceptor)
           .Write(dir + "/openfst-compact-unweighted-acceptor.fst"))
    return 1;
  if (!CompactUnweightedAcceptorFst<StdArc, uint64_t>(uacceptor)
           .Write(dir + "/openfst-compact64-unweighted-acceptor.fst"))
    return 1;
  if (!CompactUnweightedFst<StdArc, uint32_t>(unweighted)
           .Write(dir + "/openfst-compact-unweighted.fst"))
    return 1;
  if (!CompactUnweightedFst<StdArc, uint64_t>(unweighted)
           .Write(dir + "/openfst-compact64-unweighted.fst"))
    return 1;

  // An edit FST: a base FST plus edits, which the format keeps apart. The final
  // weight of state 1 and one new arc are what an edited file has to carry.
  VectorFst<StdArc> base;
  BuildBase(&base, &isyms, &osyms);
  EditFst<StdArc> edited(base);
  edited.SetFinal(1, TropicalWeight(3.5));
  edited.AddArc(2, StdArc(2, 20, TropicalWeight(4.0), 0));
  if (!edited.Write(dir + "/openfst-edit.fst")) return 1;

  // Two matcher FSTs: a base FST plus the look-ahead add-on the format stores
  // beside it. Both indexes are over input labels, so the arcs are sorted that
  // way.
  //
  // The arc form keeps the labels as they are. The label form renumbers them to
  // its index and, because `ilabel_lookahead_flags` leaves
  // `kLookAheadKeepRelabelData` off, writes no map back to the originals; the
  // second fixture is there to be refused rather than read.
  VectorFst<StdArc> matcher_base;
  BuildBase(&matcher_base, &isyms, &osyms);
  VectorFst<StdArc> sorted(matcher_base);
  ArcSort(&sorted, ILabelCompare<StdArc>());
  if (!StdArcLookAheadFst(sorted).Write(dir + "/openfst-matcher-arc.fst"))
    return 1;
  if (!StdILabelLookAheadFst(sorted).Write(dir + "/openfst-matcher-ilabel.fst"))
    return 1;
  VectorFst<StdArc> osorted(matcher_base);
  ArcSort(&osorted, OLabelCompare<StdArc>());
  if (!StdOLabelLookAheadFst(osorted).Write(dir + "/openfst-matcher-olabel.fst"))
    return 1;

  printf("wrote 17 fixtures into %s\n", dir.c_str());
  return 0;
}
