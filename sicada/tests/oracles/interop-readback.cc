// Reads a file sicada wrote, with the real OpenFst library.
//
// The fixtures go the other way: OpenFst writes, sicada reads. This closes the
// loop for the one format where the two do not produce identical bytes. sicada
// fills the outer header of a matcher FST with the wrapped FST's start state
// and counts, where upstream leaves the placeholders it wrote them with; the
// values are never read back, because the nested FST carries its own header,
// and `test_openfst_interop_more.rs` pins exactly those three fields. This
// program is how that "never read back" was established rather than assumed.
//
// Build and run (from the workspace root, `B` being the OpenFst build):
//
//   g++ -std=c++17 -O2 -I vendor/openfst -I $B/_deps/abseil-cpp-src \
//       -o /tmp/readback sicada/tests/oracles/interop-readback.cc \
//       -L $B/openfst/lib -lfst \
//       $(find $B/_deps/abseil-cpp-build/absl -name 'lib*.a' -printf '-L%h ' \
//         | tr ' ' '\n' | sort -u | tr '\n' ' ') \
//       -Wl,--start-group \
//       $(find $B/_deps/abseil-cpp-build/absl -name 'lib*.a' -printf '%f ' \
//         | sed 's/lib\([^ ]*\)\.a/-l\1/g') -Wl,--end-group -lpthread -ldl
//   /tmp/readback <file written by sicada>
#include <cstdio>
#include <memory>

#include "openfst/lib/matcher-fst.h"

using namespace fst;

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage: readback <arc_lookahead fst>\n");
    return 2;
  }
  const std::unique_ptr<StdArcLookAheadFst> fst(StdArcLookAheadFst::Read(argv[1]));
  if (!fst) {
    printf("READ FAILED\n");
    return 1;
  }
  printf("start=%d\n", fst->Start());
  for (StateIterator<StdArcLookAheadFst> s(*fst); !s.Done(); s.Next()) {
    for (ArcIterator<StdArcLookAheadFst> a(*fst, s.Value()); !a.Done(); a.Next()) {
      printf("  %d: %d/%d/%g -> %d\n", s.Value(), a.Value().ilabel,
             a.Value().olabel, a.Value().weight.Value(), a.Value().nextstate);
    }
    printf("  final(%d)=%g\n", s.Value(), fst->Final(s.Value()).Value());
  }
  printf("isyms=%s\n",
         fst->InputSymbols() ? fst->InputSymbols()->Name().c_str() : "(none)");
  return 0;
}
