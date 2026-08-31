// Builds the HLG in `tests/fixtures/openfst-hlg.fst` with the real OpenFst.
//
// H o L o G is the decoding graph a monophone or CTC recogniser uses: the
// context-dependency transducer C is not in it. The lexicon is small but has
// the two shapes that make the pipeline non-trivial. "car" is a prefix of
// "cart", and "two" and "to" are the same phones, so both need disambiguation
// symbols before L o G can be determinized.
//
// `tests/test_hlg.rs` builds the same thing with sicada and checks the result
// is isomorphic to this file.
//
// Build and run (from the workspace root, with OpenFst built as the README
// describes; the link group lets the abseil archives resolve each other):
//
//   g++ -std=c++17 -O2 -I vendor/openfst -I $B/_deps/abseil-cpp-src \
//       -o /tmp/hlg sicada/tests/oracles/hlg-reference.cc \
//       -Wl,--start-group $B/openfst/lib/libfst.a \
//       $(find $B/_deps/abseil-cpp-build/absl -name 'lib*.a' | tr '\n' ' ') \
//       -Wl,--end-group -lpthread -ldl
//   /tmp/hlg sicada/tests/fixtures/openfst-hlg.fst
//
// Phones: eps=0 k=1 a=2 t=3 r=4 u=6   disambig #1=5 #2=7 #3=8
// Words:  eps=0 cat=1 car=2 cart=3 two=4 to=5
// Transition ids: phone p emits 10p+1 then 10p+2.
#include <cstdio>
#include "openfst/lib/vector-fst.h"
#include "openfst/lib/compose.h"
#include "openfst/lib/determinize.h"
#include "openfst/lib/minimize.h"
#include "openfst/lib/rmepsilon.h"
#include "openfst/lib/arcsort.h"
#include "openfst/lib/connect.h"
using namespace fst;
using Fst_ = VectorFst<StdArc>;
using W = TropicalWeight;

static Fst_ grammar() {
  Fst_ g; g.AddState(); g.SetStart(0); g.SetFinal(0, W::One());
  g.AddArc(0, StdArc(1, 1, W(1.0), 0));   // cat
  g.AddArc(0, StdArc(2, 2, W(2.0), 0));   // car
  g.AddArc(0, StdArc(3, 3, W(1.5), 0));   // cart
  g.AddArc(0, StdArc(4, 4, W(0.5), 0));   // two
  g.AddArc(0, StdArc(5, 5, W(0.25), 0));  // to
  return g;
}
static Fst_ lexicon() {
  Fst_ l; for (int i = 0; i < 15; ++i) l.AddState();
  l.SetStart(0); l.SetFinal(0, W::One());
  // cat = k a t
  l.AddArc(0, StdArc(1, 1, W::One(), 1));
  l.AddArc(1, StdArc(2, 0, W::One(), 2));
  l.AddArc(2, StdArc(3, 0, W::One(), 0));
  // car = k a r #1   (#1 because "car" is a prefix of "cart")
  l.AddArc(0, StdArc(1, 2, W::One(), 3));
  l.AddArc(3, StdArc(2, 0, W::One(), 4));
  l.AddArc(4, StdArc(4, 0, W::One(), 5));
  l.AddArc(5, StdArc(5, 0, W::One(), 0));
  // cart = k a r t
  l.AddArc(0, StdArc(1, 3, W::One(), 6));
  l.AddArc(6, StdArc(2, 0, W::One(), 7));
  l.AddArc(7, StdArc(4, 0, W::One(), 8));
  l.AddArc(8, StdArc(3, 0, W::One(), 0));
  // two = t u #2 and to = t u #3: the same phones, so they need disambiguation
  l.AddArc(0, StdArc(3, 4, W::One(), 9));
  l.AddArc(9, StdArc(6, 0, W::One(), 10));
  l.AddArc(10, StdArc(7, 0, W::One(), 0));
  l.AddArc(0, StdArc(3, 5, W::One(), 12));
  l.AddArc(12, StdArc(6, 0, W::One(), 13));
  l.AddArc(13, StdArc(8, 0, W::One(), 0));
  return l;
}
static Fst_ hmm() {
  Fst_ h; h.AddState(); h.SetStart(0); h.SetFinal(0, W::One());
  for (int p : {1, 2, 3, 4, 6}) {
    const int s = h.AddState();
    h.AddArc(0, StdArc(10 * p + 1, p, W::One(), s));
    h.AddArc(s, StdArc(10 * p + 2, 0, W::One(), 0));
  }
  for (int d : {5, 7, 8}) h.AddArc(0, StdArc(d, d, W::One(), 0));  // disambig passes through
  return h;
}
static void show(const char* name, const Fst_& f) {
  int arcs = 0;
  for (StateIterator<Fst_> s(f); !s.Done(); s.Next()) arcs += f.NumArcs(s.Value());
  printf("%-6s %d states %d arcs\n", name, f.NumStates(), arcs);
}
int main(int argc, char** argv) {
  Fst_ g = grammar(), l = lexicon(), h = hmm();
  show("G", g); show("L", l); show("H", h);

  ArcSort(&l, OLabelCompare<StdArc>());
  Fst_ lg; Compose(l, g, &lg);            show("L.G", lg);
  RmEpsilon(&lg);                          show("rmeps", lg);
  Fst_ dlg; Determinize(lg, &dlg);         show("det", dlg);
  ArcSort(&h, OLabelCompare<StdArc>());
  ArcSort(&dlg, ILabelCompare<StdArc>());
  Fst_ hlg; Compose(h, dlg, &hlg);         show("H.LG", hlg);
  RmEpsilon(&hlg);
  Fst_ dhlg; Determinize(hlg, &dhlg);      show("det", dhlg);
  Minimize(&dhlg);                         show("min", dhlg);
  Connect(&dhlg);                          show("HLG", dhlg);
  if (argc > 1 && !dhlg.Write(argv[1])) return 1;
  return 0;
}
