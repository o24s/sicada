// Verbatim extraction of the composition filters from
// vendor/openfst/openfst/lib/compose-filter.h (commit 694dc53), together with
// IntegerFilterState from filter-state.h.
//
// The matcher members are dropped: SetState and FilterArc never touch them, and
// keeping them would drag in the whole matcher hierarchy. The two input FSTs
// are replaced by a stub exposing only the three quantities SetState reads.
//
// Prints, for each filter, the decision it reaches over every combination of
// the inputs that filter looks at. Build and run with:
//
//   g++ -std=c++17 -O2 -o /tmp/cfd tests/oracles/compose-filter-decisions.cc && /tmp/cfd
#include <cstddef>
#include <cstdio>
#include <cstdint>

namespace fst {

constexpr int kNoLabel = -1;
constexpr int kNoStateId = -1;

// filter-state.h, verbatim.
template <typename T>
class IntegerFilterState {
 public:
  IntegerFilterState() : state_(kNoStateId) {}
  explicit IntegerFilterState(T s) : state_(s) {}
  static const IntegerFilterState NoState() { return IntegerFilterState(); }
  size_t Hash() const { return static_cast<size_t>(state_); }
  bool operator==(const IntegerFilterState& fs) const { return state_ == fs.state_; }
  bool operator!=(const IntegerFilterState& fs) const { return state_ != fs.state_; }
  T GetState() const { return state_; }

 private:
  T state_;
};

using CharFilterState = IntegerFilterState<signed char>;

class TrivialFilterState {
 public:
  explicit TrivialFilterState(bool state = false) : state_(state) {}
  static TrivialFilterState NoState() { return TrivialFilterState(); }
  size_t Hash() const { return 0; }
  bool operator==(const TrivialFilterState& fs) const { return state_ == fs.state_; }
  bool operator!=(const TrivialFilterState& fs) const { return state_ != fs.state_; }
  bool GetState() const { return state_; }

 private:
  bool state_;
};

// Stands in for a weight: the filters only ever compare against Zero().
struct Weight {
  bool zero;
  static Weight Zero() { return Weight{true}; }
  bool operator!=(const Weight& o) const { return zero != o.zero; }
};

struct Arc {
  using StateId = int;
  using Weight = fst::Weight;
  int ilabel;
  int olabel;
};

// Stands in for an input FST: SetState reads exactly these.
struct StubFst {
  using Arc = fst::Arc;
  size_t num_arcs = 0;
  size_t num_ieps = 0;
  size_t num_oeps = 0;
  bool is_final = false;

  size_t NumArcs(int) const { return num_arcs; }
  size_t NumInputEpsilons(int) const { return num_ieps; }
  size_t NumOutputEpsilons(int) const { return num_oeps; }
  Weight Final(int) const { return Weight{!is_final}; }
};

namespace internal {
inline Weight Final(const StubFst& fst, int s) { return fst.Final(s); }
inline size_t NumArcs(const StubFst& fst, int s) { return fst.NumArcs(s); }
inline size_t NumInputEpsilons(const StubFst& fst, int s) { return fst.NumInputEpsilons(s); }
inline size_t NumOutputEpsilons(const StubFst& fst, int s) { return fst.NumOutputEpsilons(s); }
}  // namespace internal

class SequenceComposeFilter {
 public:
  using FilterState = CharFilterState;
  using StateId = int;

  SequenceComposeFilter(const StubFst& fst1, const StubFst& fst2)
      : fst1_(fst1), s1_(kNoStateId), s2_(kNoStateId), fs_(kNoStateId) {}

  FilterState Start() const { return FilterState(0); }

  void SetState(StateId s1, StateId s2, const FilterState& fs) {
    if (s1_ == s1 && s2_ == s2 && fs == fs_) return;
    s1_ = s1;
    s2_ = s2;
    fs_ = fs;
    const auto na1 = internal::NumArcs(fst1_, s1);
    const auto ne1 = internal::NumOutputEpsilons(fst1_, s1);
    const bool fin1 = internal::Final(fst1_, s1) != Weight::Zero();
    alleps1_ = na1 == ne1 && !fin1;
    noeps1_ = ne1 == 0;
  }

  FilterState FilterArc(Arc* arc1, Arc* arc2) const {
    if (arc1->olabel == kNoLabel) {
      return alleps1_  ? FilterState::NoState()
             : noeps1_ ? FilterState(0)
                       : FilterState(1);
    } else if (arc2->ilabel == kNoLabel) {
      return fs_ != FilterState(0) ? FilterState::NoState() : FilterState(0);
    } else {
      return arc1->olabel == 0 ? FilterState::NoState() : FilterState(0);
    }
  }

 private:
  const StubFst& fst1_;
  StateId s1_;
  StateId s2_;
  FilterState fs_;
  bool alleps1_;
  bool noeps1_;
};

class AltSequenceComposeFilter {
 public:
  using FilterState = CharFilterState;
  using StateId = int;

  AltSequenceComposeFilter(const StubFst& fst1, const StubFst& fst2)
      : fst2_(fst2), s1_(kNoStateId), s2_(kNoStateId), fs_(kNoStateId) {}

  FilterState Start() const { return FilterState(0); }

  void SetState(StateId s1, StateId s2, const FilterState& fs) {
    if (s1_ == s1 && s2_ == s2 && fs == fs_) return;
    s1_ = s1;
    s2_ = s2;
    fs_ = fs;
    const auto na2 = internal::NumArcs(fst2_, s2);
    const auto ne2 = internal::NumInputEpsilons(fst2_, s2);
    const bool fin2 = internal::Final(fst2_, s2) != Weight::Zero();
    alleps2_ = na2 == ne2 && !fin2;
    noeps2_ = ne2 == 0;
  }

  FilterState FilterArc(Arc* arc1, Arc* arc2) const {
    if (arc2->ilabel == kNoLabel) {
      return alleps2_  ? FilterState::NoState()
             : noeps2_ ? FilterState(0)
                       : FilterState(1);
    } else if (arc1->olabel == kNoLabel) {
      return fs_ == FilterState(1) ? FilterState::NoState() : FilterState(0);
    } else {
      return arc1->olabel == 0 ? FilterState::NoState() : FilterState(0);
    }
  }

 private:
  const StubFst& fst2_;
  StateId s1_;
  StateId s2_;
  FilterState fs_;
  bool alleps2_;
  bool noeps2_;
};

class MatchComposeFilter {
 public:
  using FilterState = CharFilterState;
  using StateId = int;

  MatchComposeFilter(const StubFst& fst1, const StubFst& fst2)
      : fst1_(fst1), fst2_(fst2), s1_(kNoStateId), s2_(kNoStateId), fs_(kNoStateId) {}

  FilterState Start() const { return FilterState(0); }

  void SetState(StateId s1, StateId s2, const FilterState& fs) {
    if (s1_ == s1 && s2_ == s2 && fs == fs_) return;
    s1_ = s1;
    s2_ = s2;
    fs_ = fs;
    size_t na1 = internal::NumArcs(fst1_, s1);
    size_t ne1 = internal::NumOutputEpsilons(fst1_, s1);
    bool f1 = internal::Final(fst1_, s1) != Weight::Zero();
    alleps1_ = na1 == ne1 && !f1;
    noeps1_ = ne1 == 0;
    size_t na2 = internal::NumArcs(fst2_, s2);
    size_t ne2 = internal::NumInputEpsilons(fst2_, s2);
    bool f2 = internal::Final(fst2_, s2) != Weight::Zero();
    alleps2_ = na2 == ne2 && !f2;
    noeps2_ = ne2 == 0;
  }

  FilterState FilterArc(Arc* arc1, Arc* arc2) const {
    if (arc2->ilabel == kNoLabel) {  // Epsilon in FST1.
      return fs_ == FilterState(0)
                 ? (noeps2_
                        ? FilterState(0)
                        : (alleps2_ ? FilterState::NoState() : FilterState(1)))
                 : (fs_ == FilterState(1) ? FilterState(1)
                                          : FilterState::NoState());
    } else if (arc1->olabel == kNoLabel) {  // Epsilon in FST2.
      return fs_ == FilterState(0)
                 ? (noeps1_
                        ? FilterState(0)
                        : (alleps1_ ? FilterState::NoState() : FilterState(2)))
                 : (fs_ == FilterState(2) ? FilterState(2)
                                          : FilterState::NoState());
    } else if (arc1->olabel == 0) {  // Epsilon in both.
      return fs_ == FilterState(0) ? FilterState(0) : FilterState::NoState();
    } else {  // Both are non-epsilons.
      return FilterState(0);
    }
  }

 private:
  const StubFst& fst1_;
  const StubFst& fst2_;
  StateId s1_;
  StateId s2_;
  FilterState fs_;
  bool alleps1_;
  bool alleps2_;
  bool noeps1_;
  bool noeps2_;
};

class NoMatchComposeFilter {
 public:
  using FilterState = TrivialFilterState;
  FilterState Start() const { return FilterState(true); }
  void SetState(int, int, const FilterState&) {}
  FilterState FilterArc(Arc* arc1, Arc* arc2) const {
    return FilterState(arc1->olabel != 0 || arc2->ilabel != 0);
  }
};

class NullComposeFilter {
 public:
  using FilterState = TrivialFilterState;
  FilterState Start() const { return FilterState(true); }
  void SetState(int, int, const FilterState&) {}
  FilterState FilterArc(Arc* arc1, Arc* arc2) const {
    return (arc1->olabel == kNoLabel || arc2->ilabel == kNoLabel)
               ? FilterState::NoState()
               : FilterState(true);
  }
};

class TrivialComposeFilter {
 public:
  using FilterState = TrivialFilterState;
  FilterState Start() const { return FilterState(true); }
  void SetState(int, int, const FilterState&) {}
  FilterState FilterArc(Arc*, Arc*) const { return FilterState(true); }
};

}  // namespace fst

using namespace fst;

// The labels a filter distinguishes: "this side did not consume a symbol",
// epsilon, and an ordinary symbol.
static const int kLabels[3] = {kNoLabel, 0, 5};

// The four (alleps, noeps) combinations, as the state shapes that produce them:
// alleps is `num_arcs == num_eps && !final`, noeps is `num_eps == 0`.
struct Shape {
  size_t num_arcs;
  size_t num_eps;
  bool is_final;
};
static const Shape kShapes[4] = {
    {0, 0, false},  // alleps, noeps
    {1, 1, false},  // alleps, not noeps
    {1, 0, false},  // not alleps, noeps
    {2, 1, false},  // neither
};

static char Decide(CharFilterState fs) {
  return fs == CharFilterState::NoState() ? '-' : ('0' + fs.GetState());
}

static char Decide(TrivialFilterState fs) {
  return fs == TrivialFilterState::NoState() ? '-' : '1';
}

int main() {
  // Sequence: reads olabel1, ilabel2, the filter state, and side 1's shape.
  printf("sequence ");
  for (const Shape& shape : kShapes) {
    StubFst fst1{shape.num_arcs, 0, shape.num_eps, shape.is_final};
    StubFst fst2;
    for (int fs = 0; fs < 3; ++fs) {
      for (int olabel1 : kLabels) {
        for (int ilabel2 : kLabels) {
          SequenceComposeFilter filter(fst1, fst2);
          filter.SetState(0, 0, CharFilterState(fs));
          Arc arc1{0, olabel1};
          Arc arc2{ilabel2, 0};
          putchar(Decide(filter.FilterArc(&arc1, &arc2)));
        }
      }
    }
  }
  putchar('\n');

  // AltSequence: the same, but side 2's shape.
  printf("altsequence ");
  for (const Shape& shape : kShapes) {
    StubFst fst1;
    StubFst fst2{shape.num_arcs, shape.num_eps, 0, shape.is_final};
    for (int fs = 0; fs < 3; ++fs) {
      for (int olabel1 : kLabels) {
        for (int ilabel2 : kLabels) {
          AltSequenceComposeFilter filter(fst1, fst2);
          filter.SetState(0, 0, CharFilterState(fs));
          Arc arc1{0, olabel1};
          Arc arc2{ilabel2, 0};
          putchar(Decide(filter.FilterArc(&arc1, &arc2)));
        }
      }
    }
  }
  putchar('\n');

  // Match: both shapes.
  printf("match ");
  for (const Shape& shape1 : kShapes) {
    for (const Shape& shape2 : kShapes) {
      StubFst fst1{shape1.num_arcs, 0, shape1.num_eps, shape1.is_final};
      StubFst fst2{shape2.num_arcs, shape2.num_eps, 0, shape2.is_final};
      for (int fs = 0; fs < 3; ++fs) {
        for (int olabel1 : kLabels) {
          for (int ilabel2 : kLabels) {
            MatchComposeFilter filter(fst1, fst2);
            filter.SetState(0, 0, CharFilterState(fs));
            Arc arc1{0, olabel1};
            Arc arc2{ilabel2, 0};
            putchar(Decide(filter.FilterArc(&arc1, &arc2)));
          }
        }
      }
    }
  }
  putchar('\n');

  // The stateless three read only the two labels.
  printf("nomatch ");
  for (int olabel1 : kLabels) {
    for (int ilabel2 : kLabels) {
      NoMatchComposeFilter filter;
      Arc arc1{0, olabel1};
      Arc arc2{ilabel2, 0};
      putchar(Decide(filter.FilterArc(&arc1, &arc2)));
    }
  }
  putchar('\n');

  printf("null ");
  for (int olabel1 : kLabels) {
    for (int ilabel2 : kLabels) {
      NullComposeFilter filter;
      Arc arc1{0, olabel1};
      Arc arc2{ilabel2, 0};
      putchar(Decide(filter.FilterArc(&arc1, &arc2)));
    }
  }
  putchar('\n');

  printf("trivial ");
  for (int olabel1 : kLabels) {
    for (int ilabel2 : kLabels) {
      TrivialComposeFilter filter;
      Arc arc1{0, olabel1};
      Arc arc2{ilabel2, 0};
      putchar(Decide(filter.FilterArc(&arc1, &arc2)));
    }
  }
  putchar('\n');
  return 0;
}
