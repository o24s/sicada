// Verbatim extraction of the property propagation functions from
// vendor/openfst/openfst/lib/properties.{h,cc} (commit e77e51d), with
// absl::Span replaced by a minimal stand-in.
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace fst {
struct Span {
  const uint64_t* p; size_t n;
  Span(const std::vector<uint64_t>& v) : p(v.data()), n(v.size()) {}
  bool empty() const { return n == 0; }
  size_t size() const { return n; }
  const uint64_t* begin() const { return p; }
  const uint64_t* end() const { return p + n; }
  const uint64_t& operator[](size_t i) const { return p[i]; }
};
inline constexpr uint64_t kExpanded = 0x0000000000000001ULL;
inline constexpr uint64_t kMutable = 0x0000000000000002ULL;
inline constexpr uint64_t kError = 0x0000000000000004ULL;
inline constexpr uint64_t kAcceptor = 0x0000000000010000ULL;
inline constexpr uint64_t kNotAcceptor = 0x0000000000020000ULL;
inline constexpr uint64_t kIDeterministic = 0x0000000000040000ULL;
inline constexpr uint64_t kNonIDeterministic = 0x0000000000080000ULL;
inline constexpr uint64_t kODeterministic = 0x0000000000100000ULL;
inline constexpr uint64_t kNonODeterministic = 0x0000000000200000ULL;
inline constexpr uint64_t kEpsilons = 0x0000000000400000ULL;
inline constexpr uint64_t kNoEpsilons = 0x0000000000800000ULL;
inline constexpr uint64_t kIEpsilons = 0x0000000001000000ULL;
inline constexpr uint64_t kNoIEpsilons = 0x0000000002000000ULL;
inline constexpr uint64_t kOEpsilons = 0x0000000004000000ULL;
inline constexpr uint64_t kNoOEpsilons = 0x0000000008000000ULL;
inline constexpr uint64_t kILabelSorted = 0x0000000010000000ULL;
inline constexpr uint64_t kNotILabelSorted = 0x0000000020000000ULL;
inline constexpr uint64_t kOLabelSorted = 0x0000000040000000ULL;
inline constexpr uint64_t kNotOLabelSorted = 0x0000000080000000ULL;
inline constexpr uint64_t kWeighted = 0x0000000100000000ULL;
inline constexpr uint64_t kUnweighted = 0x0000000200000000ULL;
inline constexpr uint64_t kCyclic = 0x0000000400000000ULL;
inline constexpr uint64_t kAcyclic = 0x0000000800000000ULL;
inline constexpr uint64_t kInitialCyclic = 0x0000001000000000ULL;
inline constexpr uint64_t kInitialAcyclic = 0x0000002000000000ULL;
inline constexpr uint64_t kTopSorted = 0x0000004000000000ULL;
inline constexpr uint64_t kNotTopSorted = 0x0000008000000000ULL;
inline constexpr uint64_t kAccessible = 0x0000010000000000ULL;
inline constexpr uint64_t kNotAccessible = 0x0000020000000000ULL;
inline constexpr uint64_t kCoAccessible = 0x0000040000000000ULL;
inline constexpr uint64_t kNotCoAccessible = 0x0000080000000000ULL;
inline constexpr uint64_t kString = 0x0000100000000000ULL;
inline constexpr uint64_t kNotString = 0x0000200000000000ULL;
inline constexpr uint64_t kWeightedCycles = 0x0000400000000000ULL;
inline constexpr uint64_t kUnweightedCycles = 0x0000800000000000ULL;
inline constexpr uint64_t kNullProperties =
    kAcceptor | kIDeterministic | kODeterministic | kNoEpsilons | kNoIEpsilons |
    kNoOEpsilons | kILabelSorted | kOLabelSorted | kUnweighted | kAcyclic |
    kInitialAcyclic | kTopSorted | kAccessible | kCoAccessible | kString |
    kUnweightedCycles;
inline constexpr uint64_t kCompiledStringProperties =
    kAcceptor | kString | kUnweighted | kIDeterministic | kODeterministic |
    kILabelSorted | kOLabelSorted | kAcyclic | kInitialAcyclic |
    kUnweightedCycles | kTopSorted | kAccessible | kCoAccessible;
inline constexpr uint64_t kCopyProperties =
    kError | kAcceptor | kNotAcceptor | kIDeterministic | kNonIDeterministic |
    kODeterministic | kNonODeterministic | kEpsilons | kNoEpsilons |
    kIEpsilons | kNoIEpsilons | kOEpsilons | kNoOEpsilons | kILabelSorted |
    kNotILabelSorted | kOLabelSorted | kNotOLabelSorted | kWeighted |
    kUnweighted | kCyclic | kAcyclic | kInitialCyclic | kInitialAcyclic |
    kTopSorted | kNotTopSorted | kAccessible | kNotAccessible | kCoAccessible |
    kNotCoAccessible | kString | kNotString | kWeightedCycles |
    kUnweightedCycles;
inline constexpr uint64_t kIntrinsicProperties =
    kExpanded | kMutable | kAcceptor | kNotAcceptor | kIDeterministic |
    kNonIDeterministic | kODeterministic | kNonODeterministic | kEpsilons |
    kNoEpsilons | kIEpsilons | kNoIEpsilons | kOEpsilons | kNoOEpsilons |
    kILabelSorted | kNotILabelSorted | kOLabelSorted | kNotOLabelSorted |
    kWeighted | kUnweighted | kCyclic | kAcyclic | kInitialCyclic |
    kInitialAcyclic | kTopSorted | kNotTopSorted | kAccessible |
    kNotAccessible | kCoAccessible | kNotCoAccessible | kString | kNotString |
    kWeightedCycles | kUnweightedCycles;
inline constexpr uint64_t kExtrinsicProperties = kError;
inline constexpr uint64_t kSetStartProperties =
    kExpanded | kMutable | kError | kAcceptor | kNotAcceptor | kIDeterministic |
    kNonIDeterministic | kODeterministic | kNonODeterministic | kEpsilons |
    kNoEpsilons | kIEpsilons | kNoIEpsilons | kOEpsilons | kNoOEpsilons |
    kILabelSorted | kNotILabelSorted | kOLabelSorted | kNotOLabelSorted |
    kWeighted | kUnweighted | kCyclic | kAcyclic | kTopSorted | kNotTopSorted |
    kCoAccessible | kNotCoAccessible | kWeightedCycles | kUnweightedCycles;
inline constexpr uint64_t kSetFinalProperties =
    kExpanded | kMutable | kError | kAcceptor | kNotAcceptor | kIDeterministic |
    kNonIDeterministic | kODeterministic | kNonODeterministic | kEpsilons |
    kNoEpsilons | kIEpsilons | kNoIEpsilons | kOEpsilons | kNoOEpsilons |
    kILabelSorted | kNotILabelSorted | kOLabelSorted | kNotOLabelSorted |
    kCyclic | kAcyclic | kInitialCyclic | kInitialAcyclic | kTopSorted |
    kNotTopSorted | kAccessible | kNotAccessible | kWeightedCycles |
    kUnweightedCycles;
inline constexpr uint64_t kAddStateProperties =
    kExpanded | kMutable | kError | kAcceptor | kNotAcceptor | kIDeterministic |
    kNonIDeterministic | kODeterministic | kNonODeterministic | kEpsilons |
    kNoEpsilons | kIEpsilons | kNoIEpsilons | kOEpsilons | kNoOEpsilons |
    kILabelSorted | kNotILabelSorted | kOLabelSorted | kNotOLabelSorted |
    kWeighted | kUnweighted | kCyclic | kAcyclic | kInitialCyclic |
    kInitialAcyclic | kTopSorted | kNotTopSorted | kNotAccessible |
    kNotCoAccessible | kNotString | kWeightedCycles | kUnweightedCycles;
inline constexpr uint64_t kAddArcProperties =
    kExpanded | kMutable | kError | kNotAcceptor | kNonIDeterministic |
    kNonODeterministic | kEpsilons | kIEpsilons | kOEpsilons |
    kNotILabelSorted | kNotOLabelSorted | kWeighted | kCyclic | kInitialCyclic |
    kNotTopSorted | kAccessible | kCoAccessible | kWeightedCycles;
inline constexpr uint64_t kSetArcProperties = kExpanded | kMutable | kError;
inline constexpr uint64_t kDeleteStatesProperties =
    kExpanded | kMutable | kError | kAcceptor | kIDeterministic |
    kODeterministic | kNoEpsilons | kNoIEpsilons | kNoOEpsilons |
    kILabelSorted | kOLabelSorted | kUnweighted | kAcyclic | kInitialAcyclic |
    kTopSorted | kUnweightedCycles;
inline constexpr uint64_t kDeleteArcsProperties =
    kExpanded | kMutable | kError | kAcceptor | kIDeterministic |
    kODeterministic | kNoEpsilons | kNoIEpsilons | kNoOEpsilons |
    kILabelSorted | kOLabelSorted | kUnweighted | kAcyclic | kInitialAcyclic |
    kTopSorted | kNotAccessible | kNotCoAccessible | kUnweightedCycles;
inline constexpr uint64_t kStateSortProperties =
    kExpanded | kMutable | kError | kAcceptor | kNotAcceptor | kIDeterministic |
    kNonIDeterministic | kODeterministic | kNonODeterministic | kEpsilons |
    kNoEpsilons | kIEpsilons | kNoIEpsilons | kOEpsilons | kNoOEpsilons |
    kILabelSorted | kNotILabelSorted | kOLabelSorted | kNotOLabelSorted |
    kWeighted | kUnweighted | kCyclic | kAcyclic | kInitialCyclic |
    kInitialAcyclic | kAccessible | kNotAccessible | kCoAccessible |
    kNotCoAccessible | kWeightedCycles | kUnweightedCycles;
inline constexpr uint64_t kArcSortProperties =
    kExpanded | kMutable | kError | kAcceptor | kNotAcceptor | kIDeterministic |
    kNonIDeterministic | kODeterministic | kNonODeterministic | kEpsilons |
    kNoEpsilons | kIEpsilons | kNoIEpsilons | kOEpsilons | kNoOEpsilons |
    kWeighted | kUnweighted | kCyclic | kAcyclic | kInitialCyclic |
    kInitialAcyclic | kTopSorted | kNotTopSorted | kAccessible |
    kNotAccessible | kCoAccessible | kNotCoAccessible | kString | kNotString |
    kWeightedCycles | kUnweightedCycles;
inline constexpr uint64_t kILabelInvariantProperties =
    kExpanded | kMutable | kError | kODeterministic | kNonODeterministic |
    kOEpsilons | kNoOEpsilons | kOLabelSorted | kNotOLabelSorted | kWeighted |
    kUnweighted | kCyclic | kAcyclic | kInitialCyclic | kInitialAcyclic |
    kTopSorted | kNotTopSorted | kAccessible | kNotAccessible | kCoAccessible |
    kNotCoAccessible | kString | kNotString | kWeightedCycles |
    kUnweightedCycles;
inline constexpr uint64_t kOLabelInvariantProperties =
    kExpanded | kMutable | kError | kIDeterministic | kNonIDeterministic |
    kIEpsilons | kNoIEpsilons | kILabelSorted | kNotILabelSorted | kWeighted |
    kUnweighted | kCyclic | kAcyclic | kInitialCyclic | kInitialAcyclic |
    kTopSorted | kNotTopSorted | kAccessible | kNotAccessible | kCoAccessible |
    kNotCoAccessible | kString | kNotString | kWeightedCycles |
    kUnweightedCycles;
inline constexpr uint64_t kWeightInvariantProperties =
    kExpanded | kMutable | kError | kAcceptor | kNotAcceptor | kIDeterministic |
    kNonIDeterministic | kODeterministic | kNonODeterministic | kEpsilons |
    kNoEpsilons | kIEpsilons | kNoIEpsilons | kOEpsilons | kNoOEpsilons |
    kILabelSorted | kNotILabelSorted | kOLabelSorted | kNotOLabelSorted |
    kCyclic | kAcyclic | kInitialCyclic | kInitialAcyclic | kTopSorted |
    kNotTopSorted | kAccessible | kNotAccessible | kCoAccessible |
    kNotCoAccessible | kString | kNotString;
inline constexpr uint64_t kAddSuperFinalProperties =
    kExpanded | kMutable | kError | kAcceptor | kNotAcceptor |
    kNonIDeterministic | kNonODeterministic | kEpsilons | kIEpsilons |
    kOEpsilons | kNotILabelSorted | kNotOLabelSorted | kWeighted | kUnweighted |
    kCyclic | kAcyclic | kInitialCyclic | kInitialAcyclic | kNotTopSorted |
    kNotAccessible | kCoAccessible | kNotCoAccessible | kNotString |
    kWeightedCycles | kUnweightedCycles;
inline constexpr uint64_t kRmSuperFinalProperties =
    kExpanded | kMutable | kError | kAcceptor | kNotAcceptor | kIDeterministic |
    kODeterministic | kNoEpsilons | kNoIEpsilons | kNoOEpsilons |
    kILabelSorted | kOLabelSorted | kWeighted | kUnweighted | kCyclic |
    kAcyclic | kInitialCyclic | kInitialAcyclic | kTopSorted | kAccessible |
    kCoAccessible | kNotCoAccessible | kString | kWeightedCycles |
    kUnweightedCycles;
inline constexpr uint64_t kBinaryProperties = 0x0000000000000007ULL;
inline constexpr uint64_t kTrinaryProperties = 0x0000ffffffff0000ULL;
inline constexpr uint64_t kPosTrinaryProperties =
    kTrinaryProperties & 0x5555555555555555ULL;
inline constexpr uint64_t kNegTrinaryProperties =
    kTrinaryProperties & 0xaaaaaaaaaaaaaaaaULL;
inline constexpr uint64_t kFstProperties =
    kBinaryProperties | kTrinaryProperties;
// Properties for a concatenatively-closed FST.
uint64_t ClosureProperties(uint64_t inprops, bool, bool delayed) {
  auto outprops = (kError | kAcceptor | kUnweighted | kAccessible) & inprops;
  if (inprops & kUnweighted) outprops |= kUnweightedCycles;
  if (!delayed) {
    outprops |=
        (kExpanded | kMutable | kCoAccessible | kNotTopSorted | kNotString) &
        inprops;
  }
  if (!delayed || inprops & kAccessible) {
    outprops |= (kNotAcceptor | kNonIDeterministic | kNonODeterministic |
                 kNotILabelSorted | kNotOLabelSorted | kWeighted |
                 kWeightedCycles | kNotAccessible | kNotCoAccessible) &
                inprops;
    if ((inprops & kWeighted) && (inprops & kAccessible) &&
        (inprops & kCoAccessible)) {
      outprops |= kWeightedCycles;
    }
  }
  return outprops;
}

// Properties for a complemented FST.
uint64_t ComplementProperties(uint64_t inprops) {
  auto outprops = kAcceptor | kUnweighted | kUnweightedCycles | kNoEpsilons |
                  kNoIEpsilons | kNoOEpsilons | kIDeterministic |
                  kODeterministic | kAccessible;
  outprops |=
      (kError | kILabelSorted | kOLabelSorted | kInitialCyclic) & inprops;
  if (inprops & kAccessible) {
    outprops |= kNotILabelSorted | kNotOLabelSorted | kCyclic;
  }
  return outprops;
}

// Properties for a composed FST.
uint64_t ComposeProperties(uint64_t inprops1, uint64_t inprops2) {
  auto outprops = kError & (inprops1 | inprops2);
  if (inprops1 & kAcceptor && inprops2 & kAcceptor) {
    outprops |= kAcceptor | kAccessible;
    outprops |= (kNoEpsilons | kNoIEpsilons | kNoOEpsilons | kAcyclic |
                 kInitialAcyclic) &
                inprops1 & inprops2;
    if (kNoIEpsilons & inprops1 & inprops2) {
      outprops |= (kIDeterministic | kODeterministic) & inprops1 & inprops2;
    }
  } else {
    outprops |= kAccessible;
    outprops |= (kAcceptor | kNoIEpsilons | kAcyclic | kInitialAcyclic) &
                inprops1 & inprops2;
    if (kNoIEpsilons & inprops1 & inprops2) {
      outprops |= kIDeterministic & inprops1 & inprops2;
    }
  }
  return outprops;
}

// Properties for a concatenated FST.
uint64_t ConcatProperties(uint64_t inprops1, uint64_t inprops2, bool delayed) {
  auto outprops = (kAcceptor | kUnweighted | kUnweightedCycles | kAcyclic) &
                  inprops1 & inprops2;
  outprops |= kError & (inprops1 | inprops2);
  const bool empty1 = delayed;  // Can the first FST be the empty machine?
  const bool empty2 = delayed;  // Can the second FST be the empty machine?
  if (!delayed) {
    outprops |= (kExpanded | kMutable | kNotTopSorted | kNotString) & inprops1;
    outprops |= (kNotTopSorted | kNotString) & inprops2;
  }
  if (!empty1) outprops |= (kInitialAcyclic | kInitialCyclic) & inprops1;
  if (!delayed || inprops1 & kAccessible) {
    outprops |= (kNotAcceptor | kNonIDeterministic | kNonODeterministic |
                 kEpsilons | kIEpsilons | kOEpsilons | kNotILabelSorted |
                 kNotOLabelSorted | kWeighted | kWeightedCycles | kCyclic |
                 kNotAccessible | kNotCoAccessible) &
                inprops1;
  }
  if ((inprops1 & (kAccessible | kCoAccessible)) ==
          (kAccessible | kCoAccessible) &&
      !empty1) {
    outprops |= kAccessible & inprops2;
    if (!empty2) outprops |= kCoAccessible & inprops2;
    if (!delayed || inprops2 & kAccessible) {
      outprops |= (kNotAcceptor | kNonIDeterministic | kNonODeterministic |
                   kEpsilons | kIEpsilons | kOEpsilons | kNotILabelSorted |
                   kNotOLabelSorted | kWeighted | kWeightedCycles | kCyclic |
                   kNotAccessible | kNotCoAccessible) &
                  inprops2;
    }
  }
  return outprops;
}

// Properties for a determinized FST.
uint64_t DeterminizeProperties(uint64_t inprops, bool has_subsequential_label,
                               bool distinct_psubsequential_labels) {
  auto outprops = kAccessible;
  if ((kAcceptor & inprops) ||
      ((kNoIEpsilons & inprops) && distinct_psubsequential_labels) ||
      (has_subsequential_label && distinct_psubsequential_labels)) {
    outprops |= kIDeterministic;
  }
  outprops |= (kError | kAcceptor | kAcyclic | kInitialAcyclic | kCoAccessible |
               kString) &
              inprops;
  if ((inprops & kNoIEpsilons) && distinct_psubsequential_labels) {
    outprops |= kNoEpsilons & inprops;
  }
  if (inprops & kAccessible) {
    outprops |= (kIEpsilons | kOEpsilons | kCyclic) & inprops;
  }
  if (inprops & kAcceptor) outprops |= (kNoIEpsilons | kNoOEpsilons) & inprops;
  if ((inprops & kNoIEpsilons) && has_subsequential_label) {
    outprops |= kNoIEpsilons;
  }
  return outprops;
}

// Properties for factored weight FST.
uint64_t FactorWeightProperties(uint64_t inprops) {
  auto outprops = (kExpanded | kMutable | kError | kAcceptor | kAcyclic |
                   kAccessible | kCoAccessible) &
                  inprops;
  if (inprops & kAccessible) {
    outprops |= (kNotAcceptor | kNonIDeterministic | kNonODeterministic |
                 kEpsilons | kIEpsilons | kOEpsilons | kCyclic |
                 kNotILabelSorted | kNotOLabelSorted) &
                inprops;
  }
  return outprops;
}

// Properties for an inverted FST.
uint64_t InvertProperties(uint64_t inprops) {
  auto outprops = (kExpanded | kMutable | kError | kAcceptor | kNotAcceptor |
                   kEpsilons | kNoEpsilons | kWeighted | kUnweighted |
                   kWeightedCycles | kUnweightedCycles | kCyclic | kAcyclic |
                   kInitialCyclic | kInitialAcyclic | kTopSorted |
                   kNotTopSorted | kAccessible | kNotAccessible |
                   kCoAccessible | kNotCoAccessible | kString | kNotString) &
                  inprops;
  if (kIDeterministic & inprops) outprops |= kODeterministic;
  if (kNonIDeterministic & inprops) outprops |= kNonODeterministic;
  if (kODeterministic & inprops) outprops |= kIDeterministic;
  if (kNonODeterministic & inprops) outprops |= kNonIDeterministic;

  if (kIEpsilons & inprops) outprops |= kOEpsilons;
  if (kNoIEpsilons & inprops) outprops |= kNoOEpsilons;
  if (kOEpsilons & inprops) outprops |= kIEpsilons;
  if (kNoOEpsilons & inprops) outprops |= kNoIEpsilons;

  if (kILabelSorted & inprops) outprops |= kOLabelSorted;
  if (kNotILabelSorted & inprops) outprops |= kNotOLabelSorted;
  if (kOLabelSorted & inprops) outprops |= kILabelSorted;
  if (kNotOLabelSorted & inprops) outprops |= kNotILabelSorted;
  return outprops;
}

// Properties for a projected FST.
uint64_t ProjectProperties(uint64_t inprops, bool project_input) {
  auto outprops = kAcceptor;
  outprops |= (kExpanded | kMutable | kError | kWeighted | kUnweighted |
               kWeightedCycles | kUnweightedCycles | kCyclic | kAcyclic |
               kInitialCyclic | kInitialAcyclic | kTopSorted | kNotTopSorted |
               kAccessible | kNotAccessible | kCoAccessible | kNotCoAccessible |
               kString | kNotString) &
              inprops;
  if (project_input) {
    outprops |= (kIDeterministic | kNonIDeterministic | kIEpsilons |
                 kNoIEpsilons | kILabelSorted | kNotILabelSorted) &
                inprops;

    if (kIDeterministic & inprops) outprops |= kODeterministic;
    if (kNonIDeterministic & inprops) outprops |= kNonODeterministic;

    if (kIEpsilons & inprops) outprops |= kOEpsilons | kEpsilons;
    if (kNoIEpsilons & inprops) outprops |= kNoOEpsilons | kNoEpsilons;

    if (kILabelSorted & inprops) outprops |= kOLabelSorted;
    if (kNotILabelSorted & inprops) outprops |= kNotOLabelSorted;
  } else {
    outprops |= (kODeterministic | kNonODeterministic | kOEpsilons |
                 kNoOEpsilons | kOLabelSorted | kNotOLabelSorted) &
                inprops;

    if (kODeterministic & inprops) outprops |= kIDeterministic;
    if (kNonODeterministic & inprops) outprops |= kNonIDeterministic;

    if (kOEpsilons & inprops) outprops |= kIEpsilons | kEpsilons;
    if (kNoOEpsilons & inprops) outprops |= kNoIEpsilons | kNoEpsilons;

    if (kOLabelSorted & inprops) outprops |= kILabelSorted;
    if (kNotOLabelSorted & inprops) outprops |= kNotILabelSorted;
  }
  return outprops;
}

// Properties for a randgen FST.
uint64_t RandGenProperties(uint64_t inprops, bool weighted) {
  auto outprops = kAcyclic | kInitialAcyclic | kAccessible | kUnweightedCycles;
  outprops |= inprops & kError;
  if (weighted) {
    outprops |= kTopSorted;
    outprops |=
        (kAcceptor | kNoEpsilons | kNoIEpsilons | kNoOEpsilons |
         kIDeterministic | kODeterministic | kILabelSorted | kOLabelSorted) &
        inprops;
  } else {
    outprops |= kUnweighted;
    outprops |= (kAcceptor | kILabelSorted | kOLabelSorted) & inprops;
  }
  return outprops;
}

// Properties for a replace FST.
uint64_t ReplaceProperties(Span inprops, size_t root,
                           bool epsilon_on_call, bool epsilon_on_return,
                           bool out_epsilon_on_call, bool out_epsilon_on_return,
                           bool replace_transducer, bool no_empty_fsts,
                           bool all_ilabel_sorted, bool all_olabel_sorted,
                           bool all_negative_or_dense) {
  if (inprops.empty()) return kNullProperties;
  uint64_t outprops = 0;
  for (auto inprop : inprops) outprops |= kError & inprop;
  uint64_t access_props = no_empty_fsts ? kAccessible | kCoAccessible : 0;
  for (auto inprop : inprops) {
    access_props &= (inprop & (kAccessible | kCoAccessible));
  }
  if (access_props == (kAccessible | kCoAccessible)) {
    outprops |= access_props;
    if (inprops[root] & kInitialCyclic) outprops |= kInitialCyclic;
    uint64_t props = 0;
    bool string = true;
    for (auto inprop : inprops) {
      if (replace_transducer) props |= kNotAcceptor & inprop;
      props |= (kNonIDeterministic | kNonODeterministic | kEpsilons |
                kIEpsilons | kOEpsilons | kWeighted | kWeightedCycles |
                kCyclic | kNotTopSorted | kNotString) &
               inprop;
      if (!(inprop & kString)) string = false;
    }
    outprops |= props;
    if (string) outprops |= kString;
  }
  bool acceptor = !replace_transducer;
  bool ideterministic = !epsilon_on_call && epsilon_on_return;
  bool no_iepsilons = !epsilon_on_call && !epsilon_on_return;
  bool acyclic = true;
  bool unweighted = true;
  for (size_t i = 0; i < inprops.size(); ++i) {
    if (!(inprops[i] & kAcceptor)) acceptor = false;
    if (!(inprops[i] & kIDeterministic)) ideterministic = false;
    if (!(inprops[i] & kNoIEpsilons)) no_iepsilons = false;
    if (!(inprops[i] & kAcyclic)) acyclic = false;
    if (!(inprops[i] & kUnweighted)) unweighted = false;
    if (i != root && !(inprops[i] & kNoIEpsilons)) ideterministic = false;
  }
  if (acceptor) outprops |= kAcceptor;
  if (ideterministic) outprops |= kIDeterministic;
  if (no_iepsilons) outprops |= kNoIEpsilons;
  if (acyclic) outprops |= kAcyclic;
  if (unweighted) outprops |= kUnweighted;
  if (inprops[root] & kInitialAcyclic) outprops |= kInitialAcyclic;
  // We assume that all terminals are positive. The resulting ReplaceFst is
  // known to be kILabelSorted when: (1) all sub-FSTs are kILabelSorted, (2) the
  // input label of the return arc is epsilon, and (3) one of the 3 following
  // conditions is satisfied:
  //
  //  1. the input label of the call arc is not epsilon
  //  2. all non-terminals are negative, or
  //  3. all non-terninals are positive and form a dense range containing 1.
  if (all_ilabel_sorted && epsilon_on_return &&
      (!epsilon_on_call || all_negative_or_dense)) {
    outprops |= kILabelSorted;
  }
  // Similarly, the resulting ReplaceFst is known to be kOLabelSorted when: (1)
  // all sub-FSTs are kOLabelSorted, (2) the output label of the return arc is
  // epsilon, and (3) one of the 3 following conditions is satisfied:
  //
  //  1. the output label of the call arc is not epsilon
  //  2. all non-terminals are negative, or
  //  3. all non-terninals are positive and form a dense range containing 1.
  if (all_olabel_sorted && out_epsilon_on_return &&
      (!out_epsilon_on_call || all_negative_or_dense)) {
    outprops |= kOLabelSorted;
  }
  return outprops;
}

// Properties for a relabeled FST.
uint64_t RelabelProperties(uint64_t inprops) {
  static constexpr auto outprops =
      kExpanded | kMutable | kError | kWeighted | kUnweighted |
      kWeightedCycles | kUnweightedCycles | kCyclic | kAcyclic |
      kInitialCyclic | kInitialAcyclic | kTopSorted | kNotTopSorted |
      kAccessible | kNotAccessible | kCoAccessible | kNotCoAccessible |
      kString | kNotString;
  return outprops & inprops;
}

// Properties for a reversed FST (the superinitial state limits this set).
uint64_t ReverseProperties(uint64_t inprops, bool has_superinitial) {
  auto outprops = (kExpanded | kMutable | kError | kAcceptor | kNotAcceptor |
                   kEpsilons | kIEpsilons | kOEpsilons | kUnweighted | kCyclic |
                   kAcyclic | kWeightedCycles | kUnweightedCycles) &
                  inprops;
  if (has_superinitial) outprops |= kWeighted & inprops;
  return outprops;
}

// Properties for re-weighted FST.
uint64_t ReweightProperties(uint64_t inprops, bool added_start_epsilon) {
  auto outprops = inprops & kWeightInvariantProperties;
  outprops = outprops & ~kCoAccessible;
  if (added_start_epsilon) {
    outprops &= ~(kNoEpsilons | kNoIEpsilons | kNoOEpsilons | kInitialCyclic);
    outprops |= kEpsilons | kIEpsilons | kOEpsilons | kInitialAcyclic;
  }
  return outprops;
}

// Properties for an epsilon-removed FST.
uint64_t RmEpsilonProperties(uint64_t inprops, bool delayed) {
  auto outprops = kNoEpsilons;
  outprops |= (kError | kAcceptor | kAcyclic | kInitialAcyclic) & inprops;
  if (inprops & kAcceptor) outprops |= kNoIEpsilons | kNoOEpsilons;
  if (!delayed) {
    outprops |= kExpanded | kMutable;
    outprops |= kTopSorted & inprops;
  }
  if (!delayed || inprops & kAccessible) outprops |= kNotAcceptor & inprops;
  return outprops;
}

// Properties for shortest path. This function computes how the properties of
// the output of shortest path need to be updated, given that 'props' is already
// known.
uint64_t ShortestPathProperties(uint64_t props, bool tree) {
  auto outprops =
      props | kAcyclic | kInitialAcyclic | kAccessible | kUnweightedCycles;
  if (!tree) outprops |= kCoAccessible;
  return outprops;
}

// Properties for a synchronized FST.
uint64_t SynchronizeProperties(uint64_t inprops) {
  auto outprops = (kError | kAcceptor | kAcyclic | kAccessible | kCoAccessible |
                   kUnweighted | kUnweightedCycles) &
                  inprops;
  if (inprops & kAccessible) {
    outprops |=
        (kCyclic | kNotCoAccessible | kWeighted | kWeightedCycles) & inprops;
  }
  return outprops;
}

// Properties for a unioned FST.
uint64_t UnionProperties(uint64_t inprops1, uint64_t inprops2, bool delayed) {
  auto outprops =
      (kAcceptor | kUnweighted | kUnweightedCycles | kAcyclic | kAccessible) &
      inprops1 & inprops2;
  outprops |= kError & (inprops1 | inprops2);
  outprops |= kInitialAcyclic;
  bool empty1 = delayed;  // Can the first FST be the empty machine?
  bool empty2 = delayed;  // Can the second FST be the empty machine?
  if (!delayed) {
    outprops |= (kExpanded | kMutable | kNotTopSorted) & inprops1;
    outprops |= kNotTopSorted & inprops2;
  }
  if (!empty1 && !empty2) {
    outprops |= kEpsilons | kIEpsilons | kOEpsilons;
    outprops |= kCoAccessible & inprops1 & inprops2;
  }
  // Note kNotCoAccessible does not hold because of kInitialAcyclic option.
  if (!delayed || inprops1 & kAccessible) {
    outprops |=
        (kNotAcceptor | kNonIDeterministic | kNonODeterministic | kEpsilons |
         kIEpsilons | kOEpsilons | kNotILabelSorted | kNotOLabelSorted |
         kWeighted | kWeightedCycles | kCyclic | kNotAccessible) &
        inprops1;
  }
  if (!delayed || inprops2 & kAccessible) {
    outprops |= (kNotAcceptor | kNonIDeterministic | kNonODeterministic |
                 kEpsilons | kIEpsilons | kOEpsilons | kNotILabelSorted |
                 kNotOLabelSorted | kWeighted | kWeightedCycles | kCyclic |
                 kNotAccessible | kNotCoAccessible) &
                inprops2;
  }
  return outprops;
}


}  // namespace fst

static uint64_t next_rand(uint64_t& state) {
  state ^= state << 13; state ^= state >> 7; state ^= state << 17;
  return state;
}

int main() {
  uint64_t state = 0x1234'5678'9ABC'DEF0ull;
  for (int i = 0; i < 400; ++i) {
    const uint64_t a = next_rand(state) & fst::kFstProperties;
    const uint64_t b = next_rand(state) & fst::kFstProperties;
    const bool f1 = next_rand(state) & 1;
    const bool f2 = next_rand(state) & 1;
    printf("%016llx %016llx %d %d", (unsigned long long)a,
           (unsigned long long)b, (int)f1, (int)f2);
    printf(" %016llx", (unsigned long long)fst::ClosureProperties(a, f1, f2));
    printf(" %016llx", (unsigned long long)fst::ComplementProperties(a));
    printf(" %016llx", (unsigned long long)fst::ComposeProperties(a, b));
    printf(" %016llx", (unsigned long long)fst::ConcatProperties(a, b, f1));
    printf(" %016llx", (unsigned long long)fst::DeterminizeProperties(a, f1, f2));
    printf(" %016llx", (unsigned long long)fst::FactorWeightProperties(a));
    printf(" %016llx", (unsigned long long)fst::InvertProperties(a));
    printf(" %016llx", (unsigned long long)fst::ProjectProperties(a, f1));
    printf(" %016llx", (unsigned long long)fst::RandGenProperties(a, f1));
    printf(" %016llx", (unsigned long long)fst::RelabelProperties(a));
    printf(" %016llx", (unsigned long long)fst::ReverseProperties(a, f1));
    printf(" %016llx", (unsigned long long)fst::ReweightProperties(a, f1));
    printf(" %016llx", (unsigned long long)fst::RmEpsilonProperties(a, f1));
    printf(" %016llx", (unsigned long long)fst::ShortestPathProperties(a, f1));
    printf(" %016llx", (unsigned long long)fst::SynchronizeProperties(a));
    printf(" %016llx", (unsigned long long)fst::UnionProperties(a, b, f1));
    std::vector<uint64_t> v{a, b, a ^ b};
    printf(" %016llx", (unsigned long long)fst::ReplaceProperties(
        fst::Span(v), 1, f1, f2, !f1, !f2, f1, f2, !f1, !f2, f1));
    printf("\n");
  }
  return 0;
}
