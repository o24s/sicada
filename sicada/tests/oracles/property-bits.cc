#include <cstdint>
#include <cstdio>
namespace fst {
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
}  // namespace fst
int main(){
  printf("kExpanded %016llx\n", (unsigned long long)fst::kExpanded);
  printf("kMutable %016llx\n", (unsigned long long)fst::kMutable);
  printf("kError %016llx\n", (unsigned long long)fst::kError);
  printf("kAcceptor %016llx\n", (unsigned long long)fst::kAcceptor);
  printf("kNotAcceptor %016llx\n", (unsigned long long)fst::kNotAcceptor);
  printf("kIDeterministic %016llx\n", (unsigned long long)fst::kIDeterministic);
  printf("kNonIDeterministic %016llx\n", (unsigned long long)fst::kNonIDeterministic);
  printf("kODeterministic %016llx\n", (unsigned long long)fst::kODeterministic);
  printf("kNonODeterministic %016llx\n", (unsigned long long)fst::kNonODeterministic);
  printf("kEpsilons %016llx\n", (unsigned long long)fst::kEpsilons);
  printf("kNoEpsilons %016llx\n", (unsigned long long)fst::kNoEpsilons);
  printf("kIEpsilons %016llx\n", (unsigned long long)fst::kIEpsilons);
  printf("kNoIEpsilons %016llx\n", (unsigned long long)fst::kNoIEpsilons);
  printf("kOEpsilons %016llx\n", (unsigned long long)fst::kOEpsilons);
  printf("kNoOEpsilons %016llx\n", (unsigned long long)fst::kNoOEpsilons);
  printf("kILabelSorted %016llx\n", (unsigned long long)fst::kILabelSorted);
  printf("kNotILabelSorted %016llx\n", (unsigned long long)fst::kNotILabelSorted);
  printf("kOLabelSorted %016llx\n", (unsigned long long)fst::kOLabelSorted);
  printf("kNotOLabelSorted %016llx\n", (unsigned long long)fst::kNotOLabelSorted);
  printf("kWeighted %016llx\n", (unsigned long long)fst::kWeighted);
  printf("kUnweighted %016llx\n", (unsigned long long)fst::kUnweighted);
  printf("kCyclic %016llx\n", (unsigned long long)fst::kCyclic);
  printf("kAcyclic %016llx\n", (unsigned long long)fst::kAcyclic);
  printf("kInitialCyclic %016llx\n", (unsigned long long)fst::kInitialCyclic);
  printf("kInitialAcyclic %016llx\n", (unsigned long long)fst::kInitialAcyclic);
  printf("kTopSorted %016llx\n", (unsigned long long)fst::kTopSorted);
  printf("kNotTopSorted %016llx\n", (unsigned long long)fst::kNotTopSorted);
  printf("kAccessible %016llx\n", (unsigned long long)fst::kAccessible);
  printf("kNotAccessible %016llx\n", (unsigned long long)fst::kNotAccessible);
  printf("kCoAccessible %016llx\n", (unsigned long long)fst::kCoAccessible);
  printf("kNotCoAccessible %016llx\n", (unsigned long long)fst::kNotCoAccessible);
  printf("kString %016llx\n", (unsigned long long)fst::kString);
  printf("kNotString %016llx\n", (unsigned long long)fst::kNotString);
  printf("kWeightedCycles %016llx\n", (unsigned long long)fst::kWeightedCycles);
  printf("kUnweightedCycles %016llx\n", (unsigned long long)fst::kUnweightedCycles);
  printf("kNullProperties %016llx\n", (unsigned long long)fst::kNullProperties);
  printf("kCompiledStringProperties %016llx\n", (unsigned long long)fst::kCompiledStringProperties);
  printf("kCopyProperties %016llx\n", (unsigned long long)fst::kCopyProperties);
  printf("kIntrinsicProperties %016llx\n", (unsigned long long)fst::kIntrinsicProperties);
  printf("kExtrinsicProperties %016llx\n", (unsigned long long)fst::kExtrinsicProperties);
  printf("kSetStartProperties %016llx\n", (unsigned long long)fst::kSetStartProperties);
  printf("kSetFinalProperties %016llx\n", (unsigned long long)fst::kSetFinalProperties);
  printf("kAddStateProperties %016llx\n", (unsigned long long)fst::kAddStateProperties);
  printf("kAddArcProperties %016llx\n", (unsigned long long)fst::kAddArcProperties);
  printf("kSetArcProperties %016llx\n", (unsigned long long)fst::kSetArcProperties);
  printf("kDeleteStatesProperties %016llx\n", (unsigned long long)fst::kDeleteStatesProperties);
  printf("kDeleteArcsProperties %016llx\n", (unsigned long long)fst::kDeleteArcsProperties);
  printf("kStateSortProperties %016llx\n", (unsigned long long)fst::kStateSortProperties);
  printf("kArcSortProperties %016llx\n", (unsigned long long)fst::kArcSortProperties);
  printf("kILabelInvariantProperties %016llx\n", (unsigned long long)fst::kILabelInvariantProperties);
  printf("kOLabelInvariantProperties %016llx\n", (unsigned long long)fst::kOLabelInvariantProperties);
  printf("kWeightInvariantProperties %016llx\n", (unsigned long long)fst::kWeightInvariantProperties);
  printf("kAddSuperFinalProperties %016llx\n", (unsigned long long)fst::kAddSuperFinalProperties);
  printf("kRmSuperFinalProperties %016llx\n", (unsigned long long)fst::kRmSuperFinalProperties);
  printf("kBinaryProperties %016llx\n", (unsigned long long)fst::kBinaryProperties);
  printf("kTrinaryProperties %016llx\n", (unsigned long long)fst::kTrinaryProperties);
  printf("kPosTrinaryProperties %016llx\n", (unsigned long long)fst::kPosTrinaryProperties);
  printf("kNegTrinaryProperties %016llx\n", (unsigned long long)fst::kNegTrinaryProperties);
  printf("kFstProperties %016llx\n", (unsigned long long)fst::kFstProperties);
  return 0;
}
