// The bytes OpenFst's encode table writes.
//
// Verbatim extraction of `EncodeTableHeader::Write` from
// vendor/openfst/openfst/lib/encode.cc and of `EncodeTable<Arc>::Write` and
// `Triple::Write` from vendor/openfst/openfst/lib/encode.h (commit 694dc53),
// with the table replaced by a list of triples. What is pinned is the order and
// width of every field: the magic number, the arc type string, the flags byte
// (including the internal symbol-table bits), the triple count, and then each
// (ilabel, olabel, weight).
//
// Build and run:
//   g++ -std=c++17 -O2 -o /tmp/etg tests/oracles/encode-table-golden.cc && /tmp/etg
#include <cstddef>
#include <cstdint>
#include <cstdio>
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

inline std::ostream &WriteType(std::ostream &strm, const std::string &s) {
  int32_t ns = s.size();
  strm.write(reinterpret_cast<const char *>(&ns), sizeof(ns));
  return strm.write(s.data(), ns);
}

// encode.h: the magic number and the flag bits.
inline constexpr int32_t kEncodeMagicNumber = 2128178506;
inline constexpr uint8_t kEncodeLabels = 0x01;
inline constexpr uint8_t kEncodeWeights = 0x02;
inline constexpr uint8_t kEncodeHasISymbols = 0x04;
inline constexpr uint8_t kEncodeHasOSymbols = 0x08;

// float-weight.h: a weight writes its value and nothing else.
struct TropicalWeight {
  float value;
  std::ostream &Write(std::ostream &strm) const {
    return WriteType(strm, value);
  }
};

// encode.h: EncodeTable<Arc>::Triple::Write, verbatim.
struct Triple {
  int32_t ilabel;
  int32_t olabel;
  TropicalWeight weight;

  void Write(std::ostream &strm) const {
    WriteType(strm, ilabel);
    WriteType(strm, olabel);
    WriteType(strm, weight);
  }
};

// encode.cc: EncodeTableHeader::Write, verbatim but for the member names.
struct EncodeTableHeader {
  std::string arctype_;
  uint8_t flags_;
  size_t size_;

  bool Write(std::ostream &strm) const {
    WriteType(strm, kEncodeMagicNumber);
    WriteType(strm, arctype_);
    WriteType(strm, flags_);
    WriteType(strm, size_);
    strm.flush();
    return static_cast<bool>(strm);
  }
};

// encode.h: EncodeTable<Arc>::Write, with `triples_` given directly.
bool WriteTable(std::ostream &strm, const std::vector<Triple> &triples,
                uint8_t flags) {
  EncodeTableHeader hdr;
  hdr.arctype_ = "standard";
  hdr.flags_ = flags;  // Real flags, not masked ones.
  hdr.size_ = triples.size();
  if (!hdr.Write(strm)) return false;
  for (const auto &triple : triples) triple.Write(strm);
  strm.flush();
  return static_cast<bool>(strm);
}

}  // namespace fst

int main() {
  static_assert(sizeof(size_t) == 8, "the size field is written as size_t");

  const std::vector<fst::Triple> triples = {
      {1, 2, {0.5f}},
      {3, 4, {1.5f}},
  };

  std::ostringstream strm(std::ios_base::out | std::ios_base::binary);
  fst::WriteTable(strm, triples, fst::kEncodeLabels | fst::kEncodeWeights);
  const std::string bytes = strm.str();

  std::printf("%zu bytes\n", bytes.size());
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    std::printf("%02x", static_cast<unsigned char>(bytes[i]));
    std::printf((i % 16 == 15 || i + 1 == bytes.size()) ? "\n" : " ");
  }
  return 0;
}
