// Byte-level reference for the FST header, following FstHeader::Write in
// vendor/openfst/openfst/lib/fst.cc with the util.h ReadType/WriteType
// overloads.
#include <cstdint>
#include <cstdio>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>

namespace fst {
inline constexpr int32_t kFstMagicNumber = 2125659606;

template <class T, typename std::enable_if_t<std::is_arithmetic_v<T>, T>* = nullptr>
inline std::ostream& WriteType(std::ostream& strm, const T t) {
  return strm.write(reinterpret_cast<const char*>(&t), sizeof(T));
}
inline std::ostream& WriteType(std::ostream& strm, const std::string& s) {
  int32_t ns = s.size();
  WriteType(strm, ns);
  return strm.write(s.data(), ns);
}
}  // namespace fst

int main() {
  std::ostringstream o;
  fst::WriteType(o, fst::kFstMagicNumber);
  fst::WriteType(o, std::string("vector"));     // fsttype_
  fst::WriteType(o, std::string("standard"));   // arctype_
  fst::WriteType(o, (int32_t)2);                // version_
  fst::WriteType(o, (uint32_t)3);               // flags_: HAS_ISYMBOLS|HAS_OSYMBOLS
  fst::WriteType(o, (uint64_t)0x0000000000010007ull);  // properties_
  fst::WriteType(o, (int64_t)0);                // start_
  fst::WriteType(o, (int64_t)5);                // numstates_
  fst::WriteType(o, (int64_t)7);                // numarcs_
  const std::string bytes = o.str();
  printf("len %zu\nbytes ", bytes.size());
  for (unsigned char c : bytes) printf("%02x", c);
  printf("\n");
  return 0;
}
