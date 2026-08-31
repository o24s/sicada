// Byte-level reference for the ReadType/WriteType primitives in
// vendor/openfst/openfst/lib/util.h (commit e77e51d), extracted verbatim for
// the scalar, string and vector cases.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace fst {
template <class T, typename std::enable_if_t<std::is_arithmetic_v<T> ||
                                             std::is_enum_v<T>, T>* = nullptr>
inline std::ostream& WriteType(std::ostream& strm, const T t) {
  return strm.write(reinterpret_cast<const char*>(&t), sizeof(T));
}

inline std::ostream& WriteType(std::ostream& strm, const std::string& s) {
  int32_t ns = s.size();
  WriteType(strm, ns);
  return strm.write(s.data(), ns);
}

template <class T>
inline std::ostream& WriteType(std::ostream& strm, const std::vector<T>& c) {
  int64_t n = c.size();
  WriteType(strm, n);
  for (const auto& e : c) WriteType(strm, e);
  return strm;
}
}  // namespace fst

static void dump(const char* label, const std::string& bytes) {
  printf("%s ", label);
  for (unsigned char c : bytes) printf("%02x", c);
  printf("\n");
}

int main() {
  { std::ostringstream o; fst::WriteType(o, (int32_t)0x12345678); dump("i32", o.str()); }
  { std::ostringstream o; fst::WriteType(o, (int32_t)-2); dump("i32neg", o.str()); }
  { std::ostringstream o; fst::WriteType(o, (int64_t)0x0123456789ABCDEFll); dump("i64", o.str()); }
  { std::ostringstream o; fst::WriteType(o, (uint64_t)0xFEDCBA9876543210ull); dump("u64", o.str()); }
  { std::ostringstream o; fst::WriteType(o, (float)0.5f); dump("f32", o.str()); }
  { std::ostringstream o; fst::WriteType(o, (double)-0.25); dump("f64", o.str()); }
  { std::ostringstream o; fst::WriteType(o, (uint8_t)0xAB); dump("u8", o.str()); }
  { std::ostringstream o; fst::WriteType(o, std::string("abc")); dump("str", o.str()); }
  { std::ostringstream o; fst::WriteType(o, std::string("")); dump("emptystr", o.str()); }
  { std::ostringstream o; fst::WriteType(o, std::vector<int64_t>{1, -2, 3}); dump("veci64", o.str()); }
  { std::ostringstream o; fst::WriteType(o, std::vector<int32_t>{}); dump("emptyvec", o.str()); }
  return 0;
}
