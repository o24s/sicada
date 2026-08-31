// Byte-level reference for the SymbolTable binary format, following
// SymbolTableImpl::Write in vendor/openfst/openfst/lib/symbol-table.cc
// (commit e77e51d) with the ReadType/WriteType overloads from util.h.
#include <cstdint>
#include <cstdio>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace fst {
inline constexpr int32_t kSymbolTableMagicNumber = 2125658996;

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
  // A table named "test" holding a dense run 0..2 plus one sparse key.
  std::ostringstream o;
  fst::WriteType(o, fst::kSymbolTableMagicNumber);
  fst::WriteType(o, std::string("test"));
  fst::WriteType(o, (int64_t)101);  // available_key_, as AddSymbol would leave it
  fst::WriteType(o, (int64_t)4);   // size
  const char* dense[] = {"<eps>", "a", "b"};
  for (int64_t i = 0; i < 3; ++i) {
    fst::WriteType(o, std::string(dense[i]));
    fst::WriteType(o, i);
  }
  fst::WriteType(o, std::string("sparse"));
  fst::WriteType(o, (int64_t)100);

  const std::string bytes = o.str();
  printf("len %zu\nbytes ", bytes.size());
  for (unsigned char c : bytes) printf("%02x", c);
  printf("\n");
  return 0;
}
