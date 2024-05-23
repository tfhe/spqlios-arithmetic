#include "test_commons.h"

std::ostream& operator<<(std::ostream& out, __int128_t x) {
  char c[35] = {0};
  sprintf(c, "0x%016lx%016lx", uint64_t(x >> 64), uint64_t(x));
  return out << c;
}
std::ostream& operator<<(std::ostream& out, __uint128_t x) { return out << __int128_t(x); }
