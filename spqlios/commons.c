#include "commons.h"

#include <stdio.h>
#include <stdlib.h>

EXPORT void* UNDEFINED_p_ii(int32_t n, int32_t m) { UNDEFINED(); }
EXPORT void* UNDEFINED_p_uu(uint32_t n, uint32_t m) { UNDEFINED(); }
EXPORT double* UNDEFINED_dp_pi(const void* p, int32_t n) { UNDEFINED(); }
EXPORT void* UNDEFINED_vp_pi(const void* p, int32_t n) { UNDEFINED(); }
EXPORT void* UNDEFINED_vp_pu(const void* p, uint32_t n) { UNDEFINED(); }
EXPORT void UNDEFINED_v_vpdp(const void* p, double* a) { UNDEFINED(); }
EXPORT void UNDEFINED_v_vpvp(const void* p, void* a) { UNDEFINED(); }
EXPORT double* NOT_IMPLEMENTED_dp_i(int32_t n) { NOT_IMPLEMENTED(); }
EXPORT void* NOT_IMPLEMENTED_vp_i(int32_t n) { NOT_IMPLEMENTED(); }
EXPORT void* NOT_IMPLEMENTED_vp_u(uint32_t n) { NOT_IMPLEMENTED(); }
EXPORT void NOT_IMPLEMENTED_v_dp(double* a) { NOT_IMPLEMENTED(); }
EXPORT void NOT_IMPLEMENTED_v_vp(void* p) { NOT_IMPLEMENTED(); }
EXPORT void NOT_IMPLEMENTED_v_idpdpdp(int32_t n, double* a, const double* b, const double* c) { NOT_IMPLEMENTED(); }
EXPORT void NOT_IMPLEMENTED_v_uvpcvpcvp(uint32_t n, void* r, const void* a, const void* b) { NOT_IMPLEMENTED(); }
EXPORT void NOT_IMPLEMENTED_v_uvpvpcvp(uint32_t n, void* a, void* b, const void* o) { NOT_IMPLEMENTED(); }

#ifdef _WIN32
EXPORT void* aligned_alloc(size_t align, size_t n) {
  return malloc(n);
  // unfortunately, there is no alternative that gets freed with free :(
}
#define __always_inline inline __attribute((always_inline))
#endif
