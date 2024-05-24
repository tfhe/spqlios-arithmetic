#ifndef SPQLIOS_COMMONS_H
#define SPQLIOS_COMMONS_H

#ifdef __cplusplus
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define EXPORT extern "C"
#define EXPORT_DECL extern "C"
#else
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define EXPORT
#define EXPORT_DECL extern
#define nullptr 0x0;
#endif

#define UNDEFINED()                    \
  {                                    \
    fprintf(stderr, "UNDEFINED!!!\n"); \
    abort();                           \
  }
#define NOT_IMPLEMENTED()                    \
  {                                          \
    fprintf(stderr, "NOT IMPLEMENTED!!!\n"); \
    abort();                                 \
  }
#define FATAL_ERROR(MESSAGE)                   \
  {                                            \
    fprintf(stderr, "ERROR: %s\n", (MESSAGE)); \
    abort();                                   \
  }

#ifdef __x86_64__
#define CPU_SUPPORTS __builtin_cpu_supports
#else
// TODO for now, we do not have any optimization for non x86 targets
#define CPU_SUPPORTS(xxxx) 0
#endif

EXPORT void* UNDEFINED_p_ii(int32_t n, int32_t m);
EXPORT void* UNDEFINED_p_uu(uint32_t n, uint32_t m);
EXPORT double* UNDEFINED_dp_pi(const void* p, int32_t n);
EXPORT void* UNDEFINED_vp_pi(const void* p, int32_t n);
EXPORT void* UNDEFINED_vp_pu(const void* p, uint32_t n);
EXPORT void UNDEFINED_v_vpdp(const void* p, double* a);
EXPORT void UNDEFINED_v_vpvp(const void* p, void* a);
EXPORT double* NOT_IMPLEMENTED_dp_i(int32_t n);
EXPORT void* NOT_IMPLEMENTED_vp_i(int32_t n);
EXPORT void* NOT_IMPLEMENTED_vp_u(uint32_t n);
EXPORT void NOT_IMPLEMENTED_v_dp(double* a);
EXPORT void NOT_IMPLEMENTED_v_vp(void* p);
EXPORT void NOT_IMPLEMENTED_v_idpdpdp(int32_t n, double* a, const double* b, const double* c);
EXPORT void NOT_IMPLEMENTED_v_uvpcvpcvp(uint32_t n, void* r, const void* a, const void* b);
EXPORT void NOT_IMPLEMENTED_v_uvpvpcvp(uint32_t n, void* a, void* b, const void* o);

#endif  // SPQLIOS_COMMONS_H
