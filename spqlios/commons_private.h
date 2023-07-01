#ifndef SPQLIOS_COMMONS_PRIVATE_H
#define SPQLIOS_COMMONS_PRIVATE_H

#include "commons.h"

#ifdef __cplusplus
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#else
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#endif

/** @brief log2 of a power of two (UB if m is not a power of two) */
uint32_t log2m(uint32_t m);

/** @brief checks if the doublevalue is a power of two */
uint64_t is_not_pow2_double(void* doublevalue);

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
#define NOT_SUPPORTED()                    \
  {                                          \
    fprintf(stderr, "NOT SUPPORTED!!!\n"); \
    abort();                                 \
  }
#define FATAL_ERROR(MESSAGE)                   \
  {                                            \
    fprintf(stderr, "ERROR: %s\n", (MESSAGE)); \
    abort();                                   \
  }

/** @brief reports the error and returns nullptr */
EXPORT void* spqlios_error(const char* error);
/** @brief if ptr2 is not null, returns ptr, otherwise free ptr and return null */
EXPORT void* spqlios_keep_or_free(void* ptr, void* ptr2);

#ifdef __x86_64__
#define CPU_SUPPORTS __builtin_cpu_supports
#else
// TODO for now, we do not have any optimization for non x86 targets
#define CPU_SUPPORTS(xxxx) 0
#endif

#endif  // SPQLIOS_COMMONS_PRIVATE_H