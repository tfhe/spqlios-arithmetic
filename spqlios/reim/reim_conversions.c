#include <errno.h>
#include <string.h>

#include "../commons_private.h"
#include "reim_fft.h"
#include "reim_fft_private.h"

void reim_from_znx32_ref(const REIM_FROM_ZNX32_PRECOMP* precomp, void* r, const int32_t* x) { NOT_IMPLEMENTED(); }
void reim_from_znx32_avx2_fma(const REIM_FROM_ZNX32_PRECOMP* precomp, void* r, const int32_t* x) { NOT_IMPLEMENTED(); }


void* init_reim_from_znx32_precomp(REIM_FROM_ZNX32_PRECOMP* const res, uint32_t m, uint32_t log2bound) {
  if (m & (m - 1)) return spqlios_error("m must be a power of 2");
  if (log2bound > 32) return spqlios_error("Invalid log2bound error: must be in [0,32]");
  res->m = m;
  // TODO: check selection logic
  if (CPU_SUPPORTS("avx2")) {
    if (m >= 8) {
      res->function = reim_from_znx32_avx2_fma;
    } else {
      res->function = reim_from_znx32_ref;
    }
  } else {
    res->function = reim_from_znx32_ref;
  }
  return res;
}


EXPORT REIM_FROM_ZNX32_PRECOMP* new_reim_from_znx32_precomp(uint32_t m, uint32_t log2bound) {
  REIM_FROM_ZNX32_PRECOMP* res = malloc(sizeof(*res));
  if (!res) return spqlios_error(strerror(errno));
  return spqlios_keep_or_free(res, init_reim_from_znx32_precomp(res, m, log2bound));
}

void reim_from_tnx32_ref(const REIM_FROM_TNX32_PRECOMP* precomp, void* r, const int32_t* x) { NOT_IMPLEMENTED(); }
void reim_from_tnx32_avx2_fma(const REIM_FROM_TNX32_PRECOMP* precomp, void* r, const int32_t* x) { NOT_IMPLEMENTED(); }

void* init_reim_from_tnx32_precomp(REIM_FROM_TNX32_PRECOMP* const res, uint32_t m) {
  if (m & (m - 1)) return spqlios_error("m must be a power of 2");
  res->m = m;
  // TODO: check selection logic
  if (CPU_SUPPORTS("avx2")) {
    if (m >= 8) {
      res->function = reim_from_tnx32_avx2_fma;
    } else {
      res->function = reim_from_tnx32_ref;
    }
  } else {
    res->function = reim_from_tnx32_ref;
  }
  return res;
}


EXPORT REIM_FROM_TNX32_PRECOMP* new_reim_from_tnx32_precomp(uint32_t m) {
  REIM_FROM_TNX32_PRECOMP* res = malloc(sizeof(*res));
  if (!res) return spqlios_error(strerror(errno));
  return spqlios_keep_or_free(res, init_reim_from_tnx32_precomp(res, m));
}

void reim_to_tnx32_ref(const REIM_TO_TNX32_PRECOMP* precomp, int32_t* r, const void* x) { NOT_IMPLEMENTED(); }
void reim_to_tnx32_avx2_fma(const REIM_TO_TNX32_PRECOMP* precomp, int32_t* r, const void* x) { NOT_IMPLEMENTED(); }

void* init_reim_to_tnx32_precomp(REIM_TO_TNX32_PRECOMP* const res, uint32_t m, double divisor, uint32_t log2overhead) {
  if (m & (m - 1)) return spqlios_error("m must be a power of 2");
  if (is_not_pow2_double(&divisor)) return spqlios_error("divisor must be a power of 2");
  if (log2overhead > 52) return spqlios_error("log2overhead is too large");
  res->m = m;
  res->divisor = divisor;
  // TODO: check selection logic
  if (CPU_SUPPORTS("avx2")) {
    if (log2overhead <= 18) {
      if (m >= 8) {
        res->function = reim_to_tnx32_avx2_fma;
      } else {
        res->function = reim_to_tnx32_ref;
      }
    } else {
      res->function = reim_to_tnx32_ref;
    }
  } else {
    res->function = reim_to_tnx32_ref;
  }
  return res;
}

EXPORT REIM_TO_TNX32_PRECOMP* new_reim_to_tnx32_precomp(uint32_t m, double divisor, uint32_t log2overhead) {
  REIM_TO_TNX32_PRECOMP* res = malloc(sizeof(*res));
  if (!res) return spqlios_error(strerror(errno));
  return spqlios_keep_or_free(res, init_reim_to_tnx32_precomp(res, m, divisor, log2overhead));
}

EXPORT void reim_from_znx32_simple(uint32_t m, uint32_t log2bound, void* r, const int32_t* x) {
  static REIM_FROM_ZNX32_PRECOMP* p[31] = {0};
  REIM_FROM_ZNX32_PRECOMP** f = p + log2m(m);
  if (!*f) *f = new_reim_from_znx32_precomp(m, log2bound);
  (*f)->function(*f, r, x);
}

EXPORT void reim_from_tnx32_simple(uint32_t m, void* r, const int32_t* x) {
  static REIM_FROM_TNX32_PRECOMP* p[31] = {0};
  REIM_FROM_TNX32_PRECOMP** f = p + log2m(m);
  if (!*f) *f = new_reim_from_tnx32_precomp(m);
  (*f)->function(*f, r, x);
}

EXPORT void reim_to_tnx32_simple(uint32_t m, double divisor, uint32_t log2overhead, int32_t* r, const void* x) {
  static REIM_TO_TNX32_PRECOMP* p[31] = {0};
  REIM_TO_TNX32_PRECOMP** f = p + log2m(m);
  if (!*f) *f = new_reim_to_tnx32_precomp(m, divisor, log2overhead);
  (*f)->function(*f, r, x);
}
