#ifndef SPQLIOS_CPLX_FFT_PRIVATE_H
#define SPQLIOS_CPLX_FFT_PRIVATE_H

#include "cplx_fft_public.h"

typedef void (*IFFT_FUNCTION)(const CPLX_IFFT_PRECOMP*, void*);
typedef void (*FFT_FUNCTION)(const CPLX_FFT_PRECOMP*, void*);
// conversions
typedef void (*FROM_ZNX32_FUNCTION)(const CPLX_FROM_ZNX32_PRECOMP*, void*, const int32_t*);
typedef void (*TO_ZNX32_FUNCTION)(const CPLX_FROM_ZNX32_PRECOMP*, int32_t*, const void*);
typedef void (*FROM_TNX32_FUNCTION)(const CPLX_FROM_TNX32_PRECOMP*, void*, const int32_t*);
typedef void (*TO_TNX32_FUNCTION)(const CPLX_TO_TNX32_PRECOMP*, int32_t*, const void*);
typedef void (*FROM_RNX64_FUNCTION)(const CPLX_FROM_RNX64_PRECOMP* precomp, void* r, const double* x);
typedef void (*TO_RNX64_FUNCTION)(const CPLX_TO_RNX64_PRECOMP* precomp, double* r, const void* x);
typedef void (*ROUND_TO_RNX64_FUNCTION)(const CPLX_ROUND_TO_RNX64_PRECOMP* precomp, double* r, const void* x);

struct cplx_ifft_precomp {
  IFFT_FUNCTION function;
  int64_t m;
  double* powomegas;
  void* aligned_buffers;
};

struct cplx_fft_precomp {
  FFT_FUNCTION function;
  int64_t m;
  double* powomegas;
  void* aligned_buffers;
};

struct cplx_from_znx32_precomp {
  FROM_ZNX32_FUNCTION function;
  int64_t m;
};

struct cplx_to_znx32_precomp {
  TO_ZNX32_FUNCTION function;
  int64_t m;
  double divisor;
};

struct cplx_from_tnx32_precomp {
  FROM_TNX32_FUNCTION function;
  int64_t m;
};

struct cplx_to_tnx32_precomp {
  TO_TNX32_FUNCTION function;
  int64_t m;
  double divisor;
};

struct cplx_from_rnx64_precomp {
  FROM_RNX64_FUNCTION function;
  int64_t m;
};

struct cplx_to_rnx64_precomp {
  TO_RNX64_FUNCTION function;
  int64_t m;
  double divisor;
};

struct cplx_round_to_rnx64_precomp {
  ROUND_TO_RNX64_FUNCTION function;
  int64_t m;
  double divisor;
  uint32_t log2bound;
};

#endif  // SPQLIOS_CPLX_FFT_PRIVATE_H
