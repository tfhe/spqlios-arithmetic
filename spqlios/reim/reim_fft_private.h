#ifndef SPQLIOS_REIM_FFT_PRIVATE_H
#define SPQLIOS_REIM_FFT_PRIVATE_H

#include "reim_fft_public.h"

#define STATIC_ASSERT(condition) (void)sizeof(char[-1 + 2 * !!(condition)])

typedef void (*FFT_FUNC)(const REIM_FFT_PRECOMP*, double*);
typedef void (*IFFT_FUNC)(const REIM_IFFT_PRECOMP*, double*);
typedef void (*FFTVEC_MUL_FUNC)(const REIM_FFTVEC_MUL_PRECOMP*, double*, const double*, const double*);
typedef void (*FFTVEC_ADDMUL_FUNC)(const REIM_FFTVEC_ADDMUL_PRECOMP*, double*, const double*, const double*);

typedef void (*FROM_ZNX32_FUNC)(const REIM_FROM_ZNX32_PRECOMP*, void*, const int32_t*);
typedef void (*FROM_TNX32_FUNC)(const REIM_FROM_TNX32_PRECOMP*, void*, const int32_t*);
typedef void (*TO_TNX32_FUNC)(const REIM_TO_TNX32_PRECOMP*, int32_t*, const void*);

typedef struct reim_fft_precomp {
  FFT_FUNC function;
  int64_t n;  // warning: reim uses n=2N=4m
  double* aligned_trig_precomp;
  void* aligned_data;
} REIM_FFT_PRECOMP;

typedef struct reim_ifft_precomp {
  IFFT_FUNC function;
  int64_t n; // warning: reim uses n=2N=4m
  double* aligned_trig_precomp;
  void* aligned_data;
} REIM_IFFT_PRECOMP;

typedef struct reim_mul_precomp {
  FFTVEC_MUL_FUNC function;
  int64_t m;
} REIM_FFTVEC_MUL_PRECOMP;

typedef struct reim_addmul_precomp {
  FFTVEC_ADDMUL_FUNC function;
  int64_t m;
} REIM_FFTVEC_ADDMUL_PRECOMP;


struct reim_from_znx32_precomp {
  FROM_ZNX32_FUNC function;
  int64_t m;
};

struct reim_from_tnx32_precomp {
  FROM_TNX32_FUNC function;
  int64_t m;
};

struct reim_to_tnx32_precomp {
  TO_TNX32_FUNC function;
  int64_t m;
  double divisor;
};

#endif  // SPQLIOS_REIM_FFT_PRIVATE_H
