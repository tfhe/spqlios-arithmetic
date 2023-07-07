#ifndef SPQLIOS_CPLX_FFT_PRIVATE_H
#define SPQLIOS_CPLX_FFT_PRIVATE_H

#include "cplx_fft_public.h"

typedef void (*IFFT_FUNCTION)(const CPLX_IFFT_PRECOMP*, void*);
typedef void (*FFT_FUNCTION)(const CPLX_FFT_PRECOMP*, void*);

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

#endif  // SPQLIOS_CPLX_FFT_PRIVATE_H
