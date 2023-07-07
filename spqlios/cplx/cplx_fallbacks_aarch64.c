#include "../commons_private.h"
#include "cplx_fft.h"
#include "cplx_fft_private.h"

EXPORT void cplx_ifft16_avx_fma(void* data, const void* omega) { UNDEFINED(); }
EXPORT void cplx_ifft_avx2_fma(const CPLX_IFFT_PRECOMP* itables, void* data) { UNDEFINED(); }
EXPORT void cplx_fft16_avx_fma(void* data, const void* omega) { UNDEFINED(); }
EXPORT void cplx_fft_avx2_fma(const CPLX_FFT_PRECOMP* tables, void* data) { UNDEFINED() }