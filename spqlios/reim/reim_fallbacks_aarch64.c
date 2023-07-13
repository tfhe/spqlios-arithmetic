#include "reim_fft_private.h"
#include "../commons_private.h"

EXPORT void reim_fftvec_addmul_fma(const REIM_FFTVEC_ADDMUL_PRECOMP* precomp, double* r, const double* a,
                                   const double* b) {
  UNDEFINED();
}
EXPORT void reim_fftvec_mul_fma(const REIM_FFTVEC_MUL_PRECOMP* precomp, double* r, const double* a, const double* b) {
  UNDEFINED();
}

EXPORT void reim_fft_avx2_fma(const REIM_FFT_PRECOMP* tables, double* data) { UNDEFINED(); }
EXPORT void reim_ifft_avx2_fma(const REIM_IFFT_PRECOMP* tables, double* data) { UNDEFINED(); }

EXPORT void reim_fftvec_mul(const REIM_FFTVEC_MUL_PRECOMP* precomp, double* r, const double* a, const double* b) {
  precomp->function(precomp, r, a, b);
}
EXPORT void reim_fftvec_addmul(const REIM_FFTVEC_ADDMUL_PRECOMP* precomp, double* r, const double* a, const double* b) {
  precomp->function(precomp, r, a, b);
}