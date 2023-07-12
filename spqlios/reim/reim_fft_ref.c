#include "../commons_private.h"
#include "reim_fft.h"
#include "reim_fft_private.h"

EXPORT void reim_fft_simple(uint32_t m, void* data) {
  static REIM_FFT_PRECOMP* p[31] = {0};
  REIM_FFT_PRECOMP** f = p + log2m(m);
  if (!*f) *f = new_reim_fft_precomp(m, 0);
  (*f)->function(*f, data);
}

EXPORT void reim_ifft_simple(uint32_t m, void* data) {
  static REIM_IFFT_PRECOMP* p[31] = {0};
  REIM_IFFT_PRECOMP** f = p + log2m(m);
  if (!*f) *f = new_reim_ifft_precomp(m, 0);
  (*f)->function(*f, data);
}

EXPORT void reim_fftvec_mul_simple(uint32_t m, void* r, const void* a, const void* b) {
  static REIM_FFTVEC_MUL_PRECOMP* p[31] = {0};
  REIM_FFTVEC_MUL_PRECOMP** f = p + log2m(m);
  if (!*f) *f = new_reim_fftvec_mul_precomp(m);
  (*f)->function(*f, r, a, b);
}

EXPORT void reim_fftvec_addmul_simple(uint32_t m, void* r, const void* a, const void* b) {
  static REIM_FFTVEC_ADDMUL_PRECOMP* p[31] = {0};
  REIM_FFTVEC_ADDMUL_PRECOMP** f = p + log2m(m);
  if (!*f) *f = new_reim_fftvec_addmul_precomp(m);
  (*f)->function(*f, r, a, b);
}