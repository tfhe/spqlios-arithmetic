#include <immintrin.h>

#include "cplx_fft.h"
#include "cplx_fft_private.h"

EXPORT void cplx_fftvec_addmul_fma(const CPLX_FFTVEC_ADDMUL_PRECOMP* precomp, void* r, const void* a, const void* b) {
  const uint32_t m = precomp->m;
  const double(*aa)[4] = (double(*)[4])a;
  const double(*bb)[4] = (double(*)[4])b;
  double(*rr)[4] = (double(*)[4])r;
  const double(*const aend)[4] = aa + (m >> 1);
  do {
    /*
BEGIN_TEMPLATE
const __m256d ari% = _mm256_loadu_pd(aa[%]);
const __m256d bri% = _mm256_loadu_pd(bb[%]);
const __m256d rri% = _mm256_loadu_pd(rr[%]);
const __m256d bir% = _mm256_shuffle_pd(bri%,bri%, 5);
const __m256d aii% = _mm256_shuffle_pd(ari%,ari%, 15);
const __m256d pro% = _mm256_fmaddsub_pd(aii%,bir%,rri%);
const __m256d arr% = _mm256_shuffle_pd(ari%,ari%, 0);
const __m256d res% = _mm256_fmaddsub_pd(arr%,bri%,pro%);
_mm256_storeu_pd(rr[%],res%);
rr += @; // ONCE
aa += @; // ONCE
bb += @; // ONCE
END_TEMPLATE
     */
    // BEGIN_INTERLEAVE 2
    // This block is automatically generated from the template above
    // by the interleave.pl script. Please do not edit by hand
    const __m256d ari0 = _mm256_loadu_pd(aa[0]);
    const __m256d ari1 = _mm256_loadu_pd(aa[1]);
    const __m256d bri0 = _mm256_loadu_pd(bb[0]);
    const __m256d bri1 = _mm256_loadu_pd(bb[1]);
    const __m256d rri0 = _mm256_loadu_pd(rr[0]);
    const __m256d rri1 = _mm256_loadu_pd(rr[1]);
    const __m256d bir0 = _mm256_shuffle_pd(bri0, bri0, 5);
    const __m256d bir1 = _mm256_shuffle_pd(bri1, bri1, 5);
    const __m256d aii0 = _mm256_shuffle_pd(ari0, ari0, 15);
    const __m256d aii1 = _mm256_shuffle_pd(ari1, ari1, 15);
    const __m256d pro0 = _mm256_fmaddsub_pd(aii0, bir0, rri0);
    const __m256d pro1 = _mm256_fmaddsub_pd(aii1, bir1, rri1);
    const __m256d arr0 = _mm256_shuffle_pd(ari0, ari0, 0);
    const __m256d arr1 = _mm256_shuffle_pd(ari1, ari1, 0);
    const __m256d res0 = _mm256_fmaddsub_pd(arr0, bri0, pro0);
    const __m256d res1 = _mm256_fmaddsub_pd(arr1, bri1, pro1);
    _mm256_storeu_pd(rr[0], res0);
    _mm256_storeu_pd(rr[1], res1);
    rr += 2;  // ONCE
    aa += 2;  // ONCE
    bb += 2;  // ONCE
              // END_INTERLEAVE
  } while (aa < aend);
}

EXPORT void cplx_fftvec_mul_fma(const CPLX_FFTVEC_MUL_PRECOMP* precomp, void* r, const void* a, const void* b) {
  const uint32_t m = precomp->m;
  const double(*aa)[4] = (double(*)[4])a;
  const double(*bb)[4] = (double(*)[4])b;
  double(*rr)[4] = (double(*)[4])r;
  const double(*const aend)[4] = aa + (m >> 1);
  do {
    /*
BEGIN_TEMPLATE
const __m256d ari% = _mm256_loadu_pd(aa[%]);
const __m256d bri% = _mm256_loadu_pd(bb[%]);
const __m256d bir% = _mm256_shuffle_pd(bri%,bri%, 5); // conj of b
const __m256d aii% = _mm256_shuffle_pd(ari%,ari%, 15); // im of a
const __m256d pro% = _mm256_mul_pd(aii%,bir%);
const __m256d arr% = _mm256_shuffle_pd(ari%,ari%, 0); // rr of a
const __m256d res% = _mm256_fmaddsub_pd(arr%,bri%,pro%);
_mm256_storeu_pd(rr[%],res%);
rr += @;  // ONCE
aa += @;  // ONCE
bb += @;  // ONCE
END_TEMPLATE
     */
    // BEGIN_INTERLEAVE 4
    // This block is automatically generated from the template above
    // by the interleave.pl script. Please do not edit by hand
    const __m256d ari0 = _mm256_loadu_pd(aa[0]);
    const __m256d ari1 = _mm256_loadu_pd(aa[1]);
    const __m256d ari2 = _mm256_loadu_pd(aa[2]);
    const __m256d ari3 = _mm256_loadu_pd(aa[3]);
    const __m256d bri0 = _mm256_loadu_pd(bb[0]);
    const __m256d bri1 = _mm256_loadu_pd(bb[1]);
    const __m256d bri2 = _mm256_loadu_pd(bb[2]);
    const __m256d bri3 = _mm256_loadu_pd(bb[3]);
    const __m256d bir0 = _mm256_shuffle_pd(bri0, bri0, 5);   // conj of b
    const __m256d bir1 = _mm256_shuffle_pd(bri1, bri1, 5);   // conj of b
    const __m256d bir2 = _mm256_shuffle_pd(bri2, bri2, 5);   // conj of b
    const __m256d bir3 = _mm256_shuffle_pd(bri3, bri3, 5);   // conj of b
    const __m256d aii0 = _mm256_shuffle_pd(ari0, ari0, 15);  // im of a
    const __m256d aii1 = _mm256_shuffle_pd(ari1, ari1, 15);  // im of a
    const __m256d aii2 = _mm256_shuffle_pd(ari2, ari2, 15);  // im of a
    const __m256d aii3 = _mm256_shuffle_pd(ari3, ari3, 15);  // im of a
    const __m256d pro0 = _mm256_mul_pd(aii0, bir0);
    const __m256d pro1 = _mm256_mul_pd(aii1, bir1);
    const __m256d pro2 = _mm256_mul_pd(aii2, bir2);
    const __m256d pro3 = _mm256_mul_pd(aii3, bir3);
    const __m256d arr0 = _mm256_shuffle_pd(ari0, ari0, 0);  // rr of a
    const __m256d arr1 = _mm256_shuffle_pd(ari1, ari1, 0);  // rr of a
    const __m256d arr2 = _mm256_shuffle_pd(ari2, ari2, 0);  // rr of a
    const __m256d arr3 = _mm256_shuffle_pd(ari3, ari3, 0);  // rr of a
    const __m256d res0 = _mm256_fmaddsub_pd(arr0, bri0, pro0);
    const __m256d res1 = _mm256_fmaddsub_pd(arr1, bri1, pro1);
    const __m256d res2 = _mm256_fmaddsub_pd(arr2, bri2, pro2);
    const __m256d res3 = _mm256_fmaddsub_pd(arr3, bri3, pro3);
    _mm256_storeu_pd(rr[0], res0);
    _mm256_storeu_pd(rr[1], res1);
    _mm256_storeu_pd(rr[2], res2);
    _mm256_storeu_pd(rr[3], res3);
    rr += 4;  // ONCE
    aa += 4;  // ONCE
    bb += 4;  // ONCE
              // END_INTERLEAVE
  } while (aa < aend);
}
