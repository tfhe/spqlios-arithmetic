#include <immintrin.h>

#include "coeffs_arithmetic.h"

// res = a + b. dimension n must be a power of 2
EXPORT void znx_add_i64_avx(uint64_t nn, int64_t* res, const int64_t* a, const int64_t* b) {
  if (nn <= 2) {
    if (nn == 1) {
      res[0] = a[0] + b[0];
    } else {
      _mm_storeu_si128((__m128i*)res,                     //
                       _mm_add_epi64(                     //
                           _mm_loadu_si128((__m128i*)a),  //
                           _mm_loadu_si128((__m128i*)b)));
    }
  } else {
    const __m256i* aa = (__m256i*)a;
    const __m256i* bb = (__m256i*)b;
    __m256i* rr = (__m256i*)res;
    __m256i* const rrend = (__m256i*)(res + nn);
    do {
      _mm256_storeu_si256(rr,                          //
                          _mm256_add_epi64(            //
                              _mm256_loadu_si256(aa),  //
                              _mm256_loadu_si256(bb)));
      ++rr;
      ++aa;
      ++bb;
    } while (rr < rrend);
  }
}

// res = a - b. dimension n must be a power of 2
EXPORT void znx_sub_i64_avx(uint64_t nn, int64_t* res, const int64_t* a, const int64_t* b) {
  if (nn <= 2) {
    if (nn == 1) {
      res[0] = a[0] - b[0];
    } else {
      _mm_storeu_si128((__m128i*)res,                     //
                       _mm_sub_epi64(                     //
                           _mm_loadu_si128((__m128i*)a),  //
                           _mm_loadu_si128((__m128i*)b)));
    }
  } else {
    const __m256i* aa = (__m256i*)a;
    const __m256i* bb = (__m256i*)b;
    __m256i* rr = (__m256i*)res;
    __m256i* const rrend = (__m256i*)(res + nn);
    do {
      _mm256_storeu_si256(rr,                          //
                          _mm256_sub_epi64(            //
                              _mm256_loadu_si256(aa),  //
                              _mm256_loadu_si256(bb)));
      ++rr;
      ++aa;
      ++bb;
    } while (rr < rrend);
  }
}

EXPORT void znx_negate_i64_avx(uint64_t nn, int64_t* res, const int64_t* a) {
  if (nn <= 2) {
    if (nn == 1) {
      res[0] = -a[0];
    } else {
      _mm_storeu_si128((__m128i*)res,           //
                       _mm_sub_epi64(           //
                           _mm_set1_epi64x(0),  //
                           _mm_loadu_si128((__m128i*)a)));
    }
  } else {
    const __m256i* aa = (__m256i*)a;
    __m256i* rr = (__m256i*)res;
    __m256i* const rrend = (__m256i*)(res + nn);
    do {
      _mm256_storeu_si256(rr,                         //
                          _mm256_sub_epi64(           //
                              _mm256_set1_epi64x(0),  //
                              _mm256_loadu_si256(aa)));
      ++rr;
      ++aa;
    } while (rr < rrend);
  }
}
