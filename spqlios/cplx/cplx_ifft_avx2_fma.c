#include <immintrin.h>
#include <string.h>

#include "cplx_fft.h"
#include "cplx_fft_private.h"

typedef double D4MEM[4];

/**
 * @brief complex ifft via bfs strategy (for m between 2 and 8)
 * @param dat the data to run the algorithm on
 * @param omg precomputed tables (must have been filled with fill_omega)
 * @param m ring dimension of the FFT (modulo X^m-i)
 */
void cplx_ifft_avx2_fma_bfs_2(D4MEM* dat, const D4MEM** omg, uint32_t m) {
  double* data = (double*)dat;
  D4MEM* const finaldd = (D4MEM*)(data + 2 * m);
  {
    // loop with h = 1
    // we do not do any particular optimization in this loop,
    // since this function is only called for small dimensions
    D4MEM* dd = (D4MEM*)data;
    do {
      /*
  BEGIN_TEMPLATE
      const __m256d ab% = _mm256_load_pd(dd[0+2*%]);
      const __m256d cd% = _mm256_load_pd(dd[1+2*%]);
      const __m256d ac% = _mm256_permute2f128_pd(ab%, cd%, 0b100000);
      const __m256d bd% = _mm256_permute2f128_pd(ab%, cd%, 0b110001);
      const __m256d sum% = _mm256_add_pd(ac%, bd%);
      const __m256d diff% = _mm256_sub_pd(ac%, bd%);
      const __m256d diffbar% = _mm256_shuffle_pd(diff%, diff%, 5);
      const __m256d om% = _mm256_load_pd((*omg)[0+%]);
      const __m256d omre% = _mm256_unpacklo_pd(om%, om%);
      const __m256d omim% = _mm256_unpackhi_pd(om%, om%);
      const __m256d t1% = _mm256_mul_pd(diffbar%, omim%);
      const __m256d t2% = _mm256_fmaddsub_pd(diff%, omre%, t1%);
      const __m256d newab% = _mm256_permute2f128_pd(sum%, t2%, 0b100000);
      const __m256d newcd% = _mm256_permute2f128_pd(sum%, t2%, 0b110001);
      _mm256_store_pd(dd[0+2*%], newab%);
      _mm256_store_pd(dd[1+2*%], newcd%);
      dd += 2*@;
      *omg += @;
  END_TEMPLATE
     */
      // BEGIN_INTERLEAVE 1
      const __m256d ab0 = _mm256_load_pd(dd[0 + 2 * 0]);
      const __m256d cd0 = _mm256_load_pd(dd[1 + 2 * 0]);
      const __m256d ac0 = _mm256_permute2f128_pd(ab0, cd0, 0b100000);
      const __m256d bd0 = _mm256_permute2f128_pd(ab0, cd0, 0b110001);
      const __m256d sum0 = _mm256_add_pd(ac0, bd0);
      const __m256d diff0 = _mm256_sub_pd(ac0, bd0);
      const __m256d diffbar0 = _mm256_shuffle_pd(diff0, diff0, 5);
      const __m256d om0 = _mm256_load_pd((*omg)[0 + 0]);
      const __m256d omre0 = _mm256_unpacklo_pd(om0, om0);
      const __m256d omim0 = _mm256_unpackhi_pd(om0, om0);
      const __m256d t10 = _mm256_mul_pd(diffbar0, omim0);
      const __m256d t20 = _mm256_fmaddsub_pd(diff0, omre0, t10);
      const __m256d newab0 = _mm256_permute2f128_pd(sum0, t20, 0b100000);
      const __m256d newcd0 = _mm256_permute2f128_pd(sum0, t20, 0b110001);
      _mm256_store_pd(dd[0 + 2 * 0], newab0);
      _mm256_store_pd(dd[1 + 2 * 0], newcd0);
      dd += 2 * 1;
      *omg += 1;
      // END_INTERLEAVE
    } while (dd < finaldd);
#if 0
 printf("c after first: ");
 for (uint64_t ii=0; ii<nn/2; ++ii) {
   printf("%.6lf %.6lf ",ddata[ii][0],ddata[ii][1]);
 }
 printf("\n");
#endif
  }
  // general case
  const uint32_t ms2 = m >> 1;
  for (uint32_t _2nblock = 2; _2nblock <= ms2; _2nblock <<= 1) {
    // _2nblock = h in ref code
    uint32_t nblock = _2nblock >> 1;  // =h/2 in ref code
    D4MEM* dd = (D4MEM*)data;
    do {
      const __m256d om = _mm256_load_pd((*omg)[0]);
      const __m256d omre = _mm256_unpacklo_pd(om, om);
      const __m256d omim = _mm256_addsub_pd(_mm256_setzero_pd(), _mm256_unpackhi_pd(om, om));
      D4MEM* const ddend = (dd + nblock);
      D4MEM* ddmid = ddend;
      do {
        const __m256d a = _mm256_load_pd(dd[0]);
        const __m256d b = _mm256_load_pd(ddmid[0]);
        const __m256d newa = _mm256_add_pd(a, b);
        _mm256_store_pd(dd[0], newa);
        const __m256d diff = _mm256_sub_pd(a, b);
        const __m256d t1 = _mm256_mul_pd(diff, omre);
        const __m256d bardiff = _mm256_shuffle_pd(diff, diff, 5);
        const __m256d t2 = _mm256_fmadd_pd(bardiff, omim, t1);
        _mm256_store_pd(ddmid[0], t2);
        dd += 1;
        ddmid += 1;
      } while (dd < ddend);
      dd += nblock;
      *omg += 1;
    } while (dd < finaldd);
  }
}

/**
 * @brief complex fft via bfs strategy (for m >= 16)
 * @param dat the data to run the algorithm on
 * @param omg precomputed tables (must have been filled with fill_omega)
 * @param m ring dimension of the FFT (modulo X^m-i)
 */
void cplx_ifft_avx2_fma_bfs_16(D4MEM* dat, const D4MEM** omg, uint32_t m) {
  double* data = (double*)dat;
  D4MEM* const finaldd = (D4MEM*)(data + 2 * m);
  // base iteration when h = _2nblock == 8
  {
    D4MEM* dd = (D4MEM*)data;
    do {
      cplx_ifft16_avx_fma(dd, *omg);
      dd += 8;
      *omg += 4;
    } while (dd < finaldd);
  }
  // general case
  const uint32_t ms2 = m >> 1;
  for (uint32_t _2nblock = 16; _2nblock <= ms2; _2nblock <<= 1) {
    // _2nblock = h in ref code
    uint32_t nblock = _2nblock >> 1;  // =h/2 in ref code
    D4MEM* dd = (D4MEM*)data;
    do {
      const __m256d om = _mm256_load_pd((*omg)[0]);
      const __m256d omre = _mm256_unpacklo_pd(om, om);
      const __m256d omim = _mm256_addsub_pd(_mm256_setzero_pd(), _mm256_unpackhi_pd(om, om));
      D4MEM* const ddend = (dd + nblock);
      D4MEM* ddmid = ddend;
      do {
        const __m256d a = _mm256_loadu_pd(dd[0]);
        const __m256d b = _mm256_loadu_pd(ddmid[0]);
        const __m256d newa = _mm256_add_pd(a, b);
        _mm256_storeu_pd(dd[0], newa);
        const __m256d diff = _mm256_sub_pd(a, b);
        const __m256d t1 = _mm256_mul_pd(diff, omre);
        const __m256d bardiff = _mm256_shuffle_pd(diff, diff, 5);
        const __m256d t2 = _mm256_fmadd_pd(bardiff, omim, t1);
        _mm256_storeu_pd(ddmid[0], t2);
        dd += 1;
        ddmid += 1;
      } while (dd < ddend);
      dd += nblock;
      *omg += 1;
    } while (dd < finaldd);
  }
}

/**
 * @brief complex ifft via dfs recursion (for m >= 16)
 * @param dat the data to run the algorithm on
 * @param omg precomputed tables (must have been filled with fill_omega)
 * @param m ring dimension of the FFT (modulo X^m-i)
 */
void cplx_ifft_avx2_fma_rec_16(D4MEM* dat, const D4MEM** omg, uint32_t m) {
  if (m <= 8) return cplx_ifft_avx2_fma_bfs_2(dat, omg, m);
  if (m <= 2048) return cplx_ifft_avx2_fma_bfs_16(dat, omg, m);
  const uint32_t _2nblock = m >> 1;       // = h in ref code
  const uint32_t nblock = _2nblock >> 1;  // =h/2 in ref code
  cplx_ifft_avx2_fma_rec_16(dat, omg, _2nblock);
  cplx_ifft_avx2_fma_rec_16(dat + nblock, omg, _2nblock);
  {
    // final iteration
    D4MEM* dd = dat;
    const __m256d om = _mm256_load_pd((*omg)[0]);
    const __m256d omre = _mm256_unpacklo_pd(om, om);
    const __m256d omim = _mm256_addsub_pd(_mm256_setzero_pd(), _mm256_unpackhi_pd(om, om));
    D4MEM* const ddend = (dd + nblock);
    D4MEM* ddmid = ddend;
    do {
      const __m256d a = _mm256_loadu_pd(dd[0]);
      const __m256d b = _mm256_loadu_pd(ddmid[0]);
      const __m256d newa = _mm256_add_pd(a, b);
      _mm256_store_pd(dd[0], newa);
      const __m256d diff = _mm256_sub_pd(a, b);
      const __m256d t1 = _mm256_mul_pd(diff, omre);
      const __m256d bardiff = _mm256_shuffle_pd(diff, diff, 5);
      const __m256d t2 = _mm256_fmadd_pd(bardiff, omim, t1);
      _mm256_storeu_pd(ddmid[0], t2);
      dd += 1;
      ddmid += 1;
    } while (dd < ddend);
    *omg += 1;
  }
}

/**
 * @brief complex ifft via best strategy (for m>=1)
 * @param dat the data to run the algorithm on: m complex numbers
 * @param omg precomputed tables (must have been filled with fill_omega)
 * @param m ring dimension of the FFT (modulo X^m-i)
 */
EXPORT void cplx_ifft_avx2_fma(const CPLX_IFFT_PRECOMP* precomp, void* d) {
  const uint32_t m = precomp->m;
  const D4MEM* omg = (D4MEM*)precomp->powomegas;
  if (m <= 1) return;
  if (m <= 8) return cplx_ifft_avx2_fma_bfs_2(d, &omg, m);
  if (m <= 2048) return cplx_ifft_avx2_fma_bfs_16(d, &omg, m);
  cplx_ifft_avx2_fma_rec_16(d, &omg, m);
}
