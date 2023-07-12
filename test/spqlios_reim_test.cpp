#include <cmath>

#include "gtest/gtest.h"
#include "spqlios/reim.h"
#include "spqlios/reim/reim_fft_private.h"
#include "spqlios/reim/reim_fft.h"
#include "spqlios/cplx/cplx_fft.h"
#include "spqlios/commons_private.h"



#ifdef __x86_64__
TEST(fft, reim_fft_avx2_vs_fft_reim_ref) {
  for (uint64_t nn : {16, 32, 64, 1024, 8192, 65536}) {
    uint64_t m = nn / 2;
    // CPLX_FFT_PRECOMP* tables = new_cplx_fft_precomp(m, 0);
    REIM_FFT_PRECOMP* reimtables = new_reim_fft_precomp(m, 0);
    CPLX* a = (CPLX*)aligned_alloc(32, nn / 2 * sizeof(CPLX));
    double* a1 = (double*)aligned_alloc(32, nn / 2 * sizeof(CPLX));
    double* a2 = (double*)aligned_alloc(32, nn / 2 * sizeof(CPLX));
    int64_t p = 1 << 16;
    for (uint32_t i = 0; i < nn / 2; i++) {
      a[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
      a[i][1] = (rand() % p) - p / 2;
    }
    memcpy(a1, a, nn / 2 * sizeof(CPLX));
    memcpy(a2, a, nn / 2 * sizeof(CPLX));
    reim_fft_ref(reimtables, a2);
    reim_fft_avx2_fma(reimtables, a1);
    double d = 0;
    for (uint32_t i = 0; i < nn / 2; i++) {
      double dre = fabs(a1[i] - a2[i]);
      double dim = fabs(a1[nn / 2 + i] - a2[nn / 2 + i]);
      if (dre > d) d = dre;
      if (dim > d) d = dim;
      ASSERT_LE(d, 1e-8);
    }
    ASSERT_LE(d, 1e-8);
    free(a);
    free(a1);
    free(a2);
    // delete_cplx_fft_precomp(tables);
    delete_reim_fft_precomp(reimtables);
  }
}
#endif

#ifdef __x86_64__
TEST(fft, reim_ifft_avx2_vs_reim_ifft_ref) {
  for (uint64_t nn : {16, 32, 64, 1024, 8192, 65536}) {
    uint64_t m = nn / 2;
    // CPLX_FFT_PRECOMP* tables = new_cplx_fft_precomp(m, 0);
    REIM_IFFT_PRECOMP* reimtables = new_reim_ifft_precomp(m, 0);
    CPLX* a = (CPLX*)aligned_alloc(32, nn / 2 * sizeof(CPLX));
    double* a1 = (double*)aligned_alloc(32, nn / 2 * sizeof(CPLX));
    double* a2 = (double*)aligned_alloc(32, nn / 2 * sizeof(CPLX));
    int64_t p = 1 << 16;
    for (uint32_t i = 0; i < nn / 2; i++) {
      a[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
      a[i][1] = (rand() % p) - p / 2;
    }
    memcpy(a1, a, nn / 2 * sizeof(CPLX));
    memcpy(a2, a, nn / 2 * sizeof(CPLX));
    reim_ifft_ref(reimtables, a2);
    reim_ifft_avx2_fma(reimtables, a1);
    double d = 0;
    for (uint32_t i = 0; i < nn / 2; i++) {
      double dre = fabs(a1[i] - a2[i]);
      double dim = fabs(a1[nn / 2 + i] - a2[nn / 2 + i]);
      if (dre > d) d = dre;
      if (dim > d) d = dim;
      ASSERT_LE(d, 1e-8);
    }
    ASSERT_LE(d, 1e-8);
    free(a);
    free(a1);
    free(a2);
    // delete_cplx_fft_precomp(tables);
    delete_reim_fft_precomp(reimtables);
  }
}
#endif

#ifdef __x86_64__
TEST(fft, reim_vecfft_addmul_fma_vs_ref) {
  for (uint64_t nn : {16, 32, 64, 1024, 8192, 65536}) {
    uint64_t m = nn / 2;
    REIM_FFTVEC_ADDMUL_PRECOMP* tbl = new_reim_fftvec_addmul_precomp(m);
    ASSERT_TRUE(tbl != nullptr);
    double* a1 = (double*)aligned_alloc(32, nn / 2 * sizeof(CPLX));
    double* a2 = (double*)aligned_alloc(32, nn / 2 * sizeof(CPLX));
    double* b1 = (double*)aligned_alloc(32, nn / 2 * sizeof(CPLX));
    double* b2 = (double*)aligned_alloc(32, nn / 2 * sizeof(CPLX));
    double* r1 = (double*)aligned_alloc(32, nn / 2 * sizeof(CPLX));
    double* r2 = (double*)aligned_alloc(32, nn / 2 * sizeof(CPLX));
    int64_t p = 1 << 16;
    for (uint32_t i = 0; i < nn; i++) {
      a1[i] = (rand() % p) - p / 2;  // between -p/2 and p/2
      b1[i] = (rand() % p) - p / 2;
      r1[i] = (rand() % p) - p / 2;
    }
    memcpy(a2, a1, nn / 2 * sizeof(CPLX));
    memcpy(b2, b1, nn / 2 * sizeof(CPLX));
    memcpy(r2, r1, nn / 2 * sizeof(CPLX));
    reim_fftvec_addmul_ref(tbl, r1, a1, b1);
    reim_fftvec_addmul_fma(tbl, r2, a2, b2);
    double d = 0;
    for (uint32_t i = 0; i < nn; i++) {
      double di = fabs(r1[i] - r2[i]);
      if (di > d) d = di;
      ASSERT_LE(d, 1e-8);
    }
    ASSERT_LE(d, 1e-8);
    free(a1);
    free(a2);
    free(b1);
    free(b2);
    free(r1);
    free(r2);
    delete_reim_fftvec_addmul_precomp(tbl);
  }
}
#endif

#ifdef __x86_64__
TEST(fft, reim_vecfft_mul_fma_vs_ref) {
  for (uint64_t nn : {16, 32, 64, 1024, 8192, 65536}) {
    uint64_t m = nn / 2;
    REIM_FFTVEC_MUL_PRECOMP* tbl = new_reim_fftvec_mul_precomp(m);
    double* a1 = (double*)aligned_alloc(32, nn / 2 * sizeof(CPLX));
    double* a2 = (double*)aligned_alloc(32, nn / 2 * sizeof(CPLX));
    double* b1 = (double*)aligned_alloc(32, nn / 2 * sizeof(CPLX));
    double* b2 = (double*)aligned_alloc(32, nn / 2 * sizeof(CPLX));
    double* r1 = (double*)aligned_alloc(32, nn / 2 * sizeof(CPLX));
    double* r2 = (double*)aligned_alloc(32, nn / 2 * sizeof(CPLX));
    int64_t p = 1 << 16;
    for (uint32_t i = 0; i < nn; i++) {
      a1[i] = (rand() % p) - p / 2;  // between -p/2 and p/2
      b1[i] = (rand() % p) - p / 2;
      r1[i] = (rand() % p) - p / 2;
    }
    memcpy(a2, a1, nn / 2 * sizeof(CPLX));
    memcpy(b2, b1, nn / 2 * sizeof(CPLX));
    memcpy(r2, r1, nn / 2 * sizeof(CPLX));
    reim_fftvec_mul_ref(tbl, r1, a1, b1);
    reim_fftvec_mul_fma(tbl, r2, a2, b2);
    double d = 0;
    for (uint32_t i = 0; i < nn; i++) {
      double di = fabs(r1[i] - r2[i]);
      if (di > d) d = di;
      ASSERT_LE(d, 1e-8);
    }
    ASSERT_LE(d, 1e-8);
    free(a1);
    free(a2);
    free(b1);
    free(b2);
    free(r1);
    free(r2);
    delete_reim_fftvec_mul_precomp(tbl);
  }
}
#endif
