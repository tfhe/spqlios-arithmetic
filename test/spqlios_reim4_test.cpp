#include <cmath>

#include "gtest/gtest.h"
#include "spqlios/reim4.h"
#include "spqlios/reim4/reim4_fftvec_private.h"
#include "spqlios/reim4/reim4_fftvec.h"
#include "spqlios/cplx/cplx_fft.h"

#ifdef __x86_64__
TEST(fft, reim4_vecfft_addmul_fma_vs_ref) {
  for (uint64_t nn : {16, 32, 64, 1024, 8192, 65536}) {
    uint64_t m = nn / 2;
    REIM4_FFTVEC_ADDMUL_PRECOMP* tbl = new_reim4_fftvec_addmul_precomp(m);
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
    reim4_fftvec_addmul_ref(tbl, r1, a1, b1);
    reim4_fftvec_addmul_fma(tbl, r2, a2, b2);
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
    delete_reim4_fftvec_addmul_precomp(tbl);
  }
}
#endif

#ifdef __x86_64__
TEST(fft, reim4_vecfft_mul_fma_vs_ref) {
  for (uint64_t nn : {16, 32, 64, 1024, 8192, 65536}) {
    uint64_t m = nn / 2;
    REIM4_FFTVEC_MUL_PRECOMP* tbl = new_reim4_fftvec_mul_precomp(m);
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
    reim4_fftvec_mul_ref(tbl, r1, a1, b1);
    reim4_fftvec_mul_fma(tbl, r2, a2, b2);
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
    delete_reim4_fftvec_mul_precomp(tbl);
  }
}
#endif
