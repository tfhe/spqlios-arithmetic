#include <gtest/gtest.h>

#include <cstdint>

#include "../spqlios/arithmetic/vec_znx_arithmetic_private.h"
#include "spqlios/arithmetic/vec_znx_arithmetic.h"
#include "test/testlib/ntt120_dft.h"
#include "test/testlib/ntt120_layouts.h"
#include "testlib/fft64_dft.h"
#include "testlib/fft64_layouts.h"
#include "testlib/polynomial_vector.h"

static int64_t random_automorphism_param() { return uniform_i64() | 1; }

static void test_fft64_vec_znx_dft_automorphism(VEC_ZNX_DFT_AUTOMORPHISM_F func){
  const uint64_t nn = 32;
  MODULE* module = new_module_info(nn, FFT64);

  uint64_t tmp_space_size = fft64_vec_znx_big_normalize_base2k_tmp_bytes(module) | fft64_vec_znx_dft_automorphism_tmp_bytes(module) | fft64_vec_znx_idft_tmp_bytes(module);
  uint8_t* tmp_space = (uint8_t*)spqlios_alloc(tmp_space_size);
  const double scale = 32768.0;

  for (uint64_t cols_a : {3, 5, 8}) {
    for (uint64_t cols_b : {3, 5, 8}) {

      std::cout << cols_a << " " << cols_b <<std::endl;

      int64_t p = random_automorphism_param();

      fft64_vec_znx_dft_layout a_dft(nn, cols_a);
      fft64_vec_znx_dft_layout b0_dft_auto(nn, cols_b);
      fft64_vec_znx_big_layout b1_big_auto(nn, cols_b);
      fft64_vec_znx_dft_layout b1_dft_auto(nn, cols_b);

      // Sets a_dft directly
      double* a_dft_doubles = (double*)a_dft.data;
      for (uint64_t i = 0; i < cols_a; ++i) {
        for (uint64_t j = 0; j < nn; ++j){
          a_dft_doubles[i*nn+j] = (i+j)*scale;
        }
      }

      // Hash before apply function to test
      thash hash_before = a_dft.content_hash();

      // b0_dft_auto <- AUTO(a_dft)
      func(module, p, b0_dft_auto.data, cols_b, a_dft.data, cols_a, tmp_space);

      // Checks a_dft is unchanged
      ASSERT_EQ(a_dft.content_hash(), hash_before);

      // b1_dft_auto = DFT(AUTO(IDFT(a_dft)))
      fft64_vec_znx_idft(module, b1_big_auto.data, cols_b, a_dft.data, cols_b, tmp_space);
      fft64_vec_znx_big_automorphism(module, p, b1_big_auto.data, cols_b, b1_big_auto.data, cols_b);
      fft64_vec_znx_dft(module, b1_dft_auto.data, cols_b, (int64_t*)b1_big_auto.data, cols_b, nn);

      // Checks b0_dft_auto = b1_dft_auto
      for (uint64_t i = 0; i < cols_b; ++i) {
        ASSERT_LE(infty_dist(b0_dft_auto.get_copy_zext(i), b1_dft_auto.get_copy_zext(i)), 10);
      }
    }
  }
  delete_module_info(module);
}

TEST(vec_znx_dft, fft64_vec_znx_dft_automorphism_ref) { test_fft64_vec_znx_dft_automorphism(fft64_vec_znx_dft_automorphism_ref); }

static void test_fft64_vec_znx_dft(VEC_ZNX_DFT_F dft) {
  for (uint64_t n : {2, 4, 128}) {
    MODULE* module = new_module_info(n, FFT64);
    for (uint64_t sa : {3, 5, 8}) {
      for (uint64_t sr : {3, 5, 8}) {
        uint64_t a_sl = n + uniform_u64_bits(2);
        znx_vec_i64_layout a(n, sa, a_sl);
        fft64_vec_znx_dft_layout res(n, sr);
        a.fill_random(42);
        std::vector<reim_fft64vec> expect(sr);
        for (uint64_t i = 0; i < sr; ++i) {
          expect[i] = simple_fft64(a.get_copy_zext(i));
        }
        // test the function
        thash hash_before = a.content_hash();
        dft(module, res.data, sr, a.data(), sa, a_sl);
        ASSERT_EQ(a.content_hash(), hash_before);
        for (uint64_t i = 0; i < sr; ++i) {
          reim_fft64vec actual = res.get_copy_zext(i);
          ASSERT_LE(infty_dist(actual, expect[i]), 1e-10);
        }
      }
    }
    delete_module_info(module);
  }
}

#ifdef __x86_64__
// FIXME: currently, it only works on avx
static void test_ntt120_vec_znx_dft(VEC_ZNX_DFT_F dft) {
  for (uint64_t n : {2, 4, 128}) {
    MODULE* module = new_module_info(n, NTT120);
    for (uint64_t sa : {3, 5, 8}) {
      for (uint64_t sr : {3, 5, 8}) {
        uint64_t a_sl = n + uniform_u64_bits(2);
        znx_vec_i64_layout a(n, sa, a_sl);
        ntt120_vec_znx_dft_layout res(n, sr);
        a.fill_random(42);
        std::vector<q120_nttvec> expect(sr);
        for (uint64_t i = 0; i < sr; ++i) {
          expect[i] = simple_ntt120(a.get_copy_zext(i));
        }
        // test the function
        thash hash_before = a.content_hash();
        dft(module, res.data, sr, a.data(), sa, a_sl);
        ASSERT_EQ(a.content_hash(), hash_before);
        for (uint64_t i = 0; i < sr; ++i) {
          q120_nttvec actual = res.get_copy_zext(i);
          if (!(actual == expect[i])) {
            for (uint64_t j = 0; j < n; ++j) {
              std::cerr << actual.v[j] << " vs " << expect[i].v[j] << std::endl;
            }
          }
          ASSERT_EQ(actual, expect[i]);
        }
      }
    }
    delete_module_info(module);
  }
}
#endif

TEST(vec_znx_dft, fft64_vec_znx_dft) { test_fft64_vec_znx_dft(fft64_vec_znx_dft); }
#ifdef __x86_64__
// FIXME: currently, it only works on avx
TEST(vec_znx_dft, ntt120_vec_znx_dft) { test_ntt120_vec_znx_dft(ntt120_vec_znx_dft_avx); }
#endif
TEST(vec_znx_dft, vec_znx_dft) {
  test_fft64_vec_znx_dft(vec_znx_dft);
#ifdef __x86_64__
  // FIXME: currently, it only works on avx
  test_ntt120_vec_znx_dft(ntt120_vec_znx_dft_avx);
#endif
}

static void test_fft64_vec_znx_idft(VEC_ZNX_IDFT_F idft, VEC_ZNX_IDFT_TMP_A_F idft_tmp_a,
                                    VEC_ZNX_IDFT_TMP_BYTES_F idft_tmp_bytes) {
  for (uint64_t n : {2, 4, 64, 128}) {
    MODULE* module = new_module_info(n, FFT64);
    uint64_t tmp_size = idft_tmp_bytes ? idft_tmp_bytes(module) : 0;
    std::vector<uint8_t> tmp(tmp_size);
    for (uint64_t sa : {3, 5, 8}) {
      for (uint64_t sr : {3, 5, 8}) {
        fft64_vec_znx_dft_layout a(n, sa);
        fft64_vec_znx_big_layout res(n, sr);
        a.fill_dft_random_log2bound(22);
        std::vector<znx_i64> expect(sr);
        for (uint64_t i = 0; i < sr; ++i) {
          expect[i] = simple_rint_ifft64(a.get_copy_zext(i));
        }
        // test the function
        if (idft_tmp_bytes) {
          thash hash_before = a.content_hash();
          idft(module, res.data, sr, a.data, sa, tmp.data());
          ASSERT_EQ(a.content_hash(), hash_before);
        } else {
          idft_tmp_a(module, res.data, sr, a.data, sa);
        }
        for (uint64_t i = 0; i < sr; ++i) {
          znx_i64 actual = res.get_copy_zext(i);
          // ASSERT_EQ(res.get_copy_zext(i), expect[i]);
          if (!(actual == expect[i])) {
            for (uint64_t j = 0; j < n; ++j) {
              std::cerr << actual.get_coeff(j) << " dft vs. " << expect[i].get_coeff(j) << std::endl;
            }
            FAIL();
          }
        }
      }
    }
    delete_module_info(module);
  }
}

TEST(vec_znx_dft, fft64_vec_znx_idft) {
  test_fft64_vec_znx_idft(fft64_vec_znx_idft, nullptr, fft64_vec_znx_idft_tmp_bytes);
}
TEST(vec_znx_dft, fft64_vec_znx_idft_tmp_a) { test_fft64_vec_znx_idft(nullptr, fft64_vec_znx_idft_tmp_a, nullptr); }

#ifdef __x86_64__
// FIXME: currently, it only works on avx
static void test_ntt120_vec_znx_idft(VEC_ZNX_IDFT_F idft, VEC_ZNX_IDFT_TMP_A_F idft_tmp_a,
                                     VEC_ZNX_IDFT_TMP_BYTES_F idft_tmp_bytes) {
  for (uint64_t n : {2, 4, 64, 128}) {
    MODULE* module = new_module_info(n, NTT120);
    uint64_t tmp_size = idft_tmp_bytes ? idft_tmp_bytes(module) : 0;
    std::vector<uint8_t> tmp(tmp_size);
    for (uint64_t sa : {3, 5, 8}) {
      for (uint64_t sr : {3, 5, 8}) {
        ntt120_vec_znx_dft_layout a(n, sa);
        ntt120_vec_znx_big_layout res(n, sr);
        a.fill_random();
        std::vector<znx_i128> expect(sr);
        for (uint64_t i = 0; i < sr; ++i) {
          expect[i] = simple_intt120(a.get_copy_zext(i));
        }
        // test the function
        if (idft_tmp_bytes) {
          thash hash_before = a.content_hash();
          idft(module, res.data, sr, a.data, sa, tmp.data());
          ASSERT_EQ(a.content_hash(), hash_before);
        } else {
          idft_tmp_a(module, res.data, sr, a.data, sa);
        }
        for (uint64_t i = 0; i < sr; ++i) {
          znx_i128 actual = res.get_copy_zext(i);
          ASSERT_EQ(res.get_copy_zext(i), expect[i]);
          // if (!(actual == expect[i])) {
          //   for (uint64_t j = 0; j < n; ++j) {
          //     std::cerr << actual.get_coeff(j) << " dft vs. " << expect[i].get_coeff(j) << std::endl;
          //   }
          //   FAIL();
          // }
        }
      }
    }
    delete_module_info(module);
  }
}

TEST(vec_znx_dft, ntt120_vec_znx_idft) {
  test_ntt120_vec_znx_idft(ntt120_vec_znx_idft_avx, nullptr, ntt120_vec_znx_idft_tmp_bytes_avx);
}
TEST(vec_znx_dft, ntt120_vec_znx_idft_tmp_a) {
  test_ntt120_vec_znx_idft(nullptr, ntt120_vec_znx_idft_tmp_a_avx, nullptr);
}
#endif
TEST(vec_znx_dft, vec_znx_idft) {
  test_fft64_vec_znx_idft(vec_znx_idft, nullptr, vec_znx_idft_tmp_bytes);
#ifdef __x86_64__
  // FIXME: currently, only supported on avx
  test_ntt120_vec_znx_idft(vec_znx_idft, nullptr, vec_znx_idft_tmp_bytes);
#endif
}
TEST(vec_znx_dft, vec_znx_idft_tmp_a) {
  test_fft64_vec_znx_idft(nullptr, vec_znx_idft_tmp_a, nullptr);
#ifdef __x86_64__
  // FIXME: currently, only supported on avx
  test_ntt120_vec_znx_idft(nullptr, vec_znx_idft_tmp_a, nullptr);
#endif
}
