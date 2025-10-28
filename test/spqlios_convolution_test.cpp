#include <gtest/gtest.h>
#include <stdint.h>

#include "../spqlios/arithmetic/vec_znx_arithmetic_private.h"
#include "spqlios/arithmetic/vec_znx_arithmetic.h"
#include "testlib/fft64_layouts.h"
#include "testlib/polynomial_vector.h"

static void test_convolution_prepare_contiguous(CNV_PREPARE_RIGHT_CONTIGUOUS_F* prepare_contiguous,
                                                CNV_PREPARE_RIGHT_CONTIGUOUS_TMP_BYTES_F* tmp_bytes) {
  // tests when n >= 8
  for (uint64_t nn : {8, 32}) {
    MODULE* module = new_module_info(nn, FFT64);
    uint64_t nblk = nn / 8;
    for (uint64_t nrows : {1, 2, 5}) {
      znx_vec_i64_layout vec(nn, nrows, nn);
      fft64_cnv_right_layout pvec(nn, nrows);
      vec.fill_random(30);
      std::vector<uint8_t> tmp_space(tmp_bytes(module, nrows));
      thash hash_before = vec.content_hash();
      prepare_contiguous(module, pvec.data, nrows, vec.data(), nrows, nn, tmp_space.data());
      ASSERT_EQ(vec.content_hash(), hash_before);
      for (uint64_t row = 0; row < nrows; ++row) {
        reim_fft64vec tmp = simple_fft64(vec.get_copy(row));
        for (uint64_t blk = 0; blk < nblk; ++blk) {
          reim4_elem expect = tmp.get_blk(blk);
          reim4_elem actual = pvec.get(row, blk);
          ASSERT_LE(infty_dist(actual, expect), 1e-10);
        }
      }
    }
    delete_module_info(module);
  }
}

TEST(vec_znx, fft64_convolution_prepare_right_contiguous_ref) {
  test_convolution_prepare_contiguous(fft64_convolution_prepare_right_contiguous_ref,
                                      fft64_convolution_prepare_right_contiguous_tmp_bytes);
}

static void test_cnv_apply(CNV_APPLY_DFT_F* apply, CNV_APPLY_DFT_TMP_BYTES_F* tmp_bytes) {
  for (uint64_t nn : {8, 64}) {
    MODULE* module = new_module_info(nn, FFT64);
    for (uint64_t size_l : {1, 4, 7}) {
      for (uint64_t size_r : {1, 4, 7}) {
        for (uint64_t offset : {0, 1, 2}) {
          uint64_t out_size = size_l + size_r - 1;
          if (offset > out_size) {
            continue;
          }
          fft64_cnv_right_layout pvec_right(nn, size_r);
          fft64_cnv_left_layout pvec_left(nn, size_l);
          fft64_vec_znx_dft_layout out(nn, out_size);
          pvec_right.fill_random(0);
          pvec_left.fill_random(0);
          // naive computation of the convolution
          std::vector<reim_fft64vec> expect(out_size, reim_fft64vec(nn));
          for (uint64_t k = offset; k < out_size + offset; ++k) {
            reim_fft64vec ex = reim_fft64vec::zero(nn);
            if (k < size_r + size_l) {
              uint64_t jmin = k >= size_l ? k + 1 - size_l : 0;
              uint64_t jmax = k < size_r ? k + 1 : size_r;
              for (uint64_t j = jmin; j < jmax; ++j) {
                ex += pvec_left.get_zext(k - j) * pvec_right.get_zext(j);
              }
            }
            expect[k - offset] = ex;
          }
          //
          // apply the convolution
          std::vector<uint8_t> tmp(tmp_bytes(module, out_size, offset, size_l, size_r));
          apply(module, out.data, out_size, offset, pvec_left.data, size_l, pvec_right.data, size_r, tmp.data());
          // check that the output is close from the expectation
          for (uint64_t k = 0; k < out_size; ++k) {
            reim_fft64vec actual = out.get_copy_zext(k);
            ASSERT_LE(infty_dist(actual, expect[k]), 1e-10);
          }
        }
      }
    }
    delete_module_info(module);
  }
}

TEST(vec_znx, fft64_convolution_apply_dft_ref) {
  test_cnv_apply(fft64_convolution_apply_dft_ref, fft64_convolution_apply_dft_tmp_bytes);
}
