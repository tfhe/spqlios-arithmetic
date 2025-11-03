#include <gtest/gtest.h>
#include <stdint.h>

#include "../spqlios/arithmetic/vec_znx_arithmetic_private.h"
#include "spqlios/arithmetic/vec_znx_arithmetic.h"
#include "testlib/fft64_layouts.h"
#include "testlib/polynomial_vector.h"

template <typename fft64_cnv_layout, typename CNV_PREPARE_CONTIGUOUS_F, typename CNV_PREPARE_CONTIGUOUS_TMP_BYTES_F>
static void test_convolution_prepare_contiguous(CNV_PREPARE_CONTIGUOUS_F* prepare_contiguous,
                                                CNV_PREPARE_CONTIGUOUS_TMP_BYTES_F* tmp_bytes) {
  // tests when n >= 8
  for (uint64_t nn : {8, 32}) {
    MODULE* module = new_module_info(nn, FFT64);
    uint64_t nblk = nn / 8;
    for (uint64_t in_nrows : {0, 1, 2, 5}) {
      for (uint64_t out_nrows : {0, 1, 3, 5}) {
        znx_vec_i64_layout vec(nn, in_nrows, nn);
        fft64_cnv_layout pvec(nn, out_nrows);
        vec.fill_random(30);
        std::vector<uint8_t> tmp_space(tmp_bytes(module, out_nrows, in_nrows));
        thash hash_before = vec.content_hash();
        prepare_contiguous(module,                //
                           pvec.data, out_nrows,  //
                           vec.data(), in_nrows, nn, tmp_space.data());
        ASSERT_EQ(vec.content_hash(), hash_before);
        for (uint64_t row = 0; row < out_nrows; ++row) {
          reim_fft64vec tmp = simple_fft64(vec.get_copy_zext(row));
          for (uint64_t blk = 0; blk < nblk; ++blk) {
            reim4_elem expect = tmp.get_blk(blk);
            reim4_elem actual = pvec.get(row, blk);
            ASSERT_LE(infty_dist(actual, expect), 1e-10);
          }
        }
      }
    }
    delete_module_info(module);
  }
}

TEST(vec_znx, fft64_convolution_prepare_right_contiguous_ref) {
  test_convolution_prepare_contiguous<fft64_cnv_right_layout,                     //
                                      CNV_PREPARE_RIGHT_CONTIGUOUS_F,             //
                                      CNV_PREPARE_RIGHT_CONTIGUOUS_TMP_BYTES_F>(  //
      fft64_convolution_prepare_right_contiguous_ref,                             //
      fft64_convolution_prepare_right_contiguous_tmp_bytes);
}

TEST(vec_znx, fft64_convolution_prepare_left_contiguous_ref) {
  test_convolution_prepare_contiguous<fft64_cnv_left_layout,                     //
                                      CNV_PREPARE_LEFT_CONTIGUOUS_F,             //
                                      CNV_PREPARE_LEFT_CONTIGUOUS_TMP_BYTES_F>(  //
      fft64_convolution_prepare_left_contiguous_ref,                             //
      fft64_convolution_prepare_left_contiguous_tmp_bytes);
}

static void test_cnv_apply(CNV_APPLY_DFT_F* apply, CNV_APPLY_DFT_TMP_BYTES_F* tmp_bytes) {
  for (uint64_t nn : {8, 64}) {
    MODULE* module = new_module_info(nn, FFT64);
    for (uint64_t size_l : {0, 1, 4, 7}) {
      for (uint64_t size_r : {0, 1, 3, 8}) {
        fft64_cnv_right_layout pvec_right(nn, size_r);
        fft64_cnv_left_layout pvec_left(nn, size_l);
        pvec_right.fill_random(0);
        pvec_left.fill_random(0);
        // compute the full convolution vector (leave a leading zero in this test)
        uint64_t full_cnv_size = size_l + size_r;
        std::vector<reim_fft64vec> full_convolution(full_cnv_size, reim_fft64vec(nn));
        for (uint64_t k = 0; k < full_cnv_size; ++k) {
          full_convolution[k] = reim_fft64vec::zero(nn);
        }
        for (uint64_t i = 0; i < size_l; ++i) {
          for (uint64_t j = 0; j < size_r; ++j) {
            full_convolution[i + j] += pvec_left.get_zext(i) * pvec_right.get_zext(j);
          }
        }
        // now, test
        for (uint64_t offset : {0, 1, 2}) {
          for (uint64_t out_size : {0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15}) {
            fft64_vec_znx_dft_layout out(nn, out_size);
            // apply the convolution
            std::vector<uint8_t> tmp(tmp_bytes(module, out_size, offset, size_l, size_r));
            apply(module, out.data, out_size, offset, pvec_left.data, size_l, pvec_right.data, size_r, tmp.data());
            // check the result
            for (uint64_t k = 0; k < out_size; ++k) {
              reim_fft64vec actual = out.get_copy_zext(k);
              reim_fft64vec expect = reim_fft64vec::zero(nn);
              if (k + offset < full_cnv_size) {
                expect = full_convolution[k + offset];
              }
              ASSERT_LE(infty_dist(actual, expect), 1e-10);
            }
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
