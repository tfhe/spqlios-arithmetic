#include <string.h>

#include "../reim4/reim4_arithmetic.h"
#include "vec_znx_arithmetic_private.h"

/** @brief prepares the right vector for convolution  */
EXPORT void cnv_prepare_right_contiguous(const MODULE* module,                              // N
                                         CNV_PVEC_R* pvec, uint64_t nrows,                  // output
                                         const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                         uint8_t* tmp_space                                 // scratch space
) {
  return module->func.cnv_prepare_right_contiguous(module, pvec, nrows, a, a_size, a_sl, tmp_space);
}

/** @brief minimal scratch space byte-size required for the cnv_prepare_right_contiguous function */
EXPORT uint64_t cnv_prepare_right_contiguous_tmp_bytes(const MODULE* module,  // N
                                                       uint64_t nrows,        // size of output
                                                       uint64_t a_size        // size of input
) {
  return module->func.cnv_prepare_right_contiguous_tmp_bytes(module, nrows, a_size);
}

/** @brief prepares the right vector for convolution  */
EXPORT void cnv_prepare_left_contiguous(const MODULE* module,                              // N
                                        CNV_PVEC_L* pvec, uint64_t nrows,                  // output
                                        const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                        uint8_t* tmp_space                                 // scratch space
) {
  return module->func.cnv_prepare_left_contiguous(module, pvec, nrows, a, a_size, a_sl, tmp_space);
}

/** @brief minimal scratch space byte-size required for the cnv_prepare_left_contiguous function */
EXPORT uint64_t cnv_prepare_left_contiguous_tmp_bytes(const MODULE* module,  // N
                                                      uint64_t nrows,        // size of output
                                                      uint64_t a_size        // size of input
) {
  return module->func.cnv_prepare_left_contiguous_tmp_bytes(module, nrows, a_size);
}

/** @brief prepares the right vector for convolution  */
EXPORT void fft64_convolution_prepare_right_contiguous_ref(const MODULE* module,                              // N
                                                           CNV_PVEC_R* pvec, uint64_t nrows,                  // output
                                                           const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                                           uint8_t* tmp_space  // scratch space
) {
  fft64_convolution_prepare_contiguous_ref(module, (double*)pvec, nrows, a, a_size, a_sl, tmp_space);
}

/** @brief minimal scratch space byte-size required for the cnv_prepare_left_contiguous function */
EXPORT uint64_t fft64_convolution_prepare_left_contiguous_tmp_bytes(const MODULE* module,  // N
                                                                    uint64_t nrows,        // size of output
                                                                    uint64_t a_size        // size of input
) {
  const uint64_t nn = module->nn;
  const uint64_t rows = nrows < a_size ? nrows : a_size;
  return nn * sizeof(int64_t) * rows;
}

/** @brief minimal scratch space byte-size required for the cnv_prepare_right_contiguous function */
EXPORT uint64_t fft64_convolution_prepare_right_contiguous_tmp_bytes(const MODULE* module,  // N
                                                                     uint64_t nrows,        // size of output
                                                                     uint64_t a_size        // size of input
) {
  uint64_t nn = module->nn;
  const uint64_t rows = nrows < a_size ? nrows : a_size;
  return nn * sizeof(uint64_t) * rows;
}

/** @brief prepares the left vector for convolution  */
EXPORT void fft64_convolution_prepare_left_contiguous_ref(const MODULE* module,                              // N
                                                          CNV_PVEC_L* pvec, uint64_t nrows,                  // output
                                                          const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                                          uint8_t* tmp_space  // scratch space
) {
  fft64_convolution_prepare_contiguous_ref(module, (double*)pvec, nrows, a, a_size, a_sl, tmp_space);
}

/** @brief prepares a convolution vector  */
EXPORT void fft64_convolution_prepare_contiguous_ref(const MODULE* module,                              // N
                                                     double* pvec, uint64_t nrows,                      // output
                                                     const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                                     uint8_t* tmp_space                                 // scratch space
) {
  const uint64_t m = module->m;
  const uint64_t rows = nrows < a_size ? nrows : a_size;
  // dimensions where m < 4 are not implemented yet
  if (m<4) {
    NOT_IMPLEMENTED()
  }

  VEC_ZNX_DFT* a_dft = (VEC_ZNX_DFT*)tmp_space;

  fft64_vec_znx_dft(module, a_dft, rows, a, a_size, a_sl);

  for (uint64_t blk_i = 0; blk_i < m / 4; blk_i++) {
    reim4_extract_1blk_from_contiguous_reim_ref(m, rows, blk_i, pvec + blk_i * rows * 8, (const double*)a_dft);
  }
}

/** @brief applies a convolution of two prepared vectors  */
EXPORT void cnv_apply_dft(const MODULE* module,                                      // N
                          VEC_ZNX_DFT* res, uint64_t res_size, uint64_t res_offset,  // output
                          const CNV_PVEC_L* a,
                          uint64_t a_size,  // left operand and its size (in terms of reim4)
                          const CNV_PVEC_R* b,
                          uint64_t b_size,    // right operand and its size (in terms of reim4)
                          uint8_t* tmp_space  // scratch space
) {
  return module->func.cnv_apply_dft(module, res, res_size, res_offset, a, a_size, b, b_size, tmp_space);
}

/** @brief minimal scratch space byte-size required for the cnv_apply_dft function */
EXPORT uint64_t cnv_apply_dft_tmp_bytes(const MODULE* module,                    // N
                                        uint64_t res_size, uint64_t res_offset,  // output
                                        uint64_t a_size,                         // size of left operand
                                        uint64_t b_size                          // size of right operand
) {
  return module->func.cnv_apply_dft_tmp_bytes(module, res_size, res_offset, a_size, b_size);
}

/** @brief applies a convolution of two prepared vectors  */
EXPORT void fft64_convolution_apply_dft_ref(const MODULE* module,                                      // N
                                            VEC_ZNX_DFT* res, uint64_t res_size, uint64_t res_offset,  // output
                                            const CNV_PVEC_L* a,
                                            uint64_t a_size,  // left operand and its size (in terms of reim4)
                                            const CNV_PVEC_R* b,
                                            uint64_t b_size,    // right operand and its size (in terms of reim4)
                                            uint8_t* tmp_space  // scratch space
) {
  const uint64_t m = module->m;
  uint64_t size = res_size < a_size + b_size - 1 ? res_size : a_size + b_size - 1;
  uint64_t offset = res_offset < size ? res_offset : size;

  double* dst_tmp = (double*)tmp_space;
  double* dst = (double*)res;
  const double* a_double = (const double*)a;
  const double* b_double = (const double*)b;

  for (uint64_t blk_i = 0; blk_i < m / 4; blk_i++) {
    reim4_convolution_ref(dst_tmp, size, offset, a_double + blk_i * a_size * 8, a_size, b_double + blk_i * b_size * 8,
                          b_size);
    reim4_save_1blk_to_contiguous_reim_ref(m, size, blk_i, dst, dst_tmp);
  }
}

/** @brief minimal scratch space byte-size required for the cnv_apply_dft function */
EXPORT uint64_t fft64_convolution_apply_dft_tmp_bytes(const MODULE* module,                    // N
                                                      uint64_t res_size, uint64_t res_offset,  // output
                                                      uint64_t a_size,  // size of left operand (in terms of reim4)
                                                      uint64_t b_size   // size of right operand (in terms of reim4)
) {
  uint64_t size = res_size < a_size + b_size - 1 ? res_size : a_size + b_size - 1;
  return sizeof(double) * 8 * size;
}
