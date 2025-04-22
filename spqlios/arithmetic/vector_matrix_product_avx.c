#include <string.h>

#include "../reim4/reim4_arithmetic.h"
#include "vec_znx_arithmetic_private.h"

/** @brief prepares a vmp matrix (contiguous row-major version) */
EXPORT void fft64_vmp_prepare_contiguous_avx(const MODULE* module,                                // N
                                             VMP_PMAT* pmat,                                      // output
                                             const int64_t* mat, uint64_t nrows, uint64_t ncols,  // a
                                             uint8_t* tmp_space                                   // scratch space
) {
  // there is an edge case if nn < 8
  const uint64_t nn = module->nn;
  const uint64_t m = module->m;

  double* output_mat = (double*)pmat;
  double* start_addr = (double*)pmat;
  uint64_t offset = nrows * ncols * 8;

  if (nn >= 8) {
    for (uint64_t row_i = 0; row_i < nrows; row_i++) {
      for (uint64_t col_i = 0; col_i < ncols; col_i++) {
        reim_from_znx64(module->mod.fft64.p_conv, (SVP_PPOL*)tmp_space, mat + (row_i * ncols + col_i) * nn);
        reim_fft(module->mod.fft64.p_fft, (double*)tmp_space);

        if (col_i == (ncols - 1) && (ncols % 2 == 1)) {
          // special case: last column out of an odd column number
          start_addr = output_mat + col_i * nrows * 8  // col == ncols-1
                       + row_i * 8;
        } else {
          // general case: columns go by pair
          start_addr = output_mat + (col_i / 2) * (2 * nrows) * 8  // second: col pair index
                       + row_i * 2 * 8                             // third: row index
                       + (col_i % 2) * 8;
        }

        for (uint64_t blk_i = 0; blk_i < m / 4; blk_i++) {
          // extract blk from tmp and save it
          reim4_extract_1blk_from_reim_avx(m, blk_i, start_addr + blk_i * offset, (double*)tmp_space);
        }
      }
    }
  } else {
    for (uint64_t row_i = 0; row_i < nrows; row_i++) {
      for (uint64_t col_i = 0; col_i < ncols; col_i++) {
        double* res = (double*)pmat + (col_i * nrows + row_i) * nn;
        reim_from_znx64(module->mod.fft64.p_conv, (SVP_PPOL*)res, mat + (row_i * ncols + col_i) * nn);
        reim_fft(module->mod.fft64.p_fft, res);
      }
    }
  }
}

/** @brief prepares a vmp matrix (mat[row]+col*N points to the item) */
EXPORT void fft64_vmp_prepare_dblptr_avx(const MODULE* module,                                 // N
                                         VMP_PMAT* pmat,                                       // output
                                         const int64_t** mat, uint64_t nrows, uint64_t ncols,  // a
                                         uint8_t* tmp_space                                    // scratch space
) {
  for (uint64_t row_i = 0; row_i < nrows; row_i++) {
    fft64_vmp_prepare_row_avx(module, pmat, mat[row_i], row_i, nrows, ncols, tmp_space);
  }
}

/** @brief prepares the ith-row of a vmp matrix with nrows and ncols */
EXPORT void fft64_vmp_prepare_row_avx(const MODULE* module,                                                // N
                                      VMP_PMAT* pmat,                                                      // output
                                      const int64_t* row, uint64_t row_i, uint64_t nrows, uint64_t ncols,  // a
                                      uint8_t* tmp_space  // scratch space
) {
  // there is an edge case if nn < 8
  const uint64_t nn = module->nn;
  const uint64_t m = module->m;
  double* output_mat = (double*)pmat;
  double* start_addr = (double*)pmat;
  uint64_t offset = nrows * ncols * 8;

  if (nn >= 8) {
    for (uint64_t col_i = 0; col_i < ncols; col_i++) {
      reim_from_znx64(module->mod.fft64.p_conv, (SVP_PPOL*)tmp_space, row + col_i * nn);
      reim_fft(module->mod.fft64.p_fft, (double*)tmp_space);

      if (col_i == (ncols - 1) && (ncols % 2 == 1)) {
        // special case: last column out of an odd column number
        start_addr = output_mat + col_i * nrows * 8  // col == ncols-1
                     + row_i * 8;
      } else {
        // general case: columns go by pair
        start_addr = output_mat + (col_i / 2) * (2 * nrows) * 8  // second: col pair index
                     + row_i * 2 * 8                             // third: row index
                     + (col_i % 2) * 8;
      }

      for (uint64_t blk_i = 0; blk_i < m / 4; blk_i++) {
        // extract blk from tmp and save it
        reim4_extract_1blk_from_reim_avx(m, blk_i, start_addr + blk_i * offset, (double*)tmp_space);
      }
    }
  } else {
    for (uint64_t col_i = 0; col_i < ncols; col_i++) {
      double* res = (double*)pmat + (col_i * nrows + row_i) * nn;
      reim_from_znx64(module->mod.fft64.p_conv, (SVP_PPOL*)res, row + col_i * nn);
      reim_fft(module->mod.fft64.p_fft, res);
    }
  }
}

/** @brief applies a vmp product (result in DFT space) */
EXPORT void fft64_vmp_apply_dft_avx(const MODULE* module,                                  // N
                                    VEC_ZNX_DFT* res, uint64_t res_size,                   // res
                                    const int64_t* a, uint64_t a_size, uint64_t a_sl,      // a
                                    const VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,  // prep matrix
                                    uint8_t* tmp_space                                     // scratch space
) {
  const uint64_t nn = module->nn;
  const uint64_t rows = nrows < a_size ? nrows : a_size;

  VEC_ZNX_DFT* a_dft = (VEC_ZNX_DFT*)tmp_space;
  uint8_t* new_tmp_space = (uint8_t*)tmp_space + rows * nn * sizeof(double);

  fft64_vec_znx_dft(module, a_dft, rows, a, a_size, a_sl);
  fft64_vmp_apply_dft_to_dft_avx(module, res, res_size, a_dft, a_size, pmat, nrows, ncols, new_tmp_space);
}

// EXPORT void fft64_vmp_apply_dft_to_dft_add_avx(const MODULE* module,                       // N
//                                            VEC_ZNX_DFT* res, const uint64_t res_size,  // res
//                                            const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
//                                            const VMP_PMAT* pmat, const uint64_t nrows,
//                                            const uint64_t ncols,  // prep matrix
//                                            uint8_t* tmp_space     // scratch space (a_size*sizeof(reim4) bytes)
// ) {
//   fft64_vmp_apply_dft_to_dft_custom_storeop_avx(module, res, res_size, a_dft, a_size, pmat, nrows, ncols, tmp_space, reim4_add_1blk_to_reim_avx);
// }

/** @brief this inner function could be very handy */
EXPORT void fft64_vmp_apply_dft_to_dft_avx(const MODULE* module,                       // N
                                           VEC_ZNX_DFT* res, const uint64_t res_size,  // res
                                           const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
                                           const VMP_PMAT* pmat, const uint64_t nrows,
                                           const uint64_t ncols,  // prep matrix
                                           uint8_t* tmp_space     // scratch space (a_size*sizeof(reim4) bytes)
) {
  fft64_vmp_apply_dft_to_dft_custom_storeop_avx(module, res, res_size, a_dft, a_size, pmat, nrows, ncols, tmp_space, reim4_save_1blk_to_reim_avx);
}
