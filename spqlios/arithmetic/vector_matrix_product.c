#include <string.h>
#include <assert.h>
#include "../reim4/reim4_arithmetic.h"
#include "vec_znx_arithmetic_private.h"

EXPORT uint64_t bytes_of_vmp_pmat(const MODULE* module,           // N
                                  uint64_t nrows, uint64_t ncols  // dimensions
) {
  return module->func.bytes_of_vmp_pmat(module, nrows, ncols);
}

// fft64
EXPORT uint64_t fft64_bytes_of_vmp_pmat(const MODULE* module,           // N
                                        uint64_t nrows, uint64_t ncols  // dimensions
) {
  return module->nn * nrows * ncols * sizeof(double);
}

EXPORT VMP_PMAT* new_vmp_pmat(const MODULE* module,           // N
                              uint64_t nrows, uint64_t ncols  // dimensions
) {
  return spqlios_alloc(bytes_of_vmp_pmat(module, nrows, ncols));
}

EXPORT void delete_vmp_pmat(VMP_PMAT* res) { spqlios_free(res); }

/** @brief prepares a vmp matrix (contiguous row-major version) */
EXPORT void vmp_prepare_contiguous(const MODULE* module,                                // N
                                   VMP_PMAT* pmat,                                      // output
                                   const int64_t* mat, uint64_t nrows, uint64_t ncols,  // a
                                   uint8_t* tmp_space                                   // scratch space
) {
  module->func.vmp_prepare_contiguous(module, pmat, mat, nrows, ncols, tmp_space);
}

/** @brief prepares a vmp matrix (mat[row]+col*N points to the item) */
EXPORT void vmp_prepare_dblptr(const MODULE* module,                                 // N
                               VMP_PMAT* pmat,                                       // output
                               const int64_t** mat, uint64_t nrows, uint64_t ncols,  // a
                               uint8_t* tmp_space                                    // scratch space
) {
  module->func.vmp_prepare_dblptr(module, pmat, mat, nrows, ncols, tmp_space);
}

/** @brief prepares the ith-row of a vmp matrix with nrows and ncols */
EXPORT void vmp_prepare_row(const MODULE* module,                                                // N
                            VMP_PMAT* pmat,                                                      // output
                            const int64_t* row, uint64_t row_i, uint64_t nrows, uint64_t ncols,  // a
                            uint8_t* tmp_space                                                   // scratch space
) {
  module->func.vmp_prepare_row(module, pmat, row, row_i, nrows, ncols, tmp_space);
}

/** @brief prepares the ith-row of a vmp matrix with nrows and ncols */
EXPORT void vmp_prepare_row_dft(const MODULE* module,                                                   // N
                                VMP_PMAT* pmat,                                                         // output
                                const VEC_ZNX_DFT* row, uint64_t row_i, uint64_t nrows, uint64_t ncols  // a
) {
  module->func.vmp_prepare_row_dft(module, pmat, row, row_i, nrows, ncols);
}

/** @brief extracts the ith-row of a vmp matrix with nrows and ncols */
EXPORT void vmp_extract_row_dft(const MODULE* module,                                                 // N
                                VEC_ZNX_DFT* res,                                                     // output
                                const VMP_PMAT* pmat, uint64_t row_i, uint64_t nrows, uint64_t ncols  // a
) {
  module->func.vmp_extract_row_dft(module, res, pmat, row_i, nrows, ncols);
}

/** @brief extracts the ith-row of a vmp matrix with nrows and ncols */
EXPORT void vmp_extract_row(const MODULE* module,                                                 // N
                            VEC_ZNX_BIG* res,                                                     // output
                            const VMP_PMAT* pmat, uint64_t row_i, uint64_t nrows, uint64_t ncols  // a
) {
  module->func.vmp_extract_row(module, res, pmat, row_i, nrows, ncols);
}

/** @brief minimal scratch space byte-size required for the vmp_prepare function */
EXPORT uint64_t vmp_prepare_tmp_bytes(const MODULE* module,  // N
                                      uint64_t nrows, uint64_t ncols) {
  return module->func.vmp_prepare_tmp_bytes(module, nrows, ncols);
}

/** @brief minimal scratch space byte-size required for the vmp_extract function */
EXPORT uint64_t vmp_extract_tmp_bytes(const MODULE* module,  // N
                                      uint64_t nrows, uint64_t ncols) {
  return module->func.vmp_extract_tmp_bytes(module, nrows, ncols);
}

double* get_blk_addr(uint64_t row_i, uint64_t col_i, uint64_t nrows, uint64_t ncols, const VMP_PMAT* pmat) {
  double* output_mat = (double*)pmat;

  if (col_i == (ncols - 1) && (ncols % 2 == 1)) {
    // special case: last column out of an odd column number
    return output_mat + col_i * nrows * 8  // col == ncols-1
           + row_i * 8;
  } else {
    // general case: columns go by pair
    return output_mat + (col_i / 2) * (2 * nrows) * 8  // second: col pair index
           + row_i * 2 * 8                             // third: row index
           + (col_i % 2) * 8;
  }
}

void fft64_store_svp_ppol_into_vmp_pmat_row_blk_ref(uint64_t nn, uint64_t m, const SVP_PPOL* svp_ppol, uint64_t row_i,
                                                    uint64_t col_i, uint64_t nrows, uint64_t ncols, VMP_PMAT* pmat) {
  double* start_addr = get_blk_addr(row_i, col_i, nrows, ncols, pmat);
  uint64_t offset = nrows * ncols * 8;
  for (uint64_t blk_i = 0; blk_i < m / 4; blk_i++) {
    reim4_extract_1blk_from_reim_ref(m, blk_i, start_addr + blk_i * offset, (double*)svp_ppol);
  }
}

/** @brief prepares a vmp matrix (contiguous row-major version) */
EXPORT void fft64_vmp_prepare_contiguous_ref(const MODULE* module,                                // N
                                             VMP_PMAT* pmat,                                      // output
                                             const int64_t* mat, uint64_t nrows, uint64_t ncols,  // a
                                             uint8_t* tmp_space                                   // scratch space
) {
  // there is an edge case if nn < 8
  const uint64_t nn = module->nn;
  const uint64_t m = module->m;

  if (nn >= 8) {
    for (uint64_t row_i = 0; row_i < nrows; row_i++) {
      for (uint64_t col_i = 0; col_i < ncols; col_i++) {
        reim_from_znx64(module->mod.fft64.p_conv, (SVP_PPOL*)tmp_space, mat + (row_i * ncols + col_i) * nn);
        reim_fft(module->mod.fft64.p_fft, (double*)tmp_space);
        fft64_store_svp_ppol_into_vmp_pmat_row_blk_ref(nn, m, (SVP_PPOL*)tmp_space, row_i, col_i, nrows, ncols, pmat);
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
EXPORT void fft64_vmp_prepare_dblptr_ref(const MODULE* module,                                 // N
                                         VMP_PMAT* pmat,                                       // output
                                         const int64_t** mat, uint64_t nrows, uint64_t ncols,  // a
                                         uint8_t* tmp_space                                    // scratch space
) {
  for (uint64_t row_i = 0; row_i < nrows; row_i++) {
    fft64_vmp_prepare_row_ref(module, pmat, mat[row_i], row_i, nrows, ncols, tmp_space);
  }
}

/** @brief Extracts the i-th row of the vmp_pmat into a vec_znx_dft */
EXPORT void fft64_vmp_extract_row_dft_ref(const MODULE* module, VEC_ZNX_DFT* res, const VMP_PMAT* pmat, uint64_t row_i,
                                          uint64_t nrows, uint64_t ncols) {
  // there is an edge case if nn < 8
  const uint64_t nn = module->nn;
  const uint64_t m = module->m;
  const uint64_t offset = nrows * ncols * 8;

  double* res_addr = (double*)res;

  if (nn >= 8) {
    for (uint64_t col_i = 0; col_i < ncols; col_i++) {
      const double* start_addr = get_blk_addr(row_i, col_i, nrows, ncols, pmat);
      for (uint64_t blk_i = 0; blk_i < m / 4; blk_i++) {
        reim4_extract_reim_from_1blk_ref(m, blk_i, res_addr + col_i * nn, start_addr + blk_i * offset);
      }
    }
  } else {
    for (uint64_t col_i = 0; col_i < ncols; col_i++) {
      memcpy(res_addr + col_i * nn, (double*)pmat + (col_i * nrows + row_i) * nn, nn * sizeof(double));
    }
  }
}

/** @brief Extracts the i-th row of the vmp_pmat into a vec_znx_dft */
EXPORT void fft64_vmp_extract_row_ref(const MODULE* module, VEC_ZNX_BIG* res, const VMP_PMAT* pmat, uint64_t row_i,
                                      uint64_t nrows, uint64_t ncols) {
  // there is an edge case if nn < 8
  const uint64_t nn = module->nn;
  const uint64_t m = module->m;
  const uint64_t offset = nrows * ncols * 8;

  double* res_addr = (double*)res;

  if (nn >= 8) {
    for (uint64_t col_i = 0; col_i < ncols; col_i++) {
      const double* start_addr = get_blk_addr(row_i, col_i, nrows, ncols, pmat);
      for (uint64_t blk_i = 0; blk_i < m / 4; blk_i++) {
        reim4_extract_reim_from_1blk_ref(m, blk_i, ((double*)res) + col_i * nn, start_addr + blk_i * offset);
      }
      reim_ifft(module->mod.fft64.p_ifft, ((double*)res) + col_i * nn);
      reim_to_znx64(module->mod.fft64.p_reim_to_znx, ((int64_t*)res) + col_i * nn, ((int64_t*)res) + col_i * nn);
    }
  } else {
    for (uint64_t col_i = 0; col_i < ncols; col_i++) {
      memcpy(res_addr + col_i * nn, (double*)pmat + (col_i * nrows + row_i) * nn, nn * sizeof(double));
      reim_ifft(module->mod.fft64.p_ifft, ((double*)res) + col_i * nn);
      reim_to_znx64(module->mod.fft64.p_reim_to_znx, ((int64_t*)res) + col_i * nn, ((int64_t*)res) + col_i * nn);
    }
  }
}

EXPORT void fft64_vmp_prepare_row_dft_ref(const MODULE* module,  // N
                                          VMP_PMAT* pmat,        // output
                                          const VEC_ZNX_DFT* row, uint64_t row_i, uint64_t nrows, uint64_t ncols  // a
) {
  // there is an edge case if nn < 8
  const uint64_t nn = module->nn;
  const uint64_t m = module->m;
  double* row_addr = (double*)row;

  if (nn >= 8) {
    for (uint64_t col_i = 0; col_i < ncols; col_i++) {
      fft64_store_svp_ppol_into_vmp_pmat_row_blk_ref(nn, m, (SVP_PPOL*)(row_addr + col_i * nn), row_i, col_i, nrows,
                                                     ncols, pmat);
    }
  } else {
    for (uint64_t col_i = 0; col_i < ncols; col_i++) {
      memcpy((double*)pmat + (col_i * nrows + row_i) * nn, (double*)row + col_i * nn, nn * sizeof(double));
    }
  }
}

/** @brief prepares the ith-row of a vmp matrix with nrows and ncols */
EXPORT void fft64_vmp_prepare_row_ref(const MODULE* module,                                                // N
                                      VMP_PMAT* pmat,                                                      // output
                                      const int64_t* row, uint64_t row_i, uint64_t nrows, uint64_t ncols,  // a
                                      uint8_t* tmp_space  // scratch space
) {
  // there is an edge case if nn < 8
  const uint64_t nn = module->nn;
  const uint64_t m = module->m;

  if (nn >= 8) {
    for (uint64_t col_i = 0; col_i < ncols; col_i++) {
      reim_from_znx64(module->mod.fft64.p_conv, (SVP_PPOL*)tmp_space, row + col_i * nn);
      reim_fft(module->mod.fft64.p_fft, (double*)tmp_space);
      fft64_store_svp_ppol_into_vmp_pmat_row_blk_ref(nn, m, (SVP_PPOL*)tmp_space, row_i, col_i, nrows, ncols, pmat);
    }
  } else {
    for (uint64_t col_i = 0; col_i < ncols; col_i++) {
      double* res = (double*)pmat + (col_i * nrows + row_i) * nn;
      reim_from_znx64(module->mod.fft64.p_conv, (SVP_PPOL*)res, row + col_i * nn);
      reim_fft(module->mod.fft64.p_fft, res);
    }
  }
}

/** @brief minimal scratch space byte-size required for the vmp_prepare function */
EXPORT uint64_t fft64_vmp_prepare_tmp_bytes(const MODULE* module,  // N
                                            uint64_t nrows, uint64_t ncols) {
  const uint64_t nn = module->nn;
  return nn * sizeof(int64_t);
}

/** @brief minimal scratch space byte-size required for the vmp_extract function */
EXPORT uint64_t fft64_vmp_extract_tmp_bytes(const MODULE* module,  // N
                                            uint64_t nrows, uint64_t ncols) {
  const uint64_t nn = module->nn;
  return nn * sizeof(int64_t);
}

/** @brief applies a vmp product (result in DFT space) and adds to res inplace */
EXPORT void fft64_vmp_apply_dft_add_ref(const MODULE* module,                                  // N
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
  fft64_vmp_apply_dft_to_dft_add_ref(module, res, res_size, a_dft, a_size, pmat, nrows, ncols, new_tmp_space);
}

/** @brief applies a vmp product (result in DFT space) */
EXPORT void fft64_vmp_apply_dft_ref(const MODULE* module,                                  // N
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
  fft64_vmp_apply_dft_to_dft_ref(module, res, res_size, a_dft, a_size, pmat, nrows, ncols, new_tmp_space);
}

/** @brief like fft64_vmp_apply_dft_to_dft_ref but adds in place */
EXPORT void fft64_vmp_apply_dft_to_dft_add_ref(const MODULE* module,                       // N
                                           VEC_ZNX_DFT* res, const uint64_t res_size,  // res
                                           const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
                                           const VMP_PMAT* pmat, const uint64_t nrows,
                                           const uint64_t ncols,  // prep matrix
                                           uint8_t* tmp_space     // scratch space (a_size*sizeof(reim4) bytes)
) {
    const uint64_t m = module->m;
    const uint64_t nn = module->nn;
    assert(nn >= 8);

    double* mat2cols_output = (double*)tmp_space;     // 128 bytes
    double* extracted_blk = (double*)tmp_space + 16;  // 64*min(nrows,a_size) bytes

    double* mat_input = (double*)pmat;
    double* vec_input = (double*)a_dft;
    double* vec_output = (double*)res;

    // const uint64_t row_max0 = res_size < a_size ? res_size: a_size;
    const uint64_t row_max = nrows < a_size ? nrows : a_size;
    const uint64_t col_max = ncols < res_size ? ncols : res_size;

    if (nn >= 8) { 
      for (uint64_t blk_i = 0; blk_i < m / 4; blk_i++) {
        double* mat_blk_start = mat_input + blk_i * (8 * nrows * ncols);

        reim4_extract_1blk_from_contiguous_reim_ref(m, row_max, blk_i, (double*)extracted_blk, (double*)a_dft);
        // apply mat2cols
        for (uint64_t col_i = 0; col_i < col_max - 1; col_i += 2) {
          uint64_t col_offset = col_i * (8 * nrows);
          reim4_vec_mat2cols_product_ref(row_max, mat2cols_output, extracted_blk, mat_blk_start + col_offset);

          reim4_add_1blk_to_reim_ref(m, blk_i, vec_output + col_i * nn, mat2cols_output);
          reim4_add_1blk_to_reim_ref(m, blk_i, vec_output + (col_i + 1) * nn, mat2cols_output + 8);
        }

        // check if col_max is odd, then special case
        if (col_max % 2 == 1) {
          uint64_t last_col = col_max - 1;
          uint64_t col_offset = last_col * (8 * nrows);

          // the last column is alone in the pmat: vec_mat1col
          if (ncols == col_max) {
            reim4_vec_mat1col_product_ref(row_max, mat2cols_output, extracted_blk, mat_blk_start + col_offset);
          } else {
            // the last column is part of a colpair in the pmat: vec_mat2cols and ignore the second position
            reim4_vec_mat2cols_product_ref(row_max, mat2cols_output, extracted_blk, mat_blk_start + col_offset);
          }
          reim4_add_1blk_to_reim_ref(m, blk_i, vec_output + last_col * nn, mat2cols_output);
        }
      }
  }else { 
    for (uint64_t col_i = 0; col_i < col_max; col_i++) {
      double* pmat_col = mat_input + col_i * nrows * nn;
      for (uint64_t row_i = 0; row_i < row_max; row_i++) {
        reim_fftvec_addmul(module->mod.fft64.p_addmul, vec_output + col_i * nn, vec_input + row_i * nn,
                           pmat_col + row_i * nn);
      }
    }
  }

  // zero out remaining bytes
  memset(vec_output + col_max * nn, 0, (res_size - col_max) * nn * sizeof(double));
}

/** @brief this inner function could be very handy */
EXPORT void fft64_vmp_apply_dft_to_dft_ref(const MODULE* module,                       // N
                                           VEC_ZNX_DFT* res, const uint64_t res_size,  // res
                                           const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
                                           const VMP_PMAT* pmat, const uint64_t nrows,
                                           const uint64_t ncols,  // prep matrix
                                           uint8_t* tmp_space     // scratch space (a_size*sizeof(reim4) bytes)
) {
  const uint64_t m = module->m;
  const uint64_t nn = module->nn;
  assert(nn >= 8);

  double* mat2cols_output = (double*)tmp_space;     // 128 bytes
  double* extracted_blk = (double*)tmp_space + 16;  // 64*min(nrows,a_size) bytes

  double* mat_input = (double*)pmat;
  double* vec_input = (double*)a_dft;
  double* vec_output = (double*)res;

  // const uint64_t row_max0 = res_size < a_size ? res_size: a_size;
  const uint64_t row_max = nrows < a_size ? nrows : a_size;
  const uint64_t col_max = ncols < res_size ? ncols : res_size;

  if (nn >= 8) { 
    for (uint64_t blk_i = 0; blk_i < m / 4; blk_i++) {
      double* mat_blk_start = mat_input + blk_i * (8 * nrows * ncols);

      reim4_extract_1blk_from_contiguous_reim_ref(m, row_max, blk_i, (double*)extracted_blk, (double*)a_dft);
      // apply mat2cols
      for (uint64_t col_i = 0; col_i < col_max - 1; col_i += 2) {
        uint64_t col_offset = col_i * (8 * nrows);
        reim4_vec_mat2cols_product_ref(row_max, mat2cols_output, extracted_blk, mat_blk_start + col_offset);

        reim4_save_1blk_to_reim_ref(m, blk_i, vec_output + col_i * nn, mat2cols_output);
        reim4_save_1blk_to_reim_ref(m, blk_i, vec_output + (col_i + 1) * nn, mat2cols_output + 8);
      }

      // check if col_max is odd, then special case
      if (col_max % 2 == 1) {
        uint64_t last_col = col_max - 1;
        uint64_t col_offset = last_col * (8 * nrows);

        // the last column is alone in the pmat: vec_mat1col
        if (ncols == col_max) {
          reim4_vec_mat1col_product_ref(row_max, mat2cols_output, extracted_blk, mat_blk_start + col_offset);
        } else {
          // the last column is part of a colpair in the pmat: vec_mat2cols and ignore the second position
          reim4_vec_mat2cols_product_ref(row_max, mat2cols_output, extracted_blk, mat_blk_start + col_offset);
        }
        reim4_save_1blk_to_reim_ref(m, blk_i, vec_output + last_col * nn, mat2cols_output);
      }
    }
  }else {
    for (uint64_t col_i = 0; col_i < col_max; col_i++) {
      double* pmat_col = mat_input + col_i * nrows * nn;
      for (uint64_t row_i = 0; row_i < 1; row_i++) {
        reim_fftvec_mul(module->mod.fft64.mul_fft, vec_output + col_i * nn, vec_input + row_i * nn,
                        pmat_col + row_i * nn);
      }
      for (uint64_t row_i = 1; row_i < row_max; row_i++) {
        reim_fftvec_addmul(module->mod.fft64.p_addmul, vec_output + col_i * nn, vec_input + row_i * nn,
                           pmat_col + row_i * nn);
      }
    }
  }

  // zero out remaining bytes
  memset(vec_output + col_max * nn, 0, (res_size - col_max) * nn * sizeof(double));
}

/** @brief minimal size of the tmp_space */
EXPORT uint64_t fft64_vmp_apply_dft_tmp_bytes(const MODULE* module,           // N
                                              uint64_t res_size,              // res
                                              uint64_t a_size,                // a
                                              uint64_t nrows, uint64_t ncols  // prep matrix
) {
  const uint64_t nn = module->nn;
  const uint64_t row_max = nrows < a_size ? nrows : a_size;

  return (row_max * nn * sizeof(double)) + (128) + (64 * row_max);
}

/** @brief minimal size of the tmp_space */
EXPORT uint64_t fft64_vmp_apply_dft_to_dft_tmp_bytes(const MODULE* module,           // N
                                                     uint64_t res_size,              // res
                                                     uint64_t a_size,                // a
                                                     uint64_t nrows, uint64_t ncols  // prep matrix
) {
  const uint64_t row_max = nrows < a_size ? nrows : a_size;

  return (128) + (64 * row_max);
}

EXPORT void vmp_apply_dft_to_dft(const MODULE* module,                       // N
                                 VEC_ZNX_DFT* res, const uint64_t res_size,  // res
                                 const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
                                 const VMP_PMAT* pmat, const uint64_t nrows,
                                 const uint64_t ncols,  // prep matrix
                                 uint8_t* tmp_space     // scratch space (a_size*sizeof(reim4) bytes)
) {
  module->func.vmp_apply_dft_to_dft(module, res, res_size, a_dft, a_size, pmat, nrows, ncols, tmp_space);
}

EXPORT void vmp_apply_dft_to_dft_add(const MODULE* module,                       // N
                                 VEC_ZNX_DFT* res, const uint64_t res_size,  // res
                                 const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
                                 const VMP_PMAT* pmat, const uint64_t nrows,
                                 const uint64_t ncols,  // prep matrix
                                 uint8_t* tmp_space     // scratch space (a_size*sizeof(reim4) bytes)
) {
  module->func.vmp_apply_dft_to_dft_add(module, res, res_size, a_dft, a_size, pmat, nrows, ncols, tmp_space);
}

EXPORT uint64_t vmp_apply_dft_to_dft_tmp_bytes(const MODULE* module,           // N
                                               uint64_t res_size,              // res
                                               uint64_t a_size,                // a
                                               uint64_t nrows, uint64_t ncols  // prep matrix
) {
  return module->func.vmp_apply_dft_to_dft_tmp_bytes(module, res_size, a_size, nrows, ncols);
}

/** @brief applies a vmp product (result in DFT space) adds to res inplace */
EXPORT void vmp_apply_dft_add(const MODULE* module,                                  // N
                          VEC_ZNX_DFT* res, uint64_t res_size,                   // res
                          const int64_t* a, uint64_t a_size, uint64_t a_sl,      // a
                          const VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,  // prep matrix
                          uint8_t* tmp_space                                     // scratch space
) {
  module->func.vmp_apply_dft_add(module, res, res_size, a, a_size, a_sl, pmat, nrows, ncols, tmp_space);
}

/** @brief applies a vmp product (result in DFT space) */
EXPORT void vmp_apply_dft(const MODULE* module,                                  // N
                          VEC_ZNX_DFT* res, uint64_t res_size,                   // res
                          const int64_t* a, uint64_t a_size, uint64_t a_sl,      // a
                          const VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,  // prep matrix
                          uint8_t* tmp_space                                     // scratch space
) {
  module->func.vmp_apply_dft(module, res, res_size, a, a_size, a_sl, pmat, nrows, ncols, tmp_space);
}

/** @brief minimal size of the tmp_space */
EXPORT uint64_t vmp_apply_dft_tmp_bytes(const MODULE* module,           // N
                                        uint64_t res_size,              // res
                                        uint64_t a_size,                // a
                                        uint64_t nrows, uint64_t ncols  // prep matrix
) {
  return module->func.vmp_apply_dft_tmp_bytes(module, res_size, a_size, nrows, ncols);
}
