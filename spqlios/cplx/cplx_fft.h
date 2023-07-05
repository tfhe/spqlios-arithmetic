#ifndef SPQLIOS_CPLX_FFT_H
#define SPQLIOS_CPLX_FFT_H

#include "cplx_fft_public.h"

/** @brief a complex number contains two doubles real,imag */
typedef double CPLX[2];

void cplx_set(CPLX r, const CPLX a);
void cplx_neg(CPLX r, const CPLX a);
void cplx_add(CPLX r, const CPLX a, const CPLX b);
void cplx_sub(CPLX r, const CPLX a, const CPLX b);
void cplx_mul(CPLX r, const CPLX a, const CPLX b);

/**
 * @brief splits 2h evaluations of one polynomials into 2 times h evaluations of even/odd polynomial
 * Input:  Q_0(y),...,Q_{h-1}(y),Q_0(-y),...,Q_{h-1}(-y)
 * Output: P_0(z),...,P_{h-1}(z),P_h(z),...,P_{2h-1}(z)
 * where Q_i(X)=P_i(X^2)+X.P_{h+i}(X^2) and y^2 = z
 * @param h number of "coefficients" h >= 1
 * @param data 2h complex coefficients interleaved and 256b aligned
 * @param powom y represented as (yre,yim)
 */
EXPORT void cplx_split_fft_ref(int32_t h, CPLX* data, const CPLX powom);

/**
 * Input: Q(y),Q(-y)
 * Output: P_0(z),P_1(z)
 * where Q(X)=P_0(X^2)+X.P_1(X^2) and y^2 = z
 * @param data 2 complexes coefficients interleaved and 256b aligned
 * @param powom (z,-z) interleaved: (zre,zim,-zre,-zim)
 */
void split_fft_last_ref(CPLX* data, const CPLX powom);

EXPORT void cplx_ifft_naive(const uint32_t m, const double entry_pwr, CPLX* data);
EXPORT void cplx_ifft16_avx_fma(void* data, const void* omega);
EXPORT void cplx_ifft16_ref(void* data, const void* omega);

/**
 * @brief compute the ifft evaluations of P in place
 * ifft(data) = ifft_rec(data, i);
 * function ifft_rec(data, omega) {
 *   if #data = 1: return data
 *   let s = sqrt(omega) w. re(s)>0
 *   let (u,v) = data
 *   return split_fft([ifft_rec(u, s), ifft_rec(v, -s)],s)
 * }
 * @param itables precomputed tables (contains all the powers of omega in the order they are used)
 * @param data vector of m complexes (coeffs as input, evals as output)
 */
EXPORT void cplx_ifft_ref(const CPLX_IFFT_PRECOMP* itables, void* data);
EXPORT void cplx_ifft_avx2_fma(const CPLX_IFFT_PRECOMP* itables, void* data);

#endif  // SPQLIOS_CPLX_FFT_H
