#include "cplx_fft.h"

void cplx_set(CPLX r, const CPLX a) {
  r[0] = a[0];
  r[1] = a[1];
}
void cplx_neg(CPLX r, const CPLX a) {
  r[0] = -a[0];
  r[1] = -a[1];
}
void cplx_add(CPLX r, const CPLX a, const CPLX b) {
  r[0] = a[0] + b[0];
  r[1] = a[1] + b[1];
}
void cplx_sub(CPLX r, const CPLX a, const CPLX b) {
  r[0] = a[0] - b[0];
  r[1] = a[1] - b[1];
}
void cplx_mul(CPLX r, const CPLX a, const CPLX b) {
  double re = a[0] * b[0] - a[1] * b[1];
  r[1] = a[0] * b[1] + a[1] * b[0];
  r[0] = re;
}

/**
 * @brief splits 2h evaluations of one polynomials into 2 times h evaluations of even/odd polynomial
 * Input:  Q_0(y),...,Q_{h-1}(y),Q_0(-y),...,Q_{h-1}(-y)
 * Output: P_0(z),...,P_{h-1}(z),P_h(z),...,P_{2h-1}(z)
 * where Q_i(X)=P_i(X^2)+X.P_{h+i}(X^2) and y^2 = z
 * @param h number of "coefficients" h >= 1
 * @param data 2h complex coefficients interleaved and 256b aligned
 * @param powom y represented as (yre,yim)
 */
EXPORT void cplx_split_fft_ref(int32_t h, CPLX* data, const CPLX powom) {
  CPLX* d0 = data;
  CPLX* d1 = data + h;
  for (uint64_t i = 0; i < h; ++i) {
    CPLX diff;
    cplx_sub(diff, d0[i], d1[i]);
    cplx_add(d0[i], d0[i], d1[i]);
    cplx_mul(d1[i], diff, powom);
  }
}

/**
 * Input: Q(y),Q(-y)
 * Output: P_0(z),P_1(z)
 * where Q(X)=P_0(X^2)+X.P_1(X^2) and y^2 = z
 * @param data 2 complexes coefficients interleaved and 256b aligned
 * @param powom (z,-z) interleaved: (zre,zim,-zre,-zim)
 */
void split_fft_last_ref(CPLX* data, const CPLX powom) {
  CPLX diff;
  cplx_sub(diff, data[0], data[1]);
  cplx_add(data[0], data[0], data[1]);
  cplx_mul(data[1], diff, powom);
}
