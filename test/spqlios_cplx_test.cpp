#include "gtest/gtest.h"

#include "../spqlios/cplx/cplx_fft.h"
#include "../spqlios/cplx.h"

#include <cmath>

#ifdef __x86_64__
TEST(fft, ifft16_fma_vs_ref) {
  CPLX data[16];
  CPLX omega[8];
  for (uint64_t i=0; i<32; ++i) ((double*)data)[i]= 2*i+1; //(rand()%100)-50;
  for (uint64_t i=0; i<16; ++i) ((double*)omega)[i]= i+1; //(rand()%100)-50;
  CPLX copydata[16];
  CPLX copyomega[8];
  memcpy(copydata, data, sizeof(copydata));
  memcpy(copyomega, omega, sizeof(copyomega));
  cplx_ifft16_avx_fma(data, omega);
  cplx_ifft16_ref(copydata, copyomega);
  double distance = 0;
  for (uint64_t i=0; i<16; ++i) {
    double d1 =  fabs(data[i][0]-copydata[i][0]);
    double d2 =  fabs(data[i][0]-copydata[i][0]);
    if (d1>distance) distance=d1;
    if (d2>distance) distance=d2;
  }
  /*
  printf("data:\n");
  for (uint64_t i=0; i<4; ++i) {
    for (uint64_t j=0; j<8; ++j) {
      printf("%.5lf ", data[4 * i + j / 2][j % 2]);
    }
    printf("\n");
  }
  printf("copydata:\n");
  for (uint64_t i=0; i<4; ++i) {
    for (uint64_t j=0; j<8; ++j) {
      printf("%5.5lf ", copydata[4 * i + j / 2][j % 2]);
    }
    printf("\n");
  }
  */
  ASSERT_EQ(distance, 0);
}
#endif
