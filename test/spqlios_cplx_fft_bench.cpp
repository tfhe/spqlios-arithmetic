#include <benchmark/benchmark.h>

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "../spqlios/cplx/cplx_fft.h"
#include "../spqlios/cplx.h"

using namespace std;

void init_random_values(uint64_t n, double* v) {
  for (uint64_t i = 0; i < n; ++i) v[i] = rand() - (RAND_MAX >> 1);
}

void benchmark_cplx_ifft(benchmark::State& state) {
  const int32_t nn = state.range(0);
  CPLX_IFFT_PRECOMP* a = new_cplx_ifft_precomp(nn / 2, 1);
  double* c = (double*)cplx_ifft_precomp_get_buffer(a, 0);
  init_random_values(nn, c);
  for (auto _ : state) {
    cplx_ifft(a, c);
  }
  delete_cplx_ifft_precomp(a);
}

// #define ARGS Arg(1024)->Arg(8192)->Arg(32768)->Arg(65536)
#define ARGS Arg(64)->Arg(256)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192)->Arg(16384)->Arg(32768)->Arg(65536)

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  std::cout << "Dimensions n in the benchmark below are in \"real FFT\" modulo X^n+1" << std::endl;
  std::cout << "The complex dimension m (modulo X^m-i) is half of it" << std::endl;
  BENCHMARK(benchmark_cplx_ifft)->ARGS;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
