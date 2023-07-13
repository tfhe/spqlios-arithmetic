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
#include "spqlios/reim/reim_fft.h"
#include "spqlios/reim.h"


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

void benchmark_cplx_fft(benchmark::State& state) {
  const int32_t nn = state.range(0);
  CPLX_FFT_PRECOMP* a = new_cplx_fft_precomp(nn / 2, 1);
  double* c = (double*)cplx_fft_precomp_get_buffer(a, 0);
  init_random_values(nn, c);
  for (auto _ : state) {
    // cplx_fft_simple(nn/2, c);
    cplx_fft(a, c);
  }
  delete_cplx_fft_precomp(a);
}

void benchmark_reim_fft(benchmark::State& state) {
  const int32_t nn = state.range(0);
  const uint32_t m = nn / 2;
  REIM_FFT_PRECOMP* a = new_reim_fft_precomp(m, 1);
  double* c = reim_fft_precomp_get_buffer(a, 0);
  init_random_values(nn, c);
  for (auto _ : state) {
    // cplx_fft_simple(nn/2, c);
    reim_fft(a, c);
  }
  delete_reim_fft_precomp(a);
}

void benchmark_reim_ifft(benchmark::State& state) {
  const int32_t nn = state.range(0);
  const uint32_t m = nn / 2;
  REIM_IFFT_PRECOMP* a = new_reim_ifft_precomp(m, 1);
  double* c = reim_ifft_precomp_get_buffer(a, 0);
  init_random_values(nn, c);
  for (auto _ : state) {
    // cplx_ifft_simple(nn/2, c);
    reim_ifft(a, c);
  }
  delete_reim_ifft_precomp(a);
}

// #define ARGS Arg(1024)->Arg(8192)->Arg(32768)->Arg(65536)
#define ARGS Arg(64)->Arg(256)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192)->Arg(16384)->Arg(32768)->Arg(65536)

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  std::cout << "Dimensions n in the benchmark below are in \"real FFT\" modulo X^n+1" << std::endl;
  std::cout << "The complex dimension m (modulo X^m-i) is half of it" << std::endl;
  BENCHMARK(benchmark_cplx_ifft)->ARGS;
  BENCHMARK(benchmark_cplx_fft)->ARGS;
  BENCHMARK(benchmark_reim_fft)->ARGS;
  BENCHMARK(benchmark_reim_ifft)->ARGS;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
