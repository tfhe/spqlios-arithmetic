# SPQLIOS-FFT

## Description

The SPQLIOS-FFT library aims at providing capabilities for computing the Fast Fourier Transform (FTT/NTT) in the anticyclic ring $\mathbb{Z}[X]/(X^N +1)$ for N a power of 2. It uses AVX, AVX2 and FMA assembly vectorization instructions. This library was inspired by the [Project Nayuki](https://www.nayuki.io/page/fast-fourier-transform-in-x86-assembly) and was originally provided as part of the [TFHE: Fast Fully Homomorphic Encryption Library over the Torus](https://github.com/tfhe/tfhe) library. It is now provided as an independent library with extended features and optimizations. 

The library provides the following functionalities:
* Implementation of the FFT in the complex realm with a 64-bit representation of complex coefficients.
* Fast multiplication of polynomials in the ring Z[X]/(X^N+1) or R[X]/(X^N+1) with up to 52 bits of precision.


### Optimisations

The library auto-detects support for AVX, AVX2, FMA, AVX512F instructions at runtime and automatically switches to the 
most efficient implementation, without requiring to be compiled on the spot on the target machine. This allows to distribute 
not only source but also binary distributions of the library via standard package managers (e.g. apt-install) 


## Dependencies 

The SPQLIOS-FFT library is a C library that can be compiled with a standard C compiler, and depends only on libc and libm. The API 
interface can be used in a regular C code, and any other language via classical foreign APIs. 

The unittests and integration tests are in an optional part of the code, and are written in C++.  These tests rely on 
[```benchmark```](https://github.com/google/benchmark), and ```gtest``` libraries, and therefore require a C++17 compiler.

Currently, the project has been tested with the gcc,g++ >= 11.3.0 compiler under Linux (x86_64). In the future, we plan to
extend the compatibility to other compilers, platforms and operating systems. 


## Installation

The library uses a classical ```cmake``` build mechanism: use ```cmake``` to create a ```build``` folder in the top level directory and run ```make``` from inside it. This assumes that the standard tool cmake is already installed on the system, and an up-to-date c++ compiler (i.e. g++ >=11.3.0) as well.

It will compile the shared library in optimized mode, and ```make install``` install it to the desired prefix folder (by default ```/usr/local/lib```).

If you want to choose additional compile options (i.e. other installation folder, debug mode, tests), you need to run cmake manually and pass the desired options:
```
mkdir build
cd build
cmake ../src -CMAKE_INSTALL_PREFIX=/usr/
make
```
The available options are the following:

| Variable Name          | values           | 
|------------------------|-------|
| CMAKE_INSTALL_PREFIX   | */usr/local* installation folder (libs go in lib/ and headers in include/) | 
| WARNING_PARANOID | All warnings are shown and treated as errors. Off by default |
| ENABLE_TESTING | Compiles unit tests and integration tests |



## Usage

### Layouts for complex FFT

The complex FFT handles polynomials with complex coefficients `C[X] mod X^M-i`, each one represented by two consecutive doubles `(real,imag)`. Note that a real polynomial $$\sum_{j=0}^{N-1} p_j\cdot X^j \mathtt{ mod } X^N+1$$ corresponds to the complex polynomial of half degree `M=N/2`: 
$$\sum_{j=0}^{M-1} (p_{j} + i.p_{j+M}) \cdot X^j \mathtt{ mod } X^M-i$$

In both cases, the FFT of the polynomial is the vector of its (complex) evaluations on the $M$ roots of $X^M -i$.

For a complex polynomial $C(X) = \sum c_i \cdot X^i$ of degree $M-1$ or a real polynomial $A(X) = \sum a_i \cdot X^i$ of degree N, the library handles multiple layouts, as each layout is more efficient for a specific operation. Next we describe the different layouts and then show how to use them in the next section.

#### A. cplx layout:

The coefficient space is represented as the array $a_0,a_M,a_1,a_{M+1},...,a_{M-1},a_{2M-1}$ or equivalently $Re(c_0),Im(c_0),Re(c_1),Im(c_1),...Re(c_{M-1}),Im(c_{M-1})$ in the case of the complex space.

The polynomial is evaluated at the roots of unity $\omega_{0}, \dots, \omega_{M}$: $c(\omega_{0}),...,c(\omega_{M-1})$ where 
$\omega_j = \omega^{1+\mathtt{rev}_{2N}(j)}$ and $\omega = \exp(i\cdot\pi/N)$

$\mathtt{rev}_{2N}(j)$ is the number that has the $\log_2(2N)$ bits of $j$ in reverse order.

#### B. reim layout:

In this layout for the complex space the coefficients are represented by grouping real and imaginary components: $Re(c_0), Re(c_1), Re(c_2), ..., Re(c_{M-1}), Im(c_0), Im(c_1),...,Im(c_{M-1})$ 

#### C. reim4 layout:

In this layout the real and imaginary components are in groups of size $4$ : $Re(c_0), Re(c_1), ..., Re(c_4), Im(c_0), Im(c_1), ..., Im(c_4),..., Re(c_{M-4}), Re(c_{M-3}),...,$ $Re(c_{M-1}), Im(c_{M-4}), Im(c_{M-3}), ..., Im(c_{M-1})$



### Converting between layouts


Differently than in cplx or reim layouts where the evaluations of polynomials (`fftvec`) come in reverse-bit order, in case of the reim4 layout the evaluations of a polynomial (`reim4_fftvec`) come in an arbitrary order and the only guarantee is that, on a given processor, conversions `cplx <--> reim4` or `reim <--> reim4` are consistent (converting back and forth will end up in the original order). However, a change of CPU or a cross-converstion `cplx --> reim4 --> reim` does *not* guarantee recovering the original order.


### Compute a polynomial multiplication via complex FFT, using the simpleAPI

Usage:
```
#Input: 2 arrays a,b of doubles of length nn, each one representing nn/2 complex coefficients
 in the cplx layout:
# 1. Compute a's FFT
cplx_fft_simple(nn / 2, a);
# 2. Compute b's FFT
cplx_fft_simple(nn / 2, b);
# 3. Compute the coefficient-wise computation and obtain the result in array 'c'
cplx_fftvec_addmul_simple(nn / 2, c, a, b);
# 4. Compute the iFFT of 'c'
cplx_ifft_simple(nn / 2, c);  // note that coefficients are not normalized
```

Example:
```
// define the complex coefficients of two polynomials mod X^4-i
double a[4][2] = {{1.1,2.2},{3.3,4.4},{5.5,6.6},{7.7,8.8}};
double b[4][2] = {{9.,10.},{11.,12.},{13.,14.},{15.,16.}};
double c[4][2]; // for the result
cplx_fft_simple(4, a);
cplx_fft_simple(4, b);
cplx_fftvec_mul_simple(4, c, a, b);
cplx_ifft_simple(4, c);
// c contains the complex coefficients 4.a*b mod X^4-i
```

### Compute a polynomial multiplication via complex FFT, using the simpleAPI and reim layout

Using the cplx layout to represent polynomials modulo $X^8+1$ with real coefficients is equivalent to using the reim layout to represent a polynomial with complex coefficients in $X^4-i$. Hence sometimes it will be easier to use directly the reim layout. The minimum polynomial degree to use this layout will be $16$ in case of real coefficients or $8$ in case of complex ones.

Example:
```
double a[16] = {1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9,10.,11.,12.,13.,14.,15.,16.};
double b[16] = {17.,18.,19.,20.,21.,22.,23.,24.,25.,26.,27.,28.,29.,30., 31.,32.};
double c[16]; 
reim_fft_simple(8, a);
reim_fft_simple(8, b);
reim_fftvec_mul_simple(8, c, a, b);
reim_ifft_simple(8, c);
// c contains the real coefficients 8.a*b mod X^16+1
```



