set(CMAKE_CXX_STANDARD 17)

set(test_incs ..)
set(gtest_libs)
set(benchmark_libs)
# searching for libgtest
find_path(gtest_inc NAMES gtest/gtest.h)
find_library(gtest NAMES gtest)
find_library(gtest_main REQUIRED NAMES gtest_main)
if (gtest_inc AND gtest AND gtest_main)
	message(STATUS "Found gtest: I=${gtest_inc} L=${gtest},${gtest_main}")
	set(test_incs ${test_incs} ${gtest_inc})
	set(gtest_libs ${gtest_libs} ${gtest} ${gtest_main} pthread)
else()
	message(FATAL_ERROR "Libgtest not found (required if ENABLE_TESTING is on): I=${gtest_inc} L=${gtest},${gtest_main}")
endif()
# searching for libbenchmark
find_path(benchmark_inc NAMES benchmark/benchmark.h)
find_library(benchmark NAMES benchmark)
if (benchmark_inc AND benchmark)
        message(STATUS "Found benchmark: I=${benchmark_inc} L=${benchmark}")
	set(test_incs ${test_incs} ${benchmark_inc})
	set(benchmark_libs ${benchmark_libs} ${benchmark})
else()
	message(FATAL_ERROR "Libbenchmark not found (required if ENABLE_TESTING is on): I=${benchmark_inc} L=${benchmark}")
endif()
find_path(VALGRIND_DIR NAMES valgrind/valgrind.h)
if (VALGRIND_DIR)
	message(STATUS "Found valgrind header ${VALGRIND_DIR}")
else ()
	# for now, we will fail if we don't find valgrind for tests
	message(STATUS "CANNOT FIND valgrind header: ${VALGRIND_DIR}")
endif ()

add_library(spqlios-testlib SHARED
		testlib/random.cpp
		testlib/test_commons.h
		testlib/test_commons.cpp
		testlib/mod_q120.h
		testlib/mod_q120.cpp
)
if (VALGRIND_DIR)
	target_include_directories(spqlios-testlib PRIVATE ${VALGRIND_DIR})
	target_compile_definitions(spqlios-testlib PRIVATE VALGRIND_MEM_TESTS)
endif ()



# main unittest file
message(STATUS "${gtest_libs}")
add_executable(spqlios-test
		spqlios_test.cpp
		spqlios_cplx_test.cpp
		spqlios_cplx_conversions_test.cpp
		spqlios_reim_test.cpp
		spqlios_reim_conversions_test.cpp
		spqlios_q120_ntt_test.cpp
)
target_link_libraries(spqlios-test spqlios-testlib libspqlios ${gtest_libs})
target_include_directories(spqlios-test PRIVATE ${test_incs})
add_test(NAME spqlios-test COMMAND spqlios-test)


# benchmarks
add_executable(spqlios-cplx-fft-bench spqlios_cplx_fft_bench.cpp)
target_link_libraries(spqlios-cplx-fft-bench libspqlios benchmark pthread)
target_include_directories(spqlios-cplx-fft-bench PRIVATE ../)

if (${X86})
    add_executable(spqlios-q120-ntt-bench spqlios_q120_ntt_bench.cpp)
    target_link_libraries(spqlios-q120-ntt-bench libspqlios benchmark pthread)
    target_include_directories(spqlios-q120-ntt-bench PRIVATE ${test_incs})
endif ()
