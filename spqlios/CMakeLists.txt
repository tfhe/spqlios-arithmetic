enable_language(ASM)

# C source files that are compiled for all targets (i.e. reference code)
set(SRCS_GENERIC
        commons.c 
        commons_private.c 
        cplx/cplx_common.c
        cplx/cplx_ifft_ref.c
        cplx/cplx_fft_ref.c
        cplx/cplx_conversions.c
        )
# C or assembly source files compiled only on x86 targets
set(SRCS_X86
        )
# C or assembly source files compiled only on aarch64 targets
set(SRCS_AARCH64
        cplx/cplx_fallbacks_aarch64.c
        )

# C or assembly source files compiled only on x86: avx, avx2, fma targets
set(SRCS_FMA
        cplx/cplx_ifft_avx2_fma.c
        cplx/cplx_ifft16_avx_fma.s
        cplx/cplx_fft_avx2_fma.c
        cplx/cplx_fft16_avx_fma.s
        cplx/cplx_conversions_avx2_fma.c
        )
set_source_files_properties(${SRCS_FMA} PROPERTIES COMPILE_OPTIONS "-mfma;-mavx;-mavx2")

# C or assembly source files compiled only on x86: avx512f/vl/dq + fma targets
set(SRCS_AVX512
        )
set_source_files_properties(${SRCS_AVX512} PROPERTIES COMPILE_OPTIONS "-mfma;-mavx512f;-mavx512vl;-mavx512dq")

# C or assembly source files compiled only on x86: avx2 + bmi targets
set(SRCS_AVX2
        )
set_source_files_properties(${SRCS_AVX2} PROPERTIES COMPILE_OPTIONS "-mbmi2;-mavx2")

# C source files on float128 via libquadmath on x86 targets targets
set(SRCS_F128
        cplx_f128/cplx_fft_f128.c
        cplx_f128/cplx_fft_f128.h
        )

# H header files containing the public API (these headers are installed)
set(HEADERSPUBLIC
        cplx.h
        reim.h
        reim4.h
        commons.h
        cplx/cplx_fft_public.h
        reim/reim_fft_public.h
        reim4/reim4_fftvec_public.h
        )

# H header files containing the private API (these headers are used internally)
set(HEADERSPRIVATE
        commons_private.h
        cplx/cplx_fft.h
        cplx/cplx_fft_private.h
        )

set(SPQLIOSSOURCES
        ${SRCS_GENERIC}
        ${HEADERSPUBLIC}
        ${HEADERSPRIVATE}
        )
if (${X86})
    set(SPQLIOSSOURCES ${SPQLIOSSOURCES}
            ${SRCS_X86}
            ${SRCS_FMA}
            ${SRCS_AVX2}
            ${SRCS_AVX512}
            )
endif ()
if (${AARCH64})
    set(SPQLIOSSOURCES ${SPQLIOSSOURCES}
            ${SRCS_AARCH64}
            )
endif ()


set(SPQLIOSLIBDEP
        m         # libmath depencency for cosinus/sinus functions
        )

if (ENABLE_SPQLIOS_F128)
    find_library(quadmath REQUIRED NAMES quadmath)
    set(SPQLIOSSOURCES ${SPQLIOSSOURCES} ${SRCS_F128})
    set(SPQLIOSLIBDEP ${SPQLIOSLIBDEP} quadmath)
endif (ENABLE_SPQLIOS_F128)

add_library(spqlios-fft-static STATIC ${SPQLIOSSOURCES})
add_library(spqlios-fft SHARED ${SPQLIOSSOURCES})
set_property(TARGET spqlios-fft-static PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET spqlios-fft-static PROPERTY OUTPUT_NAME spqlios-fft)
set_property(TARGET spqlios-fft PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET spqlios-fft PROPERTY SOVERSION ${SPQLIOS_VERSION_MAJOR})
set_property(TARGET spqlios-fft PROPERTY VERSION ${SPQLIOS_VERSION})
target_link_options(spqlios-fft-static PUBLIC -Wl,--no-undefined)
target_link_options(spqlios-fft PUBLIC -Wl,--no-undefined)
target_link_libraries(spqlios-fft ${SPQLIOSLIBDEP})
target_link_libraries(spqlios-fft-static ${SPQLIOSLIBDEP})
install(TARGETS spqlios-fft-static DESTINATION lib)
install(TARGETS spqlios-fft DESTINATION lib)

# install the public headers only
foreach (file ${HEADERSPUBLIC})
    get_filename_component(dir ${file} DIRECTORY)
    install(FILES ${file} DESTINATION include/spqlios/${dir})
endforeach ()
