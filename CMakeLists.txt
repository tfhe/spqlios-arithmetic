cmake_minimum_required(VERSION 3.0)
project(spqlios)

# read the current version from the manifest file
file(READ "manifest.yaml" manifest)
string(REGEX MATCH "version: +(([0-9]+)\\.([0-9]+)\\.([0-9]+))" SPQLIOS_VERSION_BLAH ${manifest})
#message(STATUS "Version: ${SPQLIOS_VERSION_BLAH}")
set(SPQLIOS_VERSION ${CMAKE_MATCH_1})
set(SPQLIOS_VERSION_MAJOR ${CMAKE_MATCH_2})
set(SPQLIOS_VERSION_MINOR ${CMAKE_MATCH_3})
set(SPQLIOS_VERSION_PATCH ${CMAKE_MATCH_4})
message(STATUS "Compiling spqlios-fft version: ${SPQLIOS_VERSION_MAJOR}.${SPQLIOS_VERSION_MINOR}.${SPQLIOS_VERSION_PATCH}")

#set(ENABLE_SPQLIOS_F128 ON CACHE BOOL "Enable float128 via libquadmath")
set(WARNING_PARANOID ON CACHE BOOL "Treat all warnings as errors")
set(ENABLE_TESTING ON CACHE BOOL "Compiles unittests and integration tests")

if (WARNING_PARANOID)
    add_compile_options(-Wall -Werror)
endif()


if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(X86 ON)
    set(AARCH64 OFF)
else ()
    set(X86 OFF)
    # set(ENABLE_SPQLIOS_F128 OFF) # float128 are only supported for x86 targets
endif ()
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)")
    set(AARCH64 ON)
endif ()

if (WIN32)
    message(STATUS "win32 detected!")
    #overrides for win32
    set(X86 OFF)
    set(AARCH64 OFF)
    set(X86_WIN32 ON)
    include_directories(/usr/local/mingw/include)
endif (WIN32)

# compiles the main library in spqlios
add_subdirectory(spqlios)

# compiles and activates unittests and itests
if (${ENABLE_TESTING})
    enable_testing()
    add_subdirectory(test)
endif()

