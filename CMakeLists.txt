cmake_minimum_required(VERSION 3.0)
project(spqlios)

set(SPQLIOS_VERSION 0.1.0)
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" SPQLIOS_VERSION_BLAH ${SPQLIOS_VERSION})
set(SPQLIOS_VERSION_MAJOR ${CMAKE_MATCH_1})
set(SPQLIOS_VERSION_MINOR ${CMAKE_MATCH_2})
set(SPQLIOS_VERSION_PATCH ${CMAKE_MATCH_3})
message(STATUS "Version: ${SPQLIOS_VERSION_MAJOR}.${SPQLIOS_VERSION_MINOR}.${SPQLIOS_VERSION_PATCH}")

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


# compiles the main library in spqlios
add_subdirectory(spqlios)

# compiles and activates unittests and itests
if (${ENABLE_TESTING})
    enable_testing()
    add_subdirectory(test)
endif()

