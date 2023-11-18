cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

add_compile_options("-g" "-O3" "-DKENLM_MAX_ORDER=6" "-std=c++17" "-fPIC" "-DINCLUDE_KENLM")

# kenlm sources
file(GLOB KENLM_UTIL_LIB_SOURCES ${CMAKE_SOURCE_DIR}/third_party/kenlm/util/*.cc)
file(GLOB KENLM_LM_LIB_SOURCES ${CMAKE_SOURCE_DIR}/third_party/kenlm/lm/*.cc)
file(GLOB KENLM_DOUBLE_CONV_LIB_SOURCES ${CMAKE_SOURCE_DIR}/third_party/kenlm/util/double-conversion/*.cc)
add_library(kenlm STATIC "${KENLM_UTIL_LIB_SOURCES}" "${KENLM_LM_LIB_SOURCES}" "${KENLM_DOUBLE_CONV_LIB_SOURCES}")
target_include_directories(kenlm PUBLIC ${CMAKE_SOURCE_DIR}/third_party/kenlm ${CMAKE_SOURCE_DIR}/third_party/boost_1_67_0)

# openfst sources
file(GLOB FST_SOURCES ${CMAKE_SOURCE_DIR}/third_party/openfst-1.8.2/src/lib/*.cc)
add_library(fst STATIC "${FST_SOURCES}")
target_include_directories(fst PUBLIC ${CMAKE_SOURCE_DIR}/third_party/openfst-1.8.2/src/include ${CMAKE_SOURCE_DIR}/third_party/openfst-1.8.2/src/include/fst/script)