cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)

# project name
project(CTCBeamDecoder CXX)

# define path to the libtorch extracted folder
set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/third_party/libtorch)

# find torch library and all necessary files
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add cxxopts library for command line parsing
include(FetchContent)

FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.1.1
)

FetchContent_GetProperties(cxxopts)

if (NOT cxxopts_POPULATED)
    FetchContent_Populate(cxxopts)
    add_subdirectory(${cxxopts_SOURCE_DIR} ${cxxopts_BINARY_DIR})
endif()

# add sudirectories
add_subdirectory(ctcdecode)
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party)
add_subdirectory(${CMAKE_SOURCE_DIR}/tests/cpp)

# build_fst library
add_library(build_fst_lib ${CMAKE_SOURCE_DIR}/tools/build_fst.cpp)
target_include_directories(build_fst_lib PUBLIC ${CMAKE_SOURCE_DIR}/tools)
target_link_libraries(build_fst_lib PUBLIC ctcdecode "${TORCH_LIBRARIES}" cxxopts pthread dl)

# executable to add that we want to compile and run
add_executable(build_fst ${CMAKE_SOURCE_DIR}/tools/build_fst_main.cpp)

# link libraries to our executable
target_link_libraries(build_fst build_fst_lib)