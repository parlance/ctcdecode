cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)

# TEST EXECUTABLE

# add google test library
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Create an executable for the tests
add_executable(build_fst_test ${CMAKE_SOURCE_DIR}/tests/cpp/test_build_fst.cpp )

# Link Google Test to the test executable
target_link_libraries(build_fst_test gtest gtest_main build_fst_lib)
target_sources(build_fst_test PRIVATE ${CMAKE_SOURCE_DIR}/tools/build_fst.cpp)
target_compile_definitions(build_fst_test PUBLIC TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/cpp/fixtures")


# Add the tests to CTest
include(GoogleTest)
gtest_discover_tests(build_fst_test)