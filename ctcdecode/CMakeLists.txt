cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17) 

# find python package
find_package(Python COMPONENTS Interpreter Development)
# Check if Python was found
if(Python_FOUND)
    message("Python found: ${Python_EXECUTABLE}")
    
    # Include directories provided by Python
    include_directories(${Python_INCLUDE_DIRS})
    
else()
    message("Python not found.")
endif()

# build pybind11
include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.10.4
)

FetchContent_MakeAvailable(pybind11)

# define path to the libtorch extracted folder
set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/third_party/libtorch)

#find torch library and all necessary files
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_compile_options("-O3" "-DKENLM_MAX_ORDER=6" "-std=c++17" "-fPIC" "-DINCLUDE_KENLM")

# ctc decode
file(GLOB CTC_SOURCES ${CMAKE_SOURCE_DIR}/ctcdecode/src/*.cpp)
add_library(ctcdecode STATIC "${CTC_SOURCES}")

target_include_directories(ctcdecode PUBLIC ${CMAKE_SOURCE_DIR}/third_party/kenlm ${CMAKE_SOURCE_DIR}/third_party/openfst-1.8.2/src/include ${CMAKE_SOURCE_DIR}/third_party/utf8 ${CMAKE_SOURCE_DIR}/third_party/ThreadPool ${CMAKE_SOURCE_DIR}/third_party/boost_1_67_0  ${CMAKE_SOURCE_DIR}/ctcdecode/src ) 
target_link_libraries(ctcdecode PUBLIC "${TORCH_LIBRARIES}"  kenlm fst pybind11::module)

# message("${CTC_SOURCES}")