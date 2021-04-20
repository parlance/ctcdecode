set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

################################################################################
# OpenFST
################################################################################
add_library(libopenfst INTERFACE)
add_subdirectory(openfst)
target_include_directories(libopenfst INTERFACE ${OPENFST_INCLUDE_DIR})
target_link_libraries(libopenfst INTERFACE ${OPENFST_LIBRARIES})
list(APPEND CTCDECODE_THIRD_PARTIES libopenfst)

################################################################################
# Set third parties globally
################################################################################
set_property(GLOBAL PROPERTY CTCDECODE_THIRD_PARTIES "${CTCDECODE_THIRD_PARTIES}")
