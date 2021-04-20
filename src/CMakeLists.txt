get_property(CTCDECODE_THIRD_PARTIES GLOBAL PROPERTY CTCDECODE_THIRD_PARTIES)

################################################################################
# libctcdecode
################################################################################
set(
  CTCDECODE_SOURCES
  path_trie.cpp
  decoder_utils.cpp
  ctc_beam_search_decoder.cpp
  binding.cpp
  )

add_library(
  ctcdecode
  SHARED
  ${CTCDECODE_SOURCES}
  )

target_link_libraries(
  ctcdecode
  PRIVATE
  ${TORCH_LIBRARIES}
  ${CTCDECODE_THIRD_PARTIES}
  )

install(
  TARGETS
  ctcdecode
  LIBRARY DESTINATION .
  )

set(CTCDECODE_LIBRARY -Wl,--no-as-needed ctcdecode -Wl,--as-needed CACHE INTERNAL "")
