include(ExternalProject)

set(INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../install)
set(ARCHIVE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../archives)

set(envs
  "CFLAGS=-fPIC -fvisibility=hidden $ENV{CFLAGS}"
  )

set(OPENFST_LIBRARIES
  ${INSTALL_DIR}/lib/libfstscript.a
  ${INSTALL_DIR}/lib/libfst.a
  )

ExternalProject_Add(openfst
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  DOWNLOAD_DIR ${ARCHIVE_DIR}
  URL http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.7.9.tar.gz
  URL_HASH SHA256=9319aeb31d1e2950ae25449884e255cc2bc9dfaf987f601590763e61a10fbdde
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${CMAKE_CURRENT_BINARY_DIR}/src/openfst/configure --prefix=${INSTALL_DIR} --enable-static --disable-shared --disable-python --with-pic
  BUILD_BYPRODUCTS ${OPENFST_LIBRARIES}
  DOWNLOAD_NO_PROGRESS ON
  LOG_DOWNLOAD ON
  LOG_UPDATE ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_MERGED_STDOUTERR ON
  LOG_OUTPUT_ON_FAILURE ON
)

add_dependencies(libopenfst openfst)
set(OPENFST_INCLUDE_DIR ${INSTALL_DIR}/include PARENT_SCOPE)
set(OPENFST_LIBRARIES ${OPENFST_LIBRARIES} PARENT_SCOPE)
