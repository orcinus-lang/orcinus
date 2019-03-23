ExternalProject_Add(
    libgmp
    URL "https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --with-pic > ${ORX_CMAKE_DIR}/libgmp-configure.log 2>&1
    BUILD_COMMAND make > ${ORX_CMAKE_DIR}/libgmp-build.log 2>&1
    INSTALL_DIR ${ORX_DIST_DIR}
    INSTALL_COMMAND make install > ${ORX_CMAKE_DIR}/libgmp-install.log 2>&1
    LOG_DOWNLOAD 0
    DOWNLOAD_NO_PROGRESS 1
)

add_library(gmp SHARED IMPORTED GLOBAL)
add_dependencies(gmp libgmp)
ExternalProject_Get_Property(libgmp install_dir)

set(GMP_LIBRARIES gmp)
set(GMP_INSTALL_DIR ${install_dir})
set(GMP_INCLUDE_DIRS "${GMP_INSTALL_DIR}/include")

set_property(TARGET gmp PROPERTY IMPORTED_LOCATION "${GMP_INSTALL_DIR}/lib/libgmp.so")
target_include_directories(gmp INTERFACE ${GMP_INCLUDE_DIRS})

set(GMP_LIBRARIES    ${GMP_LIBRARIES}    PARENT_SCOPE)
set(GMP_INSTALL_DIR  ${GMP_INSTALL_DIR}  PARENT_SCOPE)
