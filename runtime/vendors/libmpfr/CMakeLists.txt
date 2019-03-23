ExternalProject_Add(
    libmpfr
    URL "http://www.mpfr.org/mpfr-current/mpfr-4.0.2.tar.xz"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --with-gmp=${GMP_INSTALL_DIR} --enable-thread-safe --with-pic > ${ORX_CMAKE_DIR}/libmpfr-configure.log 2>&1
    BUILD_COMMAND make > ${ORX_CMAKE_DIR}/libmpfr-build.log 2>&1
    INSTALL_DIR ${ORX_DIST_DIR}
    INSTALL_COMMAND make install > ${ORX_CMAKE_DIR}/libmpfr-install.log 2>&1
    LOG_DOWNLOAD 0
    DOWNLOAD_NO_PROGRESS 1
)

add_library(mpfr SHARED IMPORTED GLOBAL)
add_dependencies(libmpfr gmp)
add_dependencies(libmpfr libgmp)
add_dependencies(mpfr libmpfr)
ExternalProject_Get_Property(libmpfr source_dir)
ExternalProject_Get_Property(libmpfr binary_dir)
ExternalProject_Get_Property(libmpfr install_dir)

set(MPFR_LIBRARIES mpfr)
set(MPFR_INSTALL_DIR ${install_dir})
set(MPFR_INCLUDE_DIRS "${MPFR_INSTALL_DIR}/include")

set_property(TARGET mpfr PROPERTY IMPORTED_LOCATION "${MPFR_INSTALL_DIR}/lib/libmpfr.so")
target_include_directories(gmp INTERFACE ${MPFR_INCLUDE_DIRS})

set(MPFR_LIBRARIES    ${MPFR_LIBRARIES}    PARENT_SCOPE)
