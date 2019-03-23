ExternalProject_Add(
    libatomic_ops
    URL "https://www.hboehm.info/gc/gc_source/libatomic_ops-7.6.8.tar.gz"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --with-pic > ${ORX_CMAKE_DIR}/libatomic_ops-configure.log 2>&1
    BUILD_COMMAND make > ${ORX_CMAKE_DIR}/libatomic_ops-build.log 2>&1
    INSTALL_DIR ${ORX_DIST_DIR}
    INSTALL_COMMAND make install > ${ORX_CMAKE_DIR}/libatomic_ops-install.log 2>&1
    LOG_DOWNLOAD 0
    DOWNLOAD_NO_PROGRESS 1
)

add_library(atomic_ops STATIC IMPORTED GLOBAL)
add_dependencies(atomic_ops libatomic_ops)
ExternalProject_Get_Property(libatomic_ops source_dir)
ExternalProject_Get_Property(libatomic_ops binary_dir)
ExternalProject_Get_Property(libatomic_ops install_dir)

set(ATOMIC_LIBRARIES atomic_ops)
set(ATOMIC_SOURCE_DIR ${source_dir})
set(ATOMIC_BUILD_DIR ${binary_dir})
set(ATOMIC_INSTALL_DIR ${install_dir})
set(ATOMIC_INCLUDE_DIRS "${ATOMIC_INSTALL_DIR}/include")
set_property(TARGET atomic_ops PROPERTY IMPORTED_LOCATION "${ATOMIC_INSTALL_DIR}/lib/libatomic_ops.a")

set(ATOMIC_LIBRARIES    ${ATOMIC_LIBRARIES}    PARENT_SCOPE)
set(ATOMIC_INSTALL_DIR  ${ATOMIC_INSTALL_DIR}  PARENT_SCOPE)
set(ATOMIC_INCLUDE_DIRS ${ATOMIC_INCLUDE_DIRS} PARENT_SCOPE)
