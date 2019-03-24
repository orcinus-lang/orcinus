# CFLAGS="$CFLAGS -I${srcdir}/include -I${srcdir}/tests"

ExternalProject_Add(
    libgc
    URL "https://www.hboehm.info/gc/gc_source/gc-8.0.2.tar.gz"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND <SOURCE_DIR>/configure    --prefix=<INSTALL_DIR>
                                                --with-pic
                                                --enable-static=yes
                                                --enable-shared=no
                                                --with-libatomic-ops=none
                                                --disable-silent-rules
                                                --disable-cplusplus
                                                --enable-large-config
                                                --enable-handle-fork
                                                --enable-threads=pthreads
                                                --enable-thread-local-alloc
                                                --enable-parallel-mark
                                                --enable-gc-assertions
                                                --enable-mmap
                                                > ${ORX_CMAKE_DIR}/libgc-configure.log 2>&1
    BUILD_COMMAND make > ${ORX_CMAKE_DIR}/libgc-build.log 2>&1
    INSTALL_DIR ${ORX_DIST_DIR}
    INSTALL_COMMAND make install > ${ORX_CMAKE_DIR}/libgc-install.log 2>&1
    LOG_DOWNLOAD 0
    DOWNLOAD_NO_PROGRESS 1
)

add_library(gc STATIC IMPORTED GLOBAL)
add_dependencies(libgc atomic_ops)
add_dependencies(gc libgc)
ExternalProject_Get_Property(libgc source_dir)
ExternalProject_Get_Property(libgc binary_dir)
ExternalProject_Get_Property(libgc install_dir)

set(GC_LIBRARIES gc)
set(GC_INSTALL_DIR ${install_dir})
set(GC_INCLUDE_DIRS "${GC_INSTALL_DIR}/include")

set_property(TARGET gc PROPERTY IMPORTED_LOCATION "${GC_INSTALL_DIR}/lib/libgc.a")
target_include_directories(gc INTERFACE ${GC_INCLUDE_DIRS})

set(GC_LIBRARIES    ${GC_LIBRARIES}    PARENT_SCOPE)
