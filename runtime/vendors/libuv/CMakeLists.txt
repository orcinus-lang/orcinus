ExternalProject_Add(
    libuv
    GIT_REPOSITORY "https://github.com/libuv/libuv.git"
    GIT_TAG "v1.27.0"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND <SOURCE_DIR>/autogen.sh                                   > ${ORX_CMAKE_DIR}/libuv-configure.log 2>&1
                   && <SOURCE_DIR>/configure --with-pic --prefix=<INSTALL_DIR> >> ${ORX_CMAKE_DIR}/libuv-configure.log 2>&1
    BUILD_COMMAND make > ${ORX_CMAKE_DIR}/libuv-build.log 2>&1
    INSTALL_COMMAND make install > ${ORX_CMAKE_DIR}/libuv-install.log 2>&1
    INSTALL_DIR ${ORX_DIST_DIR}
    LOG_DOWNLOAD 0
    DOWNLOAD_NO_PROGRESS 1
)

add_library(uv STATIC IMPORTED GLOBAL)
ExternalProject_Get_Property(libuv source_dir)
ExternalProject_Get_Property(libuv binary_dir)
ExternalProject_Get_Property(libuv install_dir)
add_dependencies(uv libuv)

set(UV_LIBRARIES uv)
set(UV_INCLUDE_DIRS "${source_dir}/include")
set_property(TARGET uv PROPERTY IMPORTED_LOCATION "${install_dir}/lib/libuv.a")

set(UV_LIBRARIES    ${UV_LIBRARIES}    PARENT_SCOPE)
set(UV_INSTALL_DIR  ${UV_INSTALL_DIR}  PARENT_SCOPE)
set(UV_INCLUDE_DIRS ${UV_INCLUDE_DIRS} PARENT_SCOPE)
