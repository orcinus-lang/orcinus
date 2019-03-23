ExternalProject_Add(
    liblfds
    URL "http://www.liblfds.org/downloads/liblfds%20release%207.1.1%20source.tar.bz2"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND cd <SOURCE_DIR>/liblfds711/build/gcc_gnumake && make INSINCDIR=<INSTALL_DIR>/include/ INSLIBDIR=<INSTALL_DIR>/lib/ > ${ORX_CMAKE_DIR}/liblfds-build.log 2>&1
    INSTALL_COMMAND cd <SOURCE_DIR>/liblfds711/build/gcc_gnumake && make ar_install INSINCDIR=<INSTALL_DIR>/include/ INSLIBDIR=<INSTALL_DIR>/lib/ > ${ORX_CMAKE_DIR}/liblfds-install.log 2>&1
    INSTALL_DIR ${ORX_DIST_DIR}
    LOG_DOWNLOAD 0
    DOWNLOAD_NO_PROGRESS 1
)

add_library(lfds STATIC IMPORTED GLOBAL)
add_dependencies(lfds liblfds)
ExternalProject_Get_Property(liblfds source_dir)
ExternalProject_Get_Property(liblfds binary_dir)
ExternalProject_Get_Property(liblfds install_dir)

set(LFDS_LIBRARIES lfds)
set(LFDS_INSTALL_DIR ${install_dir})
set(LFDS_INCLUDE_DIRS "${source_dir}/include")
set_property(TARGET lfds PROPERTY IMPORTED_LOCATION "${install_dir}/lib/liblfds711.a")

set(LFDS_LIBRARIES    ${LFDS_LIBRARIES}    PARENT_SCOPE)
set(LFDS_INSTALL_DIR  ${LFDS_INSTALL_DIR}  PARENT_SCOPE)
set(LFDS_INCLUDE_DIRS ${LFDS_INCLUDE_DIRS} PARENT_SCOPE)
