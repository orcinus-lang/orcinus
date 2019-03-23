ExternalProject_Add(
    libutf8
    GIT_REPOSITORY "https://github.com/sheredom/utf8.h.git"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_DIR ${ORX_DIST_DIR}
    INSTALL_COMMAND cmake -E copy <SOURCE_DIR>/utf8.h <INSTALL_DIR>/include/
)

add_library(utf8 INTERFACE)
ExternalProject_Get_Property(libutf8 install_dir)

set(UTF8_LIBRARIES utf8)
set(UTF8_INSTALL_DIR ${install_dir})
set(UTF8_INCLUDE_DIRS "${UTF8_INSTALL_DIR}/include")

set(UTF8_LIBRARIES    ${UTF8_LIBRARIES}    PARENT_SCOPE)
set(UTF8_INSTALL_DIR  ${UTF8_INSTALL_DIR}  PARENT_SCOPE)
set(UTF8_INCLUDE_DIRS ${UTF8_INCLUDE_DIRS} PARENT_SCOPE)

target_include_directories(utf8 INTERFACE ${UTF8_INCLUDE_DIRS})
add_dependencies(utf8 libutf8)
