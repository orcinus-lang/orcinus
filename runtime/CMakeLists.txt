# This file is part of Orcinus bootstrap project.
#
# For the full copyright and license information, please view the LICENSE
# file that was distributed with this source code.

# - cmake and project settings
cmake_minimum_required(VERSION 3.8)
project(orcinus-stdlib)

# - disallow in-source builds and make sure the user doesn't play dirty with symlinks
get_filename_component(ORX_ROOT_DIR  "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(ORX_CMAKE_DIR "${CMAKE_BINARY_DIR}" REALPATH)
if (${ORX_ROOT_DIR} STREQUAL ${ORX_CMAKE_DIR})
    file(REMOVE_RECURSE ${CMAKE_SOURCE_DIR}/CMakeCache.txt ${CMAKE_SOURCE_DIR}/CMakeFiles)

    # disable in-source builds
    if (EXISTS ${CMAKE_SOURCE_DIR}/CMakeCache.txt OR EXISTS ${CMAKE_SOURCE_DIR}/CMakeFiles)
        message(ERROR "CMakeCache.txt or CMakeFiles exists in source directory!")
    endif ()
    message(FATAL_ERROR "In-source builds are forbidden!")
endif (${ORX_ROOT_DIR} STREQUAL ${ORX_CMAKE_DIR})

# - source directories
set(ORX_VENDORS_DIR         "${ORX_ROOT_DIR}/vendors")
set(ORX_SOURCE_DIR          "${ORX_ROOT_DIR}/source")

# - project output directories
set(ORX_DIST_DIR            "${ORX_ROOT_DIR}/dist")
set(LIBRARY_OUTPUT_PATH     "${ORX_DIST_DIR}/lib")

# - cmake dist directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}")

# - cmake include directories
file(MAKE_DIRECTORY "${ORX_DIST_DIR}/include/")

# - generic c compiler configuration
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# - pattern of header for source changes
list(APPEND ORX_SOURCE_PATTERN "${ORX_SOURCE_DIR}/*.h")
list(APPEND ORX_SOURCE_PATTERN "${ORX_SOURCE_DIR}/*.c")

# Find source files
file (GLOB_RECURSE ORX_SOURCE_FILES ${ORX_SOURCE_PATTERN})
list (REMOVE_DUPLICATES ORX_SOURCE_FILES)

# - cmake extensions
include(ExternalProject)

## - ICU library
#find_package(ICU REQUIRED "data" "uc" "io")

## - thread support
#find_package(Threads)

# - vendor libraries and tools
add_subdirectory(${ORX_VENDORS_DIR}/libutf8)
add_subdirectory(${ORX_VENDORS_DIR}/libcoro)
add_subdirectory(${ORX_VENDORS_DIR}/libatomic)
add_subdirectory(${ORX_VENDORS_DIR}/libgc)
add_subdirectory(${ORX_VENDORS_DIR}/libuv)

# Create executable target
set                       (ORX_MODULE orcinus-stdlib)
add_library               (${ORX_MODULE} STATIC ${ORX_SOURCE_FILES})
set_target_properties     (${ORX_MODULE} PROPERTIES CLEAN_DIRECT_OUTPUT On)
target_include_directories(${ORX_MODULE} PRIVATE ${ORX_SOURCE_DIR})
target_include_directories(${ORX_MODULE} PRIVATE ${ICU_INCLUDE_DIRS})
#target_link_libraries     (${ORX_MODULE} ${ICU_LIBRARIES})
target_link_libraries     (${ORX_MODULE} ${CORO_LIBRARIES})
target_link_libraries     (${ORX_MODULE} ${UTF8_LIBRARIES})
target_link_libraries     (${ORX_MODULE} ${GC_LIBRARIES})
target_link_libraries     (${ORX_MODULE} ${UV_LIBRARIES})
#target_link_libraries     (${ORX_MODULE} ${CMAKE_THREAD_LIBS_INIT})
#set_target_properties     (${ORX_MODULE} PROPERTIES COMPILE_FLAGS "-fsplit-stack")
