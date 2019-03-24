set(SOURCE_FILES coro.c coro.h)
set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE C)

add_library(coro STATIC ${SOURCE_FILES})
target_compile_definitions(coro PUBLIC $<$<CONFIG:Debug>:CORO_USE_VALGRIND>)
target_compile_definitions(coro PUBLIC CORO_ASM)
#target_compile_definitions(coro PUBLIC CORO_GUARDPAGES)
#target_compile_definitions(coro PUBLIC CORO_STACKALLOC=0)
target_include_directories(coro PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set(CORO_LIBRARIES coro PARENT_SCOPE)
