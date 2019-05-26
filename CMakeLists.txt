cmake_minimum_required(VERSION 3.1)
project(reloc_fusion CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(GLEW REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

add_subdirectory(fusion_lib)

# app gui
add_executable(app "")

target_sources(app
PRIVATE
    apps/system.cpp
    apps/main.cpp
    apps/window.cpp
)

target_include_directories(app
PUBLIC
    ${GLFW_INCLUDE_DIRS}
)

target_link_libraries(app
PRIVATE
    fusion_lib
    OpenGL::GL
    ${GLFW_STATIC_LIBRARIES}
    ${GLEW_LIBRARIES}
)

# app cli
add_executable(app_cli "")

target_sources(app_cli
PRIVATE
    apps/system.cpp
    apps/main.cpp
)

target_include_directories(app_cli
PUBLIC
    ${GLFW_INCLUDE_DIRS}
)

target_link_libraries(app_cli
PRIVATE
    fusion_lib
)


file (GLOB_RECURSE resources "shaders/*.*")
foreach(resource ${resources})
    get_filename_component(filename ${resource} NAME)

    set(output "${CMAKE_BINARY_DIR}/shaders/${filename}")
 
    add_custom_command(
        COMMENT "Moving updated resource-file '${filename}'"
        OUTPUT ${output}
        DEPENDS ${resource}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${resource}
        ${output}
    )
 
    add_custom_target(${filename} ALL DEPENDS ${resource} ${output})
 
endforeach()