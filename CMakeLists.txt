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
find_package(CUDA 10.1 REQUIRED)
find_package(Pangolin REQUIRED)

add_subdirectory(slam_lib)

# app gui
add_executable(app "")

target_sources(app
PRIVATE
    apps/system.cpp
    apps/main.cpp
    apps/window.cpp
)

target_include_directories(app
PRIVATE
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(app
PRIVATE
    SLAM
    OpenGL::GL
    ${Pangolin_LIBRARIES}
    ${CUDA_LIBRARIES}
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
