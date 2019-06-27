cmake_minimum_required(VERSION 3.1)
project(XFusion VERSION 0.1 LANGUAGES CXX CUDA)

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
find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(XUtils REQUIRED)
find_package(XFusion REQUIRED)

# add_subdirectory(slam_lib)

add_executable(app "")

target_sources(app
PRIVATE
    src/Frame.cpp
    src/ICPTracker.cpp
    src/KeyFrameGraph.cpp
    src/Main.cpp
    src/Mapping.cpp
    src/Odometry.cpp
    src/PointCloud.cpp
    src/System.cpp
    src/Window.cpp
    src/PoseEstimator.cpp
    src/FeatureExtractor.cpp
    src/DescriptorMatcher.cpp
    src/Relocalizer.cpp
)

set_target_properties(app
PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 11
)

target_include_directories(app
PUBLIC
    ${PROJECT_SOURCE_DIR}/include
PRIVATE
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(app
PRIVATE
    OpenGL::GL
    Eigen3::Eigen
    XUtils::XUtils
    XFusion::XFusion
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${Pangolin_LIBRARIES}
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
