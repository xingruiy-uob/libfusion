cmake_minimum_required(VERSION 3.1)
project(fusion_lib CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(fusion REQUIRED)

add_library(fusion_lib STATIC "")

target_sources(fusion_lib 
PRIVATE
    # Core module
    fusion_core/rgbd_frame.cpp
    fusion_core/device_image.cpp

    # Basic IO module
    basic_io/rgbd_camera.cpp

    # Tracking module
    icp_trackers/rgbd_tracker.cpp
    icp_trackers/dense_odometry.cpp

    # Reconstruction module
    reconstruction/dense_mapping.cpp

    # Relocalization module
    relocalization/relocalizer.cpp
    relocalization/feature_point.cpp
    feature_graph/feature_graph.cpp
)

set_target_properties(fusion_lib
PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 11
)

target_include_directories(fusion_lib
PUBLIC
    ${PROJECT_SOURCE_DIR}/basic_io
    ${PROJECT_SOURCE_DIR}/fusion_core
    ${PROJECT_SOURCE_DIR}/icp_trackers
    ${PROJECT_SOURCE_DIR}/feature_graph
    ${PROJECT_SOURCE_DIR}/relocalization
    ${PROJECT_SOURCE_DIR}/reconstruction
    ${PROJECT_SOURCE_DIR}/reconstruction/cuda
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(fusion_lib
PUBLIC
    Eigen3::Eigen
    fusion
    /usr/lib/libOpenNI2.so
    ${OpenCV_LIBRARIES}
    ${OpenNI_LIBRARIES}
    ${CUDA_LIBRARIES}
)