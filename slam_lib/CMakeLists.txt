cmake_minimum_required(VERSION 3.1)
project(SLAMLib VERSION 0.1 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(XUtils  REQUIRED)
find_package(XFusion REQUIRED)
find_package(CUDA 10.1 REQUIRED)

add_library(SLAM SHARED)

target_sources(SLAM 
PRIVATE
    # Core module
    fusion_core/rgbd_frame.cpp
    fusion_core/device_image.cpp

    # Tracking module
    icp_trackers/rgbd_tracker.cpp
    icp_trackers/dense_odometry.cpp

    # Reconstruction module
    reconstruction/dense_mapping.cpp

    # feature module
    feature_graph/feature_graph.cpp
    feature_graph/graph_optimizer.cpp
    utils/feature_extraction.cpp
)

set_target_properties(SLAM
PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 11
)

target_include_directories(SLAM
PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(SLAM
PUBLIC
    Eigen3::Eigen
    XUtils::XUtils
    XFusion::XFusion
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
)