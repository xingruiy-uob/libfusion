#ifndef __DEVICE_MAP__
#define __DEVICE_MAP__

#include "map_struct.h"
#include "intrinsic_matrix.h"
#include <sophus/se3.hpp>
#include <opencv2/cudaarithm.hpp>
#include <thrust/device_vector.h>

namespace fusion
{
namespace cuda
{

void update(
    MapStruct map_struct,
    const cv::cuda::GpuMat depth,
    const cv::cuda::GpuMat image,
    const cv::cuda::GpuMat normal,
    const Sophus::SE3d &frame_pose,
    const IntrinsicMatrix K,
    cv::cuda::GpuMat &cv_flag,
    cv::cuda::GpuMat &cv_pos_array,
    uint &visible_block_count);

void create_rendering_blocks(
    MapStruct map_struct,
    cv::cuda::GpuMat &zrange_x,
    cv::cuda::GpuMat &zrange_y,
    const Sophus::SE3d &frame_pose,
    const IntrinsicMatrix intrinsic_matrix);

void raycast(
    MapStruct map_struct,
    cv::cuda::GpuMat vmap,
    cv::cuda::GpuMat nmap,
    cv::cuda::GpuMat zrange_x,
    cv::cuda::GpuMat zrange_y,
    const Sophus::SE3d &pose,
    const IntrinsicMatrix intrinsic_matrix);

void raycast_with_colour(
    MapStruct map_struct,
    cv::cuda::GpuMat vmap,
    cv::cuda::GpuMat nmap,
    cv::cuda::GpuMat image,
    cv::cuda::GpuMat zrange_x,
    cv::cuda::GpuMat zrange_y,
    const Sophus::SE3d &pose,
    const IntrinsicMatrix intrinsic_matrix);

void create_scene_mesh(
    MapStruct map_struct,
    uint *block_count,
    int3 *block_list,
    uint *triangle_count,
    float3 *vertex_mesh);

} // namespace cuda
} // namespace fusion

#endif