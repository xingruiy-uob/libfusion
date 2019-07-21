#ifndef FUSION_VOXEL_HASHING_MAP_PROC
#define FUSION_VOXEL_HASHING_MAP_PROC

#include <sophus/se3.hpp>
#include <opencv2/cudaarithm.hpp>
#include "data_struct/map_struct.h"
#include "data_struct/intrinsic_matrix.h"

namespace fusion
{
namespace cuda
{

void create_rendering_blocks(
    uint count_visible_block,
    uint &count_redering_block,
    HashEntry *visible_blocks,
    cv::cuda::GpuMat &zrange_x,
    cv::cuda::GpuMat &zrange_y,
    RenderingBlock *rendering_blocks,
    const Sophus::SE3d &frame_pose,
    const IntrinsicMatrix cam_params);

void raycast(
    MapStorage map_struct,
    MapState state,
    cv::cuda::GpuMat vmap,
    cv::cuda::GpuMat nmap,
    cv::cuda::GpuMat zrange_x,
    cv::cuda::GpuMat zrange_y,
    const Sophus::SE3d &pose,
    const IntrinsicMatrix intrinsic_matrix);

void raycast_with_colour(
    MapStorage map_struct,
    MapState state,
    cv::cuda::GpuMat vmap,
    cv::cuda::GpuMat nmap,
    cv::cuda::GpuMat image,
    cv::cuda::GpuMat zrange_x,
    cv::cuda::GpuMat zrange_y,
    const Sophus::SE3d &pose,
    const IntrinsicMatrix intrinsic_matrix);

} // namespace cuda
} // namespace fusion

#endif