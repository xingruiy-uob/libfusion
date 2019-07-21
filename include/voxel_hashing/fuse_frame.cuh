#ifndef FUSION_VOXEL_HASHING_FUSE_FRAME_H
#define FUSION_VOXEL_HASHING_FUSE_FRAME_H

#include <sophus/se3.hpp>
#include <opencv2/cudaarithm.hpp>
#include "data_struct/map_struct.h"
#include "data_struct/intrinsic_matrix.h"

namespace fusion
{
namespace cuda
{

void update(
    MapStorage map_struct,
    MapState state,
    const cv::cuda::GpuMat depth,
    const cv::cuda::GpuMat image,
    const Sophus::SE3d &frame_pose,
    const IntrinsicMatrix K,
    cv::cuda::GpuMat &cv_flag,
    cv::cuda::GpuMat &cv_pos_array,
    HashEntry *visible_blocks,
    uint &visible_block_count);

void update_weighted(
    MapStorage map_struct,
    MapState state,
    const cv::cuda::GpuMat depth,
    const cv::cuda::GpuMat normal,
    const cv::cuda::GpuMat image,
    const Sophus::SE3d &frame_pose,
    const IntrinsicMatrix K,
    cv::cuda::GpuMat &cv_flag,
    cv::cuda::GpuMat &cv_pos_array,
    HashEntry *visible_blocks,
    uint &visible_block_count);

} // namespace cuda
} // namespace fusion

#endif