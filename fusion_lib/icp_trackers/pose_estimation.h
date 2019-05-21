#ifndef __SE3_REDUCTION__
#define __SE3_REDUCTION__

#include "intrinsic_matrix.h"
#include "sophus/se3.hpp"
#include <opencv2/cudaarithm.hpp>

namespace fusion
{

void rgb_reduce(
    const cv::cuda::GpuMat &curr_intensity,
    const cv::cuda::GpuMat &last_intensity,
    const cv::cuda::GpuMat &last_vmap,
    const cv::cuda::GpuMat &curr_vmap,
    const cv::cuda::GpuMat &intensity_dx,
    const cv::cuda::GpuMat &intensity_dy,
    cv::cuda::GpuMat &sum,
    cv::cuda::GpuMat &out,
    const Sophus::SE3d &pose,
    const IntrinsicMatrix K,
    float *jtj, float *jtr,
    float *residual);

void icp_reduce(
    const cv::cuda::GpuMat &curr_vmap,
    const cv::cuda::GpuMat &curr_nmap,
    const cv::cuda::GpuMat &last_vmap,
    const cv::cuda::GpuMat &last_nmap,
    cv::cuda::GpuMat &sum,
    cv::cuda::GpuMat &out,
    const Sophus::SE3d &pose,
    const IntrinsicMatrix K,
    float *jtj, float *jtr,
    float *residual);

} // namespace fusion

#endif