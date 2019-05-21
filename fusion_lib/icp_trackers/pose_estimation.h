#ifndef __SE3_REDUCTION__
#define __SE3_REDUCTION__

#include "intrinsic_matrix.h"
#include "sophus/se3.hpp"
#include <opencv2/cudaarithm.hpp>

namespace fusion
{

// Simple Dense Image Alignment
// NOTE: easily affected by outliers
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

// Point-to-Plane ICP
// This computes one icp step and returns hessian
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

// TODO: Robust RGB fitting
void compute_rgb_correspondence(
    const cv::cuda::GpuMat curr_intensity,
    const cv::cuda::GpuMat last_intensity,
    const cv::cuda::GpuMat curr_intensity_dx,
    const cv::cuda::GpuMat curr_intensity_dy,
    const cv::cuda::GpuMat last_vmap,
    const Sophus::SE3d pose,
    const IntrinsicMatrix K);

} // namespace fusion

#endif