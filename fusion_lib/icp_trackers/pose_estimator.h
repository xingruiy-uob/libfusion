#ifndef __SE3_REDUCTION__
#define __SE3_REDUCTION__

#include "intrinsic_matrix.h"
#include "sophus/se3.hpp"
#include <opencv2/cudaarithm.hpp>
#include <thrust/device_vector.h>

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
void compute_rgb_corresp(
    const cv::cuda::GpuMat last_vmap,
    const cv::cuda::GpuMat last_intensity,
    const cv::cuda::GpuMat curr_intensity,
    const cv::cuda::GpuMat curr_intensity_dx,
    const cv::cuda::GpuMat curr_intensity_dy,
    const Sophus::SE3d &frame_pose,
    const IntrinsicMatrix K,
    float4 *transformed_points,
    float4 *image_corresp_data,
    float *error_term_array,
    float *variance_term_array,
    float &mean_estimate,
    float &variance_estimate,
    uint &num_corresp);

void compute_least_square_RGB(
    const uint num_corresp,
    float4 *transformed_points,
    float4 *image_corresp_data,
    const float mean_estimated,
    const float stdev_estimated,
    const IntrinsicMatrix K,
    cv::cuda::GpuMat sum,
    cv::cuda::GpuMat out,
    float *hessian_estimated,
    float *residual_estimated,
    float *residual);

} // namespace fusion

#endif