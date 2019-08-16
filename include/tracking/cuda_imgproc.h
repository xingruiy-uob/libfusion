#ifndef FUSION_CORE_CUDA_IMGPROC_H
#define FUSION_CORE_CUDA_IMGPROC_H

#include <sophus/se3.hpp>
#include <opencv2/cudaarithm.hpp>
#include <math/vector_type.h>
#include "data_struct/intrinsic_matrix.h"

namespace fusion
{

FUSION_HOST void filterDepthBilateral(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst);
FUSION_HOST void pyrDownDepth(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst);
FUSION_HOST void pyrDownImage(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst);
FUSION_HOST void pyrDownVMap(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst);
FUSION_HOST void computeDerivative(const cv::cuda::GpuMat image, cv::cuda::GpuMat &dx, cv::cuda::GpuMat &dy);
FUSION_HOST void backProjectDepth(const cv::cuda::GpuMat depth, cv::cuda::GpuMat &vmap, const IntrinsicMatrix &K);
FUSION_HOST void computeNMap(const cv::cuda::GpuMat vmap, cv::cuda::GpuMat &nmap);
FUSION_HOST void renderScene(const cv::cuda::GpuMat vmap, const cv::cuda::GpuMat nmap, cv::cuda::GpuMat &image);
FUSION_HOST void renderSceneTextured(const cv::cuda::GpuMat vmap, const cv::cuda::GpuMat nmap, const cv::cuda::GpuMat image, cv::cuda::GpuMat &out);

// Create semi-dense view of images
// NOTE: selection criteria (dx > th_dx || dy > th_dy)
void build_semi_dense_pyramid(
    const std::vector<cv::cuda::GpuMat> image_pyr,
    const std::vector<cv::cuda::GpuMat> dx_pyr,
    const std::vector<cv::cuda::GpuMat> dy_pyr,
    std::vector<cv::cuda::GpuMat> &semi_pyr,
    float th_dx,
    float th_dy);

// Warp image based on the pose
// NOTE: the pose is from dst to src
// i.e. T = T_{src}^{-1} \dot T_{dst}
void warp_image(
    const cv::cuda::GpuMat src,
    const cv::cuda::GpuMat vmap_dst,
    const Sophus::SE3d pose,
    const IntrinsicMatrix K,
    cv::cuda::GpuMat &dst);

void select_point(
    const cv::cuda::GpuMat intensity,
    const cv::cuda::GpuMat depth,
    const cv::cuda::GpuMat dx,
    const cv::cuda::GpuMat dy,
    Vector4f *selected_points);

float check_covisibility(
    const cv::cuda::GpuMat vmap,
    Eigen::Matrix3f R,
    Eigen::Vector3f t,
    IntrinsicMatrix &K);

void compute_residual(
    const cv::cuda::GpuMat ref_image,
    const cv::cuda::GpuMat ref_vmap,
    const cv::cuda::GpuMat src_image,
    const IntrinsicMatrix &K,
    const Eigen::Matrix4d T_ref2src,
    cv::cuda::GpuMat &out_image);

} // namespace fusion

#endif