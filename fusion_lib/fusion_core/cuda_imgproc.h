#ifndef __IMAGE_OPS__
#define __IMAGE_OPS__

#include "intrinsic_matrix.h"
#include <sophus/se3.hpp>
#include <opencv2/cudaarithm.hpp>

namespace fusion
{

void bilateral_filter_depth(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst);

void build_depth_pyramid(const cv::cuda::GpuMat &base_depth, std::vector<cv::cuda::GpuMat> &pyramid, const int &max_level);

void build_intensity_pyramid(const cv::cuda::GpuMat &base_intensity, std::vector<cv::cuda::GpuMat> &pyramid, const int &max_level);

void build_intensity_derivative_pyramid(const std::vector<cv::cuda::GpuMat> &intensity, std::vector<cv::cuda::GpuMat> &sobel_x, std::vector<cv::cuda::GpuMat> &sobel_y);

void build_point_cloud_pyramid(const std::vector<cv::cuda::GpuMat> &depth, std::vector<cv::cuda::GpuMat> &pyramid, const IntrinsicMatrixPyramidPtr intrinsics_pyr);

void build_normal_pyramid(const std::vector<cv::cuda::GpuMat> &vmap_pyr, std::vector<cv::cuda::GpuMat> &nmap_pyr);

void resize_device_map(std::vector<cv::cuda::GpuMat> &map_pyr);

// Shaded scene without texures
void render_scene(
    const cv::cuda::GpuMat vmap,
    const cv::cuda::GpuMat nmap,
    cv::cuda::GpuMat &image);

// Shaded scene with textures pasted on the model
void render_scene_textured(
    const cv::cuda::GpuMat vmap,
    const cv::cuda::GpuMat nmap,
    const cv::cuda::GpuMat image,
    cv::cuda::GpuMat &out);

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

} // namespace fusion

#endif