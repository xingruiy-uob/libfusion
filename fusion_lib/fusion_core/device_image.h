#ifndef DEVICE_IMAGE_H
#define DEVICE_IMAGE_H

#include <memory>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include "rgbd_frame.h"
#include "intrinsic_matrix.h"

namespace fusion
{

class DeviceImage;
typedef std::shared_ptr<DeviceImage> RgbdImagePtr;

class DeviceImage
{
public:
    DeviceImage(const DeviceImage &) = delete;
    DeviceImage &operator=(const DeviceImage &) = delete;

    DeviceImage();
    DeviceImage(const int &max_level);

    void resize_pyramid(const int &max_level);

    void resize_device_map();
    void upload(const RgbdFramePtr frame, const IntrinsicMatrixPyramidPtr intrinsics_pyr);

    RgbdFramePtr get_reference_frame() const;
    cv::cuda::GpuMat get_rendered_image();
    cv::cuda::GpuMat get_rendered_scene_textured();
    cv::cuda::GpuMat get_depth(const int &level = 0) const;
    cv::cuda::GpuMat get_raw_depth() const;
    cv::cuda::GpuMat get_image(const int &level = 0) const;
    cv::cuda::GpuMat get_vmap(const int &level = 0) const;
    cv::cuda::GpuMat get_nmap(const int &level = 0) const;
    cv::cuda::GpuMat get_intensity(const int &level = 0) const;
    cv::cuda::GpuMat get_intensity_dx(const int &level = 0) const;
    cv::cuda::GpuMat get_intensity_dy(const int &level = 0) const;

    // TODO: move functions here
    void create_depth_pyramid(const int max_level, const bool use_filter = true);
    void create_intensity_pyramid(const int max_level);
    void create_vmap_pyramid(const int max_level);
    void create_nmap_pyramid(const int max_level);

private:
    RgbdFramePtr reference_frame;

    // original image in CV_8UC3
    cv::cuda::GpuMat image;

    // original image in CV_32FC3
    // this is needed when converting to grayscale
    cv::cuda::GpuMat image_float;

    // original depth in CV_32FC1
    cv::cuda::GpuMat depth_float;

    // base intensity in CV_32FC1
    cv::cuda::GpuMat intensity_float;

    // for tracking
    std::vector<cv::cuda::GpuMat> depth_pyr;        // CV_32FC1
    std::vector<cv::cuda::GpuMat> intensity_pyr;    // CV_32FC1
    std::vector<cv::cuda::GpuMat> intensity_dx_pyr; // CV_32FC1
    std::vector<cv::cuda::GpuMat> intensity_dy_pyr; // CV_32FC1
    std::vector<cv::cuda::GpuMat> vmap_pyr;         // CV_32FC4
    std::vector<cv::cuda::GpuMat> nmap_pyr;         // CV_32FC4

    // for debugging
    cv::cuda::GpuMat rendered_image;
    cv::cuda::GpuMat rendered_image_textured;
    std::vector<cv::cuda::GpuMat> semi_dense_image;
};

} // namespace fusion

#endif