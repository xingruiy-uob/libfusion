#include "device_image.h"
#include "cuda_imgproc.h"

namespace fusion
{

void DeviceImage::resize_pyramid(const int &max_level)
{
    depth_pyr.resize(max_level);
    intensity_pyr.resize(max_level);
    intensity_dx_pyr.resize(max_level);
    intensity_dy_pyr.resize(max_level);
    vmap_pyr.resize(max_level);
    nmap_pyr.resize(max_level);
}

void DeviceImage::create_depth_pyramid(const int max_level, const bool use_filter)
{
    if (depth_float.empty() || depth_float.type() != CV_32FC1)
    {
        std::cout << "depth not supplied." << std::endl;
    }

    if (depth_pyr.size() != max_level)
        depth_pyr.resize(max_level);

    if (use_filter)
        bilateral_filter_depth(depth_float, depth_pyr[0]);
    else
        depth_float.copyTo(depth_pyr[0]);

    for (int level = 1; level < max_level; ++level)
    {
        cv::cuda::resize(depth_pyr[level - 1], depth_pyr[level], cv::Size(0, 0), 0.5, 0.5);
    }
}

void DeviceImage::create_intensity_pyramid(const int max_level)
{
    if (intensity_pyr.size() != max_level)
        intensity_pyr.resize(max_level);

    intensity_float.copyTo(intensity_pyr[0]);

    for (int level = 1; level < max_level; ++level)
    {
        cv::cuda::pyrDown(intensity_pyr[level - 1], intensity_pyr[level]);
    }
}

void DeviceImage::create_vmap_pyramid(const int max_level)
{
}

void DeviceImage::create_nmap_pyramid(const int max_level)
{
}

cv::cuda::GpuMat DeviceImage::get_vmap(const int &level) const
{
    return vmap_pyr[level];
}

cv::cuda::GpuMat DeviceImage::get_nmap(const int &level) const
{
    return nmap_pyr[level];
}

cv::cuda::GpuMat DeviceImage::get_rendered_image()
{
    render_scene(vmap_pyr[0], nmap_pyr[0], rendered_image);
    return rendered_image;
}

cv::cuda::GpuMat DeviceImage::get_rendered_scene_textured()
{
    render_scene_textured(vmap_pyr[0], nmap_pyr[0], image, rendered_image);
    return rendered_image;
}

cv::cuda::GpuMat DeviceImage::get_intensity(const int &level) const
{
    return intensity_pyr[level];
}

cv::cuda::GpuMat DeviceImage::get_intensity_dx(const int &level) const
{
    return intensity_dx_pyr[level];
}

cv::cuda::GpuMat DeviceImage::get_intensity_dy(const int &level) const
{
    return intensity_dy_pyr[level];
}

void DeviceImage::upload(const RgbdFramePtr frame, const IntrinsicMatrixPyramidPtr intrinsics_pyr)
{
    assert(frame != nullptr);
    assert(intrinsics_pyr != nullptr);

    if (frame == reference_frame)
        return;

    const int max_level = intrinsics_pyr->get_max_level();

    if (max_level != this->depth_pyr.size())
        resize_pyramid(max_level);

    cv::Mat image = frame->get_image();
    cv::Mat depth = frame->get_depth();

    this->image.upload(image);
    depth_float.upload(depth);
    this->image.convertTo(image_float, CV_32FC3);
    cv::cuda::cvtColor(image_float, intensity_float, cv::COLOR_RGB2GRAY);

    create_depth_pyramid(max_level);
    create_intensity_pyramid(max_level);

    build_intensity_derivative_pyramid(intensity_pyr, intensity_dx_pyr, intensity_dy_pyr);
    build_point_cloud_pyramid(depth_pyr, vmap_pyr, intrinsics_pyr);
    build_normal_pyramid(vmap_pyr, nmap_pyr);

    reference_frame = frame;
}

DeviceImage::DeviceImage(const int &max_level)
{
    resize_pyramid(max_level);
}

void DeviceImage::resize_device_map()
{
    fusion::resize_device_map(vmap_pyr);
    // resize_device_map(nmap);
    build_normal_pyramid(vmap_pyr, nmap_pyr);

    // image_.convertTo(image_float_, CV_32FC3);
    // cv::cuda::cvtColor(image_float_, intensity_float_, CV_RGB2GRAY);
    // build_intensity_pyramid(intensity_float_, intensity_, 5);
    // build_intensity_derivative_pyramid(intensity_, intensity_dx_, intensity_dy_);
}

cv::cuda::GpuMat DeviceImage::get_depth(const int &level) const
{
    if (level < depth_pyr.size())
        return depth_pyr[level];
}

cv::cuda::GpuMat DeviceImage::get_raw_depth() const
{
    return depth_float;
}

cv::cuda::GpuMat DeviceImage::get_image() const
{
    return image;
}

RgbdFramePtr DeviceImage::get_reference_frame() const
{
    return reference_frame;
}

} // namespace fusion