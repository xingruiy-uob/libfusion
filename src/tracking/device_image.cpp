#include "tracking/device_image.h"
#include "tracking/cuda_imgproc.h"

namespace fusion
{

DeviceImage::DeviceImage(const fusion::IntrinsicMatrix &K, const int NUM_PYRS)
{
    fusion::BuildIntrinsicPyramid(K, cam_params, NUM_PYRS);
}

DeviceImage::DeviceImage(const DeviceImage &other)
{
    depth_pyr = other.depth_pyr;
    intensity_pyr = other.intensity_pyr;
    intensity_dx_pyr = other.intensity_dx_pyr;
    intensity_dy_pyr = other.intensity_dy_pyr;
    vmap_pyr = other.vmap_pyr;
    nmap_pyr = other.nmap_pyr;
    semi_dense_image = other.semi_dense_image;
    cam_params = other.cam_params;
    image = other.image;
    image_float = other.image_float;
    depth_float = other.depth_float;
    intensity_float = other.intensity_float;
    rendered_image = other.rendered_image;
    rendered_image_textured = other.rendered_image_textured;
}

DeviceImage &DeviceImage::operator=(DeviceImage other)
{
    if (this != &other)
    {
        swap(*this, other);
    }
}

void swap(DeviceImage &first, DeviceImage &second)
{
    first.image.swap(second.image);
    first.image_float.swap(second.image_float);
    first.depth_float.swap(second.depth_float);
    first.intensity_float.swap(second.intensity_float);
    std::swap(first.depth_pyr, second.depth_pyr);
    std::swap(first.intensity_pyr, second.intensity_pyr);
    std::swap(first.intensity_dx_pyr, second.intensity_dx_pyr);
    std::swap(first.intensity_dy_pyr, second.intensity_dy_pyr);
    std::swap(first.vmap_pyr, second.vmap_pyr);
    std::swap(first.nmap_pyr, second.nmap_pyr);
    std::swap(first.semi_dense_image, second.semi_dense_image);
    first.rendered_image.swap(second.rendered_image);
    first.rendered_image_textured.swap(second.rendered_image_textured);
}

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
        filterDepthBilateral(depth_float, depth_pyr[0]);
    else
        depth_float.copyTo(depth_pyr[0]);

    for (int level = 1; level < max_level; ++level)
    {
        // cv::cuda::resize(depth_pyr[level - 1], depth_pyr[level], cv::Size(0, 0), 0.5, 0.5);
        pyrDownDepth(depth_pyr[level - 1], depth_pyr[level]);
    }
}

void DeviceImage::create_intensity_pyramid(const int max_level)
{
    if (intensity_pyr.size() != max_level)
        intensity_pyr.resize(max_level);

    intensity_float.copyTo(intensity_pyr[0]);

    for (int level = 1; level < max_level; ++level)
    {
        // cv::cuda::pyrDown(intensity_pyr[level - 1], intensity_pyr[level]);
        pyrDownImage(intensity_pyr[level - 1], intensity_pyr[level]);
    }
}

void DeviceImage::create_vmap_pyramid(const int max_level)
{
}

void DeviceImage::create_nmap_pyramid(const int max_level)
{
}

cv::cuda::GpuMat &DeviceImage::get_vmap(const int &level)
{
    return vmap_pyr[level];
}

cv::cuda::GpuMat &DeviceImage::get_nmap(const int &level)
{
    return nmap_pyr[level];
}

cv::cuda::GpuMat DeviceImage::get_rendered_image()
{
    renderScene(vmap_pyr[0], nmap_pyr[0], rendered_image);
    return rendered_image;
}

cv::cuda::GpuMat DeviceImage::get_rendered_scene_textured()
{
    renderSceneTextured(vmap_pyr[0], nmap_pyr[0], image, rendered_image);
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

void DeviceImage::upload(const std::shared_ptr<RgbdFrame> frame)
{
    if (frame == reference_frame)
        return;

    const int max_level = cam_params.size();

    if (max_level != this->depth_pyr.size())
        resize_pyramid(max_level);

    cv::Mat image = frame->image;
    cv::Mat depth = frame->depth;

    this->image.upload(image);
    depth_float.upload(depth);
    this->image.convertTo(image_float, CV_32FC3);
    cv::cuda::cvtColor(image_float, intensity_float, cv::COLOR_RGB2GRAY);

    create_depth_pyramid(max_level);
    create_intensity_pyramid(max_level);

    if (intensity_dx_pyr.size() != max_level)
        intensity_dx_pyr.resize(max_level);

    if (intensity_dy_pyr.size() != max_level)
        intensity_dy_pyr.resize(max_level);

    for (int i = 0; i < max_level; ++i)
    {
        computeDerivative(intensity_pyr[i], intensity_dx_pyr[i], intensity_dy_pyr[i]);
        backProjectDepth(depth_pyr[i], vmap_pyr[i], cam_params[i]);
        computeNMap(vmap_pyr[i], nmap_pyr[i]);
    }

    reference_frame = frame;
}

// DeviceImage::DeviceImage(const int &max_level)
// {
//     resize_pyramid(max_level);
// }

void DeviceImage::resize_device_map()
{
    for (int i = 1; i < vmap_pyr.size(); ++i)
    {
        pyrDownVMap(vmap_pyr[i - 1], vmap_pyr[i]);
    }

    for (int i = 0; i < vmap_pyr.size(); ++i)
    {
        computeNMap(vmap_pyr[i], nmap_pyr[i]);
    }
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