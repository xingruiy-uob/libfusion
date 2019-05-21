#include "rgbd_image.h"
#include "image_ops.h"

namespace fusion
{

class RgbdImage::RgbdImageImpl
{
public:
    RgbdImageImpl();
    RgbdImageImpl(const int &max_level);

    void resize_pyramid(const int &max_level);
    void upload(const RgbdFramePtr frame, const IntrinsicMatrixPyramidPtr intrinsics_pyr);
    void render_synthetic_view();

    RgbdFramePtr reference_frame_;

    cv::cuda::GpuMat image_;
    cv::cuda::GpuMat rendered_image_;
    cv::cuda::GpuMat image_float_;
    cv::cuda::GpuMat depth_float_;
    cv::cuda::GpuMat intensity_float_;

    std::vector<cv::cuda::GpuMat> depth_;
    std::vector<cv::cuda::GpuMat> intensity_;
    std::vector<cv::cuda::GpuMat> intensity_dx_;
    std::vector<cv::cuda::GpuMat> intensity_dy_;
    std::vector<cv::cuda::GpuMat> point_cloud_;
    std::vector<cv::cuda::GpuMat> normal_;
    std::vector<cv::cuda::GpuMat> semi_dense_;
};

RgbdImage::RgbdImageImpl::RgbdImageImpl()
{
}

RgbdImage::RgbdImageImpl::RgbdImageImpl(const int &max_level)
{
    resize_pyramid(max_level);
}

void RgbdImage::RgbdImageImpl::resize_pyramid(const int &max_level)
{
    depth_.resize(max_level);
    intensity_.resize(max_level);
    intensity_dx_.resize(max_level);
    intensity_dy_.resize(max_level);
    point_cloud_.resize(max_level);
    normal_.resize(max_level);
}

void RgbdImage::RgbdImageImpl::render_synthetic_view()
{
    image_rendering_phong_shading(point_cloud_[0], normal_[0], rendered_image_);
}

cv::cuda::GpuMat RgbdImage::get_vmap(const int &level) const
{
    return impl->point_cloud_[level];
}

cv::cuda::GpuMat RgbdImage::get_nmap(const int &level) const
{
    return impl->normal_[level];
}

cv::cuda::GpuMat RgbdImage::get_rendered_image() const
{
    impl->render_synthetic_view();
    return impl->rendered_image_;
}

cv::cuda::GpuMat RgbdImage::get_rendered_scene_textured() const
{
    render_scene_textured(impl->point_cloud_[0], impl->normal_[0], impl->image_, impl->rendered_image_);
    return impl->rendered_image_;
}

cv::cuda::GpuMat RgbdImage::get_intensity(const int &level) const
{
    return impl->intensity_[level];
}

cv::cuda::GpuMat RgbdImage::get_intensity_dx(const int &level) const
{
    return impl->intensity_dx_[level];
}

cv::cuda::GpuMat RgbdImage::get_intensity_dy(const int &level) const
{
    return impl->intensity_dy_[level];
}

void RgbdImage::RgbdImageImpl::upload(const RgbdFramePtr frame, const IntrinsicMatrixPyramidPtr intrinsics_pyr)
{
    assert(frame != nullptr);
    assert(intrinsics_pyr != nullptr);

    if (frame == reference_frame_)
        return;

    const int max_level = intrinsics_pyr->get_max_level();

    if (max_level != depth_.size())
        resize_pyramid(max_level);

    cv::Mat image = frame->get_image();
    cv::Mat depth = frame->get_depth();

    image_.upload(image);
    depth_float_.upload(depth);
    image_.convertTo(image_float_, CV_32FC3);
    cv::cuda::cvtColor(image_float_, intensity_float_, cv::COLOR_RGB2GRAY);

    build_depth_pyramid(depth_float_, depth_, max_level);
    build_intensity_pyramid(intensity_float_, intensity_, max_level);
    build_intensity_derivative_pyramid(intensity_, intensity_dx_, intensity_dy_);
    build_point_cloud_pyramid(depth_, point_cloud_, intrinsics_pyr);
    build_normal_pyramid(point_cloud_, normal_);
    // build_semi_dense_pyramid(intensity_, intensity_dx_, intensity_dy_, semi_dense_, 1, 1);

    // for (int i = 0; i < 5; ++i)
    // {
    //     cv::Mat img(semi_dense_[i]);
    //     img.convertTo(img, CV_8UC1);
    //     cv::imshow("semi", img);
    //     cv::waitKey(0);
    // }

    reference_frame_ = frame;
}

RgbdImage::RgbdImage() : impl(new RgbdImageImpl())
{
}

RgbdImage::RgbdImage(const int &max_level) : impl(new RgbdImageImpl(max_level))
{
}

void RgbdImage::upload(const RgbdFramePtr frame, const IntrinsicMatrixPyramidPtr intrinsics_pyr)
{
    impl->upload(frame, intrinsics_pyr);
}

void RgbdImage::resize_device_map()
{
    fusion::resize_device_map(impl->point_cloud_);
    // resize_device_map(impl->normal_);
    build_normal_pyramid(impl->point_cloud_, impl->normal_);

    // impl->image_.convertTo(impl->image_float_, CV_32FC3);
    // cv::cuda::cvtColor(impl->image_float_, impl->intensity_float_, CV_RGB2GRAY);
    // build_intensity_pyramid(impl->intensity_float_, impl->intensity_, 5);
    // build_intensity_derivative_pyramid(impl->intensity_, impl->intensity_dx_, impl->intensity_dy_);
}

cv::cuda::GpuMat RgbdImage::get_depth(const int &level) const
{
    if (level < impl->depth_.size())
        return impl->depth_[level];
}

cv::cuda::GpuMat RgbdImage::get_raw_depth() const
{
    return impl->depth_float_;
}

cv::cuda::GpuMat RgbdImage::get_image(const int &level) const
{
    if (level == 0)
        return impl->image_;
    else
    {
        return impl->image_;
    }
}

RgbdFramePtr RgbdImage::get_reference_frame() const
{
    return impl->reference_frame_;
}

class RgbdFrame::RgbdFrameImpl
{
public:
    RgbdFrameImpl(const cv::Mat &image, const cv::Mat &depth_float, size_t id, double time_stamp);

    cv::Mat image_;
    cv::Mat depth_;
    unsigned long id_;
    double time_stamp_;
    Sophus::SE3d pose_;
    RgbdFramePtr reference_frame_;
    IntrinsicMatrixPyramidPtr intrinsics_pyr_;
};

RgbdFrame::RgbdFrameImpl::RgbdFrameImpl(const cv::Mat &image, const cv::Mat &depth_float, size_t id, double time_stamp)
    : image_(image.clone()), depth_(depth_float.clone()), id_(id), time_stamp_(time_stamp)
{
}

RgbdFrame::RgbdFrame(const cv::Mat &image, const cv::Mat &depth_float, size_t id, double time_stamp) : impl(new RgbdFrameImpl(image, depth_float, id, time_stamp))
{
}

RgbdFrame::~RgbdFrame()
{
    // std::cout << "frame id: " << impl->id_ << " is released!" << std::endl;
}

size_t RgbdFrame::get_id() const
{
    return impl->id_;
}

cv::Mat RgbdFrame::get_image() const
{
    return impl->image_;
}

cv::Mat RgbdFrame::get_depth() const
{
    return impl->depth_;
}

Sophus::SE3d RgbdFrame::get_pose() const
{
    return impl->pose_;
}

void RgbdFrame::set_pose(const Sophus::SE3d &pose)
{
    impl->pose_ = pose;
}

RgbdFramePtr RgbdFrame::get_reference_frame() const
{
    return impl->reference_frame_;
}

void RgbdFrame::set_reference_frame(RgbdFramePtr reference)
{
    impl->reference_frame_ = reference;
}

} // namespace fusion