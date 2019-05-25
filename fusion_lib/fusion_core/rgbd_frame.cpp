#include "rgbd_frame.h"
#include "cuda_imgproc.h"

namespace fusion
{

RgbdFrame::RgbdFrame(const cv::Mat &depth, const cv::Mat &image, size_t id, double ts)
    : frame_id(id), time_stamp(ts), pose(Sophus::SE3d())
{
    if (depth.type() != CV_32FC1)
    {
        std::cout << "depth format must be CV_32FC1" << std::endl;
        exit(0);
    }

    if (image.type() != CV_8UC3)
    {
        std::cout << "image format must be CV_8UC3" << std::endl;
        exit(0);
    }

    source_image = image.clone();
    source_depth = depth.clone();
}

bool RgbdFrame::has_scene_data() const
{
    return !vmap.empty();
}

size_t RgbdFrame::get_id() const
{
    return frame_id;
}

cv::Mat RgbdFrame::get_image() const
{
    return source_image;
}

cv::Mat RgbdFrame::get_depth() const
{
    return source_depth;
}

cv::Mat RgbdFrame::get_vmap() const
{
    return vmap;
}

cv::Mat RgbdFrame::get_nmap() const
{
    return nmap;
}

Sophus::SE3d RgbdFrame::get_pose() const
{
    return pose;
}

RgbdFramePtr RgbdFrame::get_reference_frame() const
{
    reference;
}

void RgbdFrame::set_pose(const Sophus::SE3d &pose)
{
    this->pose = pose;
}

void RgbdFrame::set_reference_frame(RgbdFramePtr reference)
{
    this->reference = reference;
}

} // namespace fusion