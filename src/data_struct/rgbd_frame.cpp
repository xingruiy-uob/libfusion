#include "data_struct/rgbd_frame.h"

namespace fusion
{

RgbdFrame::RgbdFrame(const cv::Mat &depth, const cv::Mat &image, const size_t id, const double ts)
    : id(id), timeStamp(ts)
{
    this->image = image.clone();
    this->depth = depth.clone();
}

} // namespace fusion