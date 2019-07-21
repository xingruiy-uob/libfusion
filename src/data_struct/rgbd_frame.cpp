#include "data_struct/rgbd_frame.h"

namespace fusion
{

RgbdFrame::RgbdFrame(const cv::Mat &depth, const cv::Mat &image, const size_t id)
    : id(id)
{
    this->image = image.clone();
    this->depth = depth.clone();
}

RgbdFrame::RgbdFrame(const RgbdFrame &other)
    : depth(other.depth), image(other.image), id(other.id), timeStamp(other.timeStamp)
{
}

RgbdFrame &RgbdFrame::operator=(RgbdFrame other)
{
    if (this != &other)
    {
        swap(*this, other);
    }

    return *this;
}

void swap(RgbdFrame &, RgbdFrame &)
{
    using std::swap;
}

} // namespace fusion