#include "system_new.h"
#include "tracking/cuda_imgproc.h"

namespace fusion
{

SystemNew::SystemNew(const IntrinsicMatrix K, const int NUM_PYR)
    : initialized(false), current_frame_id(0)
{
    mapper = std::make_shared<DenseMapping>(K);
    tracker = std::make_shared<DenseTracking>(K, NUM_PYR);
}

void SystemNew::spawn_work(const cv::Mat &depth, const cv::Mat &image)
{
    cv::Mat depth_float;
    depth.convertTo(depth_float, CV_32FC1, 1 / 1000.f);

    current = std::make_shared<RgbdFrame>(depth_float, image, current_frame_id);

    if (!initialized)
    {
        last_tracked = keyframe = current;
    }
}

void SystemNew::reset()
{
}

void SystemNew::write_map_to_disk(const std::string) const
{
}

void SystemNew::read_map_from_disk(const std::string)
{
}

void SystemNew::create_keyframe()
{
}

bool SystemNew::check_keyframe_critera() const
{
}

} // namespace fusion