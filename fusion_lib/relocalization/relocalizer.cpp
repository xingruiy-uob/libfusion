#include "relocalizer.h"

namespace fusion
{

Relocalizer::Relocalizer()
{
    BRISK = cv::BRISK::create();
    SURF = cv::xfeatures2d::SURF::create();
}

void Relocalizer::insert_keyframe(RgbdFramePtr keyframe)
{
}

void Relocalizer::set_relocalization_target(RgbdFramePtr frame)
{
    cv::Mat source_image = frame->get_image();
    std::vector<cv::KeyPoint> detected_points;
    SURF->detect(source_image, detected_points);
}

void Relocalizer::match_by_pose_constraint()
{
}

void Relocalizer::compute()
{
}

} // namespace fusion