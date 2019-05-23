#ifndef __RELOCALIZER__
#define __RELOCALIZER__

#include "rgbd_frame.h"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

namespace fusion
{

class Relocalizer
{
public:
    Relocalizer();
    void insert_keyframe(RgbdFramePtr keyframe);
    void set_relocalization_target(RgbdFramePtr frame);
    void match_by_pose_constraint();
    void compute();

private:
    cv::Ptr<cv::BRISK> BRISK;
    cv::Ptr<cv::xfeatures2d::SURF> SURF;
};

} // namespace fusion

#endif