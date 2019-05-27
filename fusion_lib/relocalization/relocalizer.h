#ifndef __RELOCALIZER__
#define __RELOCALIZER__

#include "rgbd_frame.h"
#include "feature_point.h"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

namespace fusion
{

class Relocalizer
{
public:
    Relocalizer(IntrinsicMatrix K);
    void insert_keyframe(RgbdFramePtr keyframe);
    void set_relocalization_target(RgbdFramePtr frame);
    void insert_current_frame();
    void match_by_pose_constraint();
    void compute();

private:
    IntrinsicMatrix cam_param;
    cv::Ptr<cv::BRISK> BRISK;
    cv::Ptr<cv::xfeatures2d::SURF> SURF;

    std::shared_ptr<FeaturePointFrame> current_point_struct;
    std::shared_ptr<FeaturePointFrame> reference_struct;
    std::shared_ptr<size_t> point_corresp;

    // map and points
    std::vector<std::shared_ptr<FeaturePointFrame>> frames;
};

} // namespace fusion

#endif