#ifndef __FEATURE_POINT__
#define __FEATURE_POINT__

#include "rgbd_frame.h"
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

namespace fusion
{

struct FeaturePoint
{
    float depth;
    Eigen::Vector3f pos;
    cv::Mat descriptor;
    cv::KeyPoint source;
    RgbdFramePtr reference;
};

} // namespace fusion

#endif