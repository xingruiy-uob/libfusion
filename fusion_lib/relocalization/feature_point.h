#ifndef __FEATURE_POINT__
#define __FEATURE_POINT__

#include "rgbd_frame.h"
#include <Eigen/Core>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

namespace fusion
{

struct FeaturePoint
{
    FeaturePoint();
    float depth;
    Eigen::Vector3f pos;
    Eigen::Vector3f vec_normal;
    cv::Mat descriptor;
    cv::KeyPoint source;
    bool visited;
    std::vector<std::pair<RgbdFramePtr, int>> observations;
};

struct FeaturePointFrame
{
    RgbdFramePtr reference;
    std::vector<cv::KeyPoint> cv_key_points;
    std::vector<std::shared_ptr<FeaturePoint>> key_points;
    std::unordered_map<RgbdFramePtr, int> neighbours;
};

class FeatureExtractor
{
public:
    FeatureExtractor();
    void operator()();

    cv::Mat image;
    std::vector<cv::KeyPoint> key_points;
    cv::Mat descriptors;

private:
    cv::Ptr<cv::BRISK> BRISK;
    cv::Ptr<cv::ORB> ORB;
    cv::Ptr<cv::xfeatures2d::SURF> SURF;
};

} // namespace fusion

#endif