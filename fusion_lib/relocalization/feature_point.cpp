#include "feature_point.h"

namespace fusion
{

FeaturePoint::FeaturePoint()
{
    depth = -1;
    visited = false;
}

FeatureExtractor::FeatureExtractor()
{
    BRISK = cv::BRISK::create();
    ORB = cv::ORB::create(1000);
    SURF = cv::xfeatures2d::SURF::create();
}

void FeatureExtractor::operator()()
{
    // SURF->detect(image, key_points);
    // BRISK->compute(image, key_points, descriptors);
    ORB->detectAndCompute(image, cv::Mat(), key_points, descriptors);
}

} // namespace fusion