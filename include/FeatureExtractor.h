#ifndef FUSION_FEATURE_EXTRACTION_H
#define FUSION_FEATURE_EXTRACTION_H

#include <thread>
#include <sophus/se3.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include "struct/map_point.h"

namespace fusion
{

class FeatureExtractor
{
public:
    FeatureExtractor();

    void extractFeaturesSURF(
        const cv::Mat image,
        std::vector<cv::KeyPoint> &keypoints,
        cv::Mat &descriptors);

    std::thread extractFeaturesSURFAsync(
        const cv::Mat image,
        std::vector<cv::KeyPoint> &keypoints,
        cv::Mat &descriptors);

    void computeKeyPoints(
        const cv::Mat vmap, const cv::Mat nmap,
        const std::vector<cv::KeyPoint> &rawKeypoints,
        const cv::Mat &rawDescriptors,
        std::vector<cv::KeyPoint> &refinedKeypoints,
        cv::Mat &refinedDescriptors,
        std::vector<std::shared_ptr<Point3d>> &mapPoints,
        const Sophus::SE3f Tf2w);

private:
    cv::Ptr<cv::BRISK> BRISK;
    cv::Ptr<cv::xfeatures2d::SURF> SURF;
    cv::Ptr<cv::cuda::ORB> cudaORB;
    cv::Ptr<cv::cuda::SURF_CUDA> cudaSURF;
};

} // namespace fusion

#endif