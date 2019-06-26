#ifndef FUSION_DESCRIPTOR_MATCHER_H
#define FUSION_DESCRIPTOR_MATCHER_H

#include <thread>
#include <opencv2/xfeatures2d.hpp>
#include "struct/map_point.h"

namespace fusion
{

class DescriptorMatcher
{
public:
    DescriptorMatcher();

    void matchHammingKNN(
        const cv::Mat trainDesc,
        const cv::Mat queryDesc,
        std::vector<std::vector<cv::DMatch>> &matches,
        const int k = 2);

    std::thread matchHammingKNNAsync(
        const cv::Mat trainDesc,
        const cv::Mat queryDesc,
        std::vector<std::vector<cv::DMatch>> &matches,
        const int k = 2);

    void filterMatchesPairwise(
        const std::vector<std::shared_ptr<Point3d>> &src_pts,
        const std::vector<std::shared_ptr<Point3d>> &dst_pts,
        const std::vector<std::vector<cv::DMatch>> knnMatches,
        std::vector<std::vector<cv::DMatch>> &candidates);

private:
    cv::Ptr<cv::DescriptorMatcher> hammingMatcher;
    cv::Ptr<cv::DescriptorMatcher> l2Matcher;
};

} // namespace fusion

#endif