#ifndef SLAM_RELOCALIZER_H
#define SLAM_RELOCALIZER_H

#include "Frame.h"
#include "struct/map_point.h"
#include "FeatureExtractor.h"
#include "DescriptorMatcher.h"
#include <xfusion/core/intrinsic_matrix.h>

namespace fusion
{

class Relocalizer
{
public:
    Relocalizer(const fusion::IntrinsicMatrix K);
    void setFeatureExtractor(std::shared_ptr<FeatureExtractor>);
    void setDescriptorMatcher(std::shared_ptr<DescriptorMatcher>);
    void setMapPoints(std::vector<std::shared_ptr<Point3d>> mapPoints, cv::Mat &mapDescriptors);
    void setTargetFrame(std::shared_ptr<RgbdFrame> frame);
    void computeRelocalizationCandidate(std::vector<Sophus::SE3d> &candidates);

private:
    std::shared_ptr<FeatureExtractor> extractor;
    std::shared_ptr<DescriptorMatcher> matcher;

    cv::Mat map_descriptors;
    std::shared_ptr<RgbdFrame> target_frame;
    std::vector<std::shared_ptr<Point3d>> map_points;
    fusion::IntrinsicMatrix cam_param;
};

} // namespace fusion

#endif