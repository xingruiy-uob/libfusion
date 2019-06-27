#ifndef SLAM_POSE_ESTIMATOR_H
#define SLAM_POSE_ESTIMATOR_H

#include "struct/map_point.h"

namespace fusion
{

class PoseEstimator
{
public:
    static bool AbsoluteOrientation(
        std::vector<Eigen::Vector3f> src,
        std::vector<Eigen::Vector3f> dst,
        std::vector<bool> outliers,
        Eigen::Matrix4f &estimate);

    static bool AbsoluteOrientation(
        std::vector<Eigen::Vector3f> src,
        std::vector<Eigen::Vector3f> dst,
        Eigen::Matrix4f &estimate);

    static int ValidateInliers(
        const std::vector<Eigen::Vector3f> &src,
        const std::vector<Eigen::Vector3f> &dst,
        std::vector<bool> &outliers,
        const Eigen::Matrix4f &estimate);

    static void RANSAC(
        const std::vector<Eigen::Vector3f> &src,
        const std::vector<Eigen::Vector3f> &dst,
        std::vector<bool> &outliers,
        Eigen::Matrix4f &estimate,
        float &inlier_ratio,
        float &confidence);

private:
};

} // namespace fusion

#endif