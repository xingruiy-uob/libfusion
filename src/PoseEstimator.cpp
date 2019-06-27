#include "PoseEstimator.h"
#include <Eigen/Dense>
#include <stdlib.h>
#define MAX_RANSAC_ITER 100

namespace fusion
{

bool PoseEstimator::AbsoluteOrientation(
    std::vector<Eigen::Vector3f> src,
    std::vector<Eigen::Vector3f> dst,
    std::vector<bool> outliers,
    Eigen::Matrix4f &estimate)
{
    Eigen::Vector3f src_pts_sum = Eigen::Vector3f::Zero();
    Eigen::Vector3f dst_pts_sum = Eigen::Vector3f::Zero();
    int no_inliers = 0;

    for (int i = 0; i < src.size(); ++i)
    {
        if (!outliers[i])
        {
            no_inliers++;
            src_pts_sum += src[i];
            dst_pts_sum += dst[i];
        }
    }

    src_pts_sum /= no_inliers;
    dst_pts_sum /= no_inliers;

    for (int i = 0; i < src.size(); ++i)
    {
        if (!outliers[i])
        {
            src[i] -= src_pts_sum;
            dst[i] -= dst_pts_sum;
        }
    }

    Eigen::Matrix3f LS = Eigen::Matrix3f::Zero();

    for (int i = 0; i < src.size(); ++i)
    {
        if (!outliers[i])
        {
            LS += src[i] * dst[i].transpose();
        }
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(LS.cast<double>(), Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto V = svd.matrixV();
    const auto U = svd.matrixU();
    const auto R = (V * U.transpose()).transpose().cast<float>();

    if (R.determinant() < 0)
        return false;

    const auto t = src_pts_sum - R * dst_pts_sum;

    estimate.topLeftCorner(3, 3) = R;
    estimate.topRightCorner(3, 1) = t;

    return true;
}

bool PoseEstimator::AbsoluteOrientation(
    std::vector<Eigen::Vector3f> src,
    std::vector<Eigen::Vector3f> dst,
    Eigen::Matrix4f &estimate)
{
    std::vector<bool> outliers(src.size());
    std::fill(outliers.begin(), outliers.end(), false);
    return AbsoluteOrientation(src, dst, outliers, estimate);
}

int PoseEstimator::ValidateInliers(
    const std::vector<Eigen::Vector3f> &src,
    const std::vector<Eigen::Vector3f> &dst,
    std::vector<bool> &outliers,
    const Eigen::Matrix4f &estimate)
{
    int no_inliers = 0;
    float dist_thresh = 0.1f;
    const auto &R = estimate.topLeftCorner(3, 3);
    const auto &t = estimate.topRightCorner(3, 1);

    std::fill(outliers.begin(), outliers.end(), true);
    for (int i = 0; i < src.size(); ++i)
    {
        float dist = (src[i] - (R * dst[i] + t)).norm();
        if (dist < dist_thresh)
        {
            no_inliers++;
            outliers[i] = false;
        }
    }

    return no_inliers;
}

void PoseEstimator::RANSAC(
    const std::vector<Eigen::Vector3f> &src,
    const std::vector<Eigen::Vector3f> &dst,
    std::vector<bool> &outliers,
    Eigen::Matrix4f &estimate)
{
    const auto size = src.size();
    float inlier_ratio = 0.f;
    int no_iter = 0;
    float confidence = 0.f;
    int best_no_inlier = 0;
    outliers.resize(size);

    Eigen::Matrix3f R_best = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f t_best = Eigen::Matrix3f::Zero();

    while (no_iter < MAX_RANSAC_ITER)
    {
        no_iter++;

        std::vector<size_t> sampleIdx = {rand() % size, rand() % size, rand() % size};

        if (sampleIdx[0] == sampleIdx[1] ||
            sampleIdx[1] == sampleIdx[2] ||
            sampleIdx[2] == sampleIdx[0])
            continue;

        std::vector<Eigen::Vector3f> src_pts = {src[sampleIdx[0]],
                                                src[sampleIdx[1]],
                                                src[sampleIdx[2]]};
        std::vector<Eigen::Vector3f> dst_pts = {dst[sampleIdx[0]],
                                                dst[sampleIdx[1]],
                                                dst[sampleIdx[2]]};

        float src_d = (src_pts[1] - src_pts[0]).cross(src_pts[0] - src_pts[2]).norm();
        float dst_d = (dst_pts[1] - dst_pts[0]).cross(dst_pts[0] - dst_pts[2]).norm();
        if (src_d < 1e-6 || dst_d < 1e-6)
            continue;

        Eigen::Matrix4f pose;
        const auto valid = AbsoluteOrientation(src_pts, dst_pts, pose);

        if (valid)
        {
            const auto no_inliers = ValidateInliers(src, dst, outliers, pose);
            std::cout << no_inliers << std::endl;

            if (no_inliers > best_no_inlier)
            {
                //! solve ao again for all inliers
                const auto valid = AbsoluteOrientation(src, dst, outliers, pose);
                if (valid)
                {
                    best_no_inlier = no_inliers;
                    inlier_ratio = (float)no_inliers / src.size();
                    confidence = 1 - pow((1 - pow(inlier_ratio, 3)), no_iter + 1);
                    estimate = pose;
                }
            }

            if (no_inliers >= 20 && confidence >= 0.95f)
                break;
        }
    }

    const auto no_inliers = ValidateInliers(src, dst, outliers, estimate);
    std::cout << "NO_INLIERS: " << no_inliers << std::endl;
}

}; // namespace fusion