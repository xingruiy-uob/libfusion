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
    //! Must initialize before using
    //! Eigen defaults to random numbers
    estimate = Eigen::Matrix4f::Identity();
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

    //! Compute centroid for both clouds
    src_pts_sum /= no_inliers;
    dst_pts_sum /= no_inliers;

    //! Subtract centroid from all points
    for (int i = 0; i < src.size(); ++i)
    {
        if (!outliers[i])
        {
            src[i] -= src_pts_sum;
            dst[i] -= dst_pts_sum;
        }
    }

    //! Build the linear system(LS)
    Eigen::Matrix3f LS = Eigen::Matrix3f::Zero();
    for (int i = 0; i < src.size(); ++i)
    {
        if (!outliers[i])
        {
            LS += src[i] * dst[i].transpose();
        }
    }

    //! Solve SVD #include<Eigen/SVD>
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(LS.cast<double>(), Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto V = svd.matrixV();
    const auto U = svd.matrixU();
    const auto R = (V * U.transpose()).transpose().cast<float>();

    //! Check if R is a valid rotation matrix
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
    float dist_thresh = 0.05f;
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
    Eigen::Matrix4f &estimate,
    float &inlier_ratio, float &confidence)
{
    const auto size = src.size();
    inlier_ratio = 0.f;
    int no_iter = 0;
    confidence = 0.f;
    int best_no_inlier = 0;

    if (outliers.size() != size)
        outliers.resize(size);

    while (no_iter < MAX_RANSAC_ITER)
    {
        no_iter++;

        //! randomly sample 3 point pairs
        std::vector<size_t> sampleIdx = {rand() % size, rand() % size, rand() % size};

        if (sampleIdx[0] == sampleIdx[1] ||
            sampleIdx[1] == sampleIdx[2] ||
            sampleIdx[2] == sampleIdx[0])
            continue;

        //! Src points corresponds to the frame
        std::vector<Eigen::Vector3f> src_pts = {src[sampleIdx[0]],
                                                src[sampleIdx[1]],
                                                src[sampleIdx[2]]};

        //! Dst points correspond to the map
        std::vector<Eigen::Vector3f> dst_pts = {dst[sampleIdx[0]],
                                                dst[sampleIdx[1]],
                                                dst[sampleIdx[2]]};

        //! Check if the 3 points are co-linear
        float src_d = (src_pts[1] - src_pts[0]).cross(src_pts[0] - src_pts[2]).norm();
        float dst_d = (dst_pts[1] - dst_pts[0]).cross(dst_pts[0] - dst_pts[2]).norm();
        if (src_d < 1e-6 || dst_d < 1e-6)
            continue;

        //! Compute pose estimate
        Eigen::Matrix4f pose;
        const auto valid = AbsoluteOrientation(src_pts, dst_pts, pose);

        if (valid)
        {
            //! Check for outliers
            const auto no_inliers = ValidateInliers(src, dst, outliers, pose);

            if (no_inliers > best_no_inlier)
            {
                //! Solve pose again for all inliers
                const auto valid = AbsoluteOrientation(src, dst, outliers, pose);
                if (valid)
                {
                    best_no_inlier = no_inliers;
                    inlier_ratio = (float)no_inliers / src.size();
                    confidence = 1 - pow((1 - pow(inlier_ratio, 3)), no_iter + 1);
                    estimate = pose;
                }
            }

            if (inlier_ratio >= 0.8 && confidence >= 0.95f)
                break;
        }
    }

    const auto no_inliers = ValidateInliers(src, dst, outliers, estimate);
    std::cout << "NO_INLIERS: " << no_inliers << " Confidence: " << confidence << " Ration: " << inlier_ratio << std::endl;
}

}; // namespace fusion