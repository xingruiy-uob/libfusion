#include "Relocalizer.h"
#include <xfusion/core/cuda_imgproc.h>

namespace fusion
{

Relocalizer::Relocalizer(const fusion::IntrinsicMatrix K) : cam_param(K)
{
}

void Relocalizer::setFeatureExtractor(std::shared_ptr<FeatureExtractor> ext)
{
    extractor = ext;
}

void Relocalizer::setDescriptorMatcher(std::shared_ptr<DescriptorMatcher> mcr)
{
    matcher = mcr;
}

void Relocalizer::setMapPoints(std::vector<std::shared_ptr<Point3d>> mapPoints, cv::Mat &mapDescriptors)
{
    map_points = mapPoints;
    map_descriptors = mapDescriptors;
}

void Relocalizer::setTargetFrame(std::shared_ptr<RgbdFrame> frame)
{
    target_frame = frame;
}

void Relocalizer::computeRelocalizationCandidate(std::vector<Sophus::SE3d> &candidates)
{
    target_frame->pose = Sophus::SE3d();
    std::vector<cv::KeyPoint> raw_keypoints;
    cv::Mat raw_descriptors;

    cv::cuda::GpuMat depth(target_frame->depth);
    cv::cuda::GpuMat vmap_gpu, nmap_gpu;
    backProjectDepth(depth, vmap_gpu, cam_param);
    computeNMap(vmap_gpu, nmap_gpu);

    extractor->extractFeaturesSURF(
        target_frame->image,
        raw_keypoints,
        raw_descriptors);

    extractor->computeKeyPoints(
        cv::Mat(vmap_gpu),
        cv::Mat(nmap_gpu),
        raw_keypoints,
        raw_descriptors,
        target_frame->cv_key_points,
        target_frame->descriptors,
        target_frame->key_points,
        target_frame->pose.cast<float>());

    std::vector<std::vector<cv::DMatch>> matches;
    matcher->matchHammingKNN(
        map_descriptors,
        target_frame->descriptors,
        matches, 2);

    std::vector<std::vector<cv::DMatch>> candidate_matches;
    matcher->filterMatchesPairwise(target_frame->key_points, map_points, matches, candidate_matches);

    // cv::Mat outImg;
    // cv::drawKeypoints(target_frame->image, target_frame->cv_key_points, outImg);
    // cv::imshow("outImg", outImg);
    // cv::waitKey(0);
}

} // namespace fusion
