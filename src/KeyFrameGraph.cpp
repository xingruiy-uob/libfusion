#include "KeyFrameGraph.h"
#include "PoseEstimator.h"
#include <stdlib.h>
#include <ceres/ceres.h>
#include <xfusion/core/cuda_imgproc.h>
#include <xutils/DataStruct/stop_watch.h>

namespace fusion
{

KeyFrameGraph::~KeyFrameGraph()
{
    std::cout << "feature graph released." << std::endl;
}

KeyFrameGraph::KeyFrameGraph(const IntrinsicMatrix K, const int NUM_PYR)
    : FlagShouldQuit(false), cam_param(K), FlagNeedOpt(false),
      extractor(NULL), matcher(NULL)
{
    tracker = std::make_shared<DenseTracking>(K, NUM_PYR);
}

cv::Vec4f interpolate_bilinear(cv::Mat map, float x, float y)
{
    int u = (int)(x + 0.5f);
    int v = (int)(y + 0.5f);
    if (u >= 0 && v >= 0 && u < map.cols && v < map.rows)
    {
        return map.ptr<cv::Vec4f>(v)[u];
    }
}

void KeyFrameGraph::set_all_points_unvisited()
{
    for (const auto &iter : keyframe_graph)
    {
        for (const auto &iter2 : iter->key_points)
        {
            if (iter2 != NULL)
                iter2->visited = false;
        }
    }
}

std::vector<Eigen::Matrix<float, 4, 4>> KeyFrameGraph::get_keyframe_poses() const
{
    std::vector<Eigen::Matrix<float, 4, 4>> poses;
    for (const auto &kf : keyframe_graph)
    {
        Eigen::Matrix<float, 4, 4> pose = kf->pose.cast<float>().matrix();
        poses.emplace_back(pose);
    }

    return poses;
}

cv::Mat KeyFrameGraph::get_descriptor_all(std::vector<std::shared_ptr<Point3d>> &points)
{
    cv::Mat descritpors;
    points.clear();
    set_all_points_unvisited();

    for (const auto &kf : keyframe_graph)
    {
        for (const auto &point : kf->key_points)
        {
            if (point != NULL)
            {
                if (point->observations <= 1 || point->visited)
                    continue;

                descritpors.push_back(point->descriptors);
                points.emplace_back(point);

                point->visited = true;
            }
        }
    }

    return descritpors;
}

void KeyFrameGraph::get_points(float *pt3d, size_t &count, size_t max_size)
{
    count = 0;
    set_all_points_unvisited();

    for (const auto &kf : keyframe_graph)
    {
        for (const auto &point : kf->key_points)
        {
            if (count >= max_size - 1)
                return;

            if (point != NULL)
            {
                if (point->observations <= 1 || point->visited)
                    continue;

                pt3d[count * 3 + 0] = point->pos(0);
                pt3d[count * 3 + 1] = point->pos(1);
                pt3d[count * 3 + 2] = point->pos(2);
                count++;

                point->visited = true;
            }
        }
    }

    std::cout << "NUM KEY POINTS: " << count << std::endl;
}

void KeyFrameGraph::add_keyframe(std::shared_ptr<RgbdFrame> keyframe)
{
    raw_keyframe_queue.push(keyframe);
}

void KeyFrameGraph::extract_features(RgbdFramePtr keyframe)
{
    cv::Mat source_image = keyframe->image;
    auto frame_pose = keyframe->pose.cast<float>();

    cv::Mat raw_descriptors;
    std::vector<cv::KeyPoint> raw_keypoints;
    extractor->extractFeaturesSURF(
        source_image,
        raw_keypoints,
        raw_descriptors);

    extractor->computeKeyPoints(
        keyframe->vmap,
        keyframe->nmap,
        raw_keypoints,
        raw_descriptors,
        keyframe->cv_key_points,
        keyframe->descriptors,
        keyframe->key_points,
        frame_pose);
}

void KeyFrameGraph::search_loop(RgbdFramePtr keyframe)
{
    std::lock_guard<std::mutex> lock(graphMutex);

    for (const auto &candidate : keyframe_graph)
    {
        if (candidate == NULL || candidate == referenceFrame)
            continue;

        //! KNN match between two sets of features
        std::vector<std::vector<cv::DMatch>> matches;
        std::vector<std::vector<cv::DMatch>> candidate_matches;
        matcher->matchHammingKNN(candidate->descriptors, keyframe->descriptors, matches, 2);

        //! lowe's ratio test
        std::vector<cv::DMatch> list, refined_matches;
        matcher->filter_matches_ratio_test(matches, list);
        candidate_matches.push_back(list);

        //! Shuda's pair wise constraint
        // matcher->filter_matches_pair_constraint(keyframe->key_points, candidate->key_points, matches, candidate_matches);

        tracker->set_source_vmap(cv::cuda::GpuMat(keyframe->vmap));
        tracker->set_source_image(cv::cuda::GpuMat(keyframe->image));

        for (const auto &list : candidate_matches)
        {
            std::vector<Eigen::Vector3f> src_pts, dst_pts;
            for (const auto &match : list)
            {
                src_pts.push_back(keyframe->key_points[match.queryIdx]->pos);
                dst_pts.push_back(candidate->key_points[match.trainIdx]->pos);
            }

            std::vector<bool> outliers;
            Eigen::Matrix4f estimate;
            int no_inliers = 0;
            float inlier_ratio, confidence;

            //! Compute relative pose
            PoseEstimator::RANSAC(src_pts, dst_pts, outliers, estimate, inlier_ratio, confidence);

            // for (int i = 0; i < outliers.size(); ++i)
            // {
            //     if (!outliers[i])
            //         refined_matches.push_back(list[i]);
            // }

            // auto &source_image = keyframe->image;
            // auto &reference_image = candidate->image;
            // cv::Mat outImg;
            // cv::drawMatches(source_image, keyframe->cv_key_points, reference_image, candidate->cv_key_points, refined_matches, outImg, cv::Scalar(0, 255, 0));
            // cv::cvtColor(outImg, outImg, cv::COLOR_RGB2BGR);
            // cv::imshow("outImg", outImg);
            // cv::waitKey(1);

            // no_inliers = std::count(outliers.begin(), outliers.end(), false);
            // if (inlier_ratio > 0.5)
            // {
            //     std::cout << "NO_INLIERS: " << no_inliers << " Confidence: " << confidence << " Ratio: " << inlier_ratio << std::endl;
            //     std::cout << estimate << std::endl;
            //     std::cout << (keyframe->pose.inverse() * candidate->pose).matrix() << std::endl;
            // }
            // cv::cuda::GpuMat img(keyframe->image);
            // cv::cuda::GpuMat dst_vmap(candidate->vmap), dst;
            // std::cout << estimate << std::endl;
            // fusion::warp_image(img, dst_vmap, keyframe->pose.inverse() * Sophus::SE3f(estimate).cast<double>() * candidate->pose, cam_param, dst);
            // cv::Mat img2(dst);
            // cv::imshow("img2", img2);
            // cv::imshow("img1", candidate->image);
            // cv::imshow("imgo", keyframe->image);
            // cv::waitKey(0);

            //! Dense verification && refinement
            if (inlier_ratio > 0.7 && confidence >= 0.95)
            {
                tracker->set_reference_vmap(cv::cuda::GpuMat(candidate->vmap));
                tracker->set_reference_image(cv::cuda::GpuMat(candidate->image));

                // cv::Mat img(candidate->vmap);
                // cv::imshow("img", img);
                // cv::waitKey(0);

                TrackingContext context;
                context.use_initial_guess_ = true;
                context.initial_estimate_ = keyframe->pose.inverse() * Sophus::SE3f(estimate).cast<double>() * candidate->pose;
                // context.initial_estimate_ = Sophus::SE3f(estimate).cast<double>();
                context.max_iterations_ = {10, 5, 3, 3, 3};

                auto result = tracker->compute_transform(context);
                auto pose_c2k = (keyframe->pose * result.update.inverse() * candidate->pose.inverse()).cast<float>();
                // std::cout << result.update.matrix3x4() << std::endl;
                PoseEstimator::ValidateInliers(src_pts, dst_pts, outliers, pose_c2k.matrix());

                no_inliers = std::count(outliers.begin(), outliers.end(), false);
                inlier_ratio = ((float)no_inliers / src_pts.size());

                if (result.icp_error < 10e-4 && inlier_ratio > 0.6)
                {
                    matcher->match_pose_constraint(keyframe, candidate, cam_param, pose_c2k);
                }
                // std::cout << "NO_INLIERS: " << no_inliers << " Confidence: " << confidence << " Ratio: " << inlier_ratio << std::endl;
                // std::cout << estimate << std::endl;
                // std::cout << (keyframe->pose.inverse() * candidate->pose).matrix() << std::endl;

                // for (int i = 0; i < outliers.size(); ++i)
                // {
                //     if (!outliers[i])
                //         refined_matches.push_back(list[i]);
                // }

                // auto &source_image = keyframe->image;
                // auto &reference_image = candidate->image;
                // cv::Mat outImg;
                // cv::drawMatches(source_image, keyframe->cv_key_points, reference_image, candidate->cv_key_points, refined_matches, outImg, cv::Scalar(0, 255, 0));
                // cv::cvtColor(outImg, outImg, cv::COLOR_RGB2BGR);
                // cv::imshow("outImg", outImg);
                // cv::waitKey(1);

                // cv::cuda::GpuMat img(keyframe->image);
                // cv::cuda::GpuMat dst_vmap(candidate->vmap), dst;
                // fusion::warp_image(img, dst_vmap, result.update.inverse(), cam_param, dst);
                // cv::Mat img2(dst);
                // cv::imshow("img2", img2);
                // cv::imshow("img1", candidate->image);
                // cv::imshow("imgo", keyframe->image);
                // cv::waitKey(0);
            }
        }
    }
}

void KeyFrameGraph::search_correspondence(RgbdFramePtr keyframe)
{
    if (referenceFrame == NULL)
        return;

    matcher->match_pose_constraint(
        keyframe,
        referenceFrame,
        cam_param,
        referenceFrame->pose.cast<float>());
}

void KeyFrameGraph::set_feature_extractor(std::shared_ptr<FeatureExtractor> extractor)
{
    this->extractor = extractor;
}

void KeyFrameGraph::set_descriptor_matcher(std::shared_ptr<DescriptorMatcher> matcher)
{
    this->matcher = matcher;
}

void KeyFrameGraph::reset()
{
    std::lock_guard<std::mutex> lock(graphMutex);
    keyframe_graph.clear();
    raw_keyframe_queue.clear();
}

void KeyFrameGraph::terminate()
{
    FlagShouldQuit = true;
}

void KeyFrameGraph::optimize()
{
}

void KeyFrameGraph::main_loop()
{
    while (!FlagShouldQuit)
    {
        std::shared_ptr<RgbdFrame> keyframe;
        if (raw_keyframe_queue.pop(keyframe) && keyframe != NULL)
        {
            extract_features(keyframe);
            search_correspondence(keyframe);

            search_loop(keyframe);

            referenceFrame = keyframe;
            // keyframe->vmap.release();
            keyframe->nmap.release();
            keyframe_graph.push_back(keyframe);
        }

        if (FlagNeedOpt)
        {
            optimize();
            FlagNeedOpt = false;
        }
    }
}

} // namespace fusion