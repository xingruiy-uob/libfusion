#include "KeyFrameGraph.h"
#include "PoseEstimator.h"
#include <stdlib.h>
#include <ceres/ceres.h>
#include <xutils/DataStruct/stop_watch.h>

namespace fusion
{

KeyFrameGraph::~KeyFrameGraph()
{
    std::cout << "feature graph released." << std::endl;
}

KeyFrameGraph::KeyFrameGraph(const IntrinsicMatrix K, const int NUM_PYR)
    : FlagShouldQuit(false), cam_param(K),
      FlagNeedOpt(false), extractor(NULL),
      matcher(NULL)
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
    auto rotation = frame_pose.so3();

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
        if (candidate == NULL)
            continue;

        if (candidate == referenceFrame)
            continue;

        std::vector<std::vector<cv::DMatch>> matches;
        std::vector<std::vector<cv::DMatch>> candidate_matches;
        matcher->matchHammingKNN(candidate->descriptors, keyframe->descriptors, matches, 2);

        std::vector<cv::DMatch> list;
        matcher->filter_matches_ratio_test(matches, list);
        candidate_matches.push_back(list);

        // std::vector<std::vector<cv::DMatch>> candidate_matches;
        // matcher->filterMatchesPairwise(keyframe->key_points, candidate->key_points, matches, candidate_matches);

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
            float inlier_ratio, confidence;
            PoseEstimator::RANSAC(src_pts, dst_pts, outliers, estimate, inlier_ratio, confidence);

            if (inlier_ratio > 0.7 && confidence >= 0.95)
            {
                tracker->set_reference_vmap(cv::cuda::GpuMat(candidate->vmap));
                tracker->set_reference_image(cv::cuda::GpuMat(keyframe->image));

                TrackingContext context;
                context.use_initial_guess_ = true;
                std::cout << estimate << std::endl;

                context.initial_estimate_ = Sophus::SE3f(estimate).cast<double>();
                context.max_iterations_ = {10, 5, 0, 0, 0};
                auto result = tracker->compute_transform(context);

                std::cout << result.update.matrix3x4() << std::endl;
            }

            // std::vector<cv::DMatch> refined_matches;
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
            // cv::waitKey(0);
        }
    }
}

void KeyFrameGraph::search_correspondence(RgbdFramePtr keyframe)
{
    if (referenceFrame == NULL)
        return;

    const auto &fx = cam_param.fx;
    const auto &fy = cam_param.fy;
    const auto &cx = cam_param.cx;
    const auto &cy = cam_param.cy;
    const auto &cols = cam_param.width;
    const auto &rows = cam_param.height;
    auto poseInvRef = referenceFrame->pose.cast<float>().inverse();

    std::vector<cv::DMatch> matches;

    for (int i = 0; i < keyframe->key_points.size(); ++i)
    {
        const auto &desc_src = keyframe->descriptors.row(i);
        auto pt_in_ref = poseInvRef * keyframe->key_points[i]->pos;
        auto x = fx * pt_in_ref(0) / pt_in_ref(2) + cx;
        auto y = fy * pt_in_ref(1) / pt_in_ref(2) + cy;

        auto th_dist = 0.1f;
        auto min_dist = 64;
        int best_idx = -1;

        if (x >= 0 && y >= 0 && x < cols - 1 && y < rows - 1)
        {
            for (int j = 0; j < referenceFrame->key_points.size(); ++j)
            {
                if (referenceFrame->key_points[j] == NULL)
                    continue;

                auto dist = (referenceFrame->key_points[j]->pos - keyframe->key_points[i]->pos).norm();

                if (dist < th_dist)
                {
                    const auto &desc_ref = referenceFrame->descriptors.row(j);
                    auto desc_dist = cv::norm(desc_src, desc_ref, cv::NormTypes::NORM_HAMMING);
                    if (desc_dist < min_dist)
                    {
                        min_dist = desc_dist;
                        best_idx = j;
                    }
                }
            }
        }

        if (best_idx >= 0)
        {
            cv::DMatch match;
            match.queryIdx = i;
            match.trainIdx = best_idx;
            matches.push_back(std::move(match));
        }
    }

    // std::vector<cv::DMatch> refined_matches;
    for (int i = 0; i < matches.size(); ++i)
    {

        const auto &match = matches[i];
        const auto query_id = match.queryIdx;
        const auto train_id = match.trainIdx;

        if (keyframe->cv_key_points[query_id].response > referenceFrame->cv_key_points[train_id].response)
        {
            keyframe->key_points[query_id]->observations += referenceFrame->key_points[train_id]->observations;
            referenceFrame->key_points[train_id] = keyframe->key_points[query_id];
            referenceFrame->cv_key_points[train_id].response = keyframe->cv_key_points[query_id].response;
        }
        else
        {
            referenceFrame->key_points[train_id]->observations += keyframe->key_points[query_id]->observations;
            keyframe->key_points[query_id] = referenceFrame->key_points[train_id];
            keyframe->cv_key_points[query_id].response = referenceFrame->cv_key_points[train_id].response;
        }

        // refined_matches.push_back(std::move(match));
    }

    // cv::Mat outImg;
    // cv::Mat src_image = keyframe->image;
    // cv::Mat ref_image = referenceFrame->image;
    // cv::drawMatches(src_image, keyframe->cv_key_points,
    //                 ref_image, referenceFrame->cv_key_points,
    //                 refined_matches, outImg, cv::Scalar(0, 255, 0));
    // cv::imshow("correspInit", outImg);
    // cv::waitKey(1);
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