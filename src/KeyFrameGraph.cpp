#include "KeyFrameGraph.h"
#include <stdlib.h>
#include <ceres/ceres.h>
#include <xutils/DataStruct/stop_watch.h>

namespace fusion
{

KeyFrameGraph::~KeyFrameGraph()
{
    std::cout << "feature graph released." << std::endl;
}

KeyFrameGraph::KeyFrameGraph(const IntrinsicMatrix K)
    : FlagShouldQuit(false), cam_param(K), FlagNeedOpt(false)
{
    // SURF = cv::xfeatures2d::SURF::create();
    // BRISK = cv::BRISK::create();
    // Matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
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

void KeyFrameGraph::SetAllPointsUnvisited()
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

std::vector<Eigen::Matrix<float, 4, 4>> KeyFrameGraph::getKeyFramePoses() const
{
    std::vector<Eigen::Matrix<float, 4, 4>> poses;
    for (const auto &kf : keyframe_graph)
    {
        Eigen::Matrix<float, 4, 4> pose = kf->pose.cast<float>().matrix();
        poses.emplace_back(std::move(pose));
    }

    return poses;
}

cv::Mat KeyFrameGraph::getDescriptorsAll(std::vector<std::shared_ptr<Point3d>> &points)
{
    cv::Mat descritpors;
    points.clear();
    SetAllPointsUnvisited();

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
    SetAllPointsUnvisited();

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

// TODO : more sophisticated match filter
std::vector<bool> validate_matches(
    const std::vector<cv::KeyPoint> &src,
    const std::vector<cv::KeyPoint> &dst,
    const std::vector<cv::DMatch> &matches)
{
    const auto SizeMatches = matches.size();
    std::vector<bool> validation(SizeMatches);
    std::fill(validation.begin(), validation.end(), true);

    return validation;
}

static void AbsoluteOrientation(const std::vector<Eigen::Vector3f> &src,
                                const std::vector<Eigen::Vector3f> &dst,
                                Sophus::SE3d &pose)
{
}

static bool ComputePoseRANSAC(
    const std::vector<std::shared_ptr<Point3d>> &src_pts,
    const std::vector<std::shared_ptr<Point3d>> &dst_pts,
    const std::vector<cv::DMatch> &raw_matches,
    const std::vector<cv::DMatch> final_matches,
    const int MAX_RANSAC_ITER,
    Sophus::SE3d &pose_lastcurr)
{
    const int NUM_MATCHES = raw_matches.size();
    pose_lastcurr = Sophus::SE3d();
    if (NUM_MATCHES < 3)
        return false;

    int num_iter = 0;
    int num_inliers_best = 0;
    std::vector<int> sampleIdx = {0, 0, 0};

    while (num_iter < MAX_RANSAC_ITER)
    {
        sampleIdx = {rand() % NUM_MATCHES, rand() % NUM_MATCHES, rand() % NUM_MATCHES};

        if (sampleIdx[0] == sampleIdx[1] ||
            sampleIdx[1] == sampleIdx[2] ||
            sampleIdx[2] == sampleIdx[0])
            continue;

        num_iter++;
    }
}

void KeyFrameGraph::SearchLoop(RgbdFramePtr keyframe)
{
    std::lock_guard<std::mutex> lock(graphMutex);

    for (const auto &candidate : keyframe_graph)
    {
        if (candidate == NULL)
            continue;

        if (candidate == referenceFrame)
            continue;

        std::vector<std::vector<cv::DMatch>> matches;
        matcher->matchHammingKNN(candidate->descriptors, keyframe->descriptors, matches, 2);

        std::vector<std::vector<cv::DMatch>> candidate_matches;
        matcher->filterMatchesPairwise(keyframe->key_points, candidate->key_points, matches, candidate_matches);

        for (const auto &list : candidate_matches)
        {
            std::vector<cv::DMatch> final_matches;
            Sophus::SE3d pose_refcurr;
            // const auto valid = ComputePoseRANSAC(keyframe->key_points, candidate->key_points, list, final_matches, 100, pose_refcurr);

            // auto &source_image = keyframe->image;
            // auto &reference_image = candidate->image;
            // cv::Mat outImg;
            // cv::drawMatches(source_image, keyframe->cv_key_points, reference_image, candidate->cv_key_points, list, outImg, cv::Scalar(0, 255, 0));
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

    auto validation = validate_matches(keyframe->cv_key_points, referenceFrame->cv_key_points, matches);

    std::vector<cv::DMatch> refined_matches;
    for (int i = 0; i < validation.size(); ++i)
    {
        if (validation[i])
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

            refined_matches.push_back(std::move(match));
        }
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

void KeyFrameGraph::setFeatureExtractor(std::shared_ptr<FeatureExtractor> extractor)
{
    this->extractor = extractor;
}

void KeyFrameGraph::setDescriptorMatcher(std::shared_ptr<DescriptorMatcher> matcher)
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

            // SearchLoop(keyframe);

            referenceFrame = keyframe;
            keyframe->vmap.release();
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