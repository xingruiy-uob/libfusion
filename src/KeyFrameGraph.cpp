#include "KeyFrameGraph.h"
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
    SURF = cv::xfeatures2d::SURF::create();
    BRISK = cv::BRISK::create();
    Matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
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

    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keypoints;
    SURF->detect(source_image, keypoints);
    BRISK->compute(source_image, keypoints, descriptors);

    keyframe->cv_key_points.clear();
    keyframe->key_points.clear();
    keyframe->descriptors.release();

    if (!keyframe->vmap.empty())
    {
        cv::Mat vmap = keyframe->vmap;
        cv::Mat nmap = keyframe->nmap;

        auto ibegin = keypoints.begin();
        auto iend = keypoints.end();
        for (auto iter = ibegin; iter != iend; ++iter)
        {
            float x = iter->pt.x;
            float y = iter->pt.y;

            // extract vertex and normal
            cv::Vec4f z = interpolate_bilinear(vmap, x, y);
            cv::Vec4f n = interpolate_bilinear(nmap, x, y);

            if (n(3) > 0 && z(3) > 0 && z == z && n == n)
            {
                std::shared_ptr<RgbdFrame::Point3d> point(new RgbdFrame::Point3d());
                point->pos << z(0), z(1), z(2);
                // convert point to world coordinate
                point->observations = 1;
                point->pos = frame_pose * point->pos;
                point->vec_normal << n(0), n(1), n(2);
                point->vec_normal = rotation * point->vec_normal;
                point->descriptors = descriptors.row(std::distance(keypoints.begin(), iter));
                keyframe->cv_key_points.push_back(*iter);
                keyframe->key_points.emplace_back(point);
                keyframe->descriptors.push_back(point->descriptors);
            }
        }

        std::cout << "Keyframe: " << keyframe->id << " ; Features: " << keyframe->key_points.size() << std::endl;
    }
    else
    {
        std::cout << "control flow should not reach here!" << std::endl;
        keyframe->cv_key_points = keypoints;
        keyframe->key_points.resize(keypoints.size());
    }
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

static void FilterMatchesPairwise(
    const std::vector<std::shared_ptr<RgbdFrame::Point3d>> &src_pts,
    const std::vector<std::shared_ptr<RgbdFrame::Point3d>> &dst_pts,
    const std::vector<std::vector<cv::DMatch>> knnMatches,
    std::vector<std::vector<cv::DMatch>> &candidate_matches,
    const int NUM_SEARCH_CHAIN_SIZE = 1)
{
    xutils::StopWatch sw(true);
    std::vector<cv::DMatch> rawMatch;
    candidate_matches.clear();
    for (const auto &match : knnMatches)
    {
        if (match[0].distance / match[1].distance < 0.6)
        {
            rawMatch.push_back(std::move(match[0]));
        }
        else
        {
            rawMatch.push_back(std::move(match[0]));
            rawMatch.push_back(std::move(match[1]));
        }
    }

    const int NUM_RAW_MATCHES = rawMatch.size();
    cv::Mat adjecencyMat = cv::Mat::zeros(NUM_RAW_MATCHES, NUM_RAW_MATCHES, CV_32FC1);

    for (int y = 0; y < adjecencyMat.rows; ++y)
    {
        float *row = adjecencyMat.ptr<float>(y);
        const auto &match_y = rawMatch[y];
        const auto &match_y_src = src_pts[match_y.queryIdx];
        const auto &match_y_dst = dst_pts[match_y.trainIdx];

        for (int x = 0; x < adjecencyMat.cols; ++x)
        {
            const auto &match_x = rawMatch[x];
            const auto &match_x_src = src_pts[match_x.queryIdx];
            const auto &match_x_dst = dst_pts[match_x.trainIdx];

            if (match_x.trainIdx == match_y.trainIdx || match_x.queryIdx == match_y.queryIdx)
                continue;

            if (x == y)
            {
                row[x] = std::exp(-cv::norm(match_x_src->descriptors, match_x_dst->descriptors, cv::NORM_HAMMING));
            }
            else if (y < x)
            {

                const float src_dist = (match_x_src->pos - match_y_src->pos).norm();
                const float src_angle = std::acos(match_x_src->vec_normal.dot(match_y_src->vec_normal));

                const float dst_dist = (match_x_dst->pos - match_y_dst->pos).norm();
                const float dst_angle = std::acos(match_x_dst->vec_normal.dot(match_y_dst->vec_normal));

                float score = std::exp(-(std::fabs(src_dist - dst_dist) + std::fabs(src_angle - dst_angle)));
                if (std::isnan(score))
                    score = 0;

                row[x] = score;
            }
            else
            {
                row[x] = adjecencyMat.ptr<float>(x)[y];
            }
        }
    }

    cv::Mat reducedAM;
    cv::reduce(adjecencyMat, reducedAM, 0, cv::ReduceTypes::REDUCE_SUM);
    cv::Mat idxMat;
    cv::sortIdx(reducedAM, idxMat, cv::SortFlags::SORT_DESCENDING);

    std::vector<int> idxList;
    for (int y = 0; y < NUM_SEARCH_CHAIN_SIZE; ++y)
    {
        std::vector<cv::DMatch> selectedMatches;
        int head_idx = -1;
        size_t num_selected = 0;
        for (int x = y; x < idxMat.cols; ++x)
        {
            const auto &idx = idxMat.ptr<int>(0)[x];

            if (head_idx < 0)
            {
                head_idx = idx;
                selectedMatches.push_back(rawMatch[idx]);
                num_selected += 1;
            }
            else
            {
                const float &score = adjecencyMat.ptr<float>(head_idx)[idx];
                if (score > 0.1f)
                {
                    selectedMatches.push_back(rawMatch[idx]);
                    num_selected += 1;
                }
            }

            if (num_selected >= 200)
            {
                break;
            }
        }

        candidate_matches.push_back(selectedMatches);
    }

    std::cout << sw << std::endl;
}

static bool ComputePoseRANSAC(
    const std::vector<std::shared_ptr<RgbdFrame::Point3d>> &src_pts,
    const std::vector<std::shared_ptr<RgbdFrame::Point3d>> &dst_pts,
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

    while (num_iter < MAX_RANSAC_ITER)
    {
        bool badSample = false;

        num_iter++;
    }
}

void KeyFrameGraph::SearchLoop(RgbdFramePtr keyframe)
{
    for (const auto &candidate : keyframe_graph)
    {
        if (candidate == referenceFrame)
            continue;

        std::vector<std::vector<cv::DMatch>> matches;
        Matcher->knnMatch(keyframe->descriptors, candidate->descriptors, matches, 2);

        std::vector<std::vector<cv::DMatch>> candidate_matches;
        FilterMatchesPairwise(keyframe->key_points, candidate->key_points, matches, candidate_matches, 1);

        for (const auto &list : candidate_matches)
        {
            std::vector<cv::DMatch> final_matches;
            Sophus::SE3d pose_refcurr;
            const auto valid = ComputePoseRANSAC(keyframe->key_points, candidate->key_points, list, final_matches, 100, pose_refcurr);

            auto &source_image = keyframe->image;
            auto &reference_image = candidate->image;
            cv::Mat outImg;
            cv::drawMatches(source_image, keyframe->cv_key_points, reference_image, candidate->cv_key_points, list, outImg, cv::Scalar(0, 255, 0));
            cv::cvtColor(outImg, outImg, cv::COLOR_RGB2BGR);
            cv::imshow("outImg", outImg);
            cv::waitKey(0);
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

            keyframe->key_points[query_id]->observations += referenceFrame->key_points[train_id]->observations;
            referenceFrame->key_points[train_id] = keyframe->key_points[query_id];

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

void KeyFrameGraph::reset()
{
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

            SearchLoop(keyframe);

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