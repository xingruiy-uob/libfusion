#include "feature_graph.h"

namespace fusion
{

FeatureGraph::~FeatureGraph()
{
    std::cout << "feature graph released." << std::endl;
}

FeatureGraph::FeatureGraph(const IntrinsicMatrix K)
    : should_quit(false), cam_param(K)
{
    SURF = cv::xfeatures2d::SURF::create();
    BRISK = cv::BRISK::create();
}

cv::Vec4f interpolate_bilinear(cv::Mat vmap, float x, float y)
{
    int u = (int)(x + 0.5f);
    int v = (int)(y + 0.5f);
    if (u >= 0 && v >= 0 && u < vmap.cols && v < vmap.rows)
    {
        return vmap.ptr<cv::Vec4f>(v)[u];
    }
}

void FeatureGraph::set_all_points_unvisited()
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

std::vector<Eigen::Matrix<float, 4, 4>> FeatureGraph::getKeyFramePoses() const
{
    std::vector<Eigen::Matrix<float, 4, 4>> poses;
    for (const auto &kf : keyframe_graph)
    {
        Eigen::Matrix<float, 4, 4> pose = kf->get_pose().cast<float>().matrix();
        poses.emplace_back(std::move(pose));
    }

    return poses;
}

void FeatureGraph::get_points(float *pt3d, size_t &count, size_t max_size)
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
}

void FeatureGraph::add_keyframe(std::shared_ptr<RgbdFrame> keyframe)
{
    raw_keyframe_queue.push(keyframe);
}

void FeatureGraph::extract_features(RgbdFramePtr keyframe)
{
    cv::Mat source_image = keyframe->get_image();
    auto frame_pose = keyframe->get_pose().cast<float>();

    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keypoints;
    SURF->detect(source_image, keypoints);
    BRISK->compute(source_image, keypoints, descriptors);

    keyframe->cv_key_points.clear();
    keyframe->key_points.clear();
    keyframe->descriptors.release();

    if (keyframe->has_scene_data())
    {
        cv::Mat vmap = keyframe->get_vmap();
        cv::Mat nmap = keyframe->get_nmap();

        auto ibegin = keypoints.begin();
        auto iend = keypoints.end();
        for (auto iter = ibegin; iter != iend; ++iter)
        {
            float x = iter->pt.x;
            float y = iter->pt.y;

            // extract vertex and normal
            cv::Vec4f z = interpolate_bilinear(vmap, x, y);
            cv::Vec4f n = interpolate_bilinear(nmap, x, y);

            // validate vertex and normal
            if (n(3) < 0 || z(3) < 0 || z(2) < 0.1f)
                continue;

            std::shared_ptr<RgbdFrame::Point3d> point(new RgbdFrame::Point3d());
            point->pos << z(0), z(1), z(2);
            // convert point to world coordinate
            point->observations = 1;
            point->pos = frame_pose * point->pos;
            point->vec_normal << n(0), n(1), n(2);
            keyframe->cv_key_points.push_back(*iter);
            keyframe->key_points.emplace_back(point);
            keyframe->descriptors.push_back(descriptors.row(std::distance(keypoints.begin(), iter)));
        }
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

void FeatureGraph::search_correspondence(RgbdFramePtr keyframe)
{
    if (referenceFrame == NULL)
        return;

    const auto &fx = cam_param.fx;
    const auto &fy = cam_param.fy;
    const auto &cx = cam_param.cx;
    const auto &cy = cam_param.cy;
    const auto &cols = cam_param.width;
    const auto &rows = cam_param.height;
    auto poseInvRef = referenceFrame->get_pose().cast<float>().inverse();

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
    // cv::Mat src_image = keyframe->get_image();
    // cv::Mat ref_image = referenceFrame->get_image();
    // cv::drawMatches(src_image, keyframe->cv_key_points,
    //                 ref_image, referenceFrame->cv_key_points,
    //                 refined_matches, outImg, cv::Scalar(0, 255, 0));
    // cv::imshow("outImg", outImg);
    // cv::waitKey(1);
}

void FeatureGraph::reset()
{
    keyframe_graph.clear();
    raw_keyframe_queue.clear();
}

void FeatureGraph::terminate()
{
    should_quit = true;
}

void FeatureGraph::main_loop()
{
    while (!should_quit)
    {
        std::shared_ptr<RgbdFrame> keyframe;
        if (raw_keyframe_queue.pop(keyframe) && keyframe != NULL)
        {
            extract_features(keyframe);
            search_correspondence(keyframe);

            referenceFrame = keyframe;

            keyframe->get_vmap().release();
            keyframe->get_nmap().release();
            keyframe_graph.push_back(keyframe);
        }
    }
}

} // namespace fusion