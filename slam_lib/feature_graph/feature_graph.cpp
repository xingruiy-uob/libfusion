#include "feature_graph.h"

namespace fusion
{

FeatureGraph::FeatureGraph()
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

void FeatureGraph::get_points(float *pt3d, size_t &max_size)
{
    max_size = 0;
    set_all_points_unvisited();

    for (const auto &kf : keyframe_graph)
    {
        for (const auto &point : kf->key_points)
        {
            if (point != NULL)
            {
                if (point->observations <= 0 || point->visited)
                    continue;

                pt3d[max_size * 3 + 0] = point->pos(0);
                pt3d[max_size * 3 + 1] = point->pos(1);
                pt3d[max_size * 3 + 2] = point->pos(2);
                max_size++;

                point->visited = true;
            }
        }
    }

    std::cout << max_size << std::endl;
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

    if (keyframe->has_scene_data())
    {
        std::cout << keyframe->get_id() << std::endl;
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
        }
    }
    else
    {
        // only the first frame will trigger this branch
        if (keyframe->get_id() != 0)
            std::cout << "control flow should not reach here!" << std::endl;
        keyframe->cv_key_points = keypoints;
        keyframe->key_points.resize(keypoints.size());
    }

    keyframe_graph.push_back(keyframe);
}

void FeatureGraph::search_correspondence()
{
}

void FeatureGraph::main_loop()
{
    while (!should_quit)
    {
        if (raw_keyframe_queue.size_safe() != 0)
        {
            auto keyframe = raw_keyframe_queue.front_safe();

            extract_features(keyframe);

            raw_keyframe_queue.pop_safe();
        }
    }
}

} // namespace fusion