#include "relocalizer.h"
#define MAX_SEARCH_RADIUS 5

namespace fusion
{

Relocalizer::Relocalizer(IntrinsicMatrix K) : should_quit(false)
{
    BRISK = cv::BRISK::create();
    SURF = cv::xfeatures2d::SURF::create(100);
    cam_param = K;
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

void Relocalizer::main_loop()
{
    while (!should_quit)
    {
        process_new_keyframe();
    }
}

void Relocalizer::insert_keyframe(RgbdFramePtr keyframe)
{
    // TODO: make this thread safe?
    new_keyframe_buffer.push(keyframe);
}

void Relocalizer::get_points(float *pt3d, size_t &max_size)
{
    max_size = 0;
    set_all_points_unvisited();

    for (const auto &point_frame : keypoint_map)
    {
        for (const auto &point : point_frame->key_points)
        {
            if (point->observations.size() < 2 || point->visited)
                continue;

            pt3d[max_size * 3 + 0] = point->pos(0);
            pt3d[max_size * 3 + 1] = point->pos(1);
            pt3d[max_size * 3 + 2] = point->pos(2);
            max_size++;

            point->visited = true;
        }
    }

    std::cout << max_size << std::endl;
}

void Relocalizer::get_points_and_normal(float *points, float *normals, size_t &max_size)
{
    max_size = 0;

    for (const auto &point_frame : keypoint_map)
    {
        for (const auto &point : point_frame->key_points)
        {
            if (point->observations.size() < 2 || point->visited)
                continue;

            points[max_size * 3 + 0] = point->pos(0);
            points[max_size * 3 + 1] = point->pos(1);
            points[max_size * 3 + 2] = point->pos(2);
            normals[max_size * 3 + 0] = point->vec_normal(0);
            normals[max_size * 3 + 0] = point->vec_normal(1);
            normals[max_size * 3 + 0] = point->vec_normal(2);
            max_size++;

            point->visited = true;
        }
    }
}

void Relocalizer::set_relocalization_target(RgbdFramePtr frame)
{
    relocalization_target = frame;
}

void Relocalizer::get_relocalized_result()
{
    cv::Mat source_img = relocalization_target->get_image();
    std::vector<cv::KeyPoint> detected_points;
    SURF->detect(source_img, detected_points);

    auto depth = relocalization_target->get_depth();

    auto ibegin = detected_points.begin();
    auto iend = detected_points.end();
    for (auto iter = ibegin; iter != iend; iter++)
    {
    }
}

void Relocalizer::set_all_points_unvisited()
{
    for (const auto &iter : keypoint_map)
    {
        for (const auto &iter2 : iter->key_points)
        {
            iter2->visited = false;
        }
    }
}

void Relocalizer::find_all_key_points_and_descriptors()
{
    set_all_points_unvisited();
    reference_points.clear();
    relocalization_reference.release();

    for (const auto &iter : keypoint_map)
    {
        for (const auto &iter2 : iter->key_points)
        {
            if (iter2->visited == true)
                continue;

            reference_points.push_back(iter2->pos);
            relocalization_reference.push_back(iter2->descriptor);
            iter2->visited = true;
        }
    }
}

void Relocalizer::reset_relocalizer()
{
    while (new_keyframe_buffer.size_sync())
        new_keyframe_buffer.pop_sync();

    keypoint_map.clear();
    keyframe_graph.clear();
}

void Relocalizer::extract_feature_points(RgbdFramePtr keyframe)
{
    cv::Mat source_image = keyframe->get_image();
    auto frame_pose = keyframe->get_pose().cast<float>();

    cv::Mat descriptors;
    std::vector<cv::KeyPoint> detected_points;
    SURF->detect(source_image, detected_points);
    BRISK->compute(source_image, detected_points, descriptors);

    current_point_struct = std::make_shared<FeaturePointFrame>();
    current_point_struct->cv_key_points.clear();
    current_point_struct->key_points.clear();
    current_point_struct->reference = NULL;

    if (keyframe->has_scene_data())
    {
        cv::Mat vmap = keyframe->get_vmap();
        cv::Mat nmap = keyframe->get_nmap();

        auto ibegin = detected_points.begin();
        auto iend = detected_points.end();
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

            std::shared_ptr<FeaturePoint> point(new FeaturePoint());
            point->pos << z(0), z(1), z(2);
            // convert point to world coordinate
            point->pos = frame_pose * point->pos;
            point->vec_normal << n(0), n(1), n(2);
            point->observations.emplace_back(std::make_pair(current_point_struct->reference, std::distance(ibegin, iter)));
            current_point_struct->cv_key_points.push_back(*iter);
            current_point_struct->key_points.emplace_back(point);
        }
    }
    else
    {
        // only the first frame will trigger this branch
        if (keyframe->get_id() != 0)
            std::cout << "control flow should not reach here!" << std::endl;
        current_point_struct->cv_key_points = detected_points;
    }

    // NOTE: should this be thread safe?
    keyframe_graph.push_back(keyframe);
    current_point_struct->reference = keyframe;
    // the map consists many small "observations" of it
    // NOTE: should this be thread safe?
    keypoint_map.emplace_back(current_point_struct);
}

void Relocalizer::process_new_keyframe()
{
    if (new_keyframe_buffer.size_sync() == 0)
        return;

    // thread safe front()
    auto keyframe = new_keyframe_buffer.front_sync();

    // extract feature points from the new keyframe
    extract_feature_points(keyframe);

    if (reference_struct != NULL)
    {
        search_feature_correspondence();
    }

    reference_struct = current_point_struct;
    // pop the front element
    // this is thread locked.
    new_keyframe_buffer.pop_sync();
}

void Relocalizer::search_feature_correspondence()
{
    if (reference_struct == NULL)
        return;

    auto pose_last_inv = reference_struct->reference->get_pose().inverse().cast<float>();
    auto keypoints_curr = current_point_struct->key_points;
    auto keypoints_last = reference_struct->key_points;
    auto cv_keypoints_last = reference_struct->cv_key_points;

    // for visualization only
    std::vector<cv::DMatch> matches(0);

    for (auto iter = keypoints_curr.begin(); iter != keypoints_curr.end(); ++iter)
    {
        auto pt3d = pose_last_inv * (*iter)->pos;
        float x = cam_param.fx * pt3d(0) / pt3d(2) + cam_param.cx;
        float y = cam_param.fy * pt3d(1) / pt3d(2) + cam_param.cy;

        float min_dist = MAX_SEARCH_RADIUS;
        int best_idx = -1;
        if (x >= 0 && y >= 0 && x <= cam_param.width - 1 && y <= cam_param.height - 1)
        {
            for (auto iter2 = cv_keypoints_last.begin(); iter2 != cv_keypoints_last.end(); ++iter2)
            {
                float x2 = iter2->pt.x;
                float y2 = iter2->pt.y;
                float dist = sqrt(std::pow(x2 - x, 2) + std::pow(y2 - y, 2));

                if (dist < min_dist)
                {
                    min_dist = dist;
                    best_idx = std::distance(cv_keypoints_last.begin(), iter2);
                }
            }
        }

        if (best_idx >= 0)
        {
            (*iter)->observations.emplace_back(std::make_pair(reference_struct->reference, best_idx));
            // for visualization only
            cv::DMatch match;
            // match.distance = min_dist;
            match.trainIdx = best_idx;
            match.queryIdx = std::distance(keypoints_curr.begin(), iter);
            matches.push_back(match);
        }
    }

    // for visualization only
    /*
    auto image_curr = current_point_struct->reference->get_image();
    auto image_last = reference_struct->reference->get_image();
    auto cv_key_points_curr = current_point_struct->cv_key_points;
    cv::Mat outImg;
    cv::drawMatches(image_curr, cv_key_points_curr, image_last, cv_keypoints_last, matches, outImg);
    cv::cvtColor(outImg, outImg, cv::COLOR_RGB2BGR);
    cv::imshow("img", outImg);
    cv::waitKey(0);
    */
}

} // namespace fusion