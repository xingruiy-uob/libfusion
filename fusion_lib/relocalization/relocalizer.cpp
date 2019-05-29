#include "relocalizer.h"

namespace fusion
{

Relocalizer::Relocalizer(IntrinsicMatrix K) : should_quit(false)
{
    BRISK = cv::BRISK::create();
    SURF = cv::xfeatures2d::SURF::create();
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
    new_keyframe_buffer.push(keyframe);
}

void Relocalizer::get_points(float *pt3d, size_t &max_size)
{
    max_size = 0;

    for (int i = 0; i < keypoint_map.size(); ++i)
    {
        auto point_frame = keypoint_map[i];

        for (int j = 0; j < point_frame->key_points.size(); ++j)
        {
            auto point = point_frame->key_points[j];

            pt3d[max_size * 3 + 0] = point->pos(0);
            pt3d[max_size * 3 + 1] = point->pos(1);
            pt3d[max_size * 3 + 2] = point->pos(2);
            max_size++;
        }
    }
}

void Relocalizer::get_points_and_normal(float *points, float *normals, size_t &max_size)
{
    max_size = 0;

    for (int i = 0; i < keypoint_map.size(); ++i)
    {
        auto point_frame = keypoint_map[i];

        for (int j = 0; j < point_frame->key_points.size(); ++j)
        {
            auto point = point_frame->key_points[j];

            points[max_size * 3 + 0] = point->pos(0);
            points[max_size * 3 + 1] = point->pos(1);
            points[max_size * 3 + 2] = point->pos(2);
            normals[max_size * 3 + 0] = point->vec_normal(0);
            normals[max_size * 3 + 0] = point->vec_normal(1);
            normals[max_size * 3 + 0] = point->vec_normal(2);
            max_size++;
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

    std::vector<cv::KeyPoint> detected_points;
    SURF->detect(source_image, detected_points);

    current_point_struct = std::make_shared<FeaturePointFrame>();

    if (keyframe->has_scene_data())
    {
        cv::Mat vmap = keyframe->get_vmap();
        cv::Mat nmap = keyframe->get_nmap();

        for (auto iter = detected_points.begin(); iter != detected_points.end(); ++iter)
        {
            float x = iter->pt.x;
            float y = iter->pt.y;

            // extract vertex and normal
            cv::Vec4f z = interpolate_bilinear(vmap, x, y);
            cv::Vec4f n = interpolate_bilinear(nmap, x, y);

            // validate vertex and normal
            if (n(3) < 0 || z(3) < 0)
                continue;

            std::shared_ptr<FeaturePoint> point(new FeaturePoint());
            point->pos << z(0), z(1), z(2);
            // convert point to world coordinate
            point->pos = frame_pose * point->pos;
            point->vec_normal << n(0), n(1), n(2);
            // Question: will it be better if we put all
            // opencv feature points into one separate vector?
            point->source = *iter;
            current_point_struct->key_points.emplace_back(point);
        }

        current_point_struct->reference = keyframe;
        // the map consists many small "observations" of it
        keypoint_map.emplace_back(current_point_struct);
    }
}

void Relocalizer::process_new_keyframe()
{
    if (new_keyframe_buffer.size_sync() == 0)
        return;

    auto keyframe = new_keyframe_buffer.front_sync();

    // extract feature points from the new keyframe
    extract_feature_points(keyframe);

    // pop the front element
    // this is thread locked.
    new_keyframe_buffer.pop_sync();
}

} // namespace fusion