#include "relocalizer.h"

namespace fusion
{

Relocalizer::Relocalizer(IntrinsicMatrix K) : should_quit(false)
{
    BRISK = cv::BRISK::create();
    SURF = cv::xfeatures2d::SURF::create();
    cam_param = K;
}

cv::Vec4f interpolate_depth_bilinear(cv::Mat vmap, float x, float y)
{
    int u = (int)(x + 0.5f);
    int v = (int)(y + 0.5f);
    if (u >= 0 && v >= 0 && u < vmap.cols && v < vmap.rows)
    {
        return vmap.ptr<cv::Vec4f>(v)[u];
    }
}

cv::Vec3f interpolate_normal_bilinear(cv::Mat vmap, float x, float y)
{
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

void Relocalizer::get_keypoints_world(float *pt3d, size_t &max_size)
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

void Relocalizer::reset_relocalizer()
{
    while (new_keyframe_buffer.size_sync())
        new_keyframe_buffer.pop_sync();

    keypoint_map.clear();
    keyframe_graph.clear();
}

void Relocalizer::process_new_keyframe()
{
    if (new_keyframe_buffer.size_sync() == 0)
        return;

    auto keyframe = new_keyframe_buffer.front_sync();
    cv::Mat source_image = keyframe->get_image();
    auto frame_pose = keyframe->get_pose().cast<float>();

    std::vector<cv::KeyPoint> detected_points;
    SURF->detect(source_image, detected_points);

    current_point_struct = std::make_shared<FeaturePointFrame>();

    if (keyframe->has_scene_data())
    {
        cv::Mat vmap = keyframe->get_vmap();
        cv::Mat nmap = keyframe->get_nmap();
        cv::Mat depth = keyframe->get_depth();

        for (auto iter = detected_points.begin(); iter != detected_points.end(); ++iter)
        {
            float x = iter->pt.x;
            float y = iter->pt.y;

            // NOTE: this will produce erroneous measurements
            cv::Vec4f z = interpolate_depth_bilinear(vmap, x, y);
            Eigen::Vector3f pos;
            pos << z(0), z(1), z(2);

            std::shared_ptr<FeaturePoint> fp(new FeaturePoint());
            fp->pos = frame_pose * pos;
            fp->source = *iter;
            current_point_struct->key_points.emplace_back(fp);
        }

        current_point_struct->reference = keyframe;
        keypoint_map.emplace_back(current_point_struct);
    }

    std::cout << "frame : " << keyframe->get_id() << " processed." << std::endl;
    new_keyframe_buffer.pop_sync();
}

} // namespace fusion