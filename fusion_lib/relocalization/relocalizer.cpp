#include "relocalizer.h"

namespace fusion
{

Relocalizer::Relocalizer(IntrinsicMatrix K)
{
    BRISK = cv::BRISK::create();
    SURF = cv::xfeatures2d::SURF::create();
    cam_param = K;
}

float interpolate_depth_bilinear(cv::Mat vmap, float x, float y)
{
    int u = (int)floor(x), v = (int)floor(y);
    float coeff_x = x - u, coeff_y = y - v;
    float z00 = vmap.ptr<cv::Vec4f>(v)[u](3);
    float z01 = vmap.ptr<cv::Vec4f>(v)[u + 1](3);
    float z10 = vmap.ptr<cv::Vec4f>(v + 1)[u](3);
    float z11 = vmap.ptr<cv::Vec4f>(v + 1)[u + 1](3);
    return (z00 * (1 - coeff_x) + z01 * coeff_x) * (1 - coeff_y) +
           z10 * ((1 - coeff_x) + z11 * coeff_x) * coeff_y;
}

void Relocalizer::insert_current_frame()
{
    match_by_pose_constraint();
}

void Relocalizer::insert_keyframe(RgbdFramePtr keyframe)
{
    cv::Mat source_image = keyframe->get_image();
    std::vector<cv::KeyPoint> detected_points;
    SURF->detect(source_image, detected_points);
    point_struct = std::make_shared<FeaturePointFrame>();

    if (keyframe->has_scene_data())
    {
        cv::Mat vmap = keyframe->get_vmap();
        cv::Mat nmap = keyframe->get_nmap();

        for (auto iter = detected_points.begin(); iter != detected_points.end(); ++iter)
        {
            float x = iter->pt.x;
            float y = iter->pt.y;

            // NOTE: this will produce erroneous measurements
            float z = interpolate_depth_bilinear(vmap, x, y);
            // Eigen::Vector4f normal = interpolate_normal_bilinear(nmap, x, y);
            Eigen::Vector3f pos;
            pos << (x - cam_param.cx) * cam_param.invfx * z,
                (y - cam_param.cy) * cam_param.invfy * z,
                z;

            std::shared_ptr<FeaturePoint> fp;
            fp->depth = z;
            fp->pos = pos;
            fp->source = *iter;
            point_struct->key_points.emplace_back(fp);
        }

        point_struct->reference = keyframe;
    }
}

void Relocalizer::set_relocalization_target(RgbdFramePtr frame)
{
    cv::Mat source_image = frame->get_image();
    std::vector<cv::KeyPoint> detected_points;
    SURF->detect(source_image, detected_points);
}

void Relocalizer::match_by_pose_constraint()
{
}

void Relocalizer::compute()
{
}

} // namespace fusion