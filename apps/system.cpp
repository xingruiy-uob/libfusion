#include "system.h"

namespace fusion
{

System::~System()
{
}

System::System(IntrinsicMatrix base, const int NUM_PYR)
    : frame_id(0), is_initialized(false)
{
    mapping = std::make_shared<DenseMapping>(base);
    odometry = std::make_shared<DenseOdometry>(base, NUM_PYR);
    features = std::make_shared<FeatureGraph>();
}

void System::initialization()
{
    is_initialized = true;
    last_tracked_frame = current_keyframe = current_frame;
    features->add_keyframe(current_keyframe);
}

void System::process_images(const cv::Mat depth, const cv::Mat image)
{
    cv::Mat depth_float;
    depth.convertTo(depth_float, CV_32FC1, 1 / 1000.f);

    current_frame = std::make_shared<RgbdFrame>(depth_float, image, frame_id, 0);

    if (!is_initialized)
    {
        initialization();
    }

    odometry->track_frame(current_frame);

    if (!odometry->is_tracking_lost())
    {
        auto reference_image = odometry->get_reference_image();
        auto reference_frame = reference_image->get_reference_frame();

        mapping->update(reference_image);
        // mapping->update(reference_image->depth_float, reference_image->get_image(), reference_frame->get_pose());
        mapping->raycast(reference_image->vmap_pyr[0], reference_image->nmap_pyr[0], reference_frame->get_pose());

        reference_image->resize_device_map();
        reference_frame->set_scene_data(reference_image->get_vmap(), reference_image->get_nmap());

        if (keyframe_needed())
            create_keyframe();

        last_tracked_frame = current_frame;
        frame_id += 1;
    }

    cudaDeviceSynchronize();
}

bool System::keyframe_needed() const
{
    auto pose = current_frame->get_pose();
    auto ref_pose = current_keyframe->get_pose();
    if ((pose.inverse() * ref_pose).translation().norm() > 0.1f)
        return true;
    return false;
}

void System::create_keyframe()
{
    current_keyframe = last_tracked_frame;
    features->add_keyframe(current_keyframe);
}

cv::Mat System::get_shaded_depth()
{
    if (odometry->get_current_image())
        return cv::Mat(odometry->get_current_image()->get_rendered_image());
}

cv::Mat System::get_rendered_scene() const
{
    return cv::Mat(odometry->get_reference_image()->get_rendered_image());
}

cv::Mat System::get_rendered_scene_textured() const
{
    return cv::Mat(odometry->get_reference_image()->get_rendered_scene_textured());
}

void System::restart()
{
    is_initialized = false;
    frame_id = 0;

    mapping->reset_mapping();
    odometry->reset_tracking();
}

void System::save_mesh_to_file(const char *str)
{
}

size_t System::fetch_mesh_vertex_only(float *vertex)
{
    return mapping->fetch_mesh_vertex_only(vertex);
}

size_t System::fetch_mesh_with_normal(float *vertex, float *normal)
{
    return mapping->fetch_mesh_with_normal(vertex, normal);
}

size_t System::fetch_mesh_with_colour(float *vertex, unsigned char *colour)
{
    return mapping->fetch_mesh_with_colour(vertex, colour);
}

void System::fetch_key_points(float *points, size_t &max_size)
{
    features->get_points(points, max_size);
}

void System::fetch_key_points_with_normal(float *points, float *normal, size_t &max_size)
{
}

Eigen::Matrix4f System::get_camera_pose() const
{
    Eigen::Matrix4f T;
    if (odometry->get_reference_image())
    {
        T = odometry->get_reference_image()->get_reference_frame()->get_pose().cast<float>().matrix();
    }
    return T;
}

void System::writeMapToDisk(std::string file_name) const
{
    mapping->writeMapToDisk(file_name);
}

void System::readMapFromDisk(std::string file_name)
{
    mapping->readMapFromDisk(file_name);
}

} // namespace fusion