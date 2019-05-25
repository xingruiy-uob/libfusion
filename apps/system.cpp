#include "system.h"
#include "cuda_imgproc.h"

namespace fusion
{

System::System(IntrinsicMatrix base, const int NUM_PYR)
    : processed_frame_count(0), keyframe(NULL), system_initialized(false)
{
    mapping = std::make_shared<DenseMapping>(base);
    odometry = std::make_shared<DenseOdometry>(base, NUM_PYR);
    relocalizer = std::make_shared<Relocalizer>(base);
}

void System::process_images(const cv::Mat depth, const cv::Mat image)
{
    cv::Mat depth_float;
    depth.convertTo(depth_float, CV_32FC1, 1 / 1000.f);

    current = std::make_shared<RgbdFrame>(depth_float, image, processed_frame_count, 0);

    // needs initialization
    if (system_initialized)
    {
        keyframe = current;

        // break the loop
        system_initialized = true;
    }

    odometry->track_frame(current);

    if (!odometry->is_tracking_lost())
    {
        auto reference_image = odometry->get_reference_image();

        // if (processed_frame_count == 0)
        // {
        //     cv::Mat render(reference_image->get_rendered_image());
        //     cv::Mat img2;
        //     cv::cvtColor(image, img2, cv::COLOR_RGB2BGR);
        //     cv::imwrite("depth.png", render);
        //     cv::imwrite("image.png", img2);
        // }

        // if (processed_frame_count == 100)
        // {
        //     cv::Mat render(reference_image->get_rendered_image());
        //     cv::imwrite("depth2.png", render);
        // }

        mapping->update(reference_image);
        mapping->raycast(reference_image);
        reference_image->resize_device_map();

        // if (processed_frame_count == 0)
        // {
        //     cv::Mat render(reference_image->get_rendered_image());
        //     cv::imwrite("render.png", render);
        // }

        // if (processed_frame_count == 100)
        // {
        //     cv::Mat render(reference_image->get_rendered_image());
        //     cv::imwrite("render2.png", render);
        // }

        processed_frame_count += 1;
    }
}

cv::Mat System::get_rendered_scene() const
{
    return cv::Mat(odometry->get_reference_image()->get_rendered_image());
}

cv::Mat System::get_rendered_scene_textured() const
{
}

void System::restart()
{
    mapping->restart_mapping();
    odometry->restart_tracking();
    system_initialized = false;
    processed_frame_count = 0;
}

void System::save_mesh_to_file(const char *str)
{
    mapping->create_scene_mesh();
    mapping->write_mesh_to_file(str);
}

void System::create_mesh_gl(float3 *data, uint &max_size)
{
    mapping->create_scene_mesh(data, max_size);
}

Eigen::Matrix4f System::get_current_camera_pose() const
{
    return odometry->get_current_pose_matrix();
}

} // namespace fusion