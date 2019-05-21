#include "system.h"

namespace fusion
{

System::System(IntrinsicMatrix base, const int NUM_PYR)
    : processed_frame_count(0), keyframe(NULL)
{
    cam_param = std::make_shared<IntrinsicMatrixPyramid>(base, NUM_PYR);
    mapping = std::make_shared<DenseMapping>(cam_param);
    odometry = std::make_shared<DenseOdometry>(cam_param);
}

void System::process_images(const cv::Mat depth, const cv::Mat image)
{
    cv::Mat depth_float;
    depth.convertTo(depth_float, CV_32FC1, 1 / 1000.f);

    current = std::make_shared<RgbdFrame>(image, depth_float, processed_frame_count, 0);

    // needs initialization
    if (keyframe == NULL && processed_frame_count == 0)
    {
        keyframe = current;
    }

    odometry->track_frame(current);

    if (!odometry->is_tracking_lost())
    {
        auto reference_image = odometry->get_reference_image();

        mapping->update(reference_image);
        mapping->raycast(reference_image);
        reference_image->resize_device_map();
    }

    if (odometry->keyframe_needed())
        odometry->create_keyframe();

    processed_frame_count += 1;
}

cv::Mat System::get_rendered_scene() const
{
    return cv::Mat(odometry->get_reference_image()->get_rendered_image());
}

} // namespace fusion