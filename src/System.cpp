#include "System.h"
#include <xutils/DataStruct/stop_watch.h>

namespace fusion
{

System::~System()
{
    graph->terminate();
    graphThread.join();
}

System::System(const fusion::IntrinsicMatrix base, const int NUM_PYR)
    : frame_id(0), is_initialized(false), hasNewKeyFrame(false)
{
    mapping = std::make_shared<DenseMapping>(base);
    odometry = std::make_shared<DenseOdometry>(base, NUM_PYR);
    graph = std::make_shared<KeyFrameGraph>(base);
    extractor = std::make_shared<FeatureExtractor>();
    matcher = std::make_shared<DescriptorMatcher>();

    relocalizer = std::make_shared<Relocalizer>(base);
    relocalizer->setDescriptorMatcher(matcher);
    relocalizer->setFeatureExtractor(extractor);

    graphThread = std::thread(&KeyFrameGraph::main_loop, graph.get());
}

void System::initialization()
{
    is_initialized = true;
    current_frame->pose = initialPose;
    last_tracked_frame = current_keyframe = current_frame;
    // graph->add_keyframe(current_keyframe);
    hasNewKeyFrame = true;
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

    odometry->trackFrame(current_frame);

    if (!odometry->trackingLost)
    {
        auto reference_image = odometry->get_reference_image();
        auto reference_frame = reference_image->get_reference_frame();

        mapping->update(reference_image);
        mapping->raycast(reference_image->get_vmap(), reference_image->get_nmap(0), reference_frame->pose);
        reference_image->resize_device_map();

        if (keyframe_needed())
            create_keyframe();

        last_tracked_frame = current_frame;
        frame_id += 1;
    }
    else
    {
        std::vector<std::shared_ptr<Point3d>> points;
        auto descriptors = graph->getDescriptorsAll(points);

        relocalizer->setTargetFrame(current_frame);
        relocalizer->setMapPoints(points, descriptors);

        std::vector<Sophus::SE3d> candidates;
        relocalizer->computeRelocalizationCandidate(candidates);

        odometry->trackingLost = false;
    }

    if (hasNewKeyFrame)
    {
        auto reference_image = odometry->get_reference_image();
        auto reference_frame = reference_image->get_reference_frame();
        reference_image->get_vmap().download(reference_frame->vmap);
        reference_image->get_nmap().download(reference_frame->nmap);

        // reference_frame->set_scene_data(reference_image->get_vmap(), reference_image->get_nmap());
        // reference_frame->cv_key_points = extractor->getKeyPoints();
        graph->add_keyframe(reference_frame);
        hasNewKeyFrame = false;
    }
}

bool System::keyframe_needed() const
{
    auto pose = current_frame->pose;
    auto ref_pose = current_keyframe->pose;
    if ((pose.inverse() * ref_pose).translation().norm() > 0.1f)
        return true;
    return false;
}

void System::create_keyframe()
{
    current_keyframe = last_tracked_frame;
    // graph->add_keyframe(current_keyframe);
    hasNewKeyFrame = true;
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
    // initialPose = last_tracked_frame->pose;
    is_initialized = false;
    frame_id = 0;

    mapping->reset_mapping();
    odometry->reset();
    graph->reset();
}

void System::setLost(bool lost)
{
    odometry->trackingLost = true;
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

void System::fetch_key_points(float *points, size_t &count, size_t max)
{
    graph->get_points(points, count, max);
}

void System::fetch_key_points_with_normal(float *points, float *normal, size_t &max_size)
{
}

Eigen::Matrix4f System::get_camera_pose() const
{
    Eigen::Matrix4f T;
    if (odometry->get_reference_image())
    {
        T = odometry->get_reference_image()->get_reference_frame()->pose.cast<float>().matrix();
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

std::vector<Eigen::Matrix<float, 4, 4>> System::getKeyFramePoses() const
{
    return graph->getKeyFramePoses();
}

} // namespace fusion