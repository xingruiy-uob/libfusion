#ifndef __SYSTEM__
#define __SYSTEM__

#include "rgbd_frame.h"
#include "intrinsic_matrix.h"
#include "dense_mapping.h"
#include "dense_odometry.h"
#include "relocalizer.h"
#include <thread>
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>

namespace fusion
{

class System
{
public:
    ~System();
    System(IntrinsicMatrix base, const int NUM_PYR);
    void process_images(const cv::Mat depth, const cv::Mat image);

    // get rendered ray tracing map
    cv::Mat get_rendered_scene() const;
    cv::Mat get_rendered_scene_textured() const;

    // create a mesh from the map
    // and save it to a named file
    // it only contains vertex data
    void save_mesh_to_file(const char *str);

    // create mesh and store in the address
    // users are reponsible for allocating
    // the adresses in CUDA using `cudaMalloc`
    uint get_maximum_triangle_num() const; // TODO: not implemented
    void fetch_mesh_vertex_only(float3 *data, uint &max_size);
    void fetch_mesh_with_normal(float3 *vertex, float3 *normal, uint &max_size);
    void fetch_mesh_with_colour(float3 *vertex, uchar3 *colour, uint &max_size);

    // key points
    void fetch_key_points(float *points, size_t &max_size);
    void fetch_key_points_with_normal(float *points, float *normal, size_t &max_size);

    // retrieve current camera pose
    Eigen::Matrix4f get_current_camera_pose() const;
    bool is_initialized() const;

    // system controls
    void change_colour_mode(int colour_mode = 0);
    void change_run_mode(int run_mode = 0);
    void restart();

private:
    RgbdFramePtr current_frame;
    RgbdFramePtr last_tracked_frame;
    RgbdFramePtr current_keyframe;

    bool system_initialized;
    size_t processed_frame_count;

    IntrinsicMatrixPyramidPtr cam_param;

    // System modules
    std::shared_ptr<DenseMapping> mapping;
    std::shared_ptr<DenseOdometry> odometry;
    std::shared_ptr<Relocalizer> relocalizer;
    std::thread relocalizer_thread; // put relocalizer in another thread

    // Return TRUE if a new key frame is desired
    // return FALSE otherwise
    // TODO: this needs to be redesigned.
    bool keyframe_needed() const;
    void create_new_keyframe();
};

} // namespace fusion

#endif