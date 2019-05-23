#ifndef __SYSTEM__
#define __SYSTEM__

#include "rgbd_frame.h"
#include "intrinsic_matrix.h"
#include "dense_mapping.h"
#include "dense_odometry.h"
#include "relocalizer.h"
#include <opencv2/opencv.hpp>

#define RUN_MODE_PAUSE 0
#define RUN_MODE_CONTINUOUS 1
#define RUN_MODE_SINGLESHOT 2
#define INTEGRATION_ON 0
#define INTEGRATION_HOLD 1
#define COLOUR_MODE_DEPTH 0

namespace fusion
{

class System
{
public:
    System(IntrinsicMatrix base, const int NUM_PYR);
    void process_images(const cv::Mat depth, const cv::Mat image);

    void restart();
    cv::Mat get_rendered_scene() const;
    cv::Mat get_rendered_scene_textured() const;
    void save_mesh_to_file(const char *str);

private:
    RgbdFramePtr current;
    RgbdFramePtr keyframe;

    bool system_initialized;
    size_t processed_frame_count;

    IntrinsicMatrixPyramidPtr cam_param;

    // system module
    std::shared_ptr<DenseMapping> mapping;
    std::shared_ptr<DenseOdometry> odometry;
    std::shared_ptr<Relocalizer> relocalizer;
};

} // namespace fusion

#endif