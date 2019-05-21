#ifndef __SYSTEM__
#define __SYSTEM__

#include "rgbd_image.h"
#include "intrinsic_matrix.h"
#include "dense_mapping.h"
#include "dense_odometry.h"
#include <opencv2/opencv.hpp>

namespace fusion
{

class System
{
public:
    System(IntrinsicMatrix base, const int NUM_PYR);
    void process_images(const cv::Mat depth, const cv::Mat image);

    cv::Mat get_rendered_scene() const;

private:
    RgbdFramePtr current;
    RgbdFramePtr keyframe;

    size_t processed_frame_count;

    IntrinsicMatrixPyramidPtr cam_param;

    // system module
    std::shared_ptr<DenseMapping> mapping;
    std::shared_ptr<DenseOdometry> odometry;
};

} // namespace fusion

#endif