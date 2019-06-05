#ifndef RELOCALIZER_H
#define RELOCALIZER_H

#include "rgbd_frame.h"
#include "feature_point.h"
#include "thread_queue.h"
#include <mutex>
#include <queue>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

namespace fusion
{

class Relocalizer
{
public:
    Relocalizer(IntrinsicMatrix K);
    void insert_keyframe(RgbdFramePtr keyframe);
    void reset_relocalizer();
    // NOTE: this should be running in another thread.
    void main_loop();

    // NOTE: for visualisation purpose only
    void get_points(float *points, size_t &max_size);
    // TODO: not implemented
    void get_points_and_normal(float *points, float *normals, size_t &max_size);

    bool should_quit;

private:
    IntrinsicMatrix cam_param;
    cv::Ptr<cv::BRISK> BRISK;
    cv::Ptr<cv::xfeatures2d::SURF> SURF;

    // interface
    std::mutex new_keyframe_buffer_lock;
    ThreadQueue<RgbdFramePtr> new_keyframe_buffer;
    void process_new_keyframe();
    void extract_feature_points(RgbdFramePtr keyframe);
    void search_feature_correspondence();

    // keyframe graph
    std::vector<RgbdFramePtr> keyframe_graph;

    std::shared_ptr<FeaturePointFrame> current_point_struct;
    std::shared_ptr<FeaturePointFrame> reference_struct;
    std::shared_ptr<size_t> point_corresp;

    // map and points
    std::vector<std::shared_ptr<FeaturePointFrame>> keypoint_map;
};

} // namespace fusion

#endif