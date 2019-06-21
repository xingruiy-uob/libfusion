#ifndef FEATURE_GRAPH_H
#define FEATURE_GRAPH_H

#include "fusion_core/rgbd_frame.h"
#include <Eigen/Core>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <xutils/DataStruct/safe_queue.h>

namespace fusion
{

class RgbdFrame;
using RgbdFramePtr = std::shared_ptr<RgbdFrame>;

class FeatureGraph
{
public:
    FeatureGraph();
    FeatureGraph(const FeatureGraph &) = delete;
    FeatureGraph &operator=(const FeatureGraph &) = delete;

    void add_keyframe(RgbdFramePtr keyframe);
    void get_points(float *pt3d, size_t &max_size);
    void main_loop();

    bool should_quit;

private:
    cv::Ptr<cv::BRISK> BRISK;
    cv::Ptr<cv::xfeatures2d::SURF> SURF;
    std::vector<RgbdFramePtr> keyframe_graph;

    void set_all_points_unvisited();
    void search_correspondence();
    void extract_features(RgbdFramePtr keyframe);

    xutils::SafeQueue<RgbdFramePtr> raw_keyframe_queue;
};

} // namespace fusion

#endif