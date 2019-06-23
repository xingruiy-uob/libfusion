#ifndef FEATURE_GRAPH_H
#define FEATURE_GRAPH_H

#include <Eigen/Core>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <xfusion/core/intrinsic_matrix.h>
#include <xutils/DataStruct/safe_queue.h>
#include "Frame.h"

namespace fusion
{

class RgbdFrame;
using RgbdFramePtr = std::shared_ptr<RgbdFrame>;

class FeatureGraph
{
public:
    ~FeatureGraph();
    FeatureGraph(const IntrinsicMatrix K);
    FeatureGraph(const FeatureGraph &) = delete;
    FeatureGraph(const FeatureGraph &&) = delete;
    FeatureGraph &operator=(const FeatureGraph &) = delete;

    void add_keyframe(RgbdFramePtr keyframe);
    void get_points(float *pt3d, size_t &count, size_t max_size);
    std::vector<Eigen::Matrix<float, 4, 4>> getKeyFramePoses() const;
    void main_loop();
    void terminate();
    void reset();

private:
    cv::Ptr<cv::BRISK> BRISK;
    cv::Ptr<cv::xfeatures2d::SURF> SURF;
    std::vector<RgbdFramePtr> keyframe_graph;
    RgbdFramePtr referenceFrame;
    IntrinsicMatrix cam_param;

    bool FlagShouldQuit;
    bool FlagNeedOpt;

    void optimize();
    void set_all_points_unvisited();
    void search_correspondence(RgbdFramePtr keyframe);
    void extract_features(RgbdFramePtr keyframe);

    xutils::SafeQueue<RgbdFramePtr> raw_keyframe_queue;
};

} // namespace fusion

#endif