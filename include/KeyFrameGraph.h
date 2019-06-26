#ifndef FEATURE_GRAPH_H
#define FEATURE_GRAPH_H

#include <Eigen/Core>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <xfusion/core/intrinsic_matrix.h>
#include <xutils/DataStruct/safe_queue.h>
#include "Frame.h"
#include "FeatureExtractor.h"
#include "DescriptorMatcher.h"

namespace fusion
{

class KeyFrameGraph
{
public:
    ~KeyFrameGraph();
    KeyFrameGraph(const IntrinsicMatrix K);

    KeyFrameGraph(const KeyFrameGraph &) = delete;
    KeyFrameGraph &operator=(KeyFrameGraph) = delete;

    cv::Mat getDescriptorsAll(std::vector<std::shared_ptr<Point3d>> &points);
    void add_keyframe(RgbdFramePtr keyframe);
    void get_points(float *pt3d, size_t &count, size_t max_size);
    std::vector<Eigen::Matrix<float, 4, 4>> getKeyFramePoses() const;
    void main_loop();
    void terminate();
    void reset();

private:
    FeatureExtractor extractor;
    DescriptorMatcher matcher;

    std::mutex graphMutex;
    std::vector<RgbdFramePtr> keyframe_graph;
    RgbdFramePtr referenceFrame;
    IntrinsicMatrix cam_param;

    bool FlagShouldQuit;
    bool FlagNeedOpt;

    void optimize();
    void SetAllPointsUnvisited();
    void search_correspondence(RgbdFramePtr keyframe);
    void SearchLoop(RgbdFramePtr keyframe);
    void extract_features(RgbdFramePtr keyframe);

    xutils::SafeQueue<RgbdFramePtr> raw_keyframe_queue;
};

} // namespace fusion

#endif