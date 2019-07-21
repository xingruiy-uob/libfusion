#ifndef FEATURE_GRAPH_H
#define FEATURE_GRAPH_H

#include <Eigen/Core>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "data_struct/intrinsic_matrix.h"
#include "utils/safe_queue.h"
#include "tracking/icp_tracker.h"
#include "data_struct/rgbd_frame.h"
#include "features/extractor.h"
#include "features/matcher.h"

namespace fusion
{

class KeyFrameGraph
{
public:
    ~KeyFrameGraph();
    KeyFrameGraph(const IntrinsicMatrix K, const int NUM_PYR);

    void add_keyframe(std::shared_ptr<RgbdFrame> keyframe);
    void main_loop();
    void terminate();
    void reset();

    void set_feature_extractor(std::shared_ptr<FeatureExtractor>);
    void set_descriptor_matcher(std::shared_ptr<DescriptorMatcher>);
    void get_points(float *pt3d, size_t &count, size_t max_size);
    std::vector<Eigen::Matrix<float, 4, 4>> get_keyframe_poses() const;
    cv::Mat get_descriptor_all(std::vector<std::shared_ptr<Point3d>> &points);

private:
    std::shared_ptr<FeatureExtractor> extractor;
    std::shared_ptr<DescriptorMatcher> matcher;
    std::shared_ptr<DenseTracking> tracker;

    std::mutex graphMutex;
    std::vector<std::shared_ptr<RgbdFrame>> keyframe_graph;
    std::shared_ptr<RgbdFrame> referenceFrame;
    IntrinsicMatrix cam_param;

    bool FlagShouldQuit;
    bool FlagNeedOpt;

    void optimize();
    void set_all_points_unvisited();
    void search_correspondence(std::shared_ptr<RgbdFrame> keyframe);
    void search_loop(std::shared_ptr<RgbdFrame> keyframe);
    void extract_features(std::shared_ptr<RgbdFrame> keyframe);

    SafeQueue<std::shared_ptr<RgbdFrame>> raw_keyframe_queue;
};

} // namespace fusion

#endif