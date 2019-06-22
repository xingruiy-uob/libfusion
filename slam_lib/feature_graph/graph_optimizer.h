#ifndef SLAM_GRAPH_OPTIMIZER_H
#define SLAM_GRAPH_OPTIMIZER_H

#include <vector>
#include <ceres/ceres.h>
#include <fusion_core/rgbd_frame.h>

namespace fusion
{

class GraphOptimizer
{
public:
    GraphOptimizer();
    GraphOptimizer(GraphOptimizer &) = delete;
    GraphOptimizer(GraphOptimizer &&) = delete;
    GraphOptimizer &operator=(GraphOptimizer &) = delete;

    void set_keyframe_graph(const std::vector<RgbdFramePtr> graph);
    void optimize();
    void main_loop();

    bool FlagNeedOpt;
    bool should_quit;

private:
    std::vector<RgbdFramePtr> graph;
    ceres::Solver solver;
};

} // namespace fusion

#endif