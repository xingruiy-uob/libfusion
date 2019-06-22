#include "graph_optimizer.h"

namespace fusion
{

GraphOptimizer::GraphOptimizer()
    : FlagNeedOpt(false), should_quit(false)
{
}

void GraphOptimizer::set_keyframe_graph(const std::vector<RgbdFramePtr> graph)
{
}

void GraphOptimizer::optimize()
{
    FlagNeedOpt = false;
}

void GraphOptimizer::main_loop()
{
    while (should_quit)
    {
        if (FlagNeedOpt)
        {
            optimize();
        }
    }
}

} // namespace fusion