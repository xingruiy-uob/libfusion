#include "relocalizer.h"

namespace fusion
{

Relocalizer::Relocalizer()
{
    BRISK = cv::BRISK::create();
    SURF = cv::xfeatures2d::SURF::create();
}

} // namespace fusion