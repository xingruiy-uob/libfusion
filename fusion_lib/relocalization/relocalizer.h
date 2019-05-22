#ifndef __RELOCALIZER__
#define __RELOCALIZER__

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

namespace fusion
{

class Relocalizer
{
public:
    Relocalizer();

private:
    cv::Ptr<cv::BRISK> BRISK;
    cv::Ptr<cv::xfeatures2d::SURF> SURF;
};

} // namespace fusion

#endif