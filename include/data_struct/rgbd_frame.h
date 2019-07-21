#ifndef FUSION_RGBD_FRAME_H
#define FUSION_RGBD_FRAME_H

#include <map>
#include <mutex>
#include <memory>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include "data_struct/map_point.h"

namespace fusion
{

class RgbdFrame;
using RgbdFramePtr = std::shared_ptr<RgbdFrame>;

class RgbdFrame
{
public:
  RgbdFrame(const cv::Mat &depth, const cv::Mat &image, const size_t id, const double ts);

  std::vector<cv::KeyPoint> cv_key_points;
  std::vector<std::shared_ptr<Point3d>> key_points;
  std::map<RgbdFramePtr, Eigen::Matrix4f> neighbours;
  cv::Mat descriptors;

  std::size_t id;
  double timeStamp;
  Sophus::SE3d pose;

  cv::Mat image;
  cv::Mat depth;
  cv::Mat vmap;
  cv::Mat nmap;
};

} // namespace fusion

#endif