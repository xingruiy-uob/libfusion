#ifndef FUSION_RGBD_FRAME_H
#define FUSION_RGBD_FRAME_H

#include <mutex>
#include <memory>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include "intrinsic_matrix.h"

namespace fusion
{

class RgbdFrame;
using RgbdFramePtr = std::shared_ptr<RgbdFrame>;

class RgbdFrame
{
public:
  // delete default and copy constructors
  RgbdFrame() = delete;
  RgbdFrame(const RgbdFrame &) = delete;
  RgbdFrame &operator=(const RgbdFrame &) = delete;

  // create new rgbd frame
  RgbdFrame(const cv::Mat &depth, const cv::Mat &image, size_t id, double ts);

  // if the frame contains map projections
  bool has_scene_data() const;

  // get frame id
  size_t get_id() const;

  // get source image : CV_8UC3
  cv::Mat get_image() const;

  // get source depth : CV_32FC1
  cv::Mat get_depth() const;

  cv::Mat get_vmap() const;

  cv::Mat get_nmap() const;

  // get current pose in SE3d
  Sophus::SE3d get_pose() const;

  // get reference frame
  RgbdFramePtr get_reference_frame() const;

  // set current pose
  void set_pose(const Sophus::SE3d &pose);

  // set reference frame
  void set_reference_frame(RgbdFramePtr reference);

  void set_scene_data(cv::cuda::GpuMat vmap, cv::cuda::GpuMat nmap);

  bool lock();
  void unlock();

  struct Point3d
  {
    bool visited;
    Eigen::Vector3f pos;
    Eigen::Vector3f vec_normal;
    size_t observations;
  };

  std::vector<cv::KeyPoint> cv_key_points;
  std::vector<std::shared_ptr<Point3d>> key_points;

private:
  cv::Mat source_image;
  cv::Mat source_depth;
  cv::Mat vmap;
  cv::Mat nmap;

  size_t frame_id;
  double time_stamp;
  Sophus::SE3d pose;
  RgbdFramePtr reference;
  std::mutex frame_lock;
};

} // namespace fusion

#endif