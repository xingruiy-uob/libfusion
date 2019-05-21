#ifndef __RGBD_FRAME__
#define __RGBD_FRAME__

#include <memory>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include "intrinsic_matrix.h"

namespace fusion
{

class RgbdImage;
class RgbdFrame;
typedef std::shared_ptr<RgbdImage> RgbdImagePtr;
typedef std::shared_ptr<RgbdFrame> RgbdFramePtr;

class RgbdImage
{
public:
  RgbdImage();
  RgbdImage(const RgbdImage &) = delete;
  RgbdImage(const int &max_level);

  void resize_device_map();
  void upload(const RgbdFramePtr frame, const IntrinsicMatrixPyramidPtr intrinsics_pyr);

  RgbdFramePtr get_reference_frame() const;
  cv::cuda::GpuMat get_rendered_image() const;
  cv::cuda::GpuMat get_rendered_scene_textured() const;
  cv::cuda::GpuMat get_depth(const int &level = 0) const;
  cv::cuda::GpuMat get_raw_depth() const;
  cv::cuda::GpuMat get_image(const int &level = 0) const;
  cv::cuda::GpuMat get_vmap(const int &level = 0) const;
  cv::cuda::GpuMat get_nmap(const int &level = 0) const;
  cv::cuda::GpuMat get_intensity(const int &level = 0) const;
  cv::cuda::GpuMat get_intensity_dx(const int &level = 0) const;
  cv::cuda::GpuMat get_intensity_dy(const int &level = 0) const;

private:
  class RgbdImageImpl;
  std::shared_ptr<RgbdImageImpl> impl;
};

class RgbdFrame
{
public:
  RgbdFrame() = delete;
  ~RgbdFrame();
  RgbdFrame(const RgbdFrame &other) = delete;
  RgbdFrame(const cv::Mat &image, const cv::Mat &depth_float, size_t id, double time_stamp);

  size_t get_id() const;
  cv::Mat get_image() const;
  cv::Mat get_depth() const;
  Sophus::SE3d get_pose() const;
  void set_pose(const Sophus::SE3d &pose);
  RgbdFramePtr get_reference_frame() const;
  void set_reference_frame(RgbdFramePtr reference);

private:
  class RgbdFrameImpl;
  std::shared_ptr<RgbdFrameImpl> impl;
};

} // namespace fusion

#endif