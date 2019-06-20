#ifndef __DENSE_ODOMETRY__
#define __DENSE_ODOMETRY__

#include "rgbd_frame.h"
#include "device_image.h"
#include "rgbd_tracker.h"
#include <fusion/core/cuda_imgproc.h>
#include <memory>

namespace fusion
{

class DenseOdometry
{
public:
  DenseOdometry(IntrinsicMatrix base, int NUM_PYR);
  DenseOdometry(const DenseOdometry &) = delete;
  DenseOdometry &operator=(const DenseOdometry &) = delete;

  void track_frame(RgbdFramePtr current_frame);
  bool keyframe_needed() const;
  bool is_tracking_lost() const;
  void create_keyframe();
  void reset_tracking();

  std::vector<Sophus::SE3d> get_keyframe_poses() const;
  std::vector<Sophus::SE3d> get_camera_trajectory() const;

  Eigen::Matrix4f get_current_pose_matrix() const;
  RgbdImagePtr get_current_image() const;
  RgbdImagePtr get_reference_image() const;
  RgbdFramePtr get_current_frame() const;
  RgbdFramePtr get_current_keyframe() const;

private:
  RgbdFramePtr reference_frame;

  RgbdImagePtr current_image_;
  RgbdImagePtr reference_image_;

  std::vector<IntrinsicMatrix> cam_params;
  std::unique_ptr<DenseTracking> tracker_;

  bool keyframe_needed_;
  bool tracking_lost_;

  TrackingResult result;
  TrackingContext context;

  bool initialized;
};

class RgbdOdometry
{
public:
  RgbdOdometry(IntrinsicMatrix K, const int NUM_PYR = 5);
  RgbdOdometry(const RgbdOdometry &) = delete;
  RgbdOdometry &operator=(const RgbdOdometry &) = delete;

  Sophus::SE3d computeTransform();
  void swapIntensity();
  void setCurrFrame(std::shared_ptr<RgbdFrame> frame);
  void setSceneMap(cv::cuda::GpuMat &vmap);
  cv::cuda::GpuMat getImageSource();
  cv::cuda::GpuMat getDepthSource();
  cv::cuda::GpuMat getRenderedImage();
  cv::cuda::GpuMat getRenderedMap();

private:
  std::vector<cv::cuda::GpuMat> depthCurrPyr;
  std::vector<cv::cuda::GpuMat> vmapCurrPyr;
  std::vector<cv::cuda::GpuMat> nmapCurrPyr;
  std::vector<cv::cuda::GpuMat> IntensityCurrPyr;
  std::vector<cv::cuda::GpuMat> IntensityDxCurrPyr;
  std::vector<cv::cuda::GpuMat> IntensityDyCurrPyr;

  std::vector<cv::cuda::GpuMat> vmapRefPyr;
  std::vector<cv::cuda::GpuMat> nmapRefPyr;
  std::vector<cv::cuda::GpuMat> IntensityRefPyr;

  std::vector<IntrinsicMatrix> cam_params;

  cv::cuda::GpuMat SUM_SE3, OUT_SE3;
  cv::cuda::GpuMat imageSource;
  cv::cuda::GpuMat depthSource;
  cv::cuda::GpuMat imageFloat;
  cv::cuda::GpuMat intensityFloat;
  cv::cuda::GpuMat renderedImage, renderedMap;

  std::shared_ptr<RgbdFrame> refFrame;
  std::shared_ptr<RgbdFrame> currFrame;
  std::vector<size_t> iterations;

  Eigen::Matrix<float, 6, 6> icpHessMat, rgbHessMat;
  Eigen::Matrix<float, 6, 1> icpResMat, rgbResMat;
  Sophus::SE3d result;
};

} // namespace fusion

#endif