#ifndef SLAM_LIB_DENSE_ODOMETRY_H
#define SLAM_LIB_DENSE_ODOMETRY_H

#include "Frame.h"
#include "PointCloud.h"
#include "ICPTracker.h"
#include <xfusion/core/cuda_imgproc.h>
#include <memory>

namespace fusion
{

class DenseOdometry
{
public:
  DenseOdometry(const fusion::IntrinsicMatrix base, int NUM_PYR);
  DenseOdometry(const DenseOdometry &) = delete;
  DenseOdometry &operator=(const DenseOdometry &) = delete;

  bool trackingLost;
  void trackFrame(std::shared_ptr<RgbdFrame> frame);
  void reset();

  std::vector<Sophus::SE3d> get_keyframe_poses() const;
  std::vector<Sophus::SE3d> get_camera_trajectory() const;

  // Eigen::Matrix4f get_current_pose_matrix() const;
  std::shared_ptr<DeviceImage> get_current_image() const;
  std::shared_ptr<DeviceImage> get_reference_image() const;

private:
  std::shared_ptr<RgbdFrame> lastTracedFrame;
  std::shared_ptr<DeviceImage> currDeviceMapPyramid;
  std::shared_ptr<DeviceImage> refDeviceMapPyramid;
  std::vector<fusion::IntrinsicMatrix> cam_params;
  std::unique_ptr<DenseTracking> tracker;
  bool initialized;

  TrackingResult result;
  TrackingContext context;
};

} // namespace fusion

#endif