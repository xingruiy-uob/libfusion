#ifndef __DENSE_ODOMETRY__
#define __DENSE_ODOMETRY__

#include "rgbd_frame.h"
#include "device_image.h"
#include "rgbd_tracker.h"
#include <memory>

namespace fusion
{

class DenseOdometry
{
public:
  DenseOdometry(IntrinsicMatrix base, int NUM_PYR);
  void track_frame(RgbdFramePtr current_frame);
  bool keyframe_needed() const;
  // TODO: not yet implemented
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

  IntrinsicMatrixPyramidPtr intrinsics_pyr_;
  std::unique_ptr<DenseTracking> tracker_;

  bool keyframe_needed_;
  bool tracking_lost_;

  TrackingResult result;
  TrackingContext context;

  bool initialized;
};

} // namespace fusion

#endif