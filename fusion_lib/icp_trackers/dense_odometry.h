#ifndef __DENSE_ODOMETRY__
#define __DENSE_ODOMETRY__

#include "rgbd_image.h"
#include "rgbd_tracker.h"
#include <memory>

namespace fusion
{

class DenseOdometry
{
public:
  DenseOdometry(const IntrinsicMatrixPyramidPtr intrinsics_pyr);
  void track_frame(RgbdFramePtr current_frame);
  bool keyframe_needed() const;
  bool is_tracking_lost() const;
  void create_keyframe();
  void restart_tracking();

  std::vector<Sophus::SE3d> get_keyframe_poses() const;
  std::vector<Sophus::SE3d> get_camera_trajectory() const;

  RgbdImagePtr get_current_image() const;
  RgbdImagePtr get_reference_image() const;
  RgbdFramePtr get_current_frame() const;
  RgbdFramePtr get_current_keyframe() const;

private:
  RgbdFramePtr current_keyframe_;
  RgbdFramePtr last_frame_;

  RgbdImagePtr current_image_;
  RgbdImagePtr reference_image_;

  IntrinsicMatrixPyramidPtr intrinsics_pyr_;
  std::unique_ptr<DenseTracking> tracker_;

  bool keyframe_needed_;
  bool tracking_lost_;

  TrackingResult result_;
  TrackingContext context_;
};

} // namespace fusion

#endif