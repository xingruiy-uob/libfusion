#include "tracking/rgbd_odometry.h"
#include "tracking/icp_tracker.h"

namespace fusion
{

DenseOdometry::DenseOdometry(const fusion::IntrinsicMatrix base, int NUM_PYR)
    : tracker(new DenseTracking()),
      lastTracedFrame(NULL),
      trackingLost(false),
      initialized(false)
{
  currDeviceMapPyramid = std::make_shared<DeviceImage>(base, NUM_PYR);
  refDeviceMapPyramid = std::make_shared<DeviceImage>(base, NUM_PYR);
  BuildIntrinsicPyramid(base, cam_params, NUM_PYR);
}

void DenseOdometry::trackFrame(std::shared_ptr<RgbdFrame> frame)
{
  currDeviceMapPyramid->upload(frame);

  if (!initialized)
  {
    lastTracedFrame = frame;
    currDeviceMapPyramid.swap(refDeviceMapPyramid);
    initialized = true;
    return;
  }

  context.use_initial_guess_ = true;
  context.initial_estimate_ = Sophus::SE3d();
  context.intrinsics_pyr_ = cam_params;
  context.max_iterations_ = {10, 5, 3, 3, 3};

  result = tracker->compute_transform(refDeviceMapPyramid, currDeviceMapPyramid, context);

  // if (result.sucess)
  // {
  frame->pose = lastTracedFrame->pose * result.update;
  lastTracedFrame = frame;
  currDeviceMapPyramid.swap(refDeviceMapPyramid);
  trackingLost = false;
  // }
  // else
  // {
  //   trackingLost = true;
  // }
}

std::shared_ptr<DeviceImage> DenseOdometry::get_current_image() const
{
  return currDeviceMapPyramid;
}

std::shared_ptr<DeviceImage> DenseOdometry::get_reference_image() const
{
  return refDeviceMapPyramid;
}

// Eigen::Matrix4f DenseOdometry::get_current_pose_matrix() const
// {
//   if (currDeviceMapPyramid && currDeviceMapPyramid->get_reference_frame())
//   {
//     return currDeviceMapPyramid->get_reference_frame()->pose.matrix().cast<float>();
//   }
//   else
//     return Eigen::Matrix4f::Identity();
// }

void DenseOdometry::reset()
{
  lastTracedFrame = NULL;
  initialized = false;
  trackingLost = false;
}

} // namespace fusion