#include "dense_odometry.h"
#include "rgbd_tracker.h"

namespace fusion
{

DenseOdometry::DenseOdometry(IntrinsicMatrix base, int NUM_PYR)
    : tracker_(new DenseTracking()), reference_frame(NULL),
      tracking_lost_(false), initialized(false)
{
  current_image_ = std::make_shared<DeviceImage>();
  reference_image_ = std::make_shared<DeviceImage>();
  intrinsics_pyr_ = std::make_shared<IntrinsicMatrixPyramid>(base, NUM_PYR);
}

void DenseOdometry::track_frame(RgbdFramePtr current_frame)
{
  current_image_->upload(current_frame, intrinsics_pyr_);

  if (!initialized)
  {
    reference_frame = current_frame;
    current_image_.swap(reference_image_);
    initialized = true;
    return;
  }

  context.use_initial_guess_ = true;
  context.initial_estimate_ = Sophus::SE3d();
  context.intrinsics_pyr_ = intrinsics_pyr_;
  context.max_iterations_ = {10, 5, 3, 3, 3};

  result = tracker_->compute_transform(reference_image_, current_image_, context);

  if (result.sucess)
  {
    auto pose = reference_frame->get_pose() * result.update;

    // current_frame->set_reference_frame(reference_frame);
    current_frame->set_pose(pose);

    reference_frame = current_frame;
    current_image_.swap(reference_image_);
  }
  else
  {
    tracking_lost_ = true;
  }
}

RgbdImagePtr DenseOdometry::get_current_image() const
{
  return current_image_;
}

RgbdImagePtr DenseOdometry::get_reference_image() const
{
  return reference_image_;
}

Eigen::Matrix4f DenseOdometry::get_current_pose_matrix() const
{
  if (current_image_ && current_image_->get_reference_frame())
  {
    return current_image_->get_reference_frame()->get_pose().matrix().cast<float>();
  }
  else
    return Eigen::Matrix4f::Identity();
}

RgbdFramePtr DenseOdometry::get_current_keyframe() const
{
  return NULL;
}

bool DenseOdometry::keyframe_needed() const
{
  return keyframe_needed_;
}

bool DenseOdometry::is_tracking_lost() const
{
  return tracking_lost_;
}

void DenseOdometry::reset_tracking()
{
  reference_frame = NULL;
  initialized = false;
  tracking_lost_ = false;
}

} // namespace fusion