#include "dense_odometry.h"
#include "rgbd_tracker.h"

namespace fusion
{

DenseOdometry::DenseOdometry(IntrinsicMatrix base, int NUM_PYR)
    : tracker_(new DenseTracking()), current_image_(new RgbdImage()),
      reference_image_(new RgbdImage()), current_keyframe_(NULL),
      last_frame_(NULL), keyframe_needed_(false), tracking_lost_(false)
{
  intrinsics_pyr_ = std::make_shared<IntrinsicMatrixPyramid>(base, NUM_PYR);
}

void DenseOdometry::track_frame(RgbdFramePtr current_frame)
{
  current_image_->upload(current_frame, intrinsics_pyr_);

  if (current_keyframe_ != nullptr)
  {
    context_.use_initial_guess_ = true;
    context_.initial_estimate_ = Sophus::SE3d();
    context_.intrinsics_pyr_ = intrinsics_pyr_;
    context_.max_iterations_ = {10, 5, 3, 3, 3};

    result_ = tracker_->compute_transform(reference_image_, current_image_, context_);
  }
  else
  {
    last_frame_ = current_frame;
    keyframe_needed_ = true;
    current_image_.swap(reference_image_);
    return;
  }

  if (result_.sucess)
  {
    auto pose = last_frame_->get_pose() * result_.update;
    current_frame->set_reference_frame(current_keyframe_);
    current_frame->set_pose(pose);

    last_frame_ = current_frame;
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

RgbdFramePtr DenseOdometry::get_current_keyframe() const
{
  return current_keyframe_;
}

bool DenseOdometry::keyframe_needed() const
{
  return keyframe_needed_;
}

bool DenseOdometry::is_tracking_lost() const
{
  return tracking_lost_;
}

void DenseOdometry::restart_tracking()
{
  current_keyframe_ = NULL;
  keyframe_needed_ = false;
  tracking_lost_ = false;
}

void DenseOdometry::create_keyframe()
{
  keyframe_needed_ = false;
  current_keyframe_ = last_frame_;
}

} // namespace fusion