#ifndef __DENSE_TRACKING__
#define __DENSE_TRACKING__

#include <memory>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include "rgbd_image.h"

namespace fusion
{

struct TrackingResult
{
  bool sucess;
  Sophus::SE3d update;
};

struct TrackingContext
{
  bool use_initial_guess_;
  IntrinsicMatrixPyramidPtr intrinsics_pyr_;
  std::vector<int> max_iterations_;
  Sophus::SE3d initial_estimate_;
};

class DenseTracking
{
public:
  DenseTracking();
  TrackingResult compute_transform(const RgbdImagePtr reference, const RgbdImagePtr current, const TrackingContext &c);

private:
  class DenseTrackingImpl;
  std::shared_ptr<DenseTrackingImpl> impl;

  RgbdImagePtr current;
  RgbdImagePtr reference;
};

} // namespace fusion

#endif