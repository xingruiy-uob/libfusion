#ifndef __DENSE_TRACKING__
#define __DENSE_TRACKING__

#include <memory>
#include <sophus/se3.hpp>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "rgbd_frame.h"
#include "device_image.h"

namespace fusion
{

using Matrix6x6f = Eigen::Matrix<float, 6, 6>;
using Matrix6x1f = Eigen::Matrix<float, 6, 1>;
using Matrix6x1d = Eigen::Matrix<double, 6, 1>;

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
  Matrix6x6f icp_hessian;
  Matrix6x6f rgb_hessian;
  Matrix6x6f joint_hessian;

  Matrix6x1f icp_residual;
  Matrix6x1f rgb_residual;
  Matrix6x1f joint_residual;
  Matrix6x1d update;

  Eigen::Matrix<float, 2, 1> residual_icp_;
  Eigen::Matrix<float, 2, 1> residual_rgb_;

  cv::cuda::GpuMat sum_se3;
  cv::cuda::GpuMat out_se3;

  bool failed;
  bool inaccurate;
  std::vector<int> max_iterations;
  IntrinsicMatrixPyramidPtr cam_pyr;

  float4 *transformed_points;
  float4 *image_corresp_data;
  float *error_term_array;
  float *variance_term_array;
};

} // namespace fusion

#endif