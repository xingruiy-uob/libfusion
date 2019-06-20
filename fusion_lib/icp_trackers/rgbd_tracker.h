#ifndef FUSION_ICP_TRACKER_H
#define FUSION_ICP_TRACKER_H

#include <memory>
#include <sophus/se3.hpp>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "rgbd_frame.h"
#include "device_image.h"

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
  std::vector<IntrinsicMatrix> intrinsics_pyr_;
  std::vector<int> max_iterations_;
  Sophus::SE3d initial_estimate_;
};

class DenseTracking
{
public:
  DenseTracking();
  DenseTracking(IntrinsicMatrix K, const int NUM_PYR);
  TrackingResult compute_transform(const RgbdImagePtr reference, const RgbdImagePtr current, const TrackingContext &c);

private:
  std::vector<cv::cuda::GpuMat> vMapSrcPyr;
  std::vector<cv::cuda::GpuMat> nMapSrcPyr;
  std::vector<cv::cuda::GpuMat> depthSrcPyr;
  std::vector<cv::cuda::GpuMat> imageSrcPyr;
  std::vector<cv::cuda::GpuMat> imageDxSrcPyr;
  std::vector<cv::cuda::GpuMat> imageDySrcPyr;

  std::vector<cv::cuda::GpuMat> vMapRefPyr;
  std::vector<cv::cuda::GpuMat> nMapRefPyr;
  std::vector<cv::cuda::GpuMat> imageRefPyr;

  std::vector<IntrinsicMatrix> cam_params;

  cv::cuda::GpuMat imageSrcFloat;

  Eigen::Matrix<float, 6, 6> icp_hessian;
  Eigen::Matrix<float, 6, 6> rgb_hessian;
  Eigen::Matrix<float, 6, 6> joint_hessian;

  Eigen::Matrix<float, 6, 1> icp_residual;
  Eigen::Matrix<float, 6, 1> rgb_residual;
  Eigen::Matrix<float, 6, 1> joint_residual;
  Eigen::Matrix<double, 6, 1> update;

  Eigen::Matrix<float, 2, 1> residual_icp_;
  Eigen::Matrix<float, 2, 1> residual_rgb_;

  cv::cuda::GpuMat SUM_SE3;
  cv::cuda::GpuMat OUT_SE3;

  std::vector<int> max_iterations;
};

} // namespace fusion

#endif