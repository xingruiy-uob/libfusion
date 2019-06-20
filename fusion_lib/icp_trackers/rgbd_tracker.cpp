#include "rgbd_tracker.h"
#include "rgbd_frame.h"
#include "device_image.h"
#include <xutils/DataStruct/revertable.h>
#include <fusion/icp/icp_reduction.h>

namespace fusion
{

DenseTracking::DenseTracking()
{
  SUM_SE3.create(96, 29, CV_32FC1);
  OUT_SE3.create(1, 29, CV_32FC1);
}

TrackingResult DenseTracking::compute_transform(const RgbdImagePtr reference, const RgbdImagePtr current, const TrackingContext &c)
{
  xutils::Revertable<Sophus::SE3d> estimate = xutils::Revertable<Sophus::SE3d>(Sophus::SE3d());

  if (c.use_initial_guess_)
    estimate = xutils::Revertable<Sophus::SE3d>(c.initial_estimate_);

  for (int level = c.max_iterations_.size() - 1; level >= 0; --level)
  {
    cv::cuda::GpuMat curr_vmap = current->get_vmap(level);
    cv::cuda::GpuMat last_vmap = reference->get_vmap(level);
    cv::cuda::GpuMat curr_nmap = current->get_nmap(level);
    cv::cuda::GpuMat last_nmap = reference->get_nmap(level);
    cv::cuda::GpuMat curr_intensity = current->get_intensity(level);
    cv::cuda::GpuMat last_intensity = reference->get_intensity(level);
    cv::cuda::GpuMat intensity_dx = current->get_intensity_dx(level);
    cv::cuda::GpuMat intensity_dy = current->get_intensity_dy(level);
    IntrinsicMatrix K = c.intrinsics_pyr_[level];
    float icp_error = std::numeric_limits<float>::max();
    float rgb_error = std::numeric_limits<float>::max();
    float total_error = std::numeric_limits<float>::max();
    int icp_count = 0, rgb_count = 0;
    float stddev_estimated = 0;

    for (int iter = 0; iter < c.max_iterations_[level]; ++iter)
    {
      auto last_estimate = estimate.get();
      auto last_icp_error = icp_error;
      auto last_rgb_error = rgb_error;

      icp_reduce(
          curr_vmap,
          curr_nmap,
          last_vmap,
          last_nmap,
          SUM_SE3,
          OUT_SE3,
          last_estimate,
          K,
          icp_hessian.data(),
          icp_residual.data(),
          residual_icp_.data());

      float stdev_estimated;

      rgb_step(
          curr_intensity,
          last_intensity,
          last_vmap,
          curr_vmap,
          intensity_dx,
          intensity_dy,
          SUM_SE3,
          OUT_SE3,
          stddev_estimated,
          last_estimate,
          K,
          rgb_hessian.data(),
          rgb_residual.data(),
          residual_rgb_.data());

      stddev_estimated = sqrt(residual_rgb_[0] / (residual_rgb_[1] - 6));

      auto A = 1e6 * icp_hessian + rgb_hessian;
      auto b = 1e6 * icp_residual + rgb_residual;

      update = A.cast<double>().ldlt().solve(b.cast<double>());
      estimate = Sophus::SE3d::exp(update) * last_estimate;

      icp_error = sqrt(residual_icp_(0)) / residual_icp_(1);

      if (icp_error > last_icp_error)
      {
        if (icp_count >= 2)
        {
          estimate.revert();
          break;
        }

        icp_count++;
        icp_error = last_icp_error;
      }
      else
      {
        icp_count = 0;
      }

      rgb_error = sqrt(residual_rgb_(0)) / residual_rgb_(1);

      if (rgb_error > last_rgb_error)
      {
        if (rgb_count >= 2)
        {
          estimate.revert();
          break;
        }

        rgb_count++;
        rgb_error = last_rgb_error;
      }
      else
      {
        rgb_count = 0;
      }
    }
  }

  if (estimate.get().log().transpose().norm() > 0.1)
    std::cout << estimate.get().log().transpose().norm() << std::endl;

  TrackingResult result;
  result.sucess = true;
  result.update = estimate.get().inverse();
  return result;
}

} // namespace fusion