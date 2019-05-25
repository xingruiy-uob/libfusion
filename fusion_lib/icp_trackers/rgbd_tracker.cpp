#include "rgbd_tracker.h"
#include "rgbd_frame.h"
#include "device_image.h"
#include "cuda_utils.h"
#include "revertable_var.h"
#include "pose_estimator.h"

namespace fusion
{

DenseTracking::DenseTracking()
{
  sum_se3.create(96, 29, CV_32FC1);
  out_se3.create(1, 29, CV_32FC1);
  safe_call(cudaMalloc(&transformed_points, sizeof(float4) * 640 * 480));
  safe_call(cudaMalloc(&image_corresp_data, sizeof(float4) * 640 * 480));
  safe_call(cudaMalloc(&error_term_array, sizeof(float) * 640 * 480));
  safe_call(cudaMalloc(&variance_term_array, sizeof(float) * 640 * 480));
}

TrackingResult DenseTracking::compute_transform(const RgbdImagePtr reference, const RgbdImagePtr current, const TrackingContext &c)
{
  Revertable<Sophus::SE3d> estimate = Revertable<Sophus::SE3d>(Sophus::SE3d());

  if (c.use_initial_guess_)
    estimate = Revertable<Sophus::SE3d>(c.initial_estimate_);

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
    IntrinsicMatrix K = c.intrinsics_pyr_->get_intrinsic_matrix_at(level);
    float icp_error = std::numeric_limits<float>::max();
    float rgb_error = std::numeric_limits<float>::max();
    float total_error = std::numeric_limits<float>::max();
    int count = 0;
    float stddev_estimated = 0;

    for (int iter = 0; iter < c.max_iterations_[level]; ++iter)
    {
      auto last_estimate = estimate.value();
      auto last_icp_error = icp_error;
      auto last_rgb_error = rgb_error;

      icp_reduce(
          curr_vmap,
          curr_nmap,
          last_vmap,
          last_nmap,
          sum_se3,
          out_se3,
          last_estimate,
          K,
          icp_hessian.data(),
          icp_residual.data(),
          residual_icp_.data());
      joint_hessian = icp_hessian;
      joint_residual = icp_residual;

      // rgb_reduce(
      //     curr_intensity,
      //     last_intensity,
      //     last_vmap,
      //     curr_vmap,
      //     intensity_dx,
      //     intensity_dy,
      //     sum_se3,
      //     out_se3,
      //     last_estimate,
      //     K,
      //     rgb_hessian.data(),
      //     rgb_residual.data(),
      //     residual_rgb_.data());
      // joint_hessian = rgb_hessian;
      // joint_residual = rgb_residual;

      // uint num_corresp;
      // float mean_estimated;
      // float stdev_estimated;

      // compute_rgb_corresp(
      //     last_vmap,
      //     last_intensity,
      //     curr_intensity,
      //     intensity_dx,
      //     intensity_dy,
      //     last_estimate,
      //     K,
      //     transformed_points,
      //     image_corresp_data,
      //     error_term_array,
      //     variance_term_array,
      //     mean_estimated,
      //     stdev_estimated,
      //     num_corresp);

      // rgb_step(
      //     curr_intensity,
      //     last_intensity,
      //     last_vmap,
      //     curr_vmap,
      //     intensity_dx,
      //     intensity_dy,
      //     sum_se3,
      //     out_se3,
      //     stddev_estimated,
      //     last_estimate,
      //     K,
      //     rgb_hessian.data(),
      //     rgb_residual.data(),
      //     residual_rgb_.data());

      // stddev_estimated = sqrt(residual_rgb_[0] / (residual_rgb_[1] - 6));
      // compute_least_square_RGB(
      //     num_corresp,
      //     transformed_points,
      //     image_corresp_data,
      //     mean_estimated,
      //     stdev_estimated,
      //     K,
      //     sum_se3,
      //     out_se3,
      //     rgb_hessian.data(),
      //     rgb_residual.data(),
      //     residual_rgb_.data());
      // joint_hessian = rgb_hessian;
      // joint_residual = rgb_residual;

      // compute_rgb_correspondence(curr_intensity, last_intensity, intensity_dx, intensity_dy, last_vmap, last_estimate, K);

      // joint_hessian = 1e6 * icp_hessian + rgb_hessian;
      // joint_residual = 1e6 * icp_residual + rgb_residual;

      update = joint_hessian.cast<double>().ldlt().solve(joint_residual.cast<double>());

      icp_error = sqrt(residual_icp_(0)) / residual_icp_(1);

      if (icp_error > last_icp_error)
      {
        if (count >= 2)
        {
          estimate.revert();
          break;
        }

        count++;
        icp_error = last_icp_error;
        // std::cout << "errro increases at " << level << "/" << iter << std::endl;
      }
      else
      {
        count = 0;
      }

      rgb_error = sqrt(residual_rgb_(0)) / residual_rgb_(1);

      if (rgb_error > last_rgb_error)
      {
        if (count >= 2)
        {
          estimate.revert();
          break;
        }

        count++;
        rgb_error = last_rgb_error;
        // std::cout << "rgb errors increases at " << level << "/" << iter << " with error: " << rgb_error << " and num_matches: " << residual_rgb_(1) << " and update: " << update_.transpose() << std::endl;
      }
      else
      {
        count = 0;
      }

      estimate = Sophus::SE3d::exp(update) * last_estimate;
    }
  }

  // Eigen::FullPivLU<Eigen::MatrixXf> lu(JtJ_);
  // Eigen::MatrixXf null_space = lu.kernel();
  // std::cout << null_space << std::endl;
  if (estimate.value().log().transpose().norm() > 0.1)
    std::cout << estimate.value().log().transpose().norm() << std::endl;

  TrackingResult result;
  result.sucess = true;
  result.update = estimate.value().inverse();
  return result;
}

} // namespace fusion