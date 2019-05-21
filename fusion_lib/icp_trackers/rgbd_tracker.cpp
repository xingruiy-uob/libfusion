#include "rgbd_tracker.h"
#include "rgbd_image.h"
#include "revertable.h"
#include "pose_estimation.h"

namespace fusion
{

class DenseTracking::DenseTrackingImpl
{
public:
  DenseTrackingImpl();
  TrackingResult compute_transform(const RgbdImagePtr reference, const RgbdImagePtr current, const TrackingContext &c);

  Eigen::Matrix<float, 6, 6> jtj_icp_, jtj_rgb_, JtJ_;
  Eigen::Matrix<float, 6, 1> jtr_icp_, jtr_rgb_, Jtr_;
  Eigen::Matrix<double, 6, 1> update_;
  Eigen::Matrix<float, 2, 1> residual_icp_, residual_rgb_;

  cv::cuda::GpuMat sum_se3_, out_se3_;
};

DenseTracking::DenseTrackingImpl::DenseTrackingImpl()
{
  sum_se3_.create(96, 29, CV_32FC1);
  out_se3_.create(1, 29, CV_32FC1);
}

TrackingResult DenseTracking::DenseTrackingImpl::compute_transform(const RgbdImagePtr reference, const RgbdImagePtr current, const TrackingContext &c)
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

    for (int iter = 0; iter < c.max_iterations_[level]; ++iter)
    {
      auto last_estimate = estimate.value();
      auto last_icp_error = icp_error;
      auto last_rgb_error = rgb_error;
      // icp_reduce(curr_vmap, curr_nmap, last_vmap, last_nmap, sum_se3_, out_se3_, last_estimate, K, jtj_icp_.data(), jtr_icp_.data(), residual_icp_.data());
      rgb_reduce(curr_intensity, last_intensity, last_vmap, curr_vmap, intensity_dx, intensity_dy, sum_se3_, out_se3_, last_estimate, K, jtj_rgb_.data(), jtr_rgb_.data(), residual_rgb_.data());
      // JtJ_ = 1e6 * jtj_icp_ + jtj_rgb_;
      // Jtr_ = 1e6 * jtr_icp_ + jtr_rgb_;

      // compute_rgb_correspondence(curr_intensity, last_intensity, intensity_dx, intensity_dy, last_vmap, last_estimate, K);

      // std::cout << jtj_icp_ << std::endl;
      // std::cout << jtj_rgb_ << std::endl;
      JtJ_ = jtj_rgb_;
      Jtr_ = jtr_rgb_;
      // JtJ_ = jtj_icp_;
      // Jtr_ = jtr_icp_;
      update_ = JtJ_.cast<double>().ldlt().solve(Jtr_.cast<double>());

      // icp_error = sqrt(residual_icp_(0)) / residual_icp_(1);

      // if (icp_error > last_icp_error)
      // {
      //   if (count >= 2)
      //   {
      //     estimate.revert();
      //     break;
      //   }

      //   count++;
      //   icp_error = last_icp_error;
      //   // std::cout << "errro increases at " << level << "/" << iter << std::endl;
      // }
      // else
      // {
      //   count = 0;
      // }

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

      estimate.update(Sophus::SE3d::exp(update_) * last_estimate);
    }
  }

  // Eigen::FullPivLU<Eigen::MatrixXf> lu(JtJ_);
  // Eigen::MatrixXf null_space = lu.kernel();
  // std::cout << null_space << std::endl;
  if (estimate.value().log().transpose().norm() > 0.3)
    std::cout << estimate.value().log().transpose().norm() << std::endl;

  TrackingResult result;
  result.sucess = true;
  result.update = estimate.value().inverse();
  return result;
}

DenseTracking::DenseTracking() : impl(new DenseTrackingImpl())
{
}

TrackingResult DenseTracking::compute_transform(const RgbdImagePtr reference, const RgbdImagePtr current, const TrackingContext &c)
{
  return impl->compute_transform(reference, current, c);
}

} // namespace fusion