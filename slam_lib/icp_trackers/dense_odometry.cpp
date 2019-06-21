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
  BuildIntrinsicPyramid(base, cam_params, NUM_PYR);
}

void DenseOdometry::track_frame(RgbdFramePtr current_frame)
{
  current_image_->upload(current_frame, cam_params);

  if (!initialized)
  {
    reference_frame = current_frame;
    current_image_.swap(reference_image_);
    initialized = true;
    return;
  }

  context.use_initial_guess_ = true;
  context.initial_estimate_ = Sophus::SE3d();
  context.intrinsics_pyr_ = cam_params;
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

// RgbdOdometry::RgbdOdometry(IntrinsicMatrix K, const int NUM_PYR)
// {
//   depthCurrPyr.resize(NUM_PYR);
//   vmapCurrPyr.resize(NUM_PYR);
//   nmapCurrPyr.resize(NUM_PYR);
//   IntensityCurrPyr.resize(NUM_PYR);
//   IntensityDxCurrPyr.resize(NUM_PYR);
//   IntensityDyCurrPyr.resize(NUM_PYR);

//   vmapRefPyr.resize(NUM_PYR);
//   nmapRefPyr.resize(NUM_PYR);
//   IntensityRefPyr.resize(NUM_PYR);

//   fusion::BuildIntrinsicPyramid(K, cam_params, NUM_PYR);

//   for (int level = 0; level < NUM_PYR; ++level)
//   {
//     auto Ki = cam_params[level];
//     depthCurrPyr[level].create(Ki.height, Ki.width, CV_32FC1);
//     vmapCurrPyr[level].create(Ki.height, Ki.width, CV_32FC4);
//     nmapCurrPyr[level].create(Ki.height, Ki.width, CV_32FC4);
//     IntensityCurrPyr[level].create(Ki.height, Ki.width, CV_32FC1);
//     IntensityDxCurrPyr[level].create(Ki.height, Ki.width, CV_32FC1);
//     IntensityDyCurrPyr[level].create(Ki.height, Ki.width, CV_32FC1);

//     vmapRefPyr[level].create(Ki.height, Ki.width, CV_32FC4);
//     nmapRefPyr[level].create(Ki.height, Ki.width, CV_32FC4);
//     IntensityRefPyr[level].create(Ki.height, Ki.width, CV_32FC1);
//   }

//   SUM_SE3.create(96, 29, CV_32FC1);
//   OUT_SE3.create(1, 29, CV_32FC1);

//   iterations = {10, 5, 3, 3, 3};
// }

// void RgbdOdometry::swapIntensity()
// {
//   for (int i = 0; i < cam_params.size(); ++i)
//   {
//     IntensityRefPyr[i].swap(IntensityCurrPyr[i]);
//   }
// }

// void RgbdOdometry::setCurrFrame(std::shared_ptr<RgbdFrame> frame)
// {
//   if (frame == NULL)
//     return;

//   const int NUM_PYR = cam_params.size();

//   imageSource.upload(frame->get_image());
//   depthSource.upload(frame->get_depth());
//   imageSource.convertTo(imageFloat, CV_32FC3);
//   cv::cuda::cvtColor(imageFloat, IntensityCurrPyr[0], cv::COLOR_RGB2GRAY);

//   filterDepthBilateral(depthSource, depthCurrPyr[0]);
//   for (int level = 1; level < NUM_PYR; ++level)
//   {
//     pyrDownDepth(depthCurrPyr[level - 1], depthCurrPyr[level]);
//     pyrDownImage(IntensityCurrPyr[level - 1], IntensityCurrPyr[level]);
//   }

//   for (int i = 0; i < NUM_PYR; ++i)
//   {
//     computeDerivative(IntensityCurrPyr[i], IntensityDxCurrPyr[i], IntensityDyCurrPyr[i]);
//     backProjectDepth(depthCurrPyr[i], vmapCurrPyr[i], cam_params[i]);
//     computeNMap(vmapCurrPyr[i], nmapCurrPyr[i]);
//   }

//   currFrame = frame;
// }

// void RgbdOdometry::setSceneMap(cv::cuda::GpuMat &vmap)
// {
//   vmapRefPyr[0] = vmap;

//   for (int level = 1; level < cam_params.size(); ++level)
//   {
//     pyrDownVMap(vmapRefPyr[level - 1], vmapRefPyr[level]);
//   }

//   for (int level = 0; level < cam_params.size(); ++level)
//   {
//     computeNMap(vmapRefPyr[level], nmapRefPyr[level]);
//   }
// }

// cv::cuda::GpuMat RgbdOdometry::getImageSource()
// {
//   return imageSource;
// }

// cv::cuda::GpuMat RgbdOdometry::getDepthSource()
// {
//   return depthSource;
// }

// cv::cuda::GpuMat RgbdOdometry::getRenderedImage()
// {
//   renderScene(vmapCurrPyr[0], nmapCurrPyr[0], renderedImage);
//   return renderedImage;
// }

// cv::cuda::GpuMat RgbdOdometry::getRenderedMap()
// {
//   renderScene(vmapRefPyr[0], nmapRefPyr[0], renderedMap);
//   return renderedMap;
// }

// Sophus::SE3d RgbdOdometry::computeTransform()
// {
//   xutils::Revertable<Sophus::SE3d> estimate;
//   Eigen::Matrix<float, 2, 1> error;
//   float icpError = std::numeric_limits<float>::max();
//   float rgbError = std::numeric_limits<float>::max();
//   float stddevEstimate = 0;
//   int icpCount = 0, rgbCount = 0;

//   for (int level = iterations.size() - 1; level >= 0; --level)
//   {
//     for (int iter = 0; iter < iterations[level]; ++iter)
//     {
//       auto lastIcpError = icpError;
//       auto lastRgbError = rgbError;
//       auto lastEstimate = estimate.value();

//       icp_reduce(
//           vmapCurrPyr[level],
//           nmapCurrPyr[level],
//           vmapRefPyr[level],
//           nmapRefPyr[level],
//           SUM_SE3,
//           OUT_SE3,
//           lastEstimate,
//           cam_params[level],
//           icpHessMat.data(),
//           icpResMat.data(),
//           error.data());

//       icpError = sqrt(error(0)) / error(1);

//       rgb_step(
//           IntensityCurrPyr[level],
//           IntensityRefPyr[level],
//           vmapRefPyr[level],
//           vmapCurrPyr[level],
//           IntensityDxCurrPyr[level],
//           IntensityDyCurrPyr[level],
//           SUM_SE3,
//           OUT_SE3,
//           stddevEstimate,
//           lastEstimate,
//           cam_params[level],
//           rgbHessMat.data(),
//           rgbResMat.data(),
//           error.data());

//       rgbError = sqrt(error(0)) / error(1);

//       auto A = 1e6 * icpHessMat + rgbHessMat;
//       auto b = 1e6 * icpResMat + rgbResMat;

//       auto update = A.cast<double>().ldlt().solve(b.cast<double>());
//       estimate = Sophus::SE3d::exp(update) * lastEstimate;

//       if (icpError > lastIcpError)
//       {
//         if (icpCount >= 2)
//         {
//           estimate.revert();
//           break;
//         }

//         icpCount++;
//         icpError = lastIcpError;
//       }
//       else
//         icpCount = 0;

//       if (rgbError > lastRgbError)
//       {
//         if (rgbCount >= 2)
//         {
//           estimate.revert();
//           break;
//         }

//         rgbCount++;
//         rgbError = lastRgbError;
//       }
//       else
//         rgbCount = 0;
//     }
//   }

//   result = estimate.value().inverse();
//   return result;
// }

} // namespace fusion