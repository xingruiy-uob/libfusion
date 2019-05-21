#include "dense_mapping.h"
#include "map_struct.h"
#include "map_proc.h"

namespace fusion
{

class DenseMapping::DenseMappingImpl
{
public:
  DenseMappingImpl(const IntrinsicMatrixPyramidPtr &intrinsics_pyr);
  ~DenseMappingImpl();
  void update(RgbdImagePtr current_image);
  void raycast(RgbdImagePtr current_image);

  IntrinsicMatrix intrinsic_matrix_;
  std::shared_ptr<MapStruct> map_struct_;

  // for raycast
  cv::cuda::GpuMat cast_vmap_;
  cv::cuda::GpuMat cast_nmap_;
  cv::cuda::GpuMat cast_image_;
  cv::cuda::GpuMat zrange_x_;
  cv::cuda::GpuMat zrange_y_;

  // for map udate
  cv::cuda::GpuMat flag;
  cv::cuda::GpuMat pos_array;
  const int integration_level_ = 0;

  // for key point raycast
  cv::cuda::GpuMat cast_pos_;
};

DenseMapping::DenseMappingImpl::DenseMappingImpl(const IntrinsicMatrixPyramidPtr &intrinsics_pyr)
{
  map_struct_ = std::make_shared<MapStruct>(300000, 450000, 100000, 0.005f);
  map_struct_->allocate_device_memory();
  map_struct_->reset_map_struct();

  intrinsic_matrix_ = intrinsics_pyr->get_intrinsic_matrix_at(integration_level_);
  zrange_x_.create(intrinsic_matrix_.height / 8, intrinsic_matrix_.width / 8, CV_32FC1);
  zrange_y_.create(intrinsic_matrix_.height / 8, intrinsic_matrix_.width / 8, CV_32FC1);
}

DenseMapping::DenseMappingImpl::~DenseMappingImpl()
{
  if (map_struct_)
    map_struct_->release_device_memory();
}

void DenseMapping::DenseMappingImpl::update(RgbdImagePtr current_image)
{
  RgbdFramePtr current_frame = current_image->get_reference_frame();
  if (current_frame == nullptr)
    return;

  cv::cuda::GpuMat depth = current_image->get_raw_depth();
  cv::cuda::GpuMat image = current_image->get_image(integration_level_);
  cv::cuda::GpuMat normal = current_image->get_nmap(integration_level_);
  Sophus::SE3d pose = current_frame->get_pose();
  uint visible_block_count = 0;
  cuda::update(*map_struct_, depth, image, normal, pose, intrinsic_matrix_, flag, pos_array, visible_block_count);
}

void DenseMapping::DenseMappingImpl::raycast(RgbdImagePtr current_image)
{
  RgbdFramePtr current_frame = current_image->get_reference_frame();
  uint visible_block_count = 0;
  map_struct_->get_visible_block_count(visible_block_count);
  if (current_frame == nullptr || visible_block_count == 0)
    return;

  Sophus::SE3d pose = current_frame->get_pose();
  cuda::create_rendering_blocks(*map_struct_, zrange_x_, zrange_y_, pose, intrinsic_matrix_);

  uint rendering_block_count = 0;
  map_struct_->get_rendering_block_count(rendering_block_count);
  if (rendering_block_count != 0)
  {
    cast_vmap_ = current_image->get_vmap(integration_level_);
    cast_nmap_ = current_image->get_nmap(integration_level_);
    cast_image_ = current_image->get_image();
    // raycast_with_colour(*map_struct_, cast_vmap_, cast_nmap_, cast_image_, zrange_x_, zrange_y_, pose, intrinsic_matrix_);
    cuda::raycast(*map_struct_, cast_vmap_, cast_nmap_, zrange_x_, zrange_y_, pose, intrinsic_matrix_);
  }
}

DenseMapping::DenseMapping(const IntrinsicMatrixPyramidPtr &intrinsics_pyr) : impl(new DenseMappingImpl(intrinsics_pyr))
{
}

void DenseMapping::update(RgbdImagePtr image)
{
  impl->update(image);
}

void DenseMapping::raycast(RgbdImagePtr image)
{
  impl->raycast(image);
}

} // namespace fusion