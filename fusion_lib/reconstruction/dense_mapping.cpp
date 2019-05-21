#include "dense_mapping.h"
#include "map_struct.h"
#include "cuda_utils.h"
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
  void create_scene_mesh();

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

  // for mesh generation
  // cv::cuda::GpuMat block_list;
  // cv::cuda::GpuMat block_count;
  // cv::cuda::GpuMat triangle_count;
  // cv::cuda::GpuMat triangles;
  int3 *block_list;
  uint *block_count;
  uint *triangle_count;
  float3 *triangles;
};

DenseMapping::DenseMappingImpl::DenseMappingImpl(const IntrinsicMatrixPyramidPtr &intrinsics_pyr)
{
  map_struct_ = std::make_shared<MapStruct>(300000, 450000, 100000, 0.004f);
  map_struct_->allocate_device_memory();
  map_struct_->reset_map_struct();

  intrinsic_matrix_ = intrinsics_pyr->get_intrinsic_matrix_at(integration_level_);
  zrange_x_.create(intrinsic_matrix_.height / 8, intrinsic_matrix_.width / 8, CV_32FC1);
  zrange_y_.create(intrinsic_matrix_.height / 8, intrinsic_matrix_.width / 8, CV_32FC1);

  // block_list.create(1, state.num_total_hash_entries_, CV_32SC3);
  // triangles.create(1, state.num_total_mesh_vertices(), CV_32FC3);
  // block_count.create(1, 1, CV_32SC1);
  // triangle_count.create(1, 1, CV_32SC1);
  std::cout << state.num_total_hash_entries_ << std::endl;
  safe_call(cudaMalloc(&block_count, sizeof(uint)));
  safe_call(cudaMalloc(&block_list, sizeof(int3) * state.num_total_hash_entries_));
  safe_call(cudaMalloc(&triangle_count, sizeof(uint)));
  safe_call(cudaMalloc(&triangles, sizeof(float3) * state.num_total_mesh_vertices()));
}

DenseMapping::DenseMappingImpl::~DenseMappingImpl()
{
  if (map_struct_)
    map_struct_->release_device_memory();

  safe_call(cudaFree(block_count));
  safe_call(cudaFree(block_list));
  safe_call(cudaFree(triangle_count));
  safe_call(cudaFree(triangles));
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
    // cuda::raycast_with_colour(*map_struct_, cast_vmap_, cast_nmap_, cast_image_, zrange_x_, zrange_y_, pose, intrinsic_matrix_);
    cuda::raycast(*map_struct_, cast_vmap_, cast_nmap_, zrange_x_, zrange_y_, pose, intrinsic_matrix_);
  }
}

void DenseMapping::DenseMappingImpl::create_scene_mesh()
{
  cuda::create_scene_mesh(*map_struct_, block_count, block_list, triangle_count, triangles);
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

void DenseMapping::create_scene_mesh()
{
  impl->create_scene_mesh();
}

void DenseMapping::write_mesh_to_file(const char *file_name)
{
  uint host_triangle_count;
  safe_call(cudaMemcpy(&host_triangle_count, impl->triangle_count, sizeof(uint), cudaMemcpyDeviceToHost));
  if (host_triangle_count == 0)
    return;

  float3 *host_triangles = (float3 *)malloc(sizeof(float3) * host_triangle_count * 3);
  safe_call(cudaMemcpy(host_triangles, impl->triangles, sizeof(float3) * host_triangle_count * 3, cudaMemcpyDeviceToHost));

  FILE *f = fopen(file_name, "wb+");

  if (f != NULL)
  {
    for (int i = 0; i < 80; i++)
      fwrite(" ", sizeof(char), 1, f);

    fwrite(&host_triangle_count, sizeof(int), 1, f);

    float zero = 0.0f;
    short attribute = 0;
    for (uint i = 0; i < host_triangle_count; i++)
    {
      fwrite(&zero, sizeof(float), 1, f);
      fwrite(&zero, sizeof(float), 1, f);
      fwrite(&zero, sizeof(float), 1, f);

      fwrite(&host_triangles[i * 3].x, sizeof(float), 1, f);
      fwrite(&host_triangles[i * 3].y, sizeof(float), 1, f);
      fwrite(&host_triangles[i * 3].z, sizeof(float), 1, f);

      fwrite(&host_triangles[i * 3 + 1].x, sizeof(float), 1, f);
      fwrite(&host_triangles[i * 3 + 1].y, sizeof(float), 1, f);
      fwrite(&host_triangles[i * 3 + 1].z, sizeof(float), 1, f);

      fwrite(&host_triangles[i * 3 + 2].x, sizeof(float), 1, f);
      fwrite(&host_triangles[i * 3 + 2].y, sizeof(float), 1, f);
      fwrite(&host_triangles[i * 3 + 2].z, sizeof(float), 1, f);

      fwrite(&attribute, sizeof(short), 1, f);
    }

    fclose(f);
  }
}

} // namespace fusion