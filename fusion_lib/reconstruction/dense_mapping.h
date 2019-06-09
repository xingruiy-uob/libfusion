#ifndef DENSE_MAPPING_H
#define DENSE_MAPPING_H

#include <memory>
#include <cuda_runtime.h>
#include "rgbd_frame.h"
#include "map_struct.h"
#include "device_image.h"

namespace fusion
{

class DenseMapping
{
public:
  // DenseMapping(IntrinsicMatrix cam_param);
  // void update(RgbdImagePtr image);
  // void raycast(RgbdImagePtr image);

  // void raycast(cv::cuda::GpuMat &vmap, cv::cuda::GpuMat &colour);

  // void create_scene_mesh();
  // void create_scene_mesh(float3 *data, uint &max_size);
  // void fetch_mesh_with_normal(float3 *vertex, float3 *normal, uint &max_size);
  // void create_mesh_with_colour(float3 *vertex, uchar3 *colour, uint &max_size);

  // void restart_mapping();
  // void write_mesh_to_file(const char *file_name);
  ~DenseMapping();
  DenseMapping(IntrinsicMatrix cam_params);
  void update(cv::cuda::GpuMat depth, cv::cuda::GpuMat image, const Sophus::SE3d pose);
  void raycast(cv::cuda::GpuMat &vmap, cv::cuda::GpuMat &image, const Sophus::SE3d pose);

  void restart_mapping();
  void create_new_submap();
  size_t create_mesh_with_normal(float3 *vertex, float3 *normal);

private:
  // class DenseMappingImpl;
  // std::shared_ptr<DenseMappingImpl> impl;

  const static size_t NUM_PYRS = 10;

  IntrinsicMatrix cam_params;
  size_t active_map_index;
  std::vector<std::shared_ptr<MapStruct>> device_maps;
  std::vector<std::shared_ptr<MapStruct>> host_maps;

  // for map udate
  cv::cuda::GpuMat flag;
  cv::cuda::GpuMat pos_array;
  uint count_visible_block;
  HashEntry *visible_blocks;

  // for raycast
  cv::cuda::GpuMat zrange_x;
  cv::cuda::GpuMat zrange_y;
  uint count_rendering_block;
  RenderingBlock *rendering_blocks;
};

} // namespace fusion

#endif