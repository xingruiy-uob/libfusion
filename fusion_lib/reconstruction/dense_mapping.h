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
  ~DenseMapping();
  DenseMapping(IntrinsicMatrix cam_params);
  void update(cv::cuda::GpuMat depth, cv::cuda::GpuMat image, const Sophus::SE3d pose);
  void raycast(cv::cuda::GpuMat &vmap, cv::cuda::GpuMat &image, const Sophus::SE3d pose);

  void restart_mapping();
  void create_new_submap();
  size_t create_mesh_with_normal(float3 *vertex, float3 *normal);

private:
  const size_t NUM_PYRS = 10;

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