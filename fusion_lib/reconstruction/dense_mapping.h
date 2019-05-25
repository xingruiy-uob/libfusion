#ifndef __DENSE_MAPPING__
#define __DENSE_MAPPING__

#include <memory>
#include <cuda_runtime.h>
#include "rgbd_frame.h"
#include "device_image.h"

namespace fusion
{

class DenseMapping
{
public:
  DenseMapping(IntrinsicMatrix cam_param);
  void update(RgbdImagePtr image);
  void raycast(RgbdImagePtr image);

  void raycast(cv::cuda::GpuMat &vmap, cv::cuda::GpuMat &colour);

  void create_scene_mesh();
  void create_scene_mesh(float3 *data, uint &max_size);
  void fetch_mesh_with_normal(float3 *vertex, float3 *normal, uint &max_size);

  void restart_mapping();
  void write_mesh_to_file(const char *file_name);

private:
  class DenseMappingImpl;
  std::shared_ptr<DenseMappingImpl> impl;
};

} // namespace fusion

#endif