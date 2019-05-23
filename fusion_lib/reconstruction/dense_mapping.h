#ifndef __DENSE_MAPPING__
#define __DENSE_MAPPING__

#include <memory>
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
  void restart_mapping();
  void write_mesh_to_file(const char *file_name);

private:
  class DenseMappingImpl;
  std::shared_ptr<DenseMappingImpl> impl;
};

} // namespace fusion

#endif