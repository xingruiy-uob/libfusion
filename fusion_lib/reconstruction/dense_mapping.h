#ifndef __DENSE_MAPPING__
#define __DENSE_MAPPING__

#include <memory>
#include "rgbd_image.h"

namespace fusion
{

class DenseMapping
{
public:
  DenseMapping(const IntrinsicMatrixPyramidPtr &intrinsics_pyr);
  void update(RgbdImagePtr image);
  void raycast(RgbdImagePtr image);
  void create_scene_mesh();
  void write_mesh_to_file(const char* file_name);

private:
  class DenseMappingImpl;
  std::shared_ptr<DenseMappingImpl> impl;
};

} // namespace fusion

#endif