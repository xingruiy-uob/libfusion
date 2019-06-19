#include "dense_mapping.h"
#include "map_struct.h"
#include "map_proc.h"

namespace fusion
{

DenseMapping::DenseMapping(IntrinsicMatrix cam_params) : cam_params(cam_params)
{
  device_map.create();
  zrange_x.create(cam_params.height / 8, cam_params.width / 8, CV_32FC1);
  zrange_y.create(cam_params.height / 8, cam_params.width / 8, CV_32FC1);

  cudaMalloc((void **)&visible_blocks, sizeof(HashEntry) * device_map.state.num_total_hash_entries_);
  cudaMalloc((void **)&rendering_blocks, sizeof(RenderingBlock) * 100000);

  reset_mapping();
}

DenseMapping::~DenseMapping()
{
  device_map.release();
  cudaFree((void *)visible_blocks);
  cudaFree((void *)rendering_blocks);
}

void DenseMapping::update(RgbdImagePtr frame)
{
  auto image = frame->get_image();
  auto depth = frame->get_raw_depth();
  auto normal = frame->get_nmap();
  auto pose = frame->get_reference_frame()->get_pose();

  count_visible_block = 0;

  cuda::update_weighted(
      device_map.map,
      device_map.state,
      depth,
      normal,
      image,
      pose,
      cam_params,
      flag,
      pos_array,
      visible_blocks,
      count_visible_block);
}

void DenseMapping::update(
    cv::cuda::GpuMat depth,
    cv::cuda::GpuMat image,
    const Sophus::SE3d pose)
{
  count_visible_block = 0;

  cuda::update(
      device_map.map,
      device_map.state,
      depth,
      image,
      pose,
      cam_params,
      flag,
      pos_array,
      visible_blocks,
      count_visible_block);
}

void DenseMapping::raycast(
    cv::cuda::GpuMat &vmap,
    cv::cuda::GpuMat &image,
    const Sophus::SE3d pose)
{
  if (count_visible_block == 0)
    return;

  cuda::create_rendering_blocks(
      count_visible_block,
      count_rendering_block,
      visible_blocks,
      zrange_x,
      zrange_y,
      rendering_blocks,
      pose,
      cam_params);

  if (count_rendering_block != 0)
  {

    cuda::raycast_with_colour(
        device_map.map,
        device_map.state,
        vmap,
        vmap,
        image,
        zrange_x,
        zrange_y,
        pose,
        cam_params);
  }
}

void DenseMapping::reset_mapping()
{
  device_map.reset();
}

size_t DenseMapping::fetch_mesh_vertex_only(void *vertex)
{
  uint count_triangle = 0;

  cuda::create_mesh_vertex_only(
      device_map.map,
      device_map.state,
      count_visible_block,
      visible_blocks,
      count_triangle,
      vertex);

  return (size_t)count_triangle;
}

size_t DenseMapping::fetch_mesh_with_normal(void *vertex, void *normal)
{
  uint count_triangle = 0;

  cuda::create_mesh_with_normal(
      device_map.map,
      device_map.state,
      count_visible_block,
      visible_blocks,
      count_triangle,
      vertex,
      normal);

  return (size_t)count_triangle;
}

size_t DenseMapping::fetch_mesh_with_colour(void *vertex, void *colour)
{
  uint count_triangle = 0;

  cuda::create_mesh_with_colour(
      device_map.map,
      device_map.state,
      count_visible_block,
      visible_blocks,
      count_triangle,
      vertex,
      colour);

  return (size_t)count_triangle;
}

void DenseMapping::writeMapToDisk(std::string file_name)
{
}

void DenseMapping::readMapFromDisk(std::string file_name)
{
}

} // namespace fusion