#ifndef FUSION_VOXEL_HASHING_MESH_SCENE_H
#define FUSION_VOXEL_HASHING_MESH_SCENE_H

#include <sophus/se3.hpp>
#include <opencv2/cudaarithm.hpp>
#include "data_struct/map_struct.h"
#include "data_struct/intrinsic_matrix.h"

namespace fusion
{
namespace cuda
{

void create_mesh_vertex_only(
    MapStorage map_struct,
    MapState state,
    uint &block_count,
    HashEntry *block_list,
    uint &triangle_count,
    void *vertex_data);

void create_mesh_with_normal(
    MapStorage map_struct,
    MapState state,
    uint &block_count,
    HashEntry *block_list,
    uint &triangle_count,
    void *vertex_data,
    void *vertex_normal);

void create_mesh_with_colour(
    MapStorage map_struct,
    MapState state,
    uint &block_count,
    HashEntry *block_list,
    uint &triangle_count,
    void *vertex_data,
    void *vertex_colour);

} // namespace cuda
} // namespace fusion

#endif