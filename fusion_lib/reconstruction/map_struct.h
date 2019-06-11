#ifndef __MAP_STRUCT__
#define __MAP_STRUCT__

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

// Voxel block dimensionality: 8x8x8
#define BLOCK_SIZE 8
#define BLOCK_SIZE_SUB_1 7
// Total voxels in the block
#define BLOCK_SIZE3 512
// Max allowed thread in CUDA
#define MAX_THREAD 1024

// Map info
struct MapState
{
    // The total number of buckets in the map
    // NOTE: buckets are allocated for each main entry
    // It dose not cover the excess entries
    int num_total_buckets_;

    // The total number of voxel blocks in the map
    // also determins the size of the heap memory
    // which is used for storing block addresses
    int num_total_voxel_blocks_;

    // The total number of hash entres in the map
    // This is a combination of main entries and
    // the excess entries
    int num_total_hash_entries_;

    int num_max_mesh_triangles_;
    int num_max_rendering_blocks_;

    float zmin_raycast;
    float zmax_raycast;
    float zmin_update;
    float zmax_update;
    float voxel_size;

    __device__ __host__ int num_total_voxels() const;
    __device__ __host__ int num_excess_entries() const;
    __device__ __host__ int num_total_mesh_vertices() const;
    __device__ __host__ float block_size_metric() const;
    __device__ __host__ float inverse_voxel_size() const;
    __device__ __host__ float truncation_dist() const;
    __device__ __host__ float raycast_step_scale() const;
};

struct RenderingBlock
{
    short2 upper_left;
    short2 lower_right;
    float2 zrange;
};

struct Voxel
{
    short sdf;
    unsigned char weight_;
    uchar3 rgb_;

    __device__ float get_sdf() const;
    __device__ void set_sdf(float val);
};

struct HashEntry
{
    int ptr_;
    int offset_;
    int3 pos_;

    __device__ HashEntry();
    __device__ HashEntry(int3 pos, int next, int offset);
    __device__ HashEntry(const HashEntry &other);
    __device__ HashEntry &operator=(const HashEntry &other);
    __device__ bool operator==(const int3 &pos) const;
    __device__ bool operator==(const HashEntry &other) const;
};

struct MapStruct
{
    MapStruct();
    MapStruct(const MapState &) = delete;
    MapStruct &operator=(const MapState &) = delete;

    void allocate_memory(const bool &on_device = false);
    void release_memory(const bool &on_device = false);
    void reset_map_struct();

    __device__ int compute_hash(const int3 &pos) const;
    __device__ bool lock_bucket(int *mutex);
    __device__ void unlock_bucket(int *mutex);
    __device__ bool delete_entry(HashEntry &current);
    __device__ void create_block(const int3 &blockPos, int &bucket_index);
    __device__ void delete_block(HashEntry &current);
    __device__ bool create_entry(const int3 &pos, const int &offset, HashEntry *entry);
    __device__ void find_voxel(const int3 &voxel_pos, Voxel *&out) const;
    __device__ void find_entry(const int3 &block_pos, HashEntry *&out) const;

    __device__ int local_pos_to_local_idx(const int3 &pos) const;
    __device__ int voxel_pos_to_local_idx(const int3 &pos) const;
    __device__ int3 world_pt_to_voxel_pos(float3 pos) const;
    __device__ int3 voxel_pos_to_block_pos(int3 voxel_pos) const;
    __device__ int3 block_pos_to_voxel_pos(const int3 &pos) const;
    __device__ int3 voxel_pos_to_local_pos(int3 pos) const;
    __device__ int3 local_idx_to_local_pos(const int &idx) const;
    __device__ float3 voxel_pos_to_world_pt(const int3 &voxel_pos) const;

    int *heap_mem_;
    int *excess_counter_;
    int *heap_mem_counter_;
    int *bucket_mutex_;

    Voxel *voxels_;
    HashEntry *hash_table_;
};

extern MapState state;
__device__ extern MapState param;
extern bool state_initialised;
void update_device_map_state();
std::ostream &operator<<(std::ostream &o, MapState &state);

#endif