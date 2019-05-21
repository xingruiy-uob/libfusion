#ifndef __MAP_STRUCT__
#define __MAP_STRUCT__

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 8
#define BLOCK_SIZE_SUB_1 7
#define BLOCK_SIZE3 512
#define MAX_THREAD 1024

struct MapState
{
    int num_total_buckets_;
    int num_total_voxel_blocks_;
    int num_total_hash_entries_;

    int num_max_mesh_triangles_;
    int num_max_rendering_blocks_;

    float zmin_raycast_;
    float zmax_raycast_;
    float zmin_update_;
    float zmax_update_;
    float voxel_size_;

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

// struct Voxel
// {
//     short sdf_;
//     short weight_;
//     uchar3 rgb_;

//     __device__ Voxel();
//     __device__ float get_sdf() const;
//     __device__ unsigned char get_weight() const;
//     __device__ void set_sdf(float val);
//     __device__ void set_weight(unsigned char val);
// };

struct Voxel
{
    short sdf_;
    unsigned char weight_;
    uchar3 rgb_;
    unsigned char rgb_w_;

    __device__ Voxel();
    __device__ float get_sdf() const;
    __device__ unsigned char get_weight() const;
    __device__ void set_sdf(float val);
    __device__ void set_weight(unsigned char val);
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
    MapStruct() = default;
    MapStruct(const MapState &);
    MapStruct(const int &n_buckets, const int &n_entries, const int &n_blocks, const float &voxel_size);

    void allocate_device_memory();
    void release_device_memory();
    void reset_map_struct();
    void reset_visible_block_count();
    void get_visible_block_count(uint &count) const;
    void reset_rendering_block_count();
    void get_rendering_block_count(uint &count) const;

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

    uint *visible_block_count_;
    HashEntry *visible_block_pos_;

    Voxel *voxels_;
    HashEntry *hash_table_;

    uint *rendering_block_count;
    RenderingBlock *rendering_blocks;
};

extern MapState state;
__device__ extern MapState param;
extern bool state_initialised;
void update_device_map_state();
std::ostream &operator<<(std::ostream &o, MapState &state);

#endif