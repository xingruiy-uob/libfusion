#ifndef __MAP_STRUCT__
#define __MAP_STRUCT__

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#ifdef __CUDACC__
#define FUSION_HOST __host__
#define FUSION_DEVICE __device__
#define FUSION_HOST_AND_DEVICE __host__ __device__
#else
#define FUSION_HOST
#define FUSION_DEVICE
#define FUSION_HOST_AND_DEVICE
#endif

#define BLOCK_SIZE 8
#define BLOCK_SIZE3 512
#define BLOCK_SIZE_SUB_1 7
#define WARP_SIZE 32
#define MAX_THREAD 1024

#define VOXEL_SIZE 0.04f
#define MAX_DEPTH_FUSION 3.0f
#define MIN_DEPTH_FUSION 0.3f
#define MAX_DEPTH_RAYCAST 3.0f
#define MIN_DEPTH_RAYCAST 0.3f
#define NUM_VOXEL_BLOCKS 65535
#define NUM_HASH_ENTRIES 100000
#define NUM_EXCESS_ENTREIS 20000

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

    FUSION_HOST_AND_DEVICE int num_total_voxels() const;
    FUSION_HOST_AND_DEVICE int num_excess_entries() const;
    FUSION_HOST_AND_DEVICE int num_total_mesh_vertices() const;
    FUSION_HOST_AND_DEVICE float block_size_metric() const;
    FUSION_HOST_AND_DEVICE float inverse_voxel_size() const;
    FUSION_HOST_AND_DEVICE float truncation_dist() const;
    FUSION_HOST_AND_DEVICE float raycast_step_scale() const;
};

extern MapState state;
FUSION_DEVICE extern MapState param;
extern bool state_initialised;
void update_device_map_state();
std::ostream &operator<<(std::ostream &o, MapState &state);

struct RenderingBlock
{
    short2 upper_left;
    short2 lower_right;
    float2 zrange;
};

struct Voxel
{
    FUSION_DEVICE float get_sdf() const;
    FUSION_DEVICE void set_sdf(float val);
    FUSION_DEVICE float get_weight() const;
    FUSION_DEVICE void set_weight(float val);

    short sdf;
    float weight;
    uchar3 rgb;
};

struct HashEntry
{
    FUSION_HOST_AND_DEVICE HashEntry();
    FUSION_HOST_AND_DEVICE HashEntry(int3 pos, int next, int offset);
    FUSION_HOST_AND_DEVICE HashEntry(const HashEntry &other);
    FUSION_HOST_AND_DEVICE HashEntry &operator=(const HashEntry &other);
    FUSION_HOST_AND_DEVICE bool operator==(const int3 &pos) const;
    FUSION_HOST_AND_DEVICE bool operator==(const HashEntry &other) const;

    int ptr_;
    int offset_;
    int3 pos_;
};

template <bool Device>
struct MapStruct
{
    FUSION_HOST MapStruct();
    FUSION_HOST void create();
    FUSION_HOST void release();
    FUSION_HOST bool empty();
    FUSION_HOST void copyTo(MapStruct<Device> &) const;
    FUSION_HOST void upload(MapStruct<false> &);
    FUSION_HOST void download(MapStruct<false> &) const;
    FUSION_HOST void writeToDisk(std::string, bool) const;
    FUSION_HOST void readFromDisk(std::string, bool);
    FUSION_HOST void reset();

    FUSION_DEVICE bool delete_entry(HashEntry &current);
    FUSION_DEVICE void create_block(const int3 &blockPos, int &bucket_index);
    FUSION_DEVICE void delete_block(HashEntry &current);
    FUSION_DEVICE bool create_entry(const int3 &pos, const int &offset, HashEntry *entry);
    FUSION_DEVICE void find_voxel(const int3 &voxel_pos, Voxel *&out) const;
    FUSION_DEVICE void find_entry(const int3 &block_pos, HashEntry *&out) const;

    int *heap_mem_;
    int *excess_counter_;
    int *heap_mem_counter_;
    int *bucket_mutex_;

    Voxel *voxels_;
    HashEntry *hash_table_;
};

FUSION_DEVICE int compute_hash(const int3 &pos);
FUSION_DEVICE bool lock_bucket(int *mutex);
FUSION_DEVICE void unlock_bucket(int *mutex);
FUSION_DEVICE int3 world_pt_to_voxel_pos(float3 pt);
FUSION_DEVICE float3 voxel_pos_to_world_pt(const int3 &voxel_pos);
FUSION_DEVICE int3 voxel_pos_to_block_pos(int3 voxel_pos);
FUSION_DEVICE int3 block_pos_to_voxel_pos(const int3 &block_pos);
FUSION_DEVICE int3 voxel_pos_to_local_pos(int3 pos);
FUSION_DEVICE int local_pos_to_local_idx(const int3 &pos);
FUSION_DEVICE int3 local_idx_to_local_pos(const int &idx);
FUSION_DEVICE int voxel_pos_to_local_idx(const int3 &pos);

#endif