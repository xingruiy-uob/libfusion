#include "map_struct.h"
#include "vector_math.h"
#include "cuda_utils.h"

MapState state;
bool state_initialised = false;
__device__ MapState param;

MapStruct::MapStruct()
{
    if (!state_initialised)
    {
        state.num_total_buckets_ = 200000;
        state.num_total_hash_entries_ = 250000;
        state.num_total_voxel_blocks_ = 200000;
        state.zmax_raycast = 2.f;
        state.zmin_raycast = 0.3f;
        state.zmax_update = 2.f;
        state.zmin_update = 0.3f;
        state.voxel_size = 0.004f;
        state.num_max_rendering_blocks_ = 100000;
        state.num_max_mesh_triangles_ = 20000000;

        safe_call(cudaMemcpyToSymbol(param, &state, sizeof(MapState)));
        state_initialised = true;
    }
}

void MapStruct::allocate_memory(const bool &on_device)
{
    if (on_device)
    {
        safe_call(cudaMalloc((void **)&excess_counter_, sizeof(int)));
        safe_call(cudaMalloc((void **)&heap_mem_counter_, sizeof(int)));
        safe_call(cudaMalloc((void **)&bucket_mutex_, sizeof(int) * state.num_total_buckets_));
        safe_call(cudaMalloc((void **)&heap_mem_, sizeof(int) * state.num_total_voxel_blocks_));
        safe_call(cudaMalloc((void **)&hash_table_, sizeof(HashEntry) * state.num_total_hash_entries_));
        safe_call(cudaMalloc((void **)&voxels_, sizeof(Voxel) * state.num_total_voxels()));
    }
    else
    {
    }
}

void MapStruct::release_memory(const bool &on_device)
{
    if (on_device)
    {
        safe_call(cudaFree((void *)heap_mem_));
        safe_call(cudaFree((void *)heap_mem_counter_));
        safe_call(cudaFree((void *)hash_table_));
        safe_call(cudaFree((void *)bucket_mutex_));
        safe_call(cudaFree((void *)excess_counter_));
        safe_call(cudaFree((void *)voxels_));
    }
}

__global__ void reset_hash_entries_kernel(HashEntry *hash_table, int max_num)
{
    const int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index >= max_num)
        return;

    hash_table[index].ptr_ = -1;
    hash_table[index].offset_ = -1;
}

__global__ void reset_heap_memory_kernel(int *heap, int *heap_counter)
{
    const int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index >= param.num_total_voxel_blocks_)
        return;

    heap[index] = param.num_total_voxel_blocks_ - index - 1;

    if (index == 0)
    {
        heap_counter[0] = param.num_total_voxel_blocks_ - 1;
    }
}

void MapStruct::reset_map_struct()
{
    dim3 thread(MAX_THREAD);
    dim3 block(div_up(state.num_total_hash_entries_, thread.x));

    reset_hash_entries_kernel<<<block, thread>>>(hash_table_, state.num_total_hash_entries_);

    block = dim3(div_up(state.num_total_voxel_blocks_, thread.x));
    reset_heap_memory_kernel<<<block, thread>>>(heap_mem_, heap_mem_counter_);

    safe_call(cudaMemset(excess_counter_, 0, sizeof(int)));
    safe_call(cudaMemset(bucket_mutex_, 0, sizeof(int) * state.num_total_buckets_));
    safe_call(cudaMemset(voxels_, 0, sizeof(Voxel) * state.num_total_voxels()));
}

std::ostream &operator<<(std::ostream &o, MapState &state)
{
    return o;
}

__device__ __host__ int MapState::num_total_voxels() const
{
    return num_total_voxel_blocks_ * BLOCK_SIZE3;
}

__device__ __host__ float MapState::block_size_metric() const
{
    return BLOCK_SIZE * voxel_size;
}

__device__ __host__ int MapState::num_total_mesh_vertices() const
{
    return 3 * num_max_mesh_triangles_;
}

__device__ __host__ float MapState::inverse_voxel_size() const
{
    return 1.0f / voxel_size;
}

__device__ __host__ int MapState::num_excess_entries() const
{
    return num_total_hash_entries_ - num_total_buckets_;
}

__device__ __host__ float MapState::truncation_dist() const
{
    return 3.0f * voxel_size;
}

__device__ __host__ float MapState::raycast_step_scale() const
{
    return truncation_dist() * inverse_voxel_size();
}
__device__ HashEntry::HashEntry() : ptr_(-1), offset_(-1)
{
}

__device__ HashEntry::HashEntry(int3 pos, int ptr, int offset) : pos_(pos), ptr_(ptr), offset_(offset)
{
}

__device__ HashEntry::HashEntry(const HashEntry &other) : pos_(other.pos_), ptr_(other.ptr_), offset_(other.offset_)
{
}

__device__ HashEntry &
HashEntry::operator=(const HashEntry &other)
{
    pos_ = other.pos_;
    ptr_ = other.ptr_;
    offset_ = other.offset_;
    return *this;
}

__device__ bool HashEntry::operator==(const int3 &pos) const
{
    return this->pos_ == pos;
}

__device__ bool HashEntry::operator==(const HashEntry &other) const
{
    return other.pos_ == pos_;
}

__device__ float unpack_float(short val)
{
    return val / (float)32767;
}

__device__ short pack_float(float val)
{
    return (short)(val * 32767);
}

__device__ float Voxel::get_sdf() const
{
    return unpack_float(sdf);
}

__device__ void Voxel::set_sdf(float val)
{
    sdf = pack_float(val);
}

__device__ float Voxel::get_weight() const
{
    // return unpack_float(weight);
    return weight;
}

__device__ void Voxel::set_weight(float val)
{
    // weight = pack_float(val);
    weight = val;
}

__device__ bool MapStruct::lock_bucket(int *mutex)
{
    if (atomicExch(mutex, 1) != 1)
        return true;
    else
        return false;
}

__device__ void MapStruct::unlock_bucket(int *mutex)
{
    atomicExch(mutex, 0);
}

__device__ int MapStruct::compute_hash(const int3 &pos) const
{
    int res = ((pos.x * 73856093) ^ (pos.y * 19349669) ^ (pos.z * 83492791)) % param.num_total_buckets_;
    if (res < 0)
        res += param.num_total_buckets_;

    return res;
}

__device__ bool MapStruct::delete_entry(HashEntry &current)
{
    int old = atomicAdd(heap_mem_counter_, 1);
    if (old < param.num_total_voxel_blocks_ - 1)
    {
        heap_mem_[old + 1] = current.ptr_ / BLOCK_SIZE3;
        current.ptr_ = -1;
        return true;
    }
    else
    {
        atomicSub(heap_mem_counter_, 1);
        return false;
    }
}

__device__ bool MapStruct::create_entry(const int3 &pos, const int &offset, HashEntry *entry)
{
    int old = atomicSub(heap_mem_counter_, 1);
    if (old >= 0)
    {
        int ptr = heap_mem_[old];
        if (ptr != -1 && entry != nullptr)
        {
            *entry = HashEntry(pos, ptr * BLOCK_SIZE3, offset);
            return true;
        }
    }
    else
    {
        atomicAdd(heap_mem_counter_, 1);
    }

    return false;
}

__device__ void MapStruct::create_block(const int3 &block_pos, int &bucket_index)
{
    bucket_index = compute_hash(block_pos);
    int *mutex = &bucket_mutex_[bucket_index];
    HashEntry *current = &hash_table_[bucket_index];
    HashEntry *empty_entry = nullptr;
    if (current->pos_ == block_pos && current->ptr_ != -1)
        return;

    if (current->ptr_ == -1)
        empty_entry = current;

    while (current->offset_ > 0)
    {
        bucket_index = param.num_total_buckets_ + current->offset_ - 1;
        current = &hash_table_[bucket_index];
        if (current->pos_ == block_pos && current->ptr_ != -1)
            return;

        if (current->ptr_ == -1 && !empty_entry)
            empty_entry = current;
    }

    if (empty_entry != nullptr)
    {
        if (lock_bucket(mutex))
        {
            create_entry(block_pos, current->offset_, empty_entry);
            unlock_bucket(mutex);
        }
    }
    else
    {
        if (lock_bucket(mutex))
        {
            int offset = atomicAdd(excess_counter_, 1);
            if (offset <= param.num_excess_entries())
            {
                empty_entry = &hash_table_[param.num_total_buckets_ + offset - 1];
                if (create_entry(block_pos, 0, empty_entry))
                    current->offset_ = offset;
            }
            unlock_bucket(mutex);
        }
    }
}

__device__ void MapStruct::delete_block(HashEntry &current)
{
    memset(&voxels_[current.ptr_], 0, sizeof(Voxel) * BLOCK_SIZE3);
    int hash_id = compute_hash(current.pos_);
    int *mutex = &bucket_mutex_[hash_id];
    HashEntry *reference = &hash_table_[hash_id];
    HashEntry *link_entry = nullptr;

    // The entry to be deleted is the main entry
    if (reference->pos_ == current.pos_ && reference->ptr_ != -1)
    {
        if (lock_bucket(mutex))
        {
            delete_entry(current);
            unlock_bucket(mutex);
            return;
        }
    }
    // Search the linked list for the entry
    else
    {
        while (reference->offset_ > 0)
        {
            hash_id = param.num_total_buckets_ + reference->offset_ - 1;
            link_entry = reference;
            reference = &hash_table_[hash_id];
            if (reference->pos_ == current.pos_ && reference->ptr_ != -1)
            {
                if (lock_bucket(mutex))
                {
                    link_entry->offset_ = current.offset_;
                    delete_entry(current);
                    unlock_bucket(mutex);
                    return;
                }
            }
        }
    }
}

__device__ void MapStruct::find_voxel(const int3 &voxel_pos, Voxel *&out) const
{
    HashEntry *current;
    find_entry(voxel_pos_to_block_pos(voxel_pos), current);
    if (current != nullptr)
        out = &voxels_[current->ptr_ + voxel_pos_to_local_idx(voxel_pos)];
}

__device__ void MapStruct::find_entry(const int3 &block_pos, HashEntry *&out) const
{
    uint bucket_idx = compute_hash(block_pos);
    out = &hash_table_[bucket_idx];
    if (out->ptr_ != -1 && out->pos_ == block_pos)
        return;

    while (out->offset_ > 0)
    {
        bucket_idx = param.num_total_buckets_ + out->offset_ - 1;
        out = &hash_table_[bucket_idx];
        if (out->ptr_ != -1 && out->pos_ == block_pos)
            return;
    }

    out = nullptr;
}

__device__ int3 MapStruct::world_pt_to_voxel_pos(float3 pt) const
{
    pt = pt / param.voxel_size;
    return make_int3(pt);
}

__device__ int MapStruct::voxel_pos_to_local_idx(const int3 &pos) const
{
    return local_pos_to_local_idx(voxel_pos_to_local_pos(pos));
}

__device__ float3 MapStruct::voxel_pos_to_world_pt(const int3 &voxel_pos) const
{
    return (voxel_pos)*param.voxel_size;
}

__device__ int3 MapStruct::voxel_pos_to_block_pos(int3 voxel_pos) const
{
    if (voxel_pos.x < 0)
        voxel_pos.x -= BLOCK_SIZE_SUB_1;
    if (voxel_pos.y < 0)
        voxel_pos.y -= BLOCK_SIZE_SUB_1;
    if (voxel_pos.z < 0)
        voxel_pos.z -= BLOCK_SIZE_SUB_1;

    return voxel_pos / BLOCK_SIZE;
}

__device__ int3 MapStruct::block_pos_to_voxel_pos(const int3 &block_pos) const
{
    return block_pos * BLOCK_SIZE;
}

__device__ int3 MapStruct::voxel_pos_to_local_pos(int3 pos) const
{
    pos = pos % BLOCK_SIZE;

    if (pos.x < 0)
        pos.x += BLOCK_SIZE;
    if (pos.y < 0)
        pos.y += BLOCK_SIZE;
    if (pos.z < 0)
        pos.z += BLOCK_SIZE;

    return pos;
}

__device__ int MapStruct::local_pos_to_local_idx(const int3 &pos) const
{
    return pos.z * BLOCK_SIZE * BLOCK_SIZE + pos.y * BLOCK_SIZE + pos.x;
}

__device__ int3 MapStruct::local_idx_to_local_pos(const int &idx) const
{
    uint x = idx % BLOCK_SIZE;
    uint y = idx % (BLOCK_SIZE * BLOCK_SIZE) / BLOCK_SIZE;
    uint z = idx / (BLOCK_SIZE * BLOCK_SIZE);
    return make_int3(x, y, z);
}
