#include "map_struct.h"
#include "vector_math.h"
#include "cuda_utils.h"
#include <fstream>

FUSION_DEVICE MapState param;

template <bool Device>
MapStruct<Device>::MapStruct()
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

template <bool Device>
void MapStruct<Device>::reset()
{
    if (Device)
    {
#ifdef __CUDACC__

        dim3 thread(MAX_THREAD);
        dim3 block(div_up(state.num_total_hash_entries_, thread.x));

        reset_hash_entries_kernel<<<block, thread>>>(map.hash_table_, state.num_total_hash_entries_);

        block = dim3(div_up(state.num_total_voxel_blocks_, thread.x));
        reset_heap_memory_kernel<<<block, thread>>>(map.heap_mem_, map.heap_mem_counter_);

        safe_call(cudaMemset(map.excess_counter_, 0, sizeof(int)));
        safe_call(cudaMemset(map.bucket_mutex_, 0, sizeof(int) * state.num_total_buckets_));
        safe_call(cudaMemset(map.voxels_, 0, sizeof(Voxel) * state.num_total_voxels()));
#endif
    }
}

std::ostream &operator<<(std::ostream &o, MapState &state)
{
    return o;
}

FUSION_DEVICE __host__ int MapState::num_total_voxels() const
{
    return num_total_voxel_blocks_ * BLOCK_SIZE3;
}

FUSION_DEVICE __host__ float MapState::block_size_metric() const
{
    return BLOCK_SIZE * voxel_size;
}

FUSION_DEVICE __host__ int MapState::num_total_mesh_vertices() const
{
    return 3 * num_max_mesh_triangles_;
}

FUSION_DEVICE __host__ float MapState::inverse_voxel_size() const
{
    return 1.0f / voxel_size;
}

FUSION_DEVICE __host__ int MapState::num_excess_entries() const
{
    return num_total_hash_entries_ - num_total_buckets_;
}

FUSION_DEVICE __host__ float MapState::truncation_dist() const
{
    return 3.0f * voxel_size;
}

FUSION_DEVICE __host__ float MapState::raycast_step_scale() const
{
    return truncation_dist() * inverse_voxel_size();
}
FUSION_DEVICE HashEntry::HashEntry() : ptr_(-1), offset_(-1)
{
}

FUSION_DEVICE HashEntry::HashEntry(int3 pos, int ptr, int offset) : pos_(pos), ptr_(ptr), offset_(offset)
{
}

FUSION_DEVICE HashEntry::HashEntry(const HashEntry &other) : pos_(other.pos_), ptr_(other.ptr_), offset_(other.offset_)
{
}

FUSION_DEVICE HashEntry &HashEntry::operator=(const HashEntry &other)
{
    pos_ = other.pos_;
    ptr_ = other.ptr_;
    offset_ = other.offset_;
    return *this;
}

FUSION_DEVICE bool HashEntry::operator==(const int3 &pos) const
{
    return this->pos_ == pos;
}

FUSION_DEVICE bool HashEntry::operator==(const HashEntry &other) const
{
    return other.pos_ == pos_;
}

FUSION_DEVICE float unpack_float(short val)
{
    return val / (float)32767;
}

FUSION_DEVICE short pack_float(float val)
{
    return (short)(val * 32767);
}

FUSION_DEVICE float Voxel::get_sdf() const
{
    return unpack_float(sdf);
}

FUSION_DEVICE void Voxel::set_sdf(float val)
{
    sdf = pack_float(val);
}

FUSION_DEVICE float Voxel::get_weight() const
{
    // return unpack_float(weight);
    return weight;
}

FUSION_DEVICE void Voxel::set_weight(float val)
{
    // weight = pack_float(val);
    weight = val;
}

FUSION_DEVICE bool lock_bucket(int *mutex)
{
    if (atomicExch(mutex, 1) != 1)
        return true;
    else
        return false;
}

FUSION_DEVICE void unlock_bucket(int *mutex)
{
    atomicExch(mutex, 0);
}

FUSION_DEVICE int compute_hash(const int3 &pos)
{
    int res = ((pos.x * 73856093) ^ (pos.y * 19349669) ^ (pos.z * 83492791)) % param.num_total_buckets_;
    if (res < 0)
        res += param.num_total_buckets_;

    return res;
}

FUSION_DEVICE bool DeleteHashEntry(MapStorage &map, HashEntry &current)
{
    int old = atomicAdd(map.heap_mem_counter_, 1);
    if (old < param.num_total_voxel_blocks_ - 1)
    {
        map.heap_mem_[old + 1] = current.ptr_ / BLOCK_SIZE3;
        current.ptr_ = -1;
        return true;
    }
    else
    {
        atomicSub(map.heap_mem_counter_, 1);
        return false;
    }
}

FUSION_DEVICE bool CreateHashEntry(MapStorage &map, const int3 &pos, const int &offset, HashEntry *entry)
{
    int old = atomicSub(map.heap_mem_counter_, 1);
    if (old >= 0)
    {
        int ptr = map.heap_mem_[old];
        if (ptr != -1 && entry != nullptr)
        {
            *entry = HashEntry(pos, ptr * BLOCK_SIZE3, offset);
            return true;
        }
    }
    else
    {
        atomicAdd(map.heap_mem_counter_, 1);
    }

    return false;
}

FUSION_DEVICE void create_block(MapStorage &map, const int3 &block_pos, int &bucket_index)
{
    bucket_index = compute_hash(block_pos);
    int *mutex = &map.bucket_mutex_[bucket_index];
    HashEntry *current = &map.hash_table_[bucket_index];
    HashEntry *empty_entry = nullptr;
    if (current->pos_ == block_pos && current->ptr_ != -1)
        return;

    if (current->ptr_ == -1)
        empty_entry = current;

    while (current->offset_ > 0)
    {
        bucket_index = param.num_total_buckets_ + current->offset_ - 1;
        current = &map.hash_table_[bucket_index];
        if (current->pos_ == block_pos && current->ptr_ != -1)
            return;

        if (current->ptr_ == -1 && !empty_entry)
            empty_entry = current;
    }

    if (empty_entry != nullptr)
    {
        if (lock_bucket(mutex))
        {
            CreateHashEntry(map, block_pos, current->offset_, empty_entry);
            unlock_bucket(mutex);
        }
    }
    else
    {
        if (lock_bucket(mutex))
        {
            int offset = atomicAdd(map.excess_counter_, 1);
            if (offset <= param.num_excess_entries())
            {
                empty_entry = &map.hash_table_[param.num_total_buckets_ + offset - 1];
                if (CreateHashEntry(map, block_pos, 0, empty_entry))
                    current->offset_ = offset;
            }
            unlock_bucket(mutex);
        }
    }
}

FUSION_DEVICE void delete_block(MapStorage &map, HashEntry &current)
{
    memset(&map.voxels_[current.ptr_], 0, sizeof(Voxel) * BLOCK_SIZE3);
    int hash_id = compute_hash(current.pos_);
    int *mutex = &map.bucket_mutex_[hash_id];
    HashEntry *reference = &map.hash_table_[hash_id];
    HashEntry *link_entry = nullptr;

    if (reference->pos_ == current.pos_ && reference->ptr_ != -1)
    {
        if (lock_bucket(mutex))
        {
            DeleteHashEntry(map, current);
            unlock_bucket(mutex);
            return;
        }
    }
    else
    {
        while (reference->offset_ > 0)
        {
            hash_id = param.num_total_buckets_ + reference->offset_ - 1;
            link_entry = reference;
            reference = &map.hash_table_[hash_id];
            if (reference->pos_ == current.pos_ && reference->ptr_ != -1)
            {
                if (lock_bucket(mutex))
                {
                    link_entry->offset_ = current.offset_;
                    DeleteHashEntry(map, current);
                    unlock_bucket(mutex);
                    return;
                }
            }
        }
    }
}

FUSION_DEVICE void find_voxel(const MapStorage &map, const int3 &voxel_pos, Voxel *&out)
{
    HashEntry *current;
    find_entry(map, voxel_pos_to_block_pos(voxel_pos), current);
    if (current != nullptr)
        out = &map.voxels_[current->ptr_ + voxel_pos_to_local_idx(voxel_pos)];
}

FUSION_DEVICE void find_entry(const MapStorage &map, const int3 &block_pos, HashEntry *&out)
{
    uint bucket_idx = compute_hash(block_pos);
    out = &map.hash_table_[bucket_idx];
    if (out->ptr_ != -1 && out->pos_ == block_pos)
        return;

    while (out->offset_ > 0)
    {
        bucket_idx = param.num_total_buckets_ + out->offset_ - 1;
        out = &map.hash_table_[bucket_idx];
        if (out->ptr_ != -1 && out->pos_ == block_pos)
            return;
    }

    out = nullptr;
}

template <bool Device>
FUSION_HOST void MapStruct<Device>::create()
{
    if (Device)
    {
#ifdef __CUDACC__
        safe_call(cudaMalloc((void **)&map.excess_counter_, sizeof(int)));
        safe_call(cudaMalloc((void **)&map.heap_mem_counter_, sizeof(int)));
        safe_call(cudaMalloc((void **)&map.bucket_mutex_, sizeof(int) * state.num_total_buckets_));
        safe_call(cudaMalloc((void **)&map.heap_mem_, sizeof(int) * state.num_total_voxel_blocks_));
        safe_call(cudaMalloc((void **)&map.hash_table_, sizeof(HashEntry) * state.num_total_hash_entries_));
        safe_call(cudaMalloc((void **)&map.voxels_, sizeof(Voxel) * state.num_total_voxels()));
#endif
    }
    else
    {
        map.voxels_ = new Voxel[state.num_total_voxels()];
        map.hash_table_ = new HashEntry[state.num_total_hash_entries_];
        map.heap_mem_ = new int[state.num_total_voxel_blocks_];
        map.bucket_mutex_ = new int[state.num_total_buckets_];
        map.heap_mem_counter_ = new int[1];
        map.excess_counter_ = new int[1];
    }
}

template <bool Device>
FUSION_HOST void MapStruct<Device>::release()
{
    if (Device)
    {
#ifdef __CUDACC__
        safe_call(cudaFree((void *)map.heap_mem_));
        safe_call(cudaFree((void *)map.heap_mem_counter_));
        safe_call(cudaFree((void *)map.hash_table_));
        safe_call(cudaFree((void *)map.bucket_mutex_));
        safe_call(cudaFree((void *)map.excess_counter_));
        safe_call(cudaFree((void *)map.voxels_));
#endif
    }
    else
    {
        delete[] map.heap_mem_;
        delete[] map.heap_mem_counter_;
        delete[] map.hash_table_;
        delete[] map.bucket_mutex_;
        delete[] map.excess_counter_;
        delete[] map.voxels_;
    }
}

template <bool Device>
FUSION_HOST void MapStruct<Device>::copyTo(MapStruct<Device> &other) const
{
    if (Device)
    {
#ifdef __CUDACC__
        if (other.empty())
            other.create();

        safe_call(cudaMemcpy(other.map.excess_counter_, map.excess_counter_, sizeof(int), cudaMemcpyDeviceToDevice));
        safe_call(cudaMemcpy(other.map.heap_mem_counter_, map.heap_mem_counter_, sizeof(int), cudaMemcpyDeviceToDevice));
        safe_call(cudaMemcpy(other.map.bucket_mutex_, map.bucket_mutex_, sizeof(int) * state.num_total_buckets_, cudaMemcpyDeviceToDevice));
        safe_call(cudaMemcpy(other.map.heap_mem_, map.heap_mem_, sizeof(int) * state.num_total_voxel_blocks_, cudaMemcpyDeviceToDevice));
        safe_call(cudaMemcpy(other.map.hash_table_, map.hash_table_, sizeof(HashEntry) * state.num_total_hash_entries_, cudaMemcpyDeviceToDevice));
        safe_call(cudaMemcpy(other.map.voxels_, map.voxels_, sizeof(Voxel) * state.num_total_voxels(), cudaMemcpyDeviceToDevice));
#endif
    }
    else
    {
    }
}

template <bool Device>
FUSION_HOST void MapStruct<Device>::upload(MapStruct<false> &other)
{
    if (!Device)
    {
        exit(0);
    }
    else
    {
#ifdef __CUDACC__
        if (other.empty())
            return;

        safe_call(cudaMemcpy(map.excess_counter_, other.map.excess_counter_, sizeof(int), cudaMemcpyHostToDevice));
        safe_call(cudaMemcpy(map.heap_mem_counter_, other.map.heap_mem_counter_, sizeof(int), cudaMemcpyHostToDevice));
        safe_call(cudaMemcpy(map.bucket_mutex_, other.map.bucket_mutex_, sizeof(int) * state.num_total_buckets_, cudaMemcpyHostToDevice));
        safe_call(cudaMemcpy(map.heap_mem_, other.map.heap_mem_, sizeof(int) * state.num_total_voxel_blocks_, cudaMemcpyHostToDevice));
        safe_call(cudaMemcpy(map.hash_table_, other.map.hash_table_, sizeof(HashEntry) * state.num_total_hash_entries_, cudaMemcpyHostToDevice));
        safe_call(cudaMemcpy(map.voxels_, other.map.voxels_, sizeof(Voxel) * state.num_total_voxels(), cudaMemcpyHostToDevice));
#endif
    }
}

template <bool Device>
FUSION_HOST void MapStruct<Device>::download(MapStruct<false> &other) const
{
    if (!Device)
    {
        exit(0);
    }
    else
    {
#ifdef __CUDACC__
        if (other.empty())
            other.create();

        safe_call(cudaMemcpy(other.map.excess_counter_, map.excess_counter_, sizeof(int), cudaMemcpyDeviceToHost));
        safe_call(cudaMemcpy(other.map.heap_mem_counter_, map.heap_mem_counter_, sizeof(int), cudaMemcpyDeviceToHost));
        safe_call(cudaMemcpy(other.map.bucket_mutex_, map.bucket_mutex_, sizeof(int) * state.num_total_buckets_, cudaMemcpyDeviceToHost));
        safe_call(cudaMemcpy(other.map.heap_mem_, map.heap_mem_, sizeof(int) * state.num_total_voxel_blocks_, cudaMemcpyDeviceToHost));
        safe_call(cudaMemcpy(other.map.hash_table_, map.hash_table_, sizeof(HashEntry) * state.num_total_hash_entries_, cudaMemcpyDeviceToHost));
        safe_call(cudaMemcpy(other.map.voxels_, map.voxels_, sizeof(Voxel) * state.num_total_voxels(), cudaMemcpyDeviceToHost));
#endif
    }
}

template <bool Device>
FUSION_HOST bool MapStruct<Device>::empty()
{
    return map.voxels_ == NULL;
}

template <bool Device>
FUSION_HOST void MapStruct<Device>::exportModel(std::string file_name) const
{
    if (Device)
    {
        return;
    }
}

template <bool Device>
FUSION_HOST void MapStruct<Device>::writeToDisk(std::string file_name, const bool binary) const
{
    if (Device)
    {
        return;
    }

    std::ofstream file;
    if (binary)
    {
        file.open(file_name, std::ios_base::out | std::ios_base::binary);
    }
    else
    {
        file.open(file_name, std::ios_base::out);
    }

    if (file.is_open())
    {
        file.write((const char *)map.voxels_, sizeof(Voxel) * state.num_total_voxels());
        file.write((const char *)map.hash_table_, sizeof(HashEntry) * state.num_total_hash_entries_);
        file.write((const char *)map.heap_mem_, sizeof(int) * state.num_total_voxel_blocks_);
        file.write((const char *)map.bucket_mutex_, sizeof(int) * state.num_total_buckets_);
        file.write((const char *)map.heap_mem_counter_, sizeof(int));
        file.write((const char *)map.excess_counter_, sizeof(int));
        std::cout << "file wrote to disk." << std::endl;
    }
}

template <bool Device>
FUSION_HOST void MapStruct<Device>::readFromDisk(std::string file_name, const bool binary)
{
    if (Device)
    {
        return;
    }

    std::ifstream file;
    if (binary)
    {
        file.open(file_name, std::ios_base::in | std::ios_base::binary);
    }
    else
    {
        file.open(file_name, std::ios_base::in);
    }

    if (file.is_open())
    {
        file.read((char *)map.voxels_, sizeof(Voxel) * state.num_total_voxels());
        file.read((char *)map.hash_table_, sizeof(HashEntry) * state.num_total_hash_entries_);
        file.read((char *)map.heap_mem_, sizeof(int) * state.num_total_voxel_blocks_);
        file.read((char *)map.bucket_mutex_, sizeof(int) * state.num_total_buckets_);
        file.read((char *)map.heap_mem_counter_, sizeof(int));
        file.read((char *)map.excess_counter_, sizeof(int));
        std::cout << "file read from disk." << std::endl;
    }
}

FUSION_DEVICE int3 world_pt_to_voxel_pos(float3 pt)
{
    pt = pt / param.voxel_size;
    return make_int3(pt);
}

FUSION_DEVICE float3 voxel_pos_to_world_pt(const int3 &voxel_pos)
{
    return (voxel_pos)*param.voxel_size;
}

FUSION_DEVICE int3 voxel_pos_to_block_pos(int3 voxel_pos)
{
    if (voxel_pos.x < 0)
        voxel_pos.x -= BLOCK_SIZE_SUB_1;
    if (voxel_pos.y < 0)
        voxel_pos.y -= BLOCK_SIZE_SUB_1;
    if (voxel_pos.z < 0)
        voxel_pos.z -= BLOCK_SIZE_SUB_1;

    return voxel_pos / BLOCK_SIZE;
}

FUSION_DEVICE int3 block_pos_to_voxel_pos(const int3 &block_pos)
{
    return block_pos * BLOCK_SIZE;
}

FUSION_DEVICE int3 voxel_pos_to_local_pos(int3 pos)
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

FUSION_DEVICE int local_pos_to_local_idx(const int3 &pos)
{
    return pos.z * BLOCK_SIZE * BLOCK_SIZE + pos.y * BLOCK_SIZE + pos.x;
}

FUSION_DEVICE int3 local_idx_to_local_pos(const int &idx)
{
    uint x = idx % BLOCK_SIZE;
    uint y = idx % (BLOCK_SIZE * BLOCK_SIZE) / BLOCK_SIZE;
    uint z = idx / (BLOCK_SIZE * BLOCK_SIZE);
    return make_int3(x, y, z);
}

FUSION_DEVICE int voxel_pos_to_local_idx(const int3 &pos)
{
    return local_pos_to_local_idx(voxel_pos_to_local_pos(pos));
}

template class MapStruct<true>;
template class MapStruct<false>;