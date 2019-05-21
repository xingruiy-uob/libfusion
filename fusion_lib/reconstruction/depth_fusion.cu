#include "map_proc.h"
#include "vector_math.h"
#include "cuda_utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>


namespace fusion
{
    namespace cuda
{


__device__ bool is_vertex_visible(float3 pt, DeviceMatrix3x4 inv_pose,
                                  int cols, int rows, float fx,
                                  float fy, float cx, float cy)
{
    pt = inv_pose(pt);
    float2 pt2d = make_float2(fx * pt.x / pt.z + cx, fy * pt.y / pt.z + cy);
    return !(pt2d.x < 0 || pt2d.y < 0 || pt2d.x > cols - 1 || pt2d.y > rows - 1 || pt.z < param.zmin_update_ || pt.z > param.zmax_update_);
}

__device__ bool is_block_visible(const int3 &block_pos,
                                 DeviceMatrix3x4 inv_pose,
                                 int cols, int rows, float fx,
                                 float fy, float cx, float cy)
{
    float scale = param.block_size_metric();
#pragma unroll
    for (int corner = 0; corner < 8; ++corner)
    {
        int3 tmp = block_pos;
        tmp.x += (corner & 1) ? 1 : 0;
        tmp.y += (corner & 2) ? 1 : 0;
        tmp.z += (corner & 4) ? 1 : 0;

        if (is_vertex_visible(tmp * scale, inv_pose, cols, rows, fx, fy, cx, cy))
            return true;
    }

    return false;
}

__global__ void check_visibility_flag_kernel(MapStruct map_struct, uchar *flag, DeviceMatrix3x4 inv_pose,
                                             int cols, int rows, float fx, float fy, float cx, float cy)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= param.num_total_hash_entries_)
        return;

    HashEntry &current = map_struct.hash_table_[idx];
    if (current.ptr_ != -1)
    {
        switch (flag[idx])
        {
        default:
        {
            if (is_block_visible(current.pos_, inv_pose, cols, rows, fx, fy, cx, cy))
            {
                flag[idx] = 1;
            }
            else
            {
                // map_struct.delete_block(current);
                flag[idx] = 0;
            }

            return;
        }
        case 2:
        {
            flag[idx] = 1;
            return;
        }
        }
    }
}

__global__ void copy_visible_block_kernel(HashEntry *hash_table, HashEntry *visible_block, const uchar *flag, const int *pos)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= param.num_total_hash_entries_)
        return;

    if (flag[idx] == 1)
        visible_block[pos[idx]] = hash_table[idx];
}

__device__ float2 project(float3 pt, float fx, float fy, float cx, float cy)
{
    return make_float2(fx * pt.x / pt.z + cx, fy * pt.y / pt.z + cy);
}

__device__ float3 unproject(int x, int y, float z, float invfx, float invfy, float cx, float cy)
{
    return make_float3(invfx * (x - cx) * z, invfy * (y - cy) * z, z);
}

__device__ float3 unproject_world(int x, int y, float z, float invfx,
                                  float invfy, float cx, float cy, DeviceMatrix3x4 pose)
{
    return pose(unproject(x, y, z, invfx, invfy, cx, cy));
}

__device__ __inline__ int create_block(MapStruct &map_struct, const int3 block_pos)
{
    int hash_index;
    map_struct.create_block(block_pos, hash_index);
    return hash_index;
}

__global__ void create_blocks_kernel(MapStruct map_struct, cv::cuda::PtrStepSz<float> depth,
                                     float invfx, float invfy, float cx, float cy,
                                     DeviceMatrix3x4 pose, uchar *flag)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= depth.cols || y >= depth.rows)
        return;

    float z = depth.ptr(y)[x];
    if (isnan(z) || z < param.zmin_update_ || z > param.zmax_update_)
        return;

    float z_thresh = param.truncation_dist() * 0.5;
    float z_near = min(param.zmax_update_, z - z_thresh);
    float z_far = min(param.zmax_update_, z + z_thresh);
    if (z_near >= z_far)
        return;

    int3 block_near = map_struct.voxel_pos_to_block_pos(map_struct.world_pt_to_voxel_pos(unproject_world(x, y, z_near, invfx, invfy, cx, cy, pose)));
    int3 block_far = map_struct.voxel_pos_to_block_pos(map_struct.world_pt_to_voxel_pos(unproject_world(x, y, z_far, invfx, invfy, cx, cy, pose)));

    int3 d = block_far - block_near;
    int3 increment = make_int3(d.x < 0 ? -1 : 1, d.y < 0 ? -1 : 1, d.z < 0 ? -1 : 1);
    int3 incre_abs = make_int3(abs(d.x), abs(d.y), abs(d.z));
    int3 incre_err = make_int3(incre_abs.x << 1, incre_abs.y << 1, incre_abs.z << 1);

    int err_1;
    int err_2;

    // Bresenham's line algorithm
    // details see : https://en.m.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    if ((incre_abs.x >= incre_abs.y) && (incre_abs.x >= incre_abs.z))
    {
        err_1 = incre_err.y - 1;
        err_2 = incre_err.z - 1;
        flag[create_block(map_struct, block_near)] = 2;
        for (int i = 0; i < incre_abs.x; ++i)
        {
            if (err_1 > 0)
            {
                block_near.y += increment.y;
                err_1 -= incre_err.x;
            }

            if (err_2 > 0)
            {
                block_near.z += increment.z;
                err_2 -= incre_err.x;
            }

            err_1 += incre_err.y;
            err_2 += incre_err.z;
            block_near.x += increment.x;
            flag[create_block(map_struct, block_near)] = 2;
        }
    }
    else if ((incre_abs.y >= incre_abs.x) && (incre_abs.y >= incre_abs.z))
    {
        err_1 = incre_err.x - 1;
        err_2 = incre_err.z - 1;
        flag[create_block(map_struct, block_near)] = 2;
        for (int i = 0; i < incre_abs.y; ++i)
        {
            if (err_1 > 0)
            {
                block_near.x += increment.x;
                err_1 -= incre_err.y;
            }

            if (err_2 > 0)
            {
                block_near.z += increment.z;
                err_2 -= incre_err.y;
            }

            err_1 += incre_err.x;
            err_2 += incre_err.z;
            block_near.y += increment.y;
            flag[create_block(map_struct, block_near)] = 2;
        }
    }
    else
    {
        err_1 = incre_err.y - 1;
        err_2 = incre_err.x - 1;
        flag[create_block(map_struct, block_near)] = 2;
        for (int i = 0; i < incre_abs.z; ++i)
        {
            if (err_1 > 0)
            {
                block_near.y += increment.y;
                err_1 -= incre_err.z;
            }

            if (err_2 > 0)
            {
                block_near.x += increment.x;
                err_2 -= incre_err.z;
            }

            err_1 += incre_err.y;
            err_2 += incre_err.x;
            block_near.z += increment.z;
            flag[create_block(map_struct, block_near)] = 2;
        }
    }
}

__global__ void update_map_kernel(MapStruct map_struct,
                                  cv::cuda::PtrStepSz<float> depth,
                                  DeviceMatrix3x4 inv_pose,
                                  float fx, float fy,
                                  float cx, float cy)
{
    if (blockIdx.x >= param.num_total_hash_entries_ || blockIdx.x >= *map_struct.visible_block_count_)
        return;

    HashEntry &current = map_struct.visible_block_pos_[blockIdx.x];

    int3 voxel_pos = map_struct.block_pos_to_voxel_pos(current.pos_);
    float dist_thresh = param.truncation_dist();
    float inv_dist_thresh = 1.0 / dist_thresh;

#pragma unroll
    for (int block_idx_z = 0; block_idx_z < 8; ++block_idx_z)
    {
        int3 local_pos = make_int3(threadIdx.x, threadIdx.y, block_idx_z);
        float3 pt = inv_pose(map_struct.voxel_pos_to_world_pt(voxel_pos + local_pos));

        int u = __float2int_rd(fx * pt.x / pt.z + cx + 0.5);
        int v = __float2int_rd(fy * pt.y / pt.z + cy + 0.5);
        if (u < 0 || v < 0 || u > depth.cols - 1 || v > depth.rows - 1)
            continue;

        float dist = depth.ptr(v)[u];
        if (isnan(dist) || dist < 1e-2 || dist > param.zmax_update_ || dist < param.zmin_update_)
            continue;

        float sdf = dist - pt.z;
        if (sdf < -dist_thresh)
            continue;

        sdf = min(1.0f, sdf * inv_dist_thresh);
        const int local_idx = map_struct.local_pos_to_local_idx(local_pos);
        Voxel &voxel = map_struct.voxels_[current.ptr_ + local_idx];

        auto sdf_p = voxel.get_sdf();
        int weight_p = voxel.get_weight();

        if (weight_p == 0)
        {
            voxel.set_sdf(sdf);
            voxel.set_weight(1);
            continue;
        }

        unsigned char w_curr = min(255, weight_p + 1);
        sdf_p = (sdf_p * weight_p + sdf) / (weight_p + 1);
        voxel.set_sdf(sdf_p);
        voxel.set_weight(w_curr);
    }
}

__global__ void update_map_with_colour_kernel(MapStruct map_struct,
                                              cv::cuda::PtrStepSz<float> depth,
                                              cv::cuda::PtrStepSz<uchar3> image,
                                              DeviceMatrix3x4 inv_pose,
                                              float fx, float fy,
                                              float cx, float cy)
{
    if (blockIdx.x >= param.num_total_hash_entries_ || blockIdx.x >= *map_struct.visible_block_count_)
        return;

    HashEntry &current = map_struct.visible_block_pos_[blockIdx.x];

    int3 voxel_pos = map_struct.block_pos_to_voxel_pos(current.pos_);
    float dist_thresh = param.truncation_dist();
    float inv_dist_thresh = 1.0 / dist_thresh;

#pragma unroll
    for (int block_idx_z = 0; block_idx_z < 8; ++block_idx_z)
    {
        int3 local_pos = make_int3(threadIdx.x, threadIdx.y, block_idx_z);
        float3 pt = inv_pose(map_struct.voxel_pos_to_world_pt(voxel_pos + local_pos));

        int u = __float2int_rd(fx * pt.x / pt.z + cx + 0.5);
        int v = __float2int_rd(fy * pt.y / pt.z + cy + 0.5);
        if (u < 0 || v < 0 || u > depth.cols - 1 || v > depth.rows - 1)
            continue;

        float dist = depth.ptr(v)[u];
        if (isnan(dist) || dist < 1e-2 || dist > param.zmax_update_ || dist < param.zmin_update_)
            continue;

        float sdf = dist - pt.z;
        if (sdf < -dist_thresh)
            continue;

        sdf = min(1.0f, sdf * inv_dist_thresh);
        const int local_idx = map_struct.local_pos_to_local_idx(local_pos);
        Voxel &voxel = map_struct.voxels_[current.ptr_ + local_idx];

        auto sdf_p = voxel.get_sdf();
        int weight_p = voxel.get_weight();

        // update colour
        auto colour_new = image.ptr(v)[u];
        auto colour_p = voxel.rgb_;

        if (weight_p == 0)
        {
            voxel.set_sdf(sdf);
            voxel.set_weight(1);
            // voxel.rgb_w_ = 1;
            voxel.rgb_ = colour_new;
            continue;
        }

        // fuse depth
        unsigned char w_curr = min(255, weight_p + 1);
        sdf_p = (sdf_p * weight_p + sdf) / (weight_p + 1);
        voxel.set_sdf(sdf_p);
        voxel.set_weight(w_curr);

        // fuse colour
        // unsigned char colour_w = min(255, colour_w_p + 1);
        colour_p = make_uchar3((colour_p * (float)weight_p + colour_new * 1.0f) / ((float)weight_p + 1));
        voxel.rgb_ = colour_p;
        // voxel.rgb_w_ = colour_w;
    }
}

void update(MapStruct map_struct,
            const cv::cuda::GpuMat depth,
            const cv::cuda::GpuMat image,
            const cv::cuda::GpuMat normal,
            const Sophus::SE3d &frame_pose,
            const IntrinsicMatrix K,
            cv::cuda::GpuMat &cv_flag,
            cv::cuda::GpuMat &cv_pos_array,
            uint &visible_block_count)
{
    if (cv_flag.empty())
        cv_flag.create(1, state.num_total_hash_entries_, CV_8UC1);
    if (cv_pos_array.empty())
        cv_pos_array.create(1, state.num_total_hash_entries_, CV_32SC1);

    thrust::device_ptr<uchar> flag(cv_flag.ptr<uchar>());
    thrust::device_ptr<int> pos_array(cv_pos_array.ptr<int>());

    const int cols = depth.cols;
    const int rows = depth.rows;

    dim3 thread(8, 8);
    dim3 block(div_up(cols, thread.x), div_up(rows, thread.y));

    create_blocks_kernel<<<block, thread>>>(map_struct, depth, K.invfx, K.invfy, K.cx, K.cy, frame_pose, flag.get());

    thread = dim3(MAX_THREAD);
    block = dim3(div_up(state.num_total_hash_entries_, thread.x));

    check_visibility_flag_kernel<<<block, thread>>>(map_struct, flag.get(), frame_pose.inverse(), cols, rows, K.fx, K.fy, K.cx, K.cy);
    thrust::exclusive_scan(flag, flag + state.num_total_hash_entries_, pos_array);
    copy_visible_block_kernel<<<block, thread>>>(map_struct.hash_table_, map_struct.visible_block_pos_, flag.get(), pos_array.get());
    visible_block_count = pos_array[state.num_total_hash_entries_ - 1];

    safe_call(cudaMemcpy(map_struct.visible_block_count_, &visible_block_count, sizeof(uint), cudaMemcpyHostToDevice));
    if (visible_block_count == 0)
        return;

    thread = dim3(8, 8);
    block = dim3(visible_block_count);

    // update_map_kernel<<<block, thread>>>(map_struct, depth, frame_pose.inverse(), K.fx, K.fy, K.cx, K.cy);
    update_map_with_colour_kernel<<<block, thread>>>(map_struct, depth, image, frame_pose.inverse(), K.fx, K.fy, K.cx, K.cy);
    safe_call(cudaDeviceSynchronize());
}

} // namespace map
}