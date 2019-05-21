#include "pose_estimation.h"
#include "vector_math.h"
#include "cuda_utils.h"
#include "reduce_sum.h"

namespace fusion
{

template <int rows, int cols>
void inline create_jtjjtr(cv::Mat &host_data, float *host_a, float *host_b)
{
    int shift = 0;
    for (int i = 0; i < rows; ++i)
        for (int j = i; j < cols; ++j)
        {
            float value = host_data.ptr<float>(0)[shift++];
            if (j == rows)
                host_b[i] = value;
            else
                host_a[j * rows + i] = host_a[i * rows + j] = value;
        }
}

struct RgbReduction
{
    __device__ bool find_corresp(int &x, int &y)
    {
        float4 pt = last_vmap.ptr(y)[x];
        if (pt.w < 0 || isnan(pt.x))
            return false;

        i_l = last_image.ptr(y)[x];
        if (!isfinite(i_l))
            return false;

        p_transformed = pose(make_float3(pt));
        u0 = p_transformed.x / p_transformed.z * fx + cx;
        v0 = p_transformed.y / p_transformed.z * fy + cy;
        if (u0 >= 2 && u0 < cols - 2 && v0 >= 2 && v0 < rows - 2)
        {
            // int u = __float2int_rd(u0 + 0.5f);
            // int v = __float2int_rd(v0 + 0.5f);
            // i_c = curr_image.ptr(v)[u];
            // dx = dIdx.ptr(v)[u];
            // dy = dIdy.ptr(v)[u];

            i_c = interp2(curr_image, u0, v0);
            dx = interp2(dIdx, u0, v0);
            dy = interp2(dIdy, u0, v0);

            return (dx > 1 || dy > 1) && isfinite(i_c) && isfinite(dx) && isfinite(dy);
        }

        return false;
    }

    __device__ float interp2(cv::cuda::PtrStep<float> image, float &x, float &y)
    {
        int u = floor(x), v = floor(y);
        float coeff_x = x - u, coeff_y = y - v;
        return (image.ptr(v)[u] * (1 - coeff_x) + image.ptr(v)[u + 1] * coeff_x) * (1 - coeff_y) +
               (image.ptr(v + 1)[u] * (1 - coeff_x) + image.ptr(v + 1)[u + 1] * coeff_x) * coeff_y;
    }

    // __device__ float3 bilinear_interpolate_depth(cv::cuda::PtrStepSz<float4> depth, float x, float y)
    // {
    //     int u = floor(x), v = floor(y);
    //     float coeff_x = x - u, coeff_y = y - v;
    //     return (image.ptr(v)[u] * (1 - coeff_x) + image.ptr(v)[u + 1] * coeff_x) * (1 - coeff_y) + (image.ptr(v + 1)[u] * (1 - coeff_x) + image.ptr(v + 1)[u + 1] * coeff_x) * coeff_y;
    // }

    __device__ void compute_jacobian(int &k, float *sum)
    {
        int y = k / cols;
        int x = k - y * cols;

        bool corresp_found = find_corresp(x, y);
        float row[7] = {0, 0, 0, 0, 0, 0, 0};

        if (corresp_found)
        {
            // float3 left;
            // float z_inv = 1.0 / p_last.z;
            // left.x = dx * fx * z_inv;
            // left.y = dy * fy * z_inv;
            // left.z = -(left.x * p_last.x + left.y * p_last.y) * z_inv;
            // row[6] = i_c - i_l;

            // *(float3 *)&row[0] = left;
            // *(float3 *)&row[3] = cross(p_last, left);

            float3 left;
            float z_inv = 1.0 / p_transformed.z;
            left.x = dx * fx * z_inv;
            left.y = dy * fy * z_inv;
            left.z = -(left.x * p_transformed.x + left.y * p_transformed.y) * z_inv;
            row[6] = i_l - i_c;

            *(float3 *)&row[0] = left;
            *(float3 *)&row[3] = cross(p_transformed, left);
        }

        int count = 0;
#pragma unroll
        for (int i = 0; i < 7; ++i)
#pragma unroll
            for (int j = i; j < 7; ++j)
                sum[count++] = row[i] * row[j];

        sum[count] = (float)corresp_found;
    }

    __device__ __forceinline__ void operator()()
    {
        float sum[29] = {0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0};

        float val[29];
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            compute_jacobian(i, val);
#pragma unroll
            for (int j = 0; j < 29; ++j)
                sum[j] += val[j];
        }

        BlockReduce<float, 29>(sum);

        if (threadIdx.x == 0)
#pragma unroll
            for (int i = 0; i < 29; ++i)
                out.ptr(blockIdx.x)[i] = sum[i];
    }

    int cols, rows, N;
    float u0, v0;
    DeviceMatrix3x4 pose;
    float fx, fy, cx, cy, invfx, invfy;
    cv::cuda::PtrStep<float4> point_cloud, last_vmap;
    cv::cuda::PtrStep<float> last_image, curr_image;
    cv::cuda::PtrStep<float> dIdx, dIdy;
    cv::cuda::PtrStep<float> out;
    float3 p_transformed, p_last;

private:
    float i_c, i_l, dx, dy;
};

// struct RgbReduction
// {
//     __device__ bool find_corresp(int &x, int &y)
//     {
//         p_transformed = pose(make_float3(point_cloud.ptr(y)[x]));
//         u0 = p_transformed.x / p_transformed.z * fx + cx;
//         v0 = p_transformed.y / p_transformed.z * fy + cy;
//         if (u0 >= 0 && u0 < cols - 1 && v0 >= 0 && v0 < rows - 1)
//         {
//             i_c = curr_image.ptr(y)[x];
//             i_l = interp2(last_image, u0, v0);
//             dx = interp2(dIdx, u0, v0);
//             dy = interp2(dIdy, u0, v0);
//             p_last = make_float3(last_vmap.ptr((int)(v0 + 0.5))[(int)(u0 + 0.5)]);

//             // int u1 = (int)(u0 + 0.5f);
//             // int v1 = (int)(v0 + 0.5f);
//             // if (u1 < 0 || v1 < 0 || u1 >= cols || v1 >= rows)
//             //     return false;

//             // i_c = curr_image.ptr(y)[x];
//             // i_l = last_image.ptr(v1)[u1];
//             // dx = dIdx.ptr(v1)[u1];
//             // dy = dIdy.ptr(v1)[u1];
//             // p_last = make_float3(last_vmap.ptr(v1)[u1]);

//             // if (norm(p_transformed - p_last) > 0.03f)
//             //     return false;

//             // return !isnan(p_last.x) && i_c > 0 && i_l > 0 && dx != 0 && dy != 0 && isfinite(i_c) && isfinite(i_l) && isfinite(dx) && isfinite(dy);
//             return !isnan(p_last.x) && (dx > 0 || dy > 0) && isfinite(i_c) && isfinite(i_l) && isfinite(dx) && isfinite(dy);
//         }

//         return false;
//     }

//     __device__ float interp2(cv::cuda::PtrStep<float> image, float &x, float &y)
//     {
//         int u = floor(x), v = floor(y);
//         float coeff_x = x - u, coeff_y = y - v;
//         return (image.ptr(v)[u] * (1 - coeff_x) + image.ptr(v)[u + 1] * coeff_x) * (1 - coeff_y) + (image.ptr(v + 1)[u] * (1 - coeff_x) + image.ptr(v + 1)[u + 1] * coeff_x) * coeff_y;
//     }

//     // __device__ float3 bilinear_interpolate_depth(cv::cuda::PtrStepSz<float4> depth, float x, float y)
//     // {
//     //     int u = floor(x), v = floor(y);
//     //     float coeff_x = x - u, coeff_y = y - v;
//     //     return (image.ptr(v)[u] * (1 - coeff_x) + image.ptr(v)[u + 1] * coeff_x) * (1 - coeff_y) + (image.ptr(v + 1)[u] * (1 - coeff_x) + image.ptr(v + 1)[u + 1] * coeff_x) * coeff_y;
//     // }

//     __device__ void compute_jacobian(int &k, float *sum)
//     {
//         int y = k / cols;
//         int x = k - y * cols;

//         bool corresp_found = find_corresp(x, y);
//         float row[7] = {0, 0, 0, 0, 0, 0, 0};

//         if (corresp_found)
//         {
//             float3 left;
//             float z_inv = 1.0 / p_last.z;
//             left.x = dx * fx * z_inv;
//             left.y = dy * fy * z_inv;
//             left.z = -(left.x * p_last.x + left.y * p_last.y) * z_inv;
//             row[6] = i_c - i_l;

//             *(float3 *)&row[0] = left;
//             *(float3 *)&row[3] = cross(p_last, left);

//             // float3 left;
//             // float z_inv = 1.0 / p_transformed.z;
//             // left.x = dx * fx * z_inv;
//             // left.y = dy * fy * z_inv;
//             // left.z = -(left.x * p_transformed.x + left.y * p_transformed.y) * z_inv;
//             // row[6] = i_c - i_l;

//             // *(float3 *)&row[0] = left;
//             // *(float3 *)&row[3] = cross(p_transformed, left);
//         }

//         int count = 0;
// #pragma unroll
//         for (int i = 0; i < 7; ++i)
// #pragma unroll
//             for (int j = i; j < 7; ++j)
//                 sum[count++] = row[i] * row[j];

//         sum[count] = (float)corresp_found;
//     }

//     __device__ __forceinline__ void operator()()
//     {
//         float sum[29] = {0, 0, 0, 0, 0, 0, 0, 0,
//                          0, 0, 0, 0, 0, 0, 0, 0,
//                          0, 0, 0, 0, 0, 0, 0, 0,
//                          0, 0, 0, 0, 0};

//         float val[29];
//         for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
//         {
//             compute_jacobian(i, val);
// #pragma unroll
//             for (int j = 0; j < 29; ++j)
//                 sum[j] += val[j];
//         }

//         BlockReduce<float, 29>(sum);

//         if (threadIdx.x == 0)
// #pragma unroll
//             for (int i = 0; i < 29; ++i)
//                 out.ptr(blockIdx.x)[i] = sum[i];
//     }

//     int cols, rows, N;
//     float u0, v0;
//     DeviceMatrix3x4 pose;
//     float fx, fy, cx, cy, invfx, invfy;
//     cv::cuda::PtrStep<float4> point_cloud, last_vmap;
//     cv::cuda::PtrStep<float> last_image, curr_image;
//     cv::cuda::PtrStep<float> dIdx, dIdy;
//     cv::cuda::PtrStep<float> out;
//     float3 p_transformed, p_last;

// private:
//     float i_c, i_l, dx, dy;
// };

__global__ void rgb_reduce_kernel(RgbReduction rr)
{
    rr();
}

void rgb_reduce(const cv::cuda::GpuMat &curr_intensity,
                const cv::cuda::GpuMat &last_intensity,
                const cv::cuda::GpuMat &last_vmap,
                const cv::cuda::GpuMat &curr_vmap,
                const cv::cuda::GpuMat &intensity_dx,
                const cv::cuda::GpuMat &intensity_dy,
                cv::cuda::GpuMat &sum,
                cv::cuda::GpuMat &out,
                const Sophus::SE3d &pose,
                const IntrinsicMatrix K,
                float *jtj, float *jtr,
                float *residual)
{
    int cols = curr_intensity.cols;
    int rows = curr_intensity.rows;

    RgbReduction rr;
    rr.cols = cols;
    rr.rows = rows;
    rr.N = cols * rows;
    rr.last_image = last_intensity;
    rr.curr_image = curr_intensity;
    rr.point_cloud = curr_vmap;
    rr.last_vmap = last_vmap;
    rr.dIdx = intensity_dx;
    rr.dIdy = intensity_dy;
    rr.pose = pose;
    rr.fx = K.fx;
    rr.fy = K.fy;
    rr.cx = K.cx;
    rr.cy = K.cy;
    rr.invfx = K.invfx;
    rr.invfy = K.invfy;
    rr.out = sum;

    rgb_reduce_kernel<<<96, 224>>>(rr);
    safe_call(cudaDeviceSynchronize());
    safe_call(cudaGetLastError());

    cv::cuda::reduce(sum, out, 0, cv::REDUCE_SUM);

    safe_call(cudaDeviceSynchronize());
    safe_call(cudaGetLastError());

    cv::Mat host_data;
    out.download(host_data);
    create_jtjjtr<6, 7>(host_data, jtj, jtr);
    residual[0] = host_data.ptr<float>()[27];
    residual[1] = host_data.ptr<float>()[28];
}

struct ICPReduction
{
    __device__ __inline__ bool searchPoint(int &x, int &y, float3 &vcurr_g, float3 &vlast_g, float3 &nlast_g) const
    {
        float3 vlast_c = make_float3(last_vmap_.ptr(y)[x]);
        if (isnan(vlast_c.x))
            return false;

        vlast_g = pose(vlast_c);

        float invz = 1.0 / vlast_g.z;
        int u = __float2int_rd(vlast_g.x * invz * fx + cx + 0.5);
        int v = __float2int_rd(vlast_g.y * invz * fy + cy + 0.5);
        if (u < 0 || v < 0 || u >= cols || v >= rows)
            return false;

        vcurr_g = make_float3(curr_vmap_.ptr(v)[u]);

        float3 nlast_c = make_float3(last_nmap_.ptr(y)[x]);
        nlast_g = pose.rotate(nlast_c);

        float3 ncurr_g = make_float3(curr_nmap_.ptr(v)[u]);

        float dist = norm(vlast_g - vcurr_g);
        float sine = norm(cross(ncurr_g, nlast_g));

        return (sine < angleThresh && dist <= distThresh && !isnan(ncurr_g.x) && !isnan(nlast_g.x));
    }

    __device__ __inline__ void getRow(int &i, float *sum) const
    {
        int y = i / cols;
        int x = i - y * cols;

        bool found = false;
        float3 vcurr, vlast, nlast;
        found = searchPoint(x, y, vcurr, vlast, nlast);
        float row[7] = {0, 0, 0, 0, 0, 0, 0};

        if (found)
        {
            *(float3 *)&row[0] = nlast;
            *(float3 *)&row[3] = cross(vlast, nlast);
            row[6] = nlast * (vcurr - vlast);
        }

        int count = 0;
#pragma unroll
        for (int i = 0; i < 7; ++i)
        {
#pragma unroll
            for (int j = i; j < 7; ++j)
                sum[count++] = row[i] * row[j];
        }

        sum[count] = (float)found;
    }

    __device__ __inline__ void operator()() const
    {
        float sum[29] = {0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,
                         0, 0, 0, 0};

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        float val[29];
        for (; i < N; i += blockDim.x * gridDim.x)
        {
            getRow(i, val);
#pragma unroll
            for (int j = 0; j < 29; ++j)
                sum[j] += val[j];
        }

        BlockReduce<float, 29>(sum);

        if (threadIdx.x == 0)
        {
#pragma unroll
            for (int i = 0; i < 29; ++i)
                out.ptr(blockIdx.x)[i] = sum[i];
        }
    }

    DeviceMatrix3x4 pose;
    cv::cuda::PtrStep<float4> curr_vmap_, last_vmap_;
    cv::cuda::PtrStep<float4> curr_nmap_, last_nmap_;
    int cols, rows, N;
    float fx, fy, cx, cy;
    float angleThresh, distThresh;
    mutable cv::cuda::PtrStepSz<float> out;
};

// struct ICPReduction
// {
//     __device__ __inline__ bool searchPoint(int &x, int &y, float3 &vcurr_g, float3 &vlast_g, float3 &nlast_g) const
//     {

//         float3 vcurr_c = make_float3(curr_vmap_.ptr(y)[x]);
//         if (isnan(vcurr_c.x))
//             return false;

//         vcurr_g = pose(vcurr_c);

//         float invz = 1.0 / vcurr_g.z;
//         int u = (int)(vcurr_g.x * invz * fx + cx + 0.5);
//         int v = (int)(vcurr_g.y * invz * fy + cy + 0.5);
//         if (u < 0 || v < 0 || u >= cols || v >= rows)
//             return false;

//         vlast_g = make_float3(last_vmap_.ptr(v)[u]);

//         float3 ncurr_c = make_float3(curr_nmap_.ptr(y)[x]);
//         float3 ncurr_g = pose.rotate(ncurr_c);

//         nlast_g = make_float3(last_nmap_.ptr(v)[u]);

//         float dist = norm(vlast_g - vcurr_g);
//         float sine = norm(cross(ncurr_g, nlast_g));

//         return (sine < angleThresh && dist <= distThresh && !isnan(ncurr_c.x) && !isnan(nlast_g.x));
//     }

//     __device__ __inline__ void getRow(int &i, float *sum) const
//     {
//         int y = i / cols;
//         int x = i - y * cols;

//         bool found = false;
//         float3 vcurr, vlast, nlast;
//         found = searchPoint(x, y, vcurr, vlast, nlast);
//         float row[7] = {0, 0, 0, 0, 0, 0, 0};

//         if (found)
//         {
//             *(float3 *)&row[0] = nlast;
//             *(float3 *)&row[3] = cross(vlast, nlast);
//             row[6] = -nlast * (vcurr - vlast);
//         }

//         int count = 0;
// #pragma unroll
//         for (int i = 0; i < 7; ++i)
//         {
// #pragma unroll
//             for (int j = i; j < 7; ++j)
//                 sum[count++] = row[i] * row[j];
//         }

//         sum[count] = (float)found;
//     }

//     __device__ __inline__ void operator()() const
//     {
//         float sum[29] = {0, 0, 0, 0, 0,
//                          0, 0, 0, 0, 0,
//                          0, 0, 0, 0, 0,
//                          0, 0, 0, 0, 0,
//                          0, 0, 0, 0, 0,
//                          0, 0, 0, 0};

//         int i = blockIdx.x * blockDim.x + threadIdx.x;
//         float val[29];
//         for (; i < N; i += blockDim.x * gridDim.x)
//         {
//             getRow(i, val);
// #pragma unroll
//             for (int j = 0; j < 29; ++j)
//                 sum[j] += val[j];
//         }

//         BlockReduce<float, 29>(sum);

//         if (threadIdx.x == 0)
//         {
// #pragma unroll
//             for (int i = 0; i < 29; ++i)
//                 out.ptr(blockIdx.x)[i] = sum[i];
//         }
//     }

//     DeviceMatrix3x4 pose;
//     cv::cuda::PtrStep<float4> curr_vmap_, last_vmap_;
//     cv::cuda::PtrStep<float4> curr_nmap_, last_nmap_;
//     int cols, rows, N;
//     float fx, fy, cx, cy;
//     float angleThresh, distThresh;
//     mutable cv::cuda::PtrStepSz<float> out;
// };

__global__ void icp_reduce_kernel(const ICPReduction icp)
{
    icp();
}

void icp_reduce(const cv::cuda::GpuMat &curr_vmap,
                const cv::cuda::GpuMat &curr_nmap,
                const cv::cuda::GpuMat &last_vmap,
                const cv::cuda::GpuMat &last_nmap,
                cv::cuda::GpuMat &sum,
                cv::cuda::GpuMat &out,
                const Sophus::SE3d &pose,
                const IntrinsicMatrix K,
                float *jtj, float *jtr,
                float *residual)
{
    int cols = curr_vmap.cols;
    int rows = curr_vmap.rows;

    ICPReduction icp;
    icp.out = sum;
    icp.curr_vmap_ = curr_vmap;
    icp.curr_nmap_ = curr_nmap;
    icp.last_vmap_ = last_vmap;
    icp.last_nmap_ = last_nmap;
    icp.cols = cols;
    icp.rows = rows;
    icp.N = cols * rows;
    icp.pose = pose;
    icp.angleThresh = cos(30 * 3.14 / 180);
    icp.distThresh = 0.01;
    icp.fx = K.fx;
    icp.fy = K.fy;
    icp.cx = K.cx;
    icp.cy = K.cy;

    icp_reduce_kernel<<<96, 224>>>(icp);
    cv::cuda::reduce(sum, out, 0, cv::REDUCE_SUM);

    cv::Mat host_data;
    out.download(host_data);
    create_jtjjtr<6, 7>(host_data, jtj, jtr);
    residual[0] = host_data.ptr<float>()[27];
    residual[1] = host_data.ptr<float>()[28];
}

// TODO : Robust RGB Estimation

class SelectCorrespRGB
{
public:
    __device__ inline bool find_correspondence(const int &x, const int &y, float &curr_val, float &last_val, float &dx, float &dy, float4 &pt) const
    {
        if (x >= cols || y >= rows)
            return false;

        // reference point
        pt = last_vmap.ptr(y)[x];
        if (isnan(pt.x) || pt.w < 0)
            return false;

        // reference point in curr frame
        pt = T_last_curr(pt);

        // reference intensity
        last_val = last_intensity.ptr(y)[x];

        if (!isfinite(last_val))
            return false;

        auto u = fx * pt.x / pt.z + cx;
        auto v = fy * pt.y / pt.z + cy;
        if (u >= 1 && v >= 1 && u <= cols - 2 && v <= rows - 2)
        {
            curr_val = interpolate_bilinear(curr_intensity, u, v);
            dx = interpolate_bilinear(curr_intensity_dx, u, v);
            dy = interpolate_bilinear(curr_intensity_dy, u, v);

            // point selection criteria
            // TODO : Optimise this
            return (dx > 2 || dy > 2) &&
                   isfinite(curr_val) &&
                   isfinite(dx) && isfinite(dy);
        }

        return false;
    }

    __device__ float interpolate_bilinear(cv::cuda::PtrStep<float> image, float &x, float &y) const
    {
        int u = floor(x), v = floor(y);
        float coeff_x = x - u, coeff_y = y - v;
        return (image.ptr(v)[u] * (1 - coeff_x) + image.ptr(v)[u + 1] * coeff_x) * (1 - coeff_y) +
               (image.ptr(v + 1)[u] * (1 - coeff_x) + image.ptr(v + 1)[u + 1] * coeff_x) * coeff_y;
    }

    __device__ __inline__ void operator()() const
    {
        for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += blockDim.x * gridDim.x)
        {
            const int y = k / cols;
            const int x = k - y * cols;

            float4 pt;
            float curr_val, last_val, dx, dy;
            bool corresp_found = find_correspondence(x, y, curr_val, last_val, dx, dy, pt);

            if (corresp_found)
            {
                uint index = atomicAdd(num_corresp, 1);
                array_image[index] = make_float4(last_val, curr_val, dx, dy);
                array_point[index] = pt;
            }

            error_term[k] = last_val - curr_val;
            correp_mask[k] = corresp_found ? 1 : 0;
        }
    }

public:
    cv::cuda::PtrStep<float4> last_vmap;
    cv::cuda::PtrStep<float> last_intensity;
    cv::cuda::PtrStep<float> curr_intensity;
    cv::cuda::PtrStep<float> curr_intensity_dx;
    cv::cuda::PtrStep<float> curr_intensity_dy;
    float fx, fy, cx, cy;
    DeviceMatrix3x4 T_last_curr;
    int N, cols, rows;

    int *num_corresp;
    float4 *array_image;
    float4 *array_point;

    float *error_term;
    uchar *correp_mask;
};

class ComputeLeastSquaresRGB
{
    __device__ void compute_jacobian(int &k, float *sum)
    {
        float row[7] = {0, 0, 0, 0, 0, 0, 0};

        if (k < num_corresp)
        {
            float3 pt = make_float3(array_point[k]);
            float4 image = array_image[k];
            float3 left;
            float z_inv = 1.0 / pt.z;
            left.x = image.y * fx * z_inv;
            left.y = image.w * fy * z_inv;
            left.z = -(left.x * pt.x + left.y * pt.y) * z_inv;
            row[6] = image.x - image.y;

            *(float3 *)&row[0] = left;
            *(float3 *)&row[3] = cross(pt, left);
        }

        int count = 0;
        for (int i = 0; i < 7; ++i)
            for (int j = i; j < 7; ++j)
                sum[count++] = row[i] * row[j];
    }

    __device__ __forceinline__ void operator()()
    {
        float sum[29] = {0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0};

        float val[29];
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_corresp; i += blockDim.x * gridDim.x)
        {
            compute_jacobian(i, val);
#pragma unroll
            for (int j = 0; j < 29; ++j)
                sum[j] += val[j];
        }

        BlockReduce<float, 29>(sum);

        if (threadIdx.x == 0)
#pragma unroll
            for (int i = 0; i < 29; ++i)
                out.ptr(blockIdx.x)[i] = sum[i];
    }

    cv::cuda::PtrSz<float4> array_image;
    cv::cuda::PtrSz<float4> array_point;
    cv::cuda::PtrStep<float> out;
    uint num_corresp;
    float fx, fy;
};

__global__ void compute_rgb_corresp_kernel(SelectCorrespRGB delegate)
{
    delegate();
}

void compute_rgb_correspondence(const cv::cuda::GpuMat curr_intensity,
                                const cv::cuda::GpuMat last_intensity,
                                const cv::cuda::GpuMat curr_intensity_dx,
                                const cv::cuda::GpuMat curr_intensity_dy,
                                const cv::cuda::GpuMat last_vmap,
                                const Sophus::SE3d pose,
                                const IntrinsicMatrix K)
{
    auto rows = curr_intensity.rows;
    auto cols = curr_intensity.cols;

    SelectCorrespRGB rgb_struct;
    rgb_struct.curr_intensity = curr_intensity;
    rgb_struct.last_intensity = last_intensity;
    rgb_struct.curr_intensity_dx = curr_intensity_dx;
    rgb_struct.curr_intensity_dy = curr_intensity_dy;
    rgb_struct.last_vmap = last_vmap;
    rgb_struct.cx = K.cx;
    rgb_struct.cy = K.cy;
    rgb_struct.fx = K.fx;
    rgb_struct.fy = K.fy;
    rgb_struct.T_last_curr = pose;
    rgb_struct.rows = rows;
    rgb_struct.cols = cols;
    rgb_struct.N = rows * cols;

    cv::cuda::GpuMat array_point(1, cols * rows, CV_32FC4);
    cv::cuda::GpuMat array_image(1, cols * rows, CV_32FC4);
    cv::cuda::GpuMat num_corresp(1, 1, CV_32SC1);

    cv::cuda::GpuMat corresp_mask(1, cols * rows, CV_8UC1);
    cv::cuda::GpuMat error_term(1, cols * rows, CV_32FC1);

    rgb_struct.correp_mask = corresp_mask.ptr<uchar>();
    rgb_struct.error_term = error_term.ptr<float>();
    rgb_struct.num_corresp = num_corresp.ptr<int>();
    rgb_struct.array_image = array_image.ptr<float4>();
    rgb_struct.array_point = array_point.ptr<float4>();

    compute_rgb_corresp_kernel<<<96, 224>>>(rgb_struct);

    cv::Mat corresp_count(num_corresp);

    if (corresp_count.at<int>(0, 0) == 0)
        return;

    auto total = cv::cuda::sum(error_term, corresp_mask);

    std::cout << "total: " << total << " / num_corresp: " << corresp_count << std::endl;
}

} // namespace fusion