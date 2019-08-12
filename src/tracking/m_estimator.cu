#include "math/matrix_type.h"
#include "math/vector_type.h"
#include "tracking/m_estimator.h"
#include "utils/safe_call.h"
#include "tracking/reduce_sum.h"
#include "voxel_hashing/prefix_sum.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <opencv2/opencv.hpp>

namespace fusion
{

// TODO : Robust RGB Estimation
// STATUS: On halt
// struct RGBSelection
// {
//     __device__ inline bool find_corresp(
//         const int &x,
//         const int &y,
//         float &curr_val,
//         float &last_val,
//         float &dx,
//         float &dy,
//         Vector4f &pt) const
//     {
//         // reference point
//         pt = last_vmap.ptr(y)[x];
//         if (isnan(pt.x) || pt.w < 0)
//             return false;

//         // reference point in curr frame
//         pt = T_last_curr(pt);

//         // reference intensity
//         last_val = last_intensity.ptr(y)[x];

//         if (!isfinite(last_val))
//             return false;

//         auto u = fx * pt.x / pt.z + cx;
//         auto v = fy * pt.y / pt.z + cy;
//         if (u >= 1 && v >= 1 && u <= cols - 2 && v <= rows - 2)
//         {
//             curr_val = interpolate_bilinear(curr_intensity, u, v);
//             dx = interpolate_bilinear(curr_intensity_dx, u, v);
//             dy = interpolate_bilinear(curr_intensity_dy, u, v);

//             // point selection criteria
//             // TODO : Optimise this
//             return (dx > 2 || dy > 2) &&
//                    isfinite(curr_val) &&
//                    isfinite(dx) && isfinite(dy);
//         }

//         return false;
//     }

//     __device__ float interpolate_bilinear(cv::cuda::PtrStep<float> image, float &x, float &y) const
//     {
//         int u = std::floor(x), v = std::floor(y);
//         float coeff_x = x - u, coeff_y = y - v;
//         return (image.ptr(v)[u] * (1 - coeff_x) + image.ptr(v)[u + 1] * coeff_x) * (1 - coeff_y) +
//                (image.ptr(v + 1)[u] * (1 - coeff_x) + image.ptr(v + 1)[u + 1] * coeff_x) * coeff_y;
//     }

//     __device__ __inline__ void operator()() const
//     {
//         for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += blockDim.x * gridDim.x)
//         {
//             const int y = k / cols;
//             const int x = k - y * cols;

//             if (y >= cols || x >= rows)
//                 return;

//             Vector4f pt;
//             float curr_val, last_val, dx, dy;
//             bool corresp_found = find_corresp(x, y, curr_val, last_val, dx, dy, pt);

//             if (corresp_found)
//             {
//                 uint index = atomicAdd(num_corresp, 1);
//                 array_image[index] = Vector4f(last_val, curr_val, dx, dy);
//                 array_point[index] = pt;
//                 error_term[index] = pow(curr_val - last_val, 2);
//             }
//         }
//     }

//     cv::cuda::PtrStep<Vector4f> last_vmap;
//     cv::cuda::PtrStep<float> last_intensity;
//     cv::cuda::PtrStep<float> curr_intensity;
//     cv::cuda::PtrStep<float> curr_intensity_dx;
//     cv::cuda::PtrStep<float> curr_intensity_dy;
//     float fx, fy, cx, cy;
//     DeviceMatrix3x4 T_last_curr;
//     Matrix3x3f RLastCurr;
//     Vector3f TLastCurr;
//     int N, cols, rows;

//     int *num_corresp;
//     Vector4f *array_image;
//     Vector4f *array_point;

//     float *error_term;
// };

// __global__ void compute_rgb_corresp_kernel(RGBSelection delegate)
// {
//     delegate();
// }

// __global__ void compute_variance_kernel(float *error_term, float *variance_term, float mean, uint max_idx)
// {
//     uint x = threadIdx.x + blockDim.x * blockIdx.x;
//     if (x >= max_idx)
//         return;

//     variance_term[x] = pow(error_term[x] - mean, 2);
// }

// void compute_rgb_corresp(
//     const cv::cuda::GpuMat last_vmap,
//     const cv::cuda::GpuMat last_intensity,
//     const cv::cuda::GpuMat curr_intensity,
//     const cv::cuda::GpuMat curr_intensity_dx,
//     const cv::cuda::GpuMat curr_intensity_dy,
//     const Sophus::SE3d &frame_pose,
//     const IntrinsicMatrix K,
//     Vector4f *transformed_points,
//     Vector4f *image_corresp_data,
//     float *error_term_array,
//     float *variance_term_array,
//     float &mean_estimate,
//     float &stdev_estimated,
//     uint &num_corresp)
// {
//     auto cols = last_vmap.cols;
//     auto rows = last_vmap.rows;

//     RGBSelection delegate;
//     delegate.last_vmap = last_vmap;
//     delegate.last_intensity = last_intensity;
//     delegate.curr_intensity = curr_intensity;
//     delegate.curr_intensity_dx = curr_intensity_dx;
//     delegate.curr_intensity_dy = curr_intensity_dy;
//     delegate.T_last_curr = frame_pose;
//     delegate.array_image = image_corresp_data;
//     delegate.array_point = transformed_points;
//     delegate.error_term = error_term_array;
//     delegate.fx = K.fx;
//     delegate.fy = K.fy;
//     delegate.cx = K.cx;
//     delegate.cy = K.cy;
//     delegate.cols = cols;
//     delegate.rows = rows;
//     delegate.N = cols * rows;

//     safe_call(cudaMalloc(&delegate.num_corresp, sizeof(uint)));
//     safe_call(cudaMemset(delegate.num_corresp, 0, sizeof(uint)));

//     compute_rgb_corresp_kernel<<<96, 224>>>(delegate);

//     safe_call(cudaMemcpy(&num_corresp, delegate.num_corresp, sizeof(uint), cudaMemcpyDeviceToHost));

//     if (num_corresp <= 1)
//         return;

//     thrust::device_ptr<float> error_term(error_term_array);
//     thrust::device_ptr<float> variance_term(variance_term_array);

//     float sum_error = thrust::reduce(error_term, error_term + num_corresp);
//     mean_estimate = 0;
//     stdev_estimated = std::sqrt(sum_error / (num_corresp - 6));

//     // dim3 thread(MAX_THREAD);
//     // dim3 block(div_up(num_corresp, thread.x));

//     // compute_variance_kernel<<<block, thread>>>(error_term_array, variance_term_array, mean_estimate, num_corresp);
//     // float sum_variance = thrust::reduce(variance_term, variance_term + num_corresp);
//     // stdev_estimated = sqrt(sum_variance / (num_corresp - 1));

//     std::cout << "mean : " << mean_estimate << " stddev : " << stdev_estimated << " num_corresp : " << num_corresp << std::endl;

//     safe_call(cudaFree(delegate.num_corresp));
// }

// // TODO : Robust RGB Estimation
// // STATUS: On halt
// struct RGBLeastSquares
// {
//     cv::cuda::PtrStep<float> out;

//     Vector4f *transformed_points;
//     Vector4f *image_corresp_data;
//     float mean_estimated;
//     float stdev_estimated;
//     uint num_corresp;
//     float fx, fy;
//     size_t N;

//     __device__ void compute_jacobian(const int &k, float *sum)
//     {
//         float row[7] = {0, 0, 0, 0, 0, 0, 0};
//         float weight = 0;
//         if (k < num_corresp)
//         {
//             Vector3f p_transformed = ToVector3(transformed_points[k]);
//             Vector4f image = image_corresp_data[k];

//             float z_inv = 1.0 / p_transformed.z;
//             Vector3f left;
//             left.x = image.z * fx * z_inv;
//             left.y = image.w * fy * z_inv;
//             left.z = -(left.x * p_transformed.x + left.y * p_transformed.y) * z_inv;

//             float residual = image.y - image.x; // curr_val - last_val
//             float res_normalized = residual / stdev_estimated;
//             float threshold_huber = 1.345 * stdev_estimated;

//             if (fabs(res_normalized) < threshold_huber)
//                 weight = 1;
//             else
//                 weight = threshold_huber / fabs(res_normalized);

//             row[6] = (-residual);
//             // printf("%f, %f\n", res_normalized, threshold_huber);
//             *(Vector3f *)&row[0] = left;
//             *(Vector3f *)&row[3] = p_transformed.cross(left);
//         }

//         int count = 0;
// #pragma unroll
//         for (int i = 0; i < 7; ++i)
//         {
// #pragma unroll
//             for (int j = i; j < 7; ++j)
//                 sum[count++] = row[i] * row[j];
//         }
//     }

//     __device__ void operator()()
//     {
//         float sum[29] = {0, 0, 0, 0, 0,
//                          0, 0, 0, 0, 0,
//                          0, 0, 0, 0, 0,
//                          0, 0, 0, 0, 0,
//                          0, 0, 0, 0, 0,
//                          0, 0, 0, 0};

//         float val[29];
//         for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += blockDim.x * gridDim.x)
//         {
//             compute_jacobian(k, val);

// #pragma unroll
//             for (int i = 0; i < 29; ++i)
//                 sum[i] += val[i];
//         }

//         BlockReduce<float, 29>(sum);

//         if (threadIdx.x == 0)
//         {
// #pragma unroll
//             for (int i = 0; i < 29; ++i)
//                 out.ptr(blockIdx.x)[i] = sum[i];
//         }
//     }
// }; // struct RGBLeastSquares

// __global__ void compute_least_square_RGB_kernel(RGBLeastSquares delegate)
// {
//     delegate();
// }

// // TODO : Robust RGB Estimation
// // STATUS: On halt
// void compute_least_square_RGB(
//     const uint num_corresp,
//     Vector4f *transformed_points,
//     Vector4f *image_corresp_data,
//     const float mean_estimated,
//     const float stdev_estimated,
//     const IntrinsicMatrix K,
//     cv::cuda::GpuMat sum,
//     cv::cuda::GpuMat out,
//     float *hessian_estimated,
//     float *residual_estimated,
//     float *residual)
// {
//     RGBLeastSquares delegate;
//     delegate.fx = K.fx;
//     delegate.fy = K.fy;
//     delegate.out = sum;
//     delegate.N = num_corresp;
//     delegate.num_corresp = num_corresp;
//     delegate.image_corresp_data = image_corresp_data;
//     delegate.transformed_points = transformed_points;
//     delegate.mean_estimated = mean_estimated;
//     delegate.stdev_estimated = stdev_estimated;

//     compute_least_square_RGB_kernel<<<96, 224>>>(delegate);
//     cv::cuda::reduce(sum, out, 0, cv::REDUCE_SUM);

//     cv::Mat host_data;
//     out.download(host_data);
//     create_jtjjtr<6, 7>(host_data, hessian_estimated, residual_estimated);
//     residual[0] = host_data.ptr<float>()[27];
//     residual[1] = num_corresp;

//     // std::cout << residual[0] << " : " << residual[1] << std::endl;
// }

struct RgbReduction2
{
    __device__ bool find_corresp(int &x, int &y)
    {
        Vector4f pt = last_vmap.ptr(y)[x];
        if (pt.w < 0 || isnan(pt.x))
            return false;

        i_l = last_image.ptr(y)[x];
        if (!isfinite(i_l))
            return false;

        p_transformed = pose(ToVector3(pt));
        u0 = p_transformed.x / p_transformed.z * fx + cx;
        v0 = p_transformed.y / p_transformed.z * fy + cy;
        if (u0 >= 2 && u0 < cols - 2 && v0 >= 2 && v0 < rows - 2)
        {
            i_c = interp2(curr_image, u0, v0);
            dx = interp2(dIdx, u0, v0);
            dy = interp2(dIdy, u0, v0);

            return (dx > 0 || dy > 0) && isfinite(i_c) && isfinite(dx) && isfinite(dy);
        }

        return false;
    }

    __device__ float interp2(cv::cuda::PtrStep<float> image, float &x, float &y)
    {
        int u = std::floor(x), v = std::floor(y);
        float coeff_x = x - u, coeff_y = y - v;
        return (image.ptr(v)[u] * (1 - coeff_x) + image.ptr(v)[u + 1] * coeff_x) * (1 - coeff_y) +
               (image.ptr(v + 1)[u] * (1 - coeff_x) + image.ptr(v + 1)[u + 1] * coeff_x) * coeff_y;
    }

    __device__ void compute_jacobian(int &k, float *sum)
    {
        int y = k / cols;
        int x = k - y * cols;

        bool corresp_found = find_corresp(x, y);
        float row[7] = {0, 0, 0, 0, 0, 0, 0};

        if (corresp_found)
        {
            Vector3f left;
            float z_inv = 1.0 / p_transformed.z;
            left.x = dx * fx * z_inv;
            left.y = dy * fy * z_inv;
            left.z = -(left.x * p_transformed.x + left.y * p_transformed.y) * z_inv;

            float residual = i_c - i_l;

            float normalised_res = abs(residual) / stddev;

            // float weight = fabs(normalised_res) <= 4.6851 ? pow(1 - pow(normalised_res / 4.6851, 2), 2) : 0;
            // weight = sqrt(weight);
            float huber_th = 1.345 * stddev;

            // if (stddev < 10e-5)
            float weight = 1;

            if (fabs(residual) > huber_th && stddev > 10e-6)
            {
                weight = sqrtf(huber_th / fabs(residual));
            }

            row[6] = weight * (-residual);
            *(Vector3f *)&row[0] = weight * left;
            *(Vector3f *)&row[3] = weight * p_transformed.cross(left);
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
    Matrix3x4f pose;
    float fx, fy, cx, cy, invfx, invfy;
    cv::cuda::PtrStep<Vector4f> point_cloud, last_vmap;
    cv::cuda::PtrStep<float> last_image, curr_image;
    cv::cuda::PtrStep<float> dIdx, dIdy;
    cv::cuda::PtrStep<float> out;
    Vector3f p_transformed, p_last;
    float stddev;

private:
    float i_c, i_l, dx, dy;
};

// __global__ void rgb_reduce_kernel2(RgbReduction2 rr)
// {
//     rr();
// }

void rgb_step(const cv::cuda::GpuMat &curr_intensity,
              const cv::cuda::GpuMat &last_intensity,
              const cv::cuda::GpuMat &last_vmap,
              const cv::cuda::GpuMat &curr_vmap,
              const cv::cuda::GpuMat &intensity_dx,
              const cv::cuda::GpuMat &intensity_dy,
              cv::cuda::GpuMat &sum,
              cv::cuda::GpuMat &out,
              const float stddev_estimate,
              const Sophus::SE3d &pose,
              const IntrinsicMatrix K,
              float *jtj, float *jtr,
              float *residual)
{
    int cols = curr_intensity.cols;
    int rows = curr_intensity.rows;

    RgbReduction2 rr;
    rr.cols = cols;
    rr.rows = rows;
    rr.N = cols * rows;
    rr.last_image = last_intensity;
    rr.curr_image = curr_intensity;
    rr.point_cloud = curr_vmap;
    rr.last_vmap = last_vmap;
    rr.dIdx = intensity_dx;
    rr.dIdy = intensity_dy;
    rr.pose = pose.cast<float>().matrix3x4();
    rr.stddev = stddev_estimate;
    rr.fx = K.fx;
    rr.fy = K.fy;
    rr.cx = K.cx;
    rr.cy = K.cy;
    rr.invfx = K.invfx;
    rr.invfy = K.invfy;
    rr.out = sum;

    call_device_functor<<<96, 224>>>(rr);
    cv::cuda::reduce(sum, out, 0, cv::REDUCE_SUM);

    cv::Mat host_data;
    out.download(host_data);
    create_jtjjtr<6, 7>(host_data, jtj, jtr);
    residual[0] = host_data.ptr<float>()[27];
    residual[1] = host_data.ptr<float>()[28];
}

struct ResidualIntensity
{
    Vector2f dI;
    Vector2i uv;
    Vector4f point;
};

struct ComputeResidual
{
    ResidualIntensity *residuals;
    Matrix3x4f pose;
    int cols, rows;
    float *residual_abs;
    uint *num_residuals;
    float fx, fy, cx, cy;
    cv::cuda::PtrStep<Vector4f> last_vmap;
    cv::cuda::PtrStep<float> last_image, curr_image;
    cv::cuda::PtrStep<float> dIdx, dIdy;

    __device__ bool find_corresp(int &x, int &y, ResidualIntensity &res)
    {
        Vector4f pt = last_vmap.ptr(y)[x];
        if (pt.w < 0 || isnan(pt.x))
            return false;

        float val_dst = last_image.ptr(y)[x];
        if (!isfinite(val_dst))
            return false;

        Vector3f p_transformed = pose(ToVector3(pt));
        float u0 = p_transformed.x / p_transformed.z * fx + cx;
        float v0 = p_transformed.y / p_transformed.z * fy + cy;
        if (u0 >= 2 && u0 < cols - 2 && v0 >= 2 && v0 < rows - 2)
        {
            float val_src = interp2(curr_image, u0, v0);
            res.dI = Vector2f(interp2(dIdx, u0, v0), interp2(dIdy, u0, v0));
            res.point = Vector4f(p_transformed, val_src - val_dst);
            res.uv = Vector2i(u0, v0);
            return res.dI.norm() > 1 &&
                   isfinite(val_src) &&
                   isfinite(res.dI.x) &&
                   isfinite(res.dI.y);
        }

        return false;
    }

    __device__ float interp2(cv::cuda::PtrStep<float> image, float &x, float &y)
    {
        int u = std::floor(x), v = std::floor(y);
        float coeff_x = x - u, coeff_y = y - v;
        return (image.ptr(v)[u] * (1 - coeff_x) + image.ptr(v)[u + 1] * coeff_x) * (1 - coeff_y) +
               (image.ptr(v + 1)[u] * (1 - coeff_x) + image.ptr(v + 1)[u + 1] * coeff_x) * coeff_y;
    }

    __device__ bool compute_residual(int &k, ResidualIntensity &res)
    {
        int y = k / cols;
        int x = k - y * cols;

        if (find_corresp(x, y, res))
            return true;
        else
            return false;
    }

    __device__ void operator()()
    {
        int k = threadIdx.x + blockIdx.x * blockDim.x;
        if (k >= cols * rows)
            return;

        ResidualIntensity res;
        if (compute_residual(k, res))
        {
            uint offset = atomicAdd(num_residuals, 1);
            residuals[offset] = res;
            residual_abs[offset] = res.point.w * res.point.w;
        }
    }
};

struct SolveLinearSystem
{
    ResidualIntensity *residuals;
    uint num_residual;
    float std_dev;
    float fx, fy;
    float mean;
    cv::cuda::PtrStep<unsigned char> weight_map;
    cv::cuda::PtrStepSz<float> out;

    __device__ inline void compute_jacobian(int &k, float *val)
    {
        ResidualIntensity &res = residuals[k];

        Vector3f left;
        float row[7] = {0, 0, 0, 0, 0, 0, 0};

        float z_inv = 1.0 / res.point.z;
        left.x = res.dI.x * fx * z_inv;
        left.y = res.dI.y * fy * z_inv;
        left.z = -(left.x * res.point.x + left.y * res.point.y) * z_inv;

        float res_abs = fabs(res.point.w) / std_dev;

        // float weight = res_abs <= 4.6851 ? pow(1 - pow(res_abs / 4.6851, 2), 2) : 0;
        // weight = sqrt(weight);
        float huber_th = 1.345 * std_dev;
        float weight = res_abs > huber_th ? sqrt(huber_th / res_abs) : 1;
        if (weight != 1)
            weight_map.ptr(res.uv.y)[res.uv.x] = 255;
        // if (fabs(residual) > 1.4 * std_dev && stddev > 10e-6)
        // {
        //     weight = sqrtf(huber_th / fabs(residual));
        // }

        row[6] = weight * (-res.point.w);
        *(Vector3f *)&row[0] = weight * left;
        *(Vector3f *)&row[3] = weight * ToVector3(res.point).cross(left);

        int count = 0;
#pragma unroll
        for (int i = 0; i < 7; ++i)
#pragma unroll
            for (int j = i; j < 7; ++j)
                val[count++] = row[i] * row[j];
    }

    __device__ inline void operator()()
    {
        float sum[28] = {0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0};

        float val[28];
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_residual; i += blockDim.x * gridDim.x)
        {
            compute_jacobian(i, val);
#pragma unroll
            for (int j = 0; j < 28; ++j)
                sum[j] += val[j];
        }

        BlockReduce<float, 28>(sum);

        if (threadIdx.x == 0)
#pragma unroll
            for (int i = 0; i < 28; ++i)
                out.ptr(blockIdx.x)[i] = sum[i];
    }
};

float *residual_abs;
ResidualIntensity *residual_intensity;

void compute_residual(
    const cv::cuda::GpuMat &curr_intensity,
    const cv::cuda::GpuMat &last_intensity,
    const cv::cuda::GpuMat &last_vmap,
    const cv::cuda::GpuMat &intensity_dx,
    const cv::cuda::GpuMat &intensity_dy,
    const Sophus::SE3d &pose,
    const IntrinsicMatrix K,
    float *jtj, float *jtr,
    float *residual)
{
    int cols = curr_intensity.cols;
    int rows = curr_intensity.rows;

    ComputeResidual functor;
    functor.cols = cols;
    functor.rows = rows;
    functor.last_image = last_intensity;
    functor.curr_image = curr_intensity;
    functor.last_vmap = last_vmap;
    functor.dIdx = intensity_dx;
    functor.dIdy = intensity_dy;
    functor.pose = pose.cast<float>().matrix3x4();
    functor.fx = K.fx;
    functor.fy = K.fy;
    functor.cx = K.cx;
    functor.cy = K.cy;

    // float *residuals;
    uint *num_residuals;

    if (residual_abs == NULL)
        safe_call(cudaMalloc((void **)&residual_abs, sizeof(float) * 640 * 480));

    if (residual_intensity == NULL)
        safe_call(cudaMalloc((void **)&residual_intensity, sizeof(ResidualIntensity) * 640 * 480));

    // safe_call(cudaMalloc((void **)&residuals, sizeof(float) * cols * rows));
    safe_call(cudaMalloc((void **)&num_residuals, sizeof(uint)));
    safe_call(cudaMemset((void *)num_residuals, 0, sizeof(uint)));

    functor.residuals = residual_intensity;
    functor.residual_abs = residual_abs;
    functor.num_residuals = num_residuals;

    dim3 block(MAX_THREAD);
    dim3 grid(div_up(cols * rows, block.x));

    call_device_functor<<<grid, block>>>(functor);

    uint num = 0;
    safe_call(cudaMemcpy(&num, functor.num_residuals, sizeof(uint), cudaMemcpyDeviceToHost));
    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(residual_abs);
    thrust::sort(dev_ptr, dev_ptr + num);
    float median_val = 0;
    if (num % 2 == 0)
        median_val = sqrt((dev_ptr[num / 2 - 1] + dev_ptr[num / 2]) * 0.5f);
    else
        median_val = sqrt(dev_ptr[(num) / 2]);

    // std::cout << "median: " << median_val << "  num_corresp: " << num << " ratio: " << (float)num / (cols * rows) << std::endl;
    // float std_dev = 1.4826 * (1 + 5.f / (num - 6)) * sqrt(median_val); // for tukey's biweight
    float std_dev = 1.345 * (1 + 5.f / ((float)num - 6.f)) * sqrt(median_val); // for huber norm
    std::cout << "median: " << sqrt(median_val) << "stddev: " << std_dev << std::endl;

    block = dim3(MAX_THREAD);
    grid = dim3(div_up(num, block.x));

    cv::cuda::GpuMat sum(num, 28, CV_32FC1);
    cv::cuda::GpuMat out;
    sum.setTo(0);

    cv::cuda::GpuMat weightMap(rows, cols, CV_8UC1);
    weightMap.setTo(0);

    SolveLinearSystem solver;
    solver.num_residual = num;
    solver.std_dev = std_dev;
    solver.mean = median_val;
    solver.out = sum;
    solver.fx = K.fx;
    solver.fy = K.fy;
    solver.weight_map = weightMap;
    solver.residuals = residual_intensity;

    call_device_functor<<<96, 224>>>(solver);

    cv::cuda::reduce(sum, out, 0, cv::REDUCE_SUM);

    cv::Mat host_data(out);
    create_jtjjtr<6, 7>(host_data, jtj, jtr);
    residual[0] = host_data.ptr<float>()[27];
    residual[1] = num;

    if (weightMap.cols == 640)
    {
        cv::Mat img(weightMap);
        cv::imshow("img", img);
        // std::cout << img << std::endl;
        cv::waitKey(1);
    }

    // safe_call(cudaFree(residual_intensity));
    safe_call(cudaFree(num_residuals));
}

} // namespace fusion